/*
 * oSEM on OPS Particles
 * =====================
 *
 * A 2-D Synthetic Eddy Method inlet, with the eddies represented as OPS
 * particles rather than as an ops_dat on a second block.
 *
 * WHY
 * ---
 * The reference implementation (OPS/apps/c/oSEM) keeps eddies on an
 * ops_block of shape {eddies, 1}, fetches them to the host every step, and
 * hands seven arrays to compute_fluct as ops_arg_gbl. The kernel then loops
 * over all 1718 eddies at every one of the 15000 inlet nodes. Three problems:
 *
 *   1. ~99% of that work is wasted. An eddy of radius 0.2*delta = 1.4e-3
 *      covers only pi*r^2/(dy*dz) ~ 140 of the 15000 nodes.
 *   2. ops_partition splits the RANKS across blocks, not just the grid
 *      (ops_mpi_partition.cpp:250-313). With two blocks and two ranks the
 *      inlet plane lands entirely on one rank and is never parallelised.
 *   3. The eddy data reaches the kernel as a global array, which is exactly
 *      what OPS Particles exists to avoid.
 *
 * Here the eddies are particles: owned by the rank whose (y,z) subdomain they
 * occupy, found through the mapping's cell-linked list, and delivered to the
 * kernel as ops_arg_dat_particle. No host round-trip, no all-to-all, no
 * ops_arg_gbl for eddy data.
 *
 * BUILT UP FROM
 *   ../particle_tutorial_1_drift  -- declaration order, seeding, migration
 *   ../particle_tutorial_3_deposit -- the grid-outer scatter loop
 *
 * Build:  make oSEM_particles_dev_seq   (serial, no translator)
 *         make oSEM_particles_dev_mpi   (MPI, no translator)
 *         make oSEM_particles_mpi       (MPI, via the translator)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define OPS_2D

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>
#include <ops_grid_part_seq_v2.h>

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include "TBL_data.h"
#include "sem_constants.h"
#include "scatter_loop.h"
#include "grid_kernels.h"
#include "particle_kernels.h"

/* ================================================================== *
 *  Seeding
 * ================================================================== *
 *
 * Every rank walks the same global eddy list and keeps the ones inside its own
 * subdomain. The random stream is keyed on the GLOBAL eddy index, so the
 * initial eddy field is identical for any number of ranks.
 *
 * ops_fill_random_uniform() cannot be used for this. It offsets its seed by
 * rank (ops_lib_core.cpp:2576), so the field would change with the rank count,
 * and it returns NON-NEGATIVE ints only (:2604) -- the two bugs fixed in
 * oSEM commit 6a95ea939.
 *
 * The seeding contract, in this order:
 *   1. grow storage if the count exceeds particle->Nmax (OPS_MAX_PART = 1000,
 *      and there are 1718 eddies)
 *   2. write the dats directly -- particle dats are AoS,
 *      data[dim * particle_index + component]
 *   3. set particle->no_particles yourself; nothing infers it
 */
void seed_eddies(ops_particle particle, ops_dat pos, ops_dat px, ops_dat pr,
                 ops_dat pinc, ops_dat peps, ops_dat pseed,
                 unsigned int base_seed) {

  BoundingBox<double> *box = (BoundingBox<double> *)particle->box_block;

  if (eddies > (int)particle->Nmax)
    ops_particle_realloc_data(particle, eddies);

  double *d_pos = (double *)pos->data;
  double *d_x   = (double *)px->data;
  double *d_r   = (double *)pr->data;
  double *d_inc = (double *)pinc->data;
  int    *d_eps = (int *)peps->data;
  int    *d_sd  = (int *)pseed->data;

  const double radius = 0.2 * delta;
  const double increment = u0 * dt;

  int n = 0;
  for (int e = 0; e < eddies; e++) {

    /* Decorrelate consecutive indices before drawing, so eddy e and e+1 do
       not start from adjacent LCG states.
       UNSIGNED throughout -- see the note in KerConvectEddy. */
    unsigned int s = (base_seed + 2654435761u * (unsigned int)e) & 0x7fffffffu;

    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double xe = x_min + (x_max - x_min) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double ye = eddy_y_min + (eddy_y_max - eddy_y_min) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double ze = eddy_z_min + (eddy_z_max - eddy_z_min) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  const int ex = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  const int ey = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  const int ez = ((s >> 8) & 1u) ? 1 : -1;

    /* The library's own ownership predicate, so this cannot disagree with
       what the migration machinery believes. */
    const double yz[2] = {ye, ze};
    if (!box->isCoordinateInBoundingBox(yz)) continue;

    d_pos[2 * n]     = ye;
    d_pos[2 * n + 1] = ze;
    d_x[n]           = xe;
    d_r[n]           = radius;
    d_inc[n]         = increment;
    d_eps[3 * n]     = ex;
    d_eps[3 * n + 1] = ey;
    d_eps[3 * n + 2] = ez;
    d_sd[n]          = (int)s;
    n++;
  }

  particle->no_particles = n;
}

/* ================================================================== *
 *  The per-step map / migration cycle -- unchanged from tutorial 1
 * ================================================================== */
/* Counts steps on which the ranks did not agree on `decide`. */
long g_decide_disagreements = 0;

void update_maps(ops_particle particle,
                 ops_dat *dat_border,  int nborder,
                 ops_dat *dat_forward, int nforward) {

  int decide = ops_particle_update_map_lists_actual_hybrid(particle);

#ifdef OPS_MPI
  /* THE DECISION MUST BE COLLECTIVE.
   *
   * The two branches below post DIFFERENT communication patterns: the border
   * path exchanges nborder dats, the forward path nforward. If one rank takes
   * one branch while another takes the other, the sends and receives do not
   * match and the run deadlocks.
   *
   * ops_particle_update_map_lists_actual_hybrid() is documented as deciding
   * collectively, but measurably does not always do so here -- with eddies
   * respawning at random steps, np=4 and np=8 hung intermittently (niter 6, 8
   * and 12 hung while 5, 10 and 15 completed). Reducing with MPI_MAX makes any
   * rank that needs a rebuild force one everywhere, which is the safe
   * direction: the border path is the more thorough of the two.
   *
   * g_decide_disagreements records how often this actually mattered, so the
   * workaround is visible rather than silent. */
  int local = decide, agreed = 0, any = 0;
  MPI_Allreduce(&local, &agreed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local, &any,    1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (agreed != any) g_decide_disagreements++;
  decide = any;
#endif

  ops_particle_remove_delete_maps(particle, decide);

  if (decide)
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);
  else
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);

  ops_particle_reset_flags(particle, decide);
}

/* Rectangular search stencil of half-width (hy, hz). See the mapping
   declaration in main() for what the two uses of it mean. */
int *build_box_stencil(int hy, int hz, int &npoints) {
  npoints = (2 * hy + 1) * (2 * hz + 1);
  int *s = (int *)malloc(sizeof(int) * 2 * npoints);
  int k = 0;
  for (int i = -hy; i <= hy; i++)
    for (int j = -hz; j <= hz; j++) {
      s[2 * k] = i; s[2 * k + 1] = j; k++;
    }
  return s;
}

int total_particles(ops_particle particle) {
  int n = (int)particle->no_particles;
#ifdef OPS_MPI
  int tot = 0;
  MPI_Allreduce(&n, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return tot;
#else
  return n;
#endif
}

/* ================================================================== *
 *  Diagnostics -- the eddy field should fill its box and the signs
 *  should be balanced and mutually independent.
 * ================================================================== */
int report_eddy_field(ops_particle particle, ops_dat pos, ops_dat px,
                      ops_dat peps) {

  const int n = (int)particle->no_particles;
  const double *d_pos = (const double *)pos->data;
  const double *d_x   = (const double *)px->data;
  const int    *d_eps = (const int *)peps->data;

  double lo[3] = { 1e30,  1e30,  1e30};
  double hi[3] = {-1e30, -1e30, -1e30};
  int pos_count[3] = {0, 0, 0};

  for (int p = 0; p < n; p++) {
    const double v[3] = {d_x[p], d_pos[2 * p], d_pos[2 * p + 1]};
    for (int d = 0; d < 3; d++) {
      if (v[d] < lo[d]) lo[d] = v[d];
      if (v[d] > hi[d]) hi[d] = v[d];
    }
    for (int d = 0; d < 3; d++) if (d_eps[3 * p + d] > 0) pos_count[d]++;
  }

  int ntot = n;
#ifdef OPS_MPI
  double glo[3], ghi[3];
  int gpos[3], gn;
  MPI_Allreduce(lo, glo, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(hi, ghi, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(pos_count, gpos, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int d = 0; d < 3; d++) { lo[d] = glo[d]; hi[d] = ghi[d]; pos_count[d] = gpos[d]; }
  ntot = gn;
#endif

  const double blo[3] = {x_min, eddy_y_min, eddy_z_min};
  const double bhi[3] = {x_max, eddy_y_max, eddy_z_max};
  const char  *nm[3]  = {"x", "y", "z"};

  ops_printf("\n--- eddy field (%d eddies) ---\n", ntot);
  int ok = (ntot == eddies);

  for (int d = 0; d < 3; d++) {
    const double cover = 100.0 * (hi[d] - lo[d]) / (bhi[d] - blo[d]);
    ops_printf("  %s  min=% .6e max=% .6e   box [% .6e, % .6e]  coverage %5.1f%%\n",
               nm[d], lo[d], hi[d], blo[d], bhi[d], cover);
    /* A well-spread field covers nearly all of its box, and must never
       leave it -- anything outside would have been deleted. */
    ok = ok && (cover > 95.0) && (lo[d] >= blo[d] - 1e-12)
            && (hi[d] <= bhi[d] + 1e-12);
  }

  for (int d = 0; d < 3; d++) {
    const double frac = 100.0 * pos_count[d] / ntot;
    ops_printf("  eps_%s  +1: %5d (%5.1f%%)   -1: %5d (%5.1f%%)\n",
               nm[d], pos_count[d], frac, ntot - pos_count[d], 100.0 - frac);
    ok = ok && (frac > 45.0) && (frac < 55.0);
  }

  return ok;
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  /* ---- Parameters, identical to OPS/apps/c/oSEM -------------------- */

  u0    = 823.6;
  dt    = 0.00000002;
  delta = 0.007;
  r_max = 0.41 * delta;
  ny    = 100;
  nz    = 150;
  niter = 200;
  write_output_file = 0;
  TI    = 0.01;

  x_min   = -r_max;
  x_max   =  r_max;
  x_plane =  0.0;

  y_min = 0.0;
  y_max = 0.009;
  z_min = 0.0;
  z_max = 0.05;

  eddy_y_min = y_min;
  eddy_y_max = y_max + r_max;
  eddy_z_min = z_min - r_max;
  eddy_z_max = z_max + r_max;

  vol = fabs((x_max - x_min) * (y_max - y_min + 2 * r_max)
                             * (z_max - z_min + 2 * r_max));
  rep_radius = 0.2 * delta;
  calc_eddies(eddies, vol, rep_radius);

  unsigned int base_seed = 2893328493u;
  int respawn_jump = 1;   /* 2 = global respawn, 1 = local, 0 = none */
  int validate = 0;
  int halomul = 1;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-niter") == 0 && i + 1 < argc) niter = atoi(argv[++i]);
    else if (strcmp(argv[i], "-nojump") == 0) respawn_jump = 0;
    else if (strcmp(argv[i], "-globaljump") == 0) respawn_jump = 2;
    else if (strcmp(argv[i], "-validate") == 0) validate = 1;
    else if (strcmp(argv[i], "-halomul") == 0 && i + 1 < argc) halomul = atoi(argv[++i]);
    else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
      base_seed = (unsigned int)strtoul(argv[++i], NULL, 10);
  }

  ops_decl_const("u0", 1, "double", &u0);
  ops_decl_const("dt", 1, "double", &dt);
  ops_decl_const("delta", 1, "double", &delta);
  ops_decl_const("r_max", 1, "double", &r_max);
  ops_decl_const("TI", 1, "double", &TI);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("y_min", 1, "double", &y_min);
  ops_decl_const("y_max", 1, "double", &y_max);
  ops_decl_const("z_min", 1, "double", &z_min);
  ops_decl_const("z_max", 1, "double", &z_max);
  ops_decl_const("x_min", 1, "double", &x_min);
  ops_decl_const("x_max", 1, "double", &x_max);
  ops_decl_const("x_plane", 1, "double", &x_plane);
  ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
  ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
  ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
  ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);
  ops_decl_const("eddies", 1, "int", &eddies);

  /* ---- 1. Block, grids -------------------------------------------- */

  ops_block block = ops_decl_block(2, "inlet");

  int inlet_size[] = {ny, nz};

  /* Particle coordinate dat, sized to the COARSE MAPPING.
   *
   * Two constraints:
   *   - it must span the whole eddy box (the box is exactly
   *     [first node .. last node]), so at least ny+1 / nz+1 nodes;
   *   - the strided ops_decl_mapping requires size % stride == 0
   *     (ops_particle_lib_core.cpp:1844), and 101 and 151 are both prime.
   * So round the node count UP to the next multiple of the stride. Spacing
   * stays dy/dz, so bin k still groups the same cells as inlet nodes; the box
   * just extends a little past the eddy range, which is harmless. */
  const int stride_y = 12;   /* >= ceil(r/dy) = 12 */
  const int stride_z = 5;    /* >= ceil(r/dz) = 4  */

  /* Bin counts are POWERS OF TWO, and the coords dat is sized from them.
   *
   * Under MPI the stride must divide EVERY RANK'S LOCAL subdomain, not merely
   * the global grid: ops_mpi_particle_host.cpp:1229 requires
   * local_fine_size == stride * local_bin_count exactly, and throws
   * "A non-uniform map grid is generated" otherwise. A power-of-two bin count
   * keeps that true for every power-of-two rank count.
   *
   * The dat is therefore LARGER than the eddy box needs. That is harmless: it
   * only has to span the box (the box is exactly [first node .. last node]) and
   * keep spacing dy/dz so bin k still groups the same cells as inlet node k.
   * The extra bins simply stay empty. */
  const int nbin_y = 16;     /* 16 * 12 = 192 cells >= 100 needed */
  const int nbin_z = 32;     /* 32 *  5 = 160 cells >= 150 needed */
  const int part_ny = nbin_y * stride_y + 1;
  const int part_nz = nbin_z * stride_z + 1;
  int part_size[]  = {part_ny, part_nz};

  if ((part_ny - 1) % stride_y != 0 || (part_nz - 1) % stride_z != 0) {
    ops_printf("FATAL: coords dat %d x %d nodes gives %d x %d cells, not "
               "divisible by stride %d x %d\n", part_ny, part_nz,
               part_ny - 1, part_nz - 1, stride_y, stride_z);
    ops_exit();
    return 1;
  }
  int base[]       = {0, 0};
  int d_m[]        = {0, 0};
  int d_p[]        = {0, 0};

  double *null_dbl = NULL;
  int    *null_int = NULL;

  ops_dat d_y_inlet = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p,
                                   null_dbl, "double", "y_inlet");
  ops_dat d_z_inlet = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p,
                                   null_dbl, "double", "z_inlet");
  ops_dat d_part_coords = ops_decl_dat(block, 2, part_size, base, d_m, d_p,
                                       null_dbl, "double", "part_coords");

  /* Reynolds stress factors and the fluctuation field. */
  ops_dat d_a11 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a11");
  ops_dat d_a21 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a21");
  ops_dat d_a22 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a22");
  ops_dat d_a31 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a31");
  ops_dat d_a32 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a32");
  ops_dat d_a33 = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "a33");

  ops_dat d_uprime = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "uprime");
  ops_dat d_vprime = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "vprime");
  ops_dat d_wprime = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "wprime");

  /* Reference fields, only filled under -validate. */
  ops_dat d_uref = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "uref");
  ops_dat d_vref = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "vref");
  ops_dat d_wref = ops_decl_dat(block, 1, inlet_size, base, d_m, d_p, null_dbl, "double", "wref");

  /* ---- 2. Stencils ------------------------------------------------- */

  int s2d_00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2d_00, "0,0");

  /* Search half-widths in cells: an eddy of radius r reaches ceil(r/dy) cells
     in y and ceil(r/dz) in z. For the defaults that is (12, 4), so a
     25 x 9 = 225-point stencil. */
  const double part_dy = (eddy_y_max - eddy_y_min) / (double)ny;
  const double part_dz = (eddy_z_max - eddy_z_min) / (double)nz;
  const int hy = halomul * (int)ceil(rep_radius / part_dy);
  const int hz = halomul * (int)ceil(rep_radius / part_dz);

  /* COARSE MAPPING.
   *
   * A bin-per-cell mapping needs a ghost band `hy` cells deep, and the particle
   * ghost band is only correct to depth ONE (particle_tutorial_3_deposit
   * README section 3a: exact in serial at any depth, wrong under MPI beyond
   * depth 1). So the bins are made ~one eddy radius across instead, and a +/-1
   * stencil then covers the radius with a one-bin ghost band -- the only
   * configuration that is correct under MPI (section 3b). */
  int map_stride[] = {stride_y, stride_z};
  int nsten = 0;
  int *s2d_box = build_box_stencil(1, 1, nsten);   /* 3x3 bins */
  ops_stencil S2D_BOX =
      ops_decl_prolong_stencil(2, nsten, s2d_box, map_stride, "eddy_reach");

  /* ---- 3. Bounding box, particle set, dats ------------------------- */

  double dx_box[] = {0.0, 0.0};
  BoundingBox<double> *box = ops_create_bounding_box(block, d_part_coords, 2,
                                                     dx_box);

  ops_particle eddy_parts = ops_decl_particle(block, "eddies", box);

  /* Position is privileged: one per particle set, dim == block dim, declared
     before any mapping. It holds (y, z) only -- x rides along as an ordinary
     dat because it is not a search dimension. */
  ops_dat p_pos  = ops_decl_particle_pos_dat(eddy_parts, 2, base, null_dbl,
                                             "double", "eddy_yz");
  ops_dat p_x    = ops_decl_particle_dat(eddy_parts, 1, base, null_dbl,
                                         "double", "eddy_x");
  ops_dat p_r    = ops_decl_particle_dat(eddy_parts, 1, base, null_dbl,
                                         "double", "eddy_r");
  ops_dat p_inc  = ops_decl_particle_dat(eddy_parts, 1, base, null_dbl,
                                         "double", "eddy_increment");
  ops_dat p_eps  = ops_decl_particle_dat(eddy_parts, 3, base, null_int,
                                         "int", "eddy_eps");
  ops_dat p_seed = ops_decl_particle_dat(eddy_parts, 1, base, null_int,
                                         "int", "eddy_seed");

  /* ---- 4. Mapping -------------------------------------------------- *
   * S2D_BOX does a DIFFERENT job here than in the coupling loop: it sets the
   * bin array's halo depth, i.e. how far outside its subdomain a rank tracks
   * ghost eddies. It must reach at least as far as the coupling loop reads,
   * so the same object serves both -- but they are not the same mechanism. */

  ops_particle_mapping map = ops_decl_mapping(eddy_parts, d_part_coords,
                                              S2D_BOX, map_stride,
                                              OPS_WITH_VIRTUAL,
                                              OPS_UNIFORM_STAG, 1);

  /* border  : every dat whose value must survive a change of owning rank
   * forward : the cheap per-step refresh of the existing ghost layer
   * Counts computed, never hand-typed. */
  ops_dat dat_border[]  = {p_pos, p_x, p_r, p_inc, p_eps, p_seed};
  ops_dat dat_forward[] = {p_pos, p_x, p_r, p_eps};

  const int nborder  = sizeof(dat_border)  / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);

  /* ---- 5. Partition, fill grids, set up particles ------------------ */

  ops_partition("");

  int inlet_range[] = {0, ny, 0, nz};
  int part_range[]  = {0, part_ny, 0, part_nz};

  ops_par_loop(KerInitInletGrid, "KerInitInletGrid", block, 2, inlet_range,
               ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());

  /* Spacing must stay dy/dz -- the dat is bigger than the eddy box, not
     stretched over it, otherwise bins would no longer align with inlet cells. */
  double part_dxy[2] = {part_dy, part_dz};
  ops_par_loop(KerInitPartCoords, "KerInitPartCoords", block, 2, part_range,
               ops_arg_dat(d_part_coords, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(part_dxy, 2, "double", OPS_READ),
               ops_arg_idx());

  ops_par_loop(instantiate_RST_TBL, "instantiate_RST_TBL", block, 2, inlet_range,
               ops_arg_dat(d_a11, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_a21, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_a22, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_a31, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_a32, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_a33, 1, S2D_00, "double", OPS_RW),
               ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_READ),
               ops_arg_gbl(y_inp,  260, "double", OPS_READ),
               ops_arg_gbl(uu_inp, 260, "double", OPS_READ),
               ops_arg_gbl(uv_inp, 260, "double", OPS_READ),
               ops_arg_gbl(vv_inp, 260, "double", OPS_READ),
               ops_arg_gbl(ww_inp, 260, "double", OPS_READ));

  /* Only now do real coordinates exist, so only now can the box be derived.
     Calling this before the loops above yields a box of all-zeros and
     "Defined bounding box of non-positive volume". */
  ops_particle_setup_partition();

  seed_eddies(eddy_parts, p_pos, p_x, p_r, p_inc, p_eps, p_seed, base_seed);
  ops_particle_setup_maps_with_dats(eddy_parts, dat_border, nborder);

  ops_printf("oSEM on OPS Particles\n");
  ops_printf("inlet %d x %d, %d eddies, r = %.4e\n", ny, nz, eddies, rep_radius);
  ops_printf("coords %d x %d, coarse bins %d x %d cells -> %d x %d bins, "
             "3x3 stencil, reach %d x %d cells (need %d x %d)\n",
             part_ny, part_nz, stride_y, stride_z,
             part_ny / stride_y, part_nz / stride_z,
             stride_y, stride_z, hy, hz);
  ops_printf("dy = %.4e (%d cells/radius), dz = %.4e (%d cells/radius)\n",
             part_dy, hy, part_dz, hz);

  int ok = report_eddy_field(eddy_parts, p_pos, p_x, p_eps);
  ops_printf("STAGE A (seeding)      : %s\n", ok ? "PASS" : "FAIL");

  /* ---- 6. Time loop ------------------------------------------------ */

  double range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  /* Scratch for the -validate reference only. */
  double worst_rel = 0.0;
  double *ref_x = NULL, *ref_y = NULL, *ref_z = NULL, *ref_r = NULL;
  int *ref_ex = NULL, *ref_ey = NULL, *ref_ez = NULL;
  if (validate) {
    ref_x = (double *)malloc(sizeof(double) * eddies);
    ref_y = (double *)malloc(sizeof(double) * eddies);
    ref_z = (double *)malloc(sizeof(double) * eddies);
    ref_r = (double *)malloc(sizeof(double) * eddies);
    ref_ex = (int *)malloc(sizeof(int) * eddies);
    ref_ey = (int *)malloc(sizeof(int) * eddies);
    ref_ez = (int *)malloc(sizeof(int) * eddies);
    ops_printf("\n--- STAGE C: particle scatter vs oSEM reference ---\n");
  }

  /* This rank's own slice of the eddy box, used by the local-respawn mode.
   *
   * INTERSECTED with the true eddy box. The bounding box now follows the
   * coordinate dat, which is deliberately larger than the eddy range (its size
   * is driven by the bin count, see above). Respawning into the raw local box
   * would scatter eddies into the empty margin beyond eddy_y_max/eddy_z_max. */
  double lbox[4] = {fmax(box->getLocalMin().x, eddy_y_min),
                    fmin(box->getLocalMax().x, eddy_y_max),
                    fmax(box->getLocalMin().y, eddy_z_min),
                    fmin(box->getLocalMax().y, eddy_z_max)};

  /* A rank whose slice lies entirely in the margin owns no eddies and must not
     respawn into an inverted interval. */
  if (lbox[1] <= lbox[0]) { lbox[0] = eddy_y_min; lbox[1] = eddy_y_max; }
  if (lbox[3] <= lbox[2]) { lbox[2] = eddy_z_min; lbox[3] = eddy_z_max; }

  for (int step = 1; step <= niter; step++) {

    ops_particle_par_loop(KerConvectEddy, "KerConvectEddy", eddy_parts, 2,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_x,    1, "double", eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_pos,  2, "double", eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_r,    1, "double", eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_inc,  1, "double", eddy_parts, map, OPS_READ),
                          ops_arg_dat_particle(p_eps,  3, "int",    eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_seed, 1, "int",    eddy_parts, map, OPS_RW),
                          ops_arg_gbl(&respawn_jump, 1, "int", OPS_READ),
                          ops_arg_gbl(lbox, 4, "double", OPS_READ));

    update_maps(eddy_parts, dat_border, nborder, dat_forward, nforward);

    /* ---- STAGE C: the fluctuation field ---------------------------- *
     * Zero first (the scatter kernel can only accumulate), then scatter
     * from the eddies binned near each node. Every eddy quantity arrives
     * as ops_arg_dat_particle -- no ops_arg_gbl for eddy data anywhere. */

    ops_reduction r_visit = ops_decl_reduction_handle(sizeof(double), "double", "nvisit");
    ops_reduction r_hitp  = ops_decl_reduction_handle(sizeof(double), "double", "nhitp");
    ops_reduction r_hitr  = ops_decl_reduction_handle(sizeof(double), "double", "nhitr");

    ops_par_loop(KerZeroFluct, "KerZeroFluct", block, 2, inlet_range,
                 ops_arg_dat(d_uprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(d_vprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(d_wprime, 1, S2D_00, "double", OPS_WRITE));

    ops_par_scatter_loop(KerComputeFluct, "KerComputeFluct", eddy_parts, map,
                         S2D_BOX, 2, inlet_range,
                         ops_arg_dat(d_uprime, 1, S2D_00, "double", OPS_INC),
                         ops_arg_dat(d_vprime, 1, S2D_00, "double", OPS_INC),
                         ops_arg_dat(d_wprime, 1, S2D_00, "double", OPS_INC),
                         ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a11, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a21, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a22, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a31, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a32, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat(d_a33, 1, S2D_00, "double", OPS_READ),
                         ops_arg_dat_particle(p_pos, 2, "double", eddy_parts, map, OPS_READ),
                         ops_arg_dat_particle(p_x,   1, "double", eddy_parts, map, OPS_READ),
                         ops_arg_dat_particle(p_r,   1, "double", eddy_parts, map, OPS_READ),
                         ops_arg_dat_particle(p_eps, 3, "int",    eddy_parts, map, OPS_READ),
                         ops_arg_reduce(r_visit, 1, "double", OPS_INC),
                         ops_arg_reduce(r_hitp,  1, "double", OPS_INC));

    if (validate) {
      /* Reference: oSEM's formulation, from the SAME eddy field. Serial
         only -- it needs every eddy in one host array, which is exactly the
         global gather the particle path exists to avoid. */
      const int np = (int)eddy_parts->no_particles;
      const double *dp = (const double *)p_pos->data;
      const double *dx = (const double *)p_x->data;
      const double *dr = (const double *)p_r->data;
      const int    *de = (const int *)p_eps->data;

      for (int e = 0; e < np; e++) {
        ref_x[e] = dx[e];  ref_y[e] = dp[2 * e];  ref_z[e] = dp[2 * e + 1];
        ref_r[e] = dr[e];
        ref_ex[e] = de[3 * e]; ref_ey[e] = de[3 * e + 1]; ref_ez[e] = de[3 * e + 2];
      }

      ops_par_loop(KerComputeFluctRef, "KerComputeFluctRef", block, 2, inlet_range,
                   ops_arg_dat(d_uref, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(d_vref, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(d_wref, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a11, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a21, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a22, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a31, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a32, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_a33, 1, S2D_00, "double", OPS_READ),
                   ops_arg_gbl(ref_x, eddies, "double", OPS_READ),
                   ops_arg_gbl(ref_y, eddies, "double", OPS_READ),
                   ops_arg_gbl(ref_z, eddies, "double", OPS_READ),
                   ops_arg_gbl(ref_r, eddies, "double", OPS_READ),
                   ops_arg_gbl(ref_ex, eddies, "int", OPS_READ),
                   ops_arg_gbl(ref_ey, eddies, "int", OPS_READ),
                   ops_arg_gbl(ref_ez, eddies, "int", OPS_READ),
                   ops_arg_reduce(r_hitr, 1, "double", OPS_INC));

      double h_diff = 0.0, h_val = 0.0, h_part = 0.0;
      ops_reduction r_diff = ops_decl_reduction_handle(sizeof(double), "double", "maxdiff");
      ops_reduction r_val  = ops_decl_reduction_handle(sizeof(double), "double", "maxval");
      ops_reduction r_part = ops_decl_reduction_handle(sizeof(double), "double", "maxpart");

      ops_par_loop(KerCompareFluct, "KerCompareFluct", block, 2, inlet_range,
                   ops_arg_dat(d_uprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_vprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_wprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_uref, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_vref, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(d_wref, 1, S2D_00, "double", OPS_READ),
                   ops_arg_reduce(r_diff, 1, "double", OPS_MAX),
                   ops_arg_reduce(r_val,  1, "double", OPS_MAX),
                   ops_arg_reduce(r_part, 1, "double", OPS_MAX));

      ops_reduction_result(r_diff, &h_diff);
      ops_reduction_result(r_val,  &h_val);
      ops_reduction_result(r_part, &h_part);

      double v = 0, hp = 0, hr = 0;
      ops_reduction_result(r_visit, &v);
      ops_reduction_result(r_hitp, &hp);
      ops_reduction_result(r_hitr, &hr);
      ops_printf("  step %4d  visits=%.0f  particle hits=%.0f  reference hits=%.0f\n",
                 step, v, hp, hr);

      const double rel = (h_val > 0.0) ? h_diff / h_val : 0.0;
      if (rel > worst_rel) worst_rel = rel;

      if (step % 20 == 0 || step == 1)
        ops_printf("  step %4d  max|particle| = %.4e  max|reference| = %.4e  "
                   "max|diff| = %.4e  relative = %.3e\n",
                   step, h_part, h_val, h_diff, rel);
    }
  }

  /* ---- 7. Check ---------------------------------------------------- *
   * A crossing takes 2*r_max/(u0*dt) ~ 348 steps, so a run of a few hundred
   * genuinely exercises recycling: eddies must respawn, be re-binned after
   * teleporting, and migrate between ranks -- all without losing any. */

  /* Field checksum -- the MPI check for Stage C.
   * With -nojump the eddies never change (y,z), so nothing migrates and the
   * field is fully deterministic; seeding is keyed on the global eddy index, so
   * it is identical at any rank count. Any difference between rank counts is
   * therefore a real defect in the scatter loop or its halo. */
  {
    double h_sum2 = 0.0, h_amax = 0.0;
    ops_reduction r_s = ops_decl_reduction_handle(sizeof(double), "double", "sum2");
    ops_reduction r_m = ops_decl_reduction_handle(sizeof(double), "double", "amax");
    ops_par_loop(KerFluctChecksum, "KerFluctChecksum", block, 2, inlet_range,
                 ops_arg_dat(d_uprime, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(d_vprime, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(d_wprime, 1, S2D_00, "double", OPS_READ),
                 ops_arg_reduce(r_s, 1, "double", OPS_INC),
                 ops_arg_reduce(r_m, 1, "double", OPS_MAX));
    ops_reduction_result(r_s, &h_sum2);
    ops_reduction_result(r_m, &h_amax);
    ops_printf("FIELD CHECKSUM         : sum|q|^2 = %.15e   max = %.15e\n",
               h_sum2, h_amax);
  }

  int ok_b = report_eddy_field(eddy_parts, p_pos, p_x, p_eps);
  const int nfinal = total_particles(eddy_parts);

  ops_printf("\n---------------------------------------------\n");
  ops_printf("steps run              : %d\n", niter);
  ops_printf("decide disagreements   : %ld  (steps where ranks differed)\n",
             g_decide_disagreements);
  ops_printf("eddies expected        : %d\n", eddies);
  ops_printf("eddies found           : %d\n", nfinal);
  ops_printf("STAGE B (convect+recycle): %s\n",
             (ok_b && nfinal == eddies) ? "PASS" : "FAIL");
  if (validate)
    ops_printf("STAGE C (compute_fluct): %s   worst relative difference %.3e\n",
               (worst_rel < 1e-12) ? "PASS" : "FAIL", worst_rel);
  ops_printf("---------------------------------------------\n");

  free(s2d_box);
  ops_exit();
  return (ok && ok_b && nfinal == eddies) ? 0 : 1;
}
