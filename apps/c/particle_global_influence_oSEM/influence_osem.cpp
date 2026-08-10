/*
 * OPS Particles -- oSEM with the eddies as particles
 * ==================================================
 *
 * A port of apps/c/oSEM (2-D inlet plane) in which the synthetic eddies are
 * OPS PARTICLES rather than grid dats on a second block, structured like
 * apps/c/particle_global_influence.
 *
 * THE THREE KERNELS, AND WHAT THEY CORRESPOND TO
 *
 *   before the loop   KerInitGrid / KerInitRST / KerInitEddy
 *                       <- instantiate_grid / instantiate_RST /
 *                          instantiate_eddies
 *   in the loop       KerConvectEddies      <- convect_eddies
 *                       the "advance" kernel: purely per-eddy, no coupling
 *   in the loop       KerComputeFluct       <- compute_fluct
 *                       the "influence" kernel: every inlet node sums over
 *                       EVERY eddy in the domain
 *
 * WHY THIS IS NOT JUST A RESTRUCTURING
 *
 *   oSEM hands the eddy state to compute_fluct by calling ops_dat_fetch_data
 *   on each of seven eddy dats and passing the host arrays as ops_arg_gbl.
 *   Under MPI that is wrong, not merely partial. ops_dat_fetch_data
 *   (ops_mpi_rt_support.cpp:2125) copies only THIS RANK's slice of the
 *   decomposed eddy block, and writes it starting at offset 0 -- it computes a
 *   displacement, `ldisp`, and then never uses it in the memcpy. compute_fluct
 *   then loops over the GLOBAL eddy count, so every index past the local slice
 *   reads uninitialised heap on the first step and stale values afterwards.
 *   Each rank therefore builds the inlet from a different, partly garbage set
 *   of eddies. It is correct only at np = 1.
 *
 *   Here the eddies are gathered with an array-valued ops_reduction: each rank
 *   INCs its own eddies into slots picked by global id, and the MPI_Allreduce
 *   inside ops_reduction_result returns the COMPLETE list, in id order, on
 *   every rank. That is the same array shape compute_fluct already wanted, so
 *   the kernel body ports across unchanged -- and it is correct at any rank
 *   count.
 *
 * TWO DELIBERATE DEPARTURES FROM THE REFERENCE
 *
 *   1. Randoms. oSEM calls ops_fill_random_uniform into int dats on the eddy
 *      block once per quantity per step. Eddies here migrate between ranks, so
 *      a rank-indexed random dat would hand a migrating eddy someone else's
 *      stream. Each eddy carries its own LCG state instead, so the stream
 *      belongs to the eddy and is identical at any rank count. This also
 *      sidesteps a known defect: ops_fill_random_uniform on an int dat never
 *      returns a negative value, so oSEM's `(rng < 0) ? -1 : 1` sign draws are
 *      always +1.
 *
 *   2. One block, not two. oSEM has an inlet_block and an eddy_block. The
 *      eddies are particles here, so they live inside the inlet block, and its
 *      grid already spans the full eddy box -- oSEM's instantiate_grid runs
 *      from z_min - r_max to z_max + r_max for exactly that reason. The
 *      bounding box OPS derives from the coordinate dat is what decides
 *      whether an eddy is inside the domain, so it must cover everywhere an
 *      eddy may legally be.
 *
 * THE RECYCLE IS A TELEPORT -- READ THIS BEFORE RUNNING UNDER MPI
 *
 *   convect_eddies recycles an eddy leaving the downstream face by giving it a
 *   fresh random (y, z). As a particle operation that is a jump to an
 *   arbitrary point in the plane, and OPS particle migration only hands a
 *   particle to a NEIGHBOURING rank. The drift tutorial measured what that
 *   costs (silent loss at np = 8, segfault at np = 3 for a full-width jump),
 *   and the two supported alternatives are both blocked here: halo groups
 *   express a fixed translation, not a random one, and runtime insert/delete
 *   does not compile with rearrange_for_removal hanging under MPI.
 *
 *   The kernel is written faithfully rather than worked around, and the eddy
 *   population is counted every step through an OPS reduction, so any loss is
 *   reported rather than silent. See the README for what actually happens.
 *
 * OPTIONS
 *   -niter N     timesteps                                default 2000
 *   -ny N -nz N  inlet plane resolution                   default 100 x 150
 *   -nprint N    report interval                          default 200
 *   -nout N      write an HDF5 frame every N steps        default 0 (off)
 *   -rst tbl|iso boundary-layer profile or isotropic       default tbl
 *
 * Build:  make influence_osem_dev_seq / _dev_mpi
 * Run:    ./influence_osem_dev_seq
 *         OMP_NUM_THREADS=1 mpirun -np 4 ./influence_osem_dev_mpi
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define OPS_2D

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>

#include "osem_constants.h"
#include "TBL_data.h"   /* tabulated boundary-layer Reynolds stresses */
#include "osem_common.h"
#include "particle_kernels.h"
#include "grid_kernels.h"
#include "ops_particle_random.h"
#include "osem_io.h"

typedef double Real;

/* ------------------------------------------------------------------ *
 * PROF_SLOTS -- why the stress-profile reduction is a fixed size
 * ------------------------------------------------------------------ *
 * KerFluctProfile accumulates 6 stresses per wall-normal row, so the natural
 * dimension is 6*(ny+1). It cannot be written that way: the OPS translator
 * parses the ops_arg_reduce dimension with parseIntLiteral
 * (ops_translator/ops-translator/cpp/parser.py:353), which accepts an
 * INTEGER_LITERAL or a unary +/- literal and raises "Expected int expression"
 * on anything else. `6 * (ny + 1)` is a BINARY_OPERATOR, so the translator
 * build fails to parse while the seq/dev builds compile it happily -- which is
 * how it went unnoticed.
 *
 * So the reduction is declared once at a fixed maximum and the loop names that
 * literal. Rows above 6*(ny+1) are never written and stay zero; the driver
 * reads only the first 6*(ny+1).
 *
 * KEEP THIS NUMBER SMALL. The translator UNROLLS the reduction: the generated
 * kernel carries one scalar local and one write-back per slot, plus an OpenMP
 * reduction clause naming every one of them. A first attempt at 6150 (ny up to
 * 1024) generated a 24819-line kernel that had not finished compiling at -O3
 * after 500 s; 606 generates ~2400 lines and builds in seconds. Compile time
 * scales with the literal, so raise it only as far as an actual grid needs.
 *
 * 606 = 6 * (100 + 1), the default ny. Three things must stay in step: this
 * macro, the literal in the ops_arg_reduce call, and PROF_MAX_NY. main()
 * checks ny against PROF_MAX_NY at startup and refuses rather than silently
 * truncating the profile.
 *
 * The restriction bites only with the tabulated RST, since that is the only
 * mode that runs KerFluctProfile; -rst iso leaves ny unbounded.
 */
#define PROF_SLOTS 606
#define PROF_MAX_NY 100

/* Every constant and run option is declared in osem_constants.h and given its
   value at the top of main -- nothing is defined at file scope. */

/* Optional per-row dump of the time-averaged shear profile, for averaging
   ACROSS realisations. The in-run deviation flattens onto a per-seed floor
   because the eddy (y,z) trajectories are deterministic, so no single run can
   distinguish that floor from a systematic error in a21; the ensemble mean
   over seeds can, because a lattice residual averages away and a bias does
   not. See the shear-stress block at the end of main. */
static const char *shear_dump = NULL;

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-niter") && i + 1 < argc) niter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-ny") && i + 1 < argc) ny = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nz") && i + 1 < argc) nz = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nprint") && i + 1 < argc) nprint = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nout") && i + 1 < argc) nout = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-rst") && i + 1 < argc) use_tbl = strcmp(argv[++i], "iso") != 0;
    /* A different seed is a different REALISATION of the same flow. The
       statistical checks below are only meaningful against their own
       realisation-to-realisation scatter, and this is how to measure it. */
    else if (!strcmp(argv[i], "-seed") && i + 1 < argc)
      seed_gbl = (unsigned int)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-dumpshear") && i + 1 < argc)
      shear_dump = argv[++i];
    else if (!strcmp(argv[i], "-rng") && i + 1 < argc) {
      const char *m = argv[++i];
      rng_method = !strcmp(m, "minstd") ? OPS_PRNG_MINSTD
                                 : (!strcmp(m, "shared") ? OPS_PRNG_SHARED
                                                         : OPS_PRNG_MT19937);
    }
  }
}

/* ================================================================== *
 *  Seeding
 * ================================================================== *
 * Positions must be written by hand -- the particle count is set here and
 * nothing infers it. As in the other apps in this series, every rank walks the
 * whole global sequence and keeps only the eddies inside its own subdomain, so
 * the index into that sequence is a global id: unique, dense in [0,neddy),
 * identical at any rank count, and free.
 *
 * The eddy's LCG seed is derived from its id, so its random stream is a
 * property of the eddy rather than of where it happens to live.
 */
static void seed_eddies(ops_particle particle, ops_dat pos, ops_dat gid,
                        int neddy) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  if (neddy > (int)particle->Nmax) ops_particle_realloc_data(particle, neddy);

  Real *xp = (Real *)pos->data;
  int *ip = (int *)gid->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    /* Same counter-based generator the fill uses, so the initial positions are
       drawn from the identical stream: counter 0, components 0 and 1. */
    const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) *
                                      ops_prandom_uniform(seed_gbl, i, 0u, 0);
    const Real z = eddy_z_min + (eddy_z_max - eddy_z_min) *
                                      ops_prandom_uniform(seed_gbl, i, 0u, 1);

    if (y < lo[0] || y >= hi[0] || z < lo[1] || z >= hi[1]) continue;

    xp[2 * n] = y;
    xp[2 * n + 1] = z;
    ip[n] = i;
    n++;
  }

  particle->no_particles = n;
}

/* Host copy of the kernel's interpolation, for the profile check below. Kept
   deliberately separate rather than shared: if the two ever disagree, the test
   has caught something real. */
static Real tbl_at(const double *tab, Real y) {
  int idx = 0;   /* must match KerInitRST_TBL exactly, including this
                    initialisation -- see the hazard note in grid_kernels.h */
  for (int i = 1; i < ntbl - 1; i++)
    if ((y - y_inp[i]) < 0) { idx = i - 1; break; }
  const Real w = (y - y_inp[idx]) / (y_inp[idx + 1] - y_inp[idx]);
  return tab[idx] + w * (tab[idx + 1] - tab[idx]);
}

static void update_maps(ops_particle particle, ops_dat *db, int nb,
                        ops_dat *df, int nf) {
  int decide = ops_particle_update_map_lists_actual_hybrid(particle);
  ops_particle_remove_delete_maps(particle, decide);
  if (decide) ops_particle_intrablock_border_map_update(particle, db, nb);
  else ops_particle_intrablock_forward_map_update(particle, df, nf);
  ops_particle_reset_flags(particle, decide);
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  /* ---- 1. geometry, exactly oSEM's ------------------------------- */

  /* ---- 1. constants ---------------------------------------------- *
   * Values are assigned here, not at file scope, so the whole configuration
   * of a run reads top-to-bottom in one place -- oSEM's arrangement. The
   * variables themselves are declared in osem_constants.h and registered with
   * ops_decl_const further down.
   */

  u0 = 823.6;
  dt = 0.00000002;
  delta = 0.007;
  ti = 0.01;

  y_min = 0.0;   y_max = 0.009;
  z_min = 0.0;   z_max = 0.05;

  ny = 100;
  nz = 150;

  seed_gbl = 2893328493u;   /* oSEM's seed_gbl */
  niter = 2000;
  nprint = 200;
  nout = 0;
  /* minstd_rand per particle: 130x faster than mt19937 (0.08 vs 10.99 ms/step)
     at the same rank invariance, and measured clean on the correlation that
     matters -- see ops_particle_random.h. Override with -rng. */
  rng_method = OPS_PRNG_MINSTD;
  ntbl = 260;      /* rows in TBL_data.h */
  use_tbl = 1;     /* boundary-layer profile by default; -rst iso for the
                      uniform one, whose all-equal-rms invariant is a useful
                      cross-check */

  /* Command line overrides the defaults above, before anything is derived
     from them. */
  parse_args(argc, argv);

  /* The stress-profile reduction is a fixed size (see PROF_SLOTS), so ny has a
     hard ceiling. Fail loudly rather than silently truncating the profile. */
  if (use_tbl && ny > PROF_MAX_NY) {
    ops_printf("error: -ny %d exceeds PROF_MAX_NY = %d (the stress-profile\n"
               "reduction is fixed at %d slots because the OPS translator\n"
               "requires a literal ops_arg_reduce dimension). Raise both\n"
               "PROF_SLOTS and the literal in the KerFluctProfile loop.\n",
               ny, PROF_MAX_NY, PROF_SLOTS);
    ops_exit();
    return 1;
  }

  /* Derived geometry. r_max is the eddy search radius: it sets the streamwise
     extent of the eddy box and the padding on the plane, because an eddy
     further away than that contributes exactly nothing. */
  r_max = 0.41 * delta;

  x_min = -r_max;
  x_max = r_max;
  x_plane = 0.0;

  eddy_y_min = y_min;
  eddy_y_max = y_max + r_max;
  eddy_z_min = z_min - r_max;
  eddy_z_max = z_max + r_max;

  eddy_radius = 0.2 * delta;
  increment = u0 * dt;
  shape_norm = 1.0 / 1.5829045;
  u0ti = u0 * ti;

  /* One eddy per cube of side eddy_radius, over the box volume.
   *
   * KEPT AS THE REFERENCE HAS IT, KNOWINGLY. The y extent below pads BOTH
   * sides by r_max, but the eddy box pads only the TOP (eddy_y_min = y_min,
   * eddy_y_max = y_max + r_max). So `vol` describes a box 1.2418x larger in y
   * than the one the eddies occupy, the eddy count is that much too high, and
   * every Reynolds stress is inflated with it -- measured <S^2> = 1.16 where
   * it must be 1, i.e. ~8 % high in rms.
   *
   * Do NOT "fix" this without asking. It is oSEM's
   * (apps/c/oSEM/OPS_oSEM.cpp:43-49) and the port matches it deliberately, so
   * that numbers from the two apps stay comparable. The corrected form is
   *   (y_max - y_min + r_max)
   * which makes vol exactly (x_max-x_min)(eddy_y_max-eddy_y_min)
   * (eddy_z_max-eddy_z_min); it was tried and measured, and Part 3 of
   * UNDERSTANDING_oSEM.md records what it changes. */
  vol = fabs((x_max - x_min) * (y_max - y_min + 2 * r_max) *
             (z_max - z_min + 2 * r_max));
  eddies = (int)trunc(vol / pow(eddy_radius, 3));

  /* One flow-through takes (x_max - x_min) / increment steps. Size the
     transverse drift so an eddy can cross the box in about that time -- fast
     enough to decorrelate, still well under one cell per step, which is what
     keeps migration local. */
  const Real nflow = (x_max - x_min) / increment;
  vt_y = (eddy_y_max - eddy_y_min) / nflow;
  vt_z = (eddy_z_max - eddy_z_min) / nflow;

  /* ---- 2. block, grid dats, stencils ----------------------------- */

  ops_block block = ops_decl_block(2, "osem_block");

  int size[] = {ny + 1, nz + 1};
  int base[] = {0, 0};
  int d_m[] = {-1, -1};
  int d_p[] = {1, 1};
  Real *nd = NULL;
  int *ni = NULL;

  ops_dat crd = ops_decl_dat(block, 2, size, base, d_m, d_p, nd, "double", "crd");
  ops_dat a11 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a11");
  ops_dat a21 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a21");
  ops_dat a22 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a22");
  ops_dat a31 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a31");
  ops_dat a32 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a32");
  ops_dat a33 = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "a33");
  ops_dat uprime = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "uprime");
  ops_dat vprime = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "vprime");
  ops_dat wprime = ops_decl_dat(block, 1, size, base, d_m, d_p, nd, "double", "wprime");

  int s00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s00, "0,0");
  int s9[] = {-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1};
  ops_stencil S2D_9pt = ops_decl_stencil(2, 9, s9, "9pt");

  /* ---- 3. reductions --------------------------------------------- */

  ops_reduction h_all = ops_decl_reduction_handle(
      eddies * NCOMP * sizeof(double), "double", "all_eddies");
  ops_reduction h_cnt =
      ops_decl_reduction_handle(sizeof(int), "int", "eddy_count");
  ops_reduction h_stat =
      ops_decl_reduction_handle(4 * sizeof(double), "double", "fluct_stats");
  ops_reduction h_sync = ops_decl_reduction_handle(sizeof(int), "int", "sync");
  /* Sized from PROF_SLOTS, not from ny, because the ops_arg_reduce that fills
     it has to name the same number -- and there it must be a bare literal.
     See the note on PROF_SLOTS. Slots above 6*(ny+1) stay zero. */
  ops_reduction h_prof = ops_decl_reduction_handle(
      PROF_SLOTS * sizeof(double), "double", "profile");

  /* ---- 4. the particle set --------------------------------------- */

  Real dxb[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, crd, 2, dxb);
  ops_particle particle = ops_decl_particle(block, "eddies", box);

  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 2, base, nd, "double",
                                            "position");
  ops_dat p_x = ops_decl_particle_dat(particle, 1, base, nd, "double", "x");
  ops_dat p_r = ops_decl_particle_dat(particle, 1, base, nd, "double", "radius");
  ops_dat p_eps = ops_decl_particle_dat(particle, 3, base, nd, "double", "eps");
  ops_dat p_vt = ops_decl_particle_dat(particle, 2, base, nd, "double", "vt");
  ops_dat p_gid = ops_decl_particle_dat(particle, 1, base, ni, "int", "gid");
  /* Six independent uniforms per eddy per step, filled by the driver before
     the kernels -- the same pattern as oSEM's ops_fill_random_uniform. */
  ops_dat p_rnd = ops_decl_particle_dat(particle, 6, base, nd, "double", "rnd");

  ops_particle_mapping map = ops_decl_mapping(
      particle, crd, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  /* p_rnd is deliberately NOT in either list: it is re-filled from scratch
     every step, keyed on global id, so there is nothing to preserve across a
     migration. */
  ops_dat dat_border[] = {p_pos, p_x, p_r, p_eps, p_vt, p_gid};
  ops_dat dat_forward[] = {p_pos, p_x, p_r, p_eps, p_vt, p_gid};
  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);

  /* ---- 5. partition, coordinates, particle setup ----------------- */

  /* Register the constants. Must precede ops_partition, and it is what makes
     them visible to translator-generated and device code rather than only to
     this translation unit. */
  ops_decl_const("u0", 1, "double", &u0);
  ops_decl_const("dt", 1, "double", &dt);
  ops_decl_const("delta", 1, "double", &delta);
  ops_decl_const("u0ti", 1, "double", &u0ti);
  ops_decl_const("x_min", 1, "double", &x_min);
  ops_decl_const("x_max", 1, "double", &x_max);
  ops_decl_const("x_plane", 1, "double", &x_plane);
  ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
  ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
  ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
  ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);
  ops_decl_const("eddy_radius", 1, "double", &eddy_radius);
  ops_decl_const("increment", 1, "double", &increment);
  ops_decl_const("shape_norm", 1, "double", &shape_norm);
  ops_decl_const("vt_y", 1, "double", &vt_y);
  ops_decl_const("vt_z", 1, "double", &vt_z);
  ops_decl_const("eddies", 1, "int", &eddies);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("ntbl", 1, "int", &ntbl);

  ops_partition("");

  int grid_range[] = {0, ny + 1, 0, nz + 1};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(crd, 2, S2D_00, "double", OPS_WRITE), ops_arg_idx());

  /* The Reynolds stresses. Two variants, exactly as oSEM has them: a uniform
     isotropic tensor, or a tabulated boundary-layer profile interpolated in y.
     The tables are bulk data rather than scalar parameters, so they go in as
     ops_arg_gbl -- which is how oSEM passes them too. */
  if (use_tbl)
    ops_par_loop(KerInitRST_TBL, "KerInitRST_TBL", block, 2, grid_range,
                 ops_arg_dat(a11, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a21, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a22, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a31, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a32, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a33, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(crd, 2, S2D_00, "double", OPS_READ),
                 ops_arg_gbl(y_inp, 260, "double", OPS_READ),
                 ops_arg_gbl(uu_inp, 260, "double", OPS_READ),
                 ops_arg_gbl(uv_inp, 260, "double", OPS_READ),
                 ops_arg_gbl(vv_inp, 260, "double", OPS_READ),
                 ops_arg_gbl(ww_inp, 260, "double", OPS_READ));
  else
    ops_par_loop(KerInitRST, "KerInitRST", block, 2, grid_range,
               ops_arg_dat(a11, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a21, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a22, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a31, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a32, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a33, 1, S2D_00, "double", OPS_WRITE));

  ops_particle_setup_partition();
  seed_eddies(particle, p_pos, p_gid, eddies);
  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

  Real range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  ops_printf("\n=== oSEM with OPS particles (2-D inlet) ===\n");
  ops_printf("inlet %d x %d nodes over [%.4f, %.4f] x [%.4f, %.4f]\n", ny + 1,
             nz + 1, eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max);
  ops_printf("eddies %d   radius %.5f   u0*dt %.3e   %d steps\n", eddies,
             eddy_radius, increment, niter);
  ops_printf("gather buffer %d x %d doubles = %zu bytes/step\n", eddies, NCOMP,
             eddies * NCOMP * sizeof(double));

  /* ---- 6. initialise the eddies ---------------------------------- */

  ops_prandom_shared_init(seed_gbl);
  ops_printf("rng: %s\n", rng_method == OPS_PRNG_MINSTD ? "minstd_rand per particle"
             : (rng_method == OPS_PRNG_SHARED ? "shared engine, storage order"
                                       : "mt19937 per particle"));
  ops_fill_random_uniform_particle(particle, p_rnd, p_gid, seed_gbl, 1u, rng_method);

  ops_particle_par_loop(
      KerInitEddy, "KerInitEddy", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
      range_parts, map,
      ops_arg_dat_particle(p_x, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_r, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_eps, 3, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_vt, 2, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_rnd, 6, "double", particle, map, OPS_READ));

  std::vector<Real> all_eddies(eddies * NCOMP);

  /* ---- 7. time loop ---------------------------------------------- */

  double c0, w0, c1, w1;
  double t_convect = 0, t_gather = 0, t_fluct = 0, t_rng = 0;
  int lost_at = -1;

  /* Output is off by default: the timings quoted in the README were measured
     without it, and a frame write is not part of any of the three kernels. */
  osem_io_params io = {ny,
                       nz,
                       eddies,
                       niter,
                       nout,
                       dt,
                       u0ti,
                       x_plane,
                       {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max},
                       {0.0, 0.0, 0.0},
                       use_tbl};

  /* The target profile is fixed for the run, so build it once. */
  /* PROF_SLOTS, not 6*(ny+1): ops_reduction_result writes the handle's full
     declared length, so a buffer sized to the grid would be overrun. */
  std::vector<Real> raw(PROF_SLOTS, 0.0);
  /* Running time-average of the stress profile. A single snapshot is a poor
     test of the SHEAR: <u'v'> depends on the cancellation <Sx Sy> -> 0 between
     two independent sign fields, and with only ~40 independent eddy-sized
     patches per row that cancellation scatters badly. Accumulating over the
     run averages it down. */
  std::vector<Real> pf_sum(6 * (ny + 1), 0.0);
  long pf_nsamp = 0;
  std::vector<Real> prof(3 * (ny + 1), 0.0), targ(3 * (ny + 1), 0.0);
  if (use_tbl)
    for (int i = 0; i <= ny; i++) {
      const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * (Real)i / (Real)ny;
      targ[3 * i + 0] = sqrt(tbl_at(uu_inp, y));
      targ[3 * i + 1] = sqrt(tbl_at(vv_inp, y));
      targ[3 * i + 2] = sqrt(tbl_at(ww_inp, y));
    }
  if (nout > 0) {
    remove_stale_output("osem_output", niter, nout, h_sync);
    ops_printf("writing %d HDF5 frames (osem_output_??????.h5)\n", niter / nout);
  } else {
    ops_printf("hint: add -nout 10 to write frames for plot_osem_h5.py\n");
  }

  for (int it = 1; it <= niter; it++) {

    /* -- convect: purely per-eddy, exactly like the advance kernel -- */
    /* Refresh the randoms, exactly as oSEM does before convect_eddies. Keyed
       on global id and the step, so an eddy's draw is the same whichever rank
       happens to own it. Timed separately from the convect loop below -- they
       are different costs and must not be conflated. */
    ops_timers(&c0, &w0);
    ops_fill_random_uniform_particle(particle, p_rnd, p_gid, seed_gbl,
                                     (unsigned int)it + 1u, rng_method);
    ops_timers(&c1, &w1);
    t_rng += w1 - w0;

    ops_timers(&c0, &w0);

    ops_particle_par_loop(
        KerConvectEddies, "KerConvectEddies", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_x, 1, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_r, 1, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_eps, 3, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_vt, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_rnd, 6, "double", particle, map, OPS_READ));

    update_maps(particle, dat_border, nborder, dat_forward, nforward);
    ops_timers(&c1, &w1);
    t_convect += w1 - w0;

    /* -- gather: what replaces oSEM's seven ops_dat_fetch_data calls -- */
    ops_timers(&c0, &w0);
    ops_particle_par_loop(
        KerPublishEddy, "KerPublishEddy", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_x, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_r, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_eps, 3, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_reduce(h_all, eddies * NCOMP, "double", OPS_INC));
    ops_reduction_result(h_all, all_eddies.data());
    ops_timers(&c1, &w1);
    t_gather += w1 - w0;

    /* -- the eddy population is conserved; check that it still is -- */
    int count = 0;
    ops_particle_par_loop(
        KerCountEddies, "KerCountEddies", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_reduce(h_cnt, 1, "int", OPS_INC));
    ops_reduction_result(h_cnt, &count);
    if (count != eddies && lost_at < 0) {
      lost_at = it;
      ops_printf("\n*** eddy population changed at step %d: %d of %d ***\n", it,
                 count, eddies);
      ops_printf("*** the recycle teleport is outside what OPS particle"
                 " migration can express -- see the README ***\n\n");
    }

    /* -- compute_fluct: every node sums over every eddy -- */
    ops_timers(&c0, &w0);
    ops_par_loop(KerComputeFluct, "KerComputeFluct", block, 2, grid_range,
                 ops_arg_dat(uprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(vprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(wprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(crd, 2, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a11, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a21, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a22, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a31, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a32, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a33, 1, S2D_00, "double", OPS_READ),
                 ops_arg_gbl(all_eddies.data(), eddies * NCOMP, "double",
                             OPS_READ));
    ops_timers(&c1, &w1);
    t_fluct += w1 - w0;

    if (use_tbl) {
      /* The literal in the ops_arg_reduce below cannot be written as
         PROF_SLOTS (the translator parses the token, not the preprocessed
         value), so this is what keeps the two in agreement. */
      static_assert(PROF_SLOTS == 606,
                    "PROF_SLOTS and the literal dimension in the "
                    "KerFluctProfile ops_arg_reduce must match");
      ops_par_loop(KerFluctProfile, "KerFluctProfile", block, 2, grid_range,
                   ops_arg_dat(uprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(vprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(wprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_idx(),
                   /* 606 = PROF_SLOTS, and it MUST be spelled as a literal:
                      the translator parses this token itself and cannot
                      evaluate a macro or an expression. The static_assert
                      above the loop is what stops the two drifting -- getting
                      this wrong once already produced a loop declaring 6150
                      doubles against a 606-double handle, which overruns the
                      reduction buffer without crashing. */
                   ops_arg_reduce(h_prof, 606, "double", OPS_INC));
      ops_reduction_result(h_prof, raw.data());
      /* pf_sum.size(), not raw.size(): raw is the oversized reduction buffer
         and only its first 6*(ny+1) entries are ever written. */
      for (size_t k = 0; k < pf_sum.size(); k++) pf_sum[k] += raw[k];
      pf_nsamp++;
    }

    /* -- output (not timed) ---------------------------------------- */
    if (nout > 0 && it % nout == 0) {
      Real fs[4] = {0, 0, 0, 0};
      ops_par_loop(KerFluctStats, "KerFluctStats", block, 2, grid_range,
                   ops_arg_dat(uprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(vprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(wprime, 1, S2D_00, "double", OPS_READ),
                   ops_arg_reduce(h_stat, 4, "double", OPS_INC));
      ops_reduction_result(h_stat, fs);
      const Real nn2 = (fs[3] > 0.0) ? fs[3] : 1.0;
      io.rms[0] = sqrt(fs[0] / nn2);
      io.rms[1] = sqrt(fs[1] / nn2);
      io.rms[2] = sqrt(fs[2] / nn2);

      if (use_tbl) {
        /* `raw` already holds this step's sums, from the accumulation above. */
        const Real nrow = (Real)(nz + 1);
        for (int i = 0; i <= ny; i++)
          for (int c = 0; c < 3; c++)
            prof[3 * i + c] = sqrt(raw[6 * i + c] / nrow);
      }

      write_osem_step(block, crd, uprime, vprime, wprime, all_eddies, prof,
                      targ, io, it);
    }

    if (it % nprint == 0) ops_printf("step %5d / %d\n", it, niter);
  }

  /* ---- 8. report -------------------------------------------------- */

  Real st[4] = {0, 0, 0, 0};
  ops_par_loop(KerFluctStats, "KerFluctStats", block, 2, grid_range,
               ops_arg_dat(uprime, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(vprime, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(wprime, 1, S2D_00, "double", OPS_READ),
               ops_arg_reduce(h_stat, 4, "double", OPS_INC));
  ops_reduction_result(h_stat, st);

  /* Sign balance. Each of u',v',w' is one a_ii times one eps times one shape,
     so with a11 = a22 = a33 their rms must agree statistically. If they do not,
     the eps draws are the first suspect. */
  {
    double m[3] = {0, 0, 0};
    for (int i = 0; i < eddies; i++)
      for (int k = 0; k < 3; k++) m[k] += all_eddies[NCOMP * i + E_SX + k];
    /* Pairwise sign correlation. The bug that skewed an earlier version was a
       correlation, not a bias, so the means alone are not a sufficient check. */
    double c01 = 0, c02 = 0, c12 = 0;
    for (int i = 0; i < eddies; i++) {
      const double a0 = all_eddies[NCOMP * i + E_SX];
      const double a1 = all_eddies[NCOMP * i + E_SY];
      const double a2 = all_eddies[NCOMP * i + E_SZ];
      c01 += a0 * a1; c02 += a0 * a2; c12 += a1 * a2;
    }
    ops_printf("\nmean eps (x,y,z) over %d eddies: %+.4f %+.4f %+.4f"
               "   (0 = balanced)\n", eddies, m[0] / eddies, m[1] / eddies,
               m[2] / eddies);
    ops_printf("eps correlations xy/xz/yz: %+.4f %+.4f %+.4f"
               "   (0 = independent)\n", c01 / eddies, c02 / eddies,
               c12 / eddies);

    /* THE ONE THAT MATTERS: does an eddy's SIGN correlate with its POSITION?
       That is the failure mode that skewed an earlier version -- compute_fluct
       selects eddies by x, so any x-sign coupling biases the selected set.
       The sign-vs-sign correlations above do not test it. */
    {
      double mx = 0.0;
      for (int i = 0; i < eddies; i++) mx += all_eddies[NCOMP * i + E_X];
      mx /= eddies;
      double sx = 0.0, px = 0.0, py = 0.0, pz = 0.0;
      for (int i = 0; i < eddies; i++) {
        const double xc = all_eddies[NCOMP * i + E_X] - mx;
        sx += xc * xc;
        px += xc * all_eddies[NCOMP * i + E_SX];
        py += xc * all_eddies[NCOMP * i + E_SY];
        pz += xc * all_eddies[NCOMP * i + E_SZ];
      }
      sx = sqrt(sx / eddies);
      ops_printf("corr(x, eps_x/y/z):        %+.4f %+.4f %+.4f"
                 "   (|.|>0.06 is suspicious at N=%d)\n",
                 px / eddies / sx, py / eddies / sx, pz / eddies / sx, eddies);
    }
  }

  const Real n = (st[3] > 0.0) ? st[3] : 1.0;
  /* ---- profile check ---------------------------------------------- *
   * With the TBL profile the three rms values are supposed to DIFFER, so the
   * isotropic invariant is gone. The test instead is whether the computed
   * rms(y) follows the tabulated target: rms(u') should track sqrt(R11),
   * rms(v') sqrt(R22), rms(w') sqrt(R33), since the raw sums are unit
   * variance by construction (shape_norm). This exercises the interpolation,
   * the Cholesky and the eddy summation together.
   */
  if (use_tbl) {
    /* Time-averaged, not a final snapshot. */
    std::vector<Real> pf(pf_sum);
    const Real navg = (Real)(pf_nsamp > 0 ? pf_nsamp : 1);
    for (size_t k = 0; k < pf.size(); k++) pf[k] /= navg;


    const Real nrow = (Real)(nz + 1);
    ops_printf("\n--- stress profile vs tabulated target ---------------\n");
    ops_printf("averaged over %ld steps. Note the decorrelation time is one\n"
               "flow-through, ~%d steps, so the number of INDEPENDENT samples\n"
               "is roughly %ld.\n", pf_nsamp,
               (int)((x_max - x_min) / increment),
               1L + pf_nsamp / (long)((x_max - x_min) / increment));
    ops_printf("   y        rms u'   target    rms v'   target    rms w'   target\n");

    Real se = 0.0, st = 0.0;
    int nused = 0;
    for (int i = 0; i <= ny; i++) {
      const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * (Real)i / (Real)ny;
      const Real got[3] = {sqrt(pf[6 * i + 0] / nrow),
                           sqrt(pf[6 * i + 1] / nrow),
                           sqrt(pf[6 * i + 2] / nrow)};
      const Real want[3] = {sqrt(tbl_at(uu_inp, y)), sqrt(tbl_at(vv_inp, y)),
                            sqrt(tbl_at(ww_inp, y))};
      for (int c = 0; c < 3; c++) {
        se += (got[c] - want[c]) * (got[c] - want[c]);
        st += want[c] * want[c];
      }
      nused++;
      if (i % 20 == 0)
        ops_printf("  %.5f  %7.2f %8.2f   %7.2f %8.2f   %7.2f %8.2f\n", y,
                   got[0], want[0], got[1], want[1], got[2], want[2]);
    }
    ops_printf("\nprofile agreement over %d rows: %.1f %% rms deviation\n",
               nused, 100.0 * sqrt(se / st));

    /* ---- shear stress ------------------------------------------------ *
     * <u'v'> = a11 * a21 * <S_x^2> = a11 * a21 = R21 by construction. This is
     * the ONLY check that exercises a21: the rms values are blind to it,
     * because a22 = sqrt(R22 - a21^2) makes a21^2 + a22^2 collapse to R22
     * whatever a21 happens to be. An error in the shear term would otherwise
     * pass every test in this app silently.
     *
     * <u'w'> and <v'w'> must vanish (a31 = a32 = 0), so they come free.
     *
     * SPLIT IN TWO, and the reason matters. R21 is only non-zero inside the
     * boundary layer: it peaks at 601 near the wall and is identically 0 over
     * the top third of the grid, which is padding out to y_max + r_max. A
     * single relative rms over ALL rows puts the freestream rows' noise in the
     * numerator while they contribute nothing to the denominator, so the number
     * measures how far the average has converged rather than whether a21 is
     * right -- which is why it swung 64 / 52 / 154 % across realisations of the
     * same correct code.
     *
     * So: the AGREEMENT figure covers only rows carrying real shear (|R21|
     * above 5 % of peak, y < 0.0078), and the freestream rows are reported
     * separately as a NOISE FLOOR. The floor is what has to fall as 1/sqrt(N)
     * with averaging; the agreement figure is the actual test of a21.
     */
    const Real uv_peak = 601.048494;         /* max |uv_inp|, from TBL_data.h */
    const Real uv_cut = 0.05 * uv_peak;
    Real se_uv = 0, st_uv = 0, worst_uw = 0, worst_vw = 0;
    Real noise_sq = 0.0;
    int n_sig = 0, n_free = 0;
    /* The CORRELATION COEFFICIENT is the test that actually isolates a21:
     *
     *     rho = <u'v'> / sqrt(<u'u'> <v'v'>)   ->   R21 / sqrt(R11 R22)
     *
     * Every stress here carries a common factor <S^2>, the variance of the
     * raw eddy sum, which is NOT 1 in this app (see the note printed below).
     * That factor cancels from rho and does not cancel from <u'v'> alone, so
     * the raw rms deviation above conflates an a21 error with a normalisation
     * error while rho separates them. Accumulated as a slope, sum(rho m rho t)
     * / sum(rho t^2), so the near-wall rows where rho is largest dominate. */
    Real rho_num = 0.0, rho_den = 0.0;
    Real s2_sum = 0.0;      /* mean <u'u'>/R11 over sheared rows = <S_x^2> */
    int s2_n = 0;
    /* pf comes from ops_reduction_result, which ends in an Allreduce, so every
       rank holds the same numbers -- hence the root guard, or every rank would
       write the same file. */
    FILE *dump = (shear_dump && ops_is_root()) ? fopen(shear_dump, "w") : NULL;
    /* The diagonal terms ride along so the CORRELATION COEFFICIENT
       <u'v'>/(rms u' rms v') can be formed offline. That ratio is the real
       test of a21: it is independent of the overall shape normalisation, so a
       shape_norm that leaves <S_x^2> = c != 1 cancels out of it, while it
       inflates <u'v'> and the rms values by c and sqrt(c) respectively. */
    if (dump) fprintf(dump, "# y  <u'v'>  R21  <u'u'>  R11  <v'v'>  R22\n");
    ops_printf("\n   y         <u'v'>    target R21\n");
    for (int i = 0; i <= ny; i++) {
      const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * (Real)i / (Real)ny;
      const Real uv = pf[6 * i + 3] / nrow;
      const Real want = tbl_at(uv_inp, y);
      if (fabs(want) >= uv_cut) {
        se_uv += (uv - want) * (uv - want);
        st_uv += want * want;
        n_sig++;
        const Real uu = pf[6 * i + 0] / nrow, vv = pf[6 * i + 1] / nrow;
        const Real R11 = tbl_at(uu_inp, y), R22 = tbl_at(vv_inp, y);
        if (uu > 0 && vv > 0 && R11 > 0 && R22 > 0) {
          const Real rho_m = uv / sqrt(uu * vv);
          const Real rho_t = want / sqrt(R11 * R22);
          rho_num += rho_m * rho_t;
          rho_den += rho_t * rho_t;
          s2_sum += uu / R11;
          s2_n++;
        }
      } else {
        noise_sq += (uv - want) * (uv - want);
        n_free++;
      }
      /* Normalise the cross terms by their natural scale sqrt(R11*R33). */
      const Real scale = sqrt(sqrt(tbl_at(uu_inp, y) * tbl_at(ww_inp, y)) *
                              sqrt(tbl_at(vv_inp, y) * tbl_at(ww_inp, y)));
      if (scale > 1e-6) {
        const Real a = fabs(pf[6 * i + 4] / nrow) / (scale * scale);
        const Real b = fabs(pf[6 * i + 5] / nrow) / (scale * scale);
        if (a > worst_uw) worst_uw = a;
        if (b > worst_vw) worst_vw = b;
      }
      if (dump)
        fprintf(dump, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e\n", y, uv, want,
                pf[6 * i + 0] / nrow, tbl_at(uu_inp, y), pf[6 * i + 1] / nrow,
                tbl_at(vv_inp, y));
      if (i % 20 == 0)
        ops_printf("  %.5f  %10.2f  %10.2f%s\n", y, uv, want,
                   fabs(want) >= uv_cut ? "" : "   (freestream)");
    }
    if (dump) fclose(dump);
    ops_printf("\nshear agreement over %d sheared rows: %.1f %% rms deviation"
               " from R21\n", n_sig, 100.0 * sqrt(se_uv / st_uv));
    ops_printf("noise floor from %d freestream rows (R21 = 0 there):"
               " %.1f %% of peak R21\n", n_free,
               100.0 * sqrt(noise_sq / (Real)(n_free > 0 ? n_free : 1)) /
                   uv_peak);
    ops_printf("cross terms that must vanish: max |<u'w'>| %.4f,"
               " max |<v'w'>| %.4f  (normalised)\n", worst_uw, worst_vw);

    /* ---- what the two numbers above actually mean --------------------- *
     * The rms deviation is NOT a verdict on a21. It is dominated by a
     * normalisation offset that a21 has nothing to do with:
     *
     *   eddies = vol / eddy_radius^3, with
     *   vol    = (x_max-x_min)(y_max-y_min + 2 r_max)(z_max-z_min + 2 r_max)
     *
     * pads y by 2*r_max, but the eddy box only pads y on TOP (eddy_y_min =
     * y_min, eddy_y_max = y_max + r_max). So the eddy count is sized for a box
     * 1.24x larger in y than the one the eddies occupy, and the density -- and
     * with it every stress -- comes out that much high. Measured <S^2> ~ 1.16
     * against a shape-function prediction of 0.949 * 1.242 = 1.179. This is
     * INHERITED, not introduced: apps/c/oSEM/OPS_oSEM.cpp:43-49 has the same
     * pair. Left alone deliberately -- the port holds the reference's
     * invariants, including this one.
     *
     * The correlation coefficient divides that factor out, which is why it,
     * and not the rms deviation, is the verification of a21. */
    ops_printf("mean <u'u'>/R11 over sheared rows: %.3f"
               "  (= <S_x^2>; 1.0 if the eddy density were consistent with"
               " the box -- see note in source)\n",
               s2_sum / (Real)(s2_n > 0 ? s2_n : 1));
    ops_printf("CORRELATION rho = <u'v'>/sqrt(<u'u'><v'v'>) vs R21/sqrt(R11 R22):"
               " slope %.4f\n   (normalisation-independent, so THIS is the test"
               " of a21; 1.0 = exact)\n",
               rho_den > 0 ? rho_num / rho_den : 0.0);
  }

  ops_printf("\n--- final inlet plane --------------------------------\n");
  ops_printf("rms u' = %.4f   v' = %.4f   w' = %.4f   (u0*TI = %.4f)\n",
             sqrt(st[0] / n), sqrt(st[1] / n), sqrt(st[2] / n), u0ti);
  ops_printf("eddy population: %s\n",
             lost_at < 0 ? "conserved for the whole run"
                         : "CHANGED -- see the message above");

  const double ms = 1000.0 / (double)niter;
  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("rng fill            %9.3f\n", t_rng * ms);
  ops_printf("convect + migrate   %9.3f\n", t_convect * ms);
  ops_printf("gather (allgather)  %9.3f\n", t_gather * ms);
  ops_printf("compute_fluct       %9.3f\n", t_fluct * ms);

  ops_exit();
  return (lost_at < 0) ? 0 : 1;
}
