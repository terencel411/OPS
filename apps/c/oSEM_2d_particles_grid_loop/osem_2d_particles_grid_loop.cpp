#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>

#define OPS_2D

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>
#include <ops_grid_part_seq_v2.h>   /* grid-outer / particle-inner ops_par_loop */
#include <ops_particle_grid_seq.h>  /* particle-outer / grid-inner */

#include "osem_constants.h"
#include "TBL_data.h"
#include "osem_common.h"
#include "particle_kernels.h"
#include "grid_kernels.h"
#include "ops_particle_random.h"
#include "osem_io.h"

typedef double Real;

// each rank keeps the eddies whose (y,z) lands in its own subdomain. The draw is
// a pure function of the global id
static void seed_eddies(ops_particle eddy_particle, ops_dat pos, ops_dat gid,
                        int neddy) {

  if (neddy > (int)eddy_particle->Nmax) ops_particle_realloc_data(eddy_particle, neddy);

  BoundingBox<Real> *box = (BoundingBox<Real> *)eddy_particle->box_block;
  Real *xp = (Real *)pos->data;
  int *ip = (int *)gid->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    Real p[2];
    p[0] = eddy_y_min + (eddy_y_max - eddy_y_min) * ops_prandom_uniform(seed_gbl, i, 0u, 0);
    p[1] = eddy_z_min + (eddy_z_max - eddy_z_min) * ops_prandom_uniform(seed_gbl, i, 0u, 1);
    if (!box->isCoordinateInBoundingBox(p)) continue;

    xp[2 * n] = p[0];
    xp[2 * n + 1] = p[1];
    ip[n] = i;
    n++;
  }

  eddy_particle->no_particles = n;
}

// prints rank ownership - for debug (will be removed later)
static void report_ownership(ops_particle particle, ops_dat pos, ops_dat gid,
                             ops_particle_mapping map, int nid_print) {

  const Real *xp = (const Real *)pos->data;
  const int *ip = (const int *)gid->data;
  const int *p2b = (const int *)map->parts_to_grid->data;
  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  std::vector<int> mine;
  const size_t nall = particle->no_particles + particle->no_virtual;
  int held = (int)nall;
  int binned = 0;
  for (size_t i = 0; i < nall; i++) {
    if (p2b[i] >= 0) binned++;                       // owned and ghosts alike
    if (i < particle->no_particles &&
        box->isCoordinateInBoundingBox(xp + 2 * i)) mine.push_back(ip[i]);
  }
  std::sort(mine.begin(), mine.end());
  int nown = (int)mine.size();

  int rank = 0, nranks = 1;
  std::vector<int> owncnt(1, nown), bincnt(1, binned), heldcnt(1, held);
  std::vector<int> displ(1, 0), allids(mine);

#ifdef OPS_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  owncnt.assign(nranks, 0);
  bincnt.assign(nranks, 0);
  heldcnt.assign(nranks, 0);
  displ.assign(nranks, 0);
  MPI_Gather(&nown, 1, MPI_INT, owncnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&binned, 1, MPI_INT, bincnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&held, 1, MPI_INT, heldcnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  for (int r = 0; r < nranks; r++) { displ[r] = total; total += owncnt[r]; }

  allids.assign(total > 0 ? total : 1, -1);
  MPI_Gatherv(mine.data(), nown, MPI_INT, allids.data(), owncnt.data(),
              displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
#else
  int total = nown;
#endif

  if (rank != 0) return;

  printf("  eddy ownership: held = stored here (owned + ghosts), "
         "owned = (y,z) in this rank's box, binned = reachable through its bins\n");
  printf("  %4s %8s %8s %8s   eddy ids\n", "rank", "held", "owned", "binned");
  for (int r = 0; r < nranks; r++) {
    printf("  %4d %8d %8d %8d   ", r, heldcnt[r], owncnt[r], bincnt[r]);
    const int nshow = (nid_print < 0 || nid_print > owncnt[r]) ? owncnt[r] : nid_print;
    for (int k = 0; k < nshow; k++) printf("%d ", allids[displ[r] + k]);
    if (nshow < owncnt[r]) printf("... (+%d more)", owncnt[r] - nshow);
    printf("\n");
  }
  int tbin = 0;
  for (int r = 0; r < nranks; r++) tbin += bincnt[r];
  printf("  %4s %8s %8d %8d   of %d eddies%s\n", "all", "-", total, tbin, eddies,
         total == eddies ? "" : "  <-- MISMATCH: an eddy sits on no rank's box");
  if (nranks > 1)
    printf("  %d of the %d binned entries are ghosts, the same eddy held by more "
           "than one rank: the band that keeps the boundary nodes exact\n",
           tbin - eddies, tbin);
  fflush(NULL);
}

// debug only - will be removed later
static void collect_eddies(ops_particle particle, ops_dat pos, ops_dat x,
                           ops_dat r, ops_dat eps, ops_dat gid,
                           std::vector<Real> &all) {

  const Real *xp = (const Real *)pos->data;
  const Real *xx = (const Real *)x->data;
  const Real *rr = (const Real *)r->data;
  const Real *ee = (const Real *)eps->data;
  const int *ip = (const int *)gid->data;

  std::vector<Real> mine(all.size(), 0.0);
  for (size_t i = 0; i < particle->no_particles; i++) {
    const int s = NCOMP * ip[i];
    mine[s + E_X] = xx[i];
    mine[s + E_Y] = xp[2 * i];
    mine[s + E_Z] = xp[2 * i + 1];
    mine[s + E_R] = rr[i];
    mine[s + E_SX] = ee[3 * i];
    mine[s + E_SY] = ee[3 * i + 1];
    mine[s + E_SZ] = ee[3 * i + 2];
  }

#ifdef OPS_MPI
  MPI_Allreduce(mine.data(), all.data(), (int)all.size(), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
#else
  all = mine;
#endif
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  // declare ops constants
  u0 = 823.6;
  dt = 0.00000002;
  delta = 0.007;
  ti = 0.01;

  y_min = 0.0;   y_max = 0.009;
  z_min = 0.0;   z_max = 0.05;

  ny = 100;
  nz = 150;

  seed_gbl = 2893328493u;
  niter = 2000;
  nprint = 100;
  rng_method = OPS_PRNG_MINSTD;

  // number of entries in TBL_data.h
  ntbl = 260;

  // 1 = uses instantiate_RST_TBL, 0 = uses instantiate_RST
  use_tbl = 1;

  // extra bins added to the computed eddy_radius reach of the search stencil
  int bin_margin = 1;

  // eddy ids listed per rank at each print step; -1 lists all of them
  int nid_print = 12;
  int deposit = 1;   /* 1 = particle-outer deposit, 0 = grid-outer gather */

  for (int a = 1; a < argc - 1; a++)
  {
    if (strcmp(argv[a], "-margin") == 0) bin_margin = atoi(argv[a + 1]);
    if (strcmp(argv[a], "-niter") == 0)  niter = atoi(argv[a + 1]);
    if (strcmp(argv[a], "-nprint") == 0) nprint = atoi(argv[a + 1]);
    if (strcmp(argv[a], "-nids") == 0)   nid_print = atoi(argv[a + 1]);
    if (strcmp(argv[a], "-deposit") == 0) deposit = atoi(argv[a + 1]);
  }

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
  u0ti = u0 * ti;

  vol = fabs((x_max - x_min) * (y_max - y_min + 2 * r_max) *
             (z_max - z_min + 2 * r_max));
  eddies = (int)trunc(vol / pow(eddy_radius, 3));

  // setting bins nums for the search stencil
  const double dbin_y = (eddy_y_max - eddy_y_min) / (double)(ny + 1);
  const double dbin_z = (eddy_z_max - eddy_z_min) / (double)(nz + 1);
  const int nb_y = std::max(1, (int)ceil(eddy_radius / dbin_y) + bin_margin);
  const int nb_z = std::max(1, (int)ceil(eddy_radius / dbin_z) + bin_margin);

  ops_block block = ops_decl_block(2, "osem_block");

  int size[] = {ny + 1, nz + 1};
  int base[] = {0, 0};
  int d_m[] = {-1, -1};
  int d_p[] = {1, 1};

  int d_mg[] = {-nb_y, -nb_z};
  int d_pg[] = {nb_y, nb_z};

  Real *nd = NULL;
  int *ni = NULL;

  ops_dat d_grid = ops_decl_dat(block, 2, size, base, d_mg, d_pg, nd, "double", "d_grid");
  ops_dat a11 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a11");
  ops_dat a21 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a21");
  ops_dat a22 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a22");
  ops_dat a31 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a31");
  ops_dat a32 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a32");
  ops_dat a33 = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "a33");
  ops_dat uprime = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "uprime");
  ops_dat vprime = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "vprime");
  ops_dat wprime = ops_decl_dat(block, 1, size, base, d_mg, d_pg, nd, "double", "wprime");

  int s00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s00, "0,0");

  // search stencil at that reach, with the corner bins that cannot hold an eddy
  // within eddy_radius removed
  std::vector<int> s_search;
  for (int i = -nb_y; i <= nb_y; i++)
    for (int j = -nb_z; j <= nb_z; j++) {
      const double ddy = std::max(0.0, (std::abs(i) - 1) * dbin_y);
      const double ddz = std::max(0.0, (std::abs(j) - 1) * dbin_z);
      if (ddy * ddy + ddz * ddz > eddy_radius * eddy_radius) continue;
      s_search.push_back(i);
      s_search.push_back(j);
    }
  const int n_search = (int)s_search.size() / 2;

  ops_stencil S2D_SEARCH = ops_decl_stencil(2, n_search, s_search.data(), "search");

  // declaring particle dats
  Real dxb[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, d_grid, 2, dxb);
  ops_particle eddy_particle = ops_decl_particle(block, "eddies", box);

  ops_dat eddy_particle_pos = ops_decl_particle_pos_dat(eddy_particle, 2, base, nd, "double", "position");
  ops_dat eddy_particle_x = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "x");
  ops_dat eddy_particle_r = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "radius");
  ops_dat eddy_particle_eps = ops_decl_particle_dat(eddy_particle, 3, base, nd, "double", "eps");
  ops_dat eddy_particle_id = ops_decl_particle_dat(eddy_particle, 1, base, ni, "int", "gid");

  // rng field which holds 6 random vars for eddy_particle_pos & eddy_particle_eps
  ops_dat eddy_particle_rng = ops_decl_particle_dat(eddy_particle, 6, base, nd, "double", "rnd");

  // S2D_SEARCH sets the binhead halo: how far outside its own subdomain a rank
  // tracks eddies
  ops_particle_mapping map = ops_decl_mapping(
      eddy_particle, d_grid, S2D_SEARCH, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

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
  ops_decl_const("eddies", 1, "int", &eddies);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("ntbl", 1, "int", &ntbl);

  // used in instantiate_RST_TBL kernel
  ops_decl_const("y_inp", 260, "double", y_inp);
  ops_decl_const("uu_inp", 260, "double", uu_inp);
  ops_decl_const("uv_inp", 260, "double", uv_inp);
  ops_decl_const("vv_inp", 260, "double", vv_inp);
  ops_decl_const("ww_inp", 260, "double", ww_inp);

  ops_partition("");

  int grid_range[] = {0, ny + 1, 0, nz + 1};

  ops_par_loop(instantiate_grid, "instantiate_grid", block, 2, grid_range,
               ops_arg_dat(d_grid, 2, S2D_00, "double", OPS_WRITE), ops_arg_idx());

  if (use_tbl)
    ops_par_loop(instantiate_RST_TBL, "instantiate_RST_TBL", block, 2, grid_range,
                 ops_arg_dat(a11, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a21, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a22, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a31, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a32, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(a33, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(d_grid, 2, S2D_00, "double", OPS_READ));
  else
    ops_par_loop(instantiate_RST, "instantiate_RST", block, 2, grid_range,
               ops_arg_dat(a11, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a21, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a22, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a31, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a32, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a33, 1, S2D_00, "double", OPS_WRITE));

  ops_particle_setup_partition();

  seed_eddies(eddy_particle, eddy_particle_pos, eddy_particle_id, eddies);

  ops_dat dat_border[] = {eddy_particle_pos, eddy_particle_x, eddy_particle_r,
                          eddy_particle_eps, eddy_particle_id};
  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);

  // builds the bins and the ghost band. Positions never change after this, so
  // this is the only map build of the run (yz pos don't change even after reaching x_max)
  // Need to find a way make it periodic, send from x_max to x_min
  ops_particle_setup_maps_with_dats(eddy_particle, dat_border, nborder);

  Real range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  // initialising eddies, ghosts included
  ops_fill_random_uniform_particle(eddy_particle, eddy_particle_rng, eddy_particle_id, seed_gbl, 1u, rng_method, true);

  ops_particle_par_loop(
      KerInitEddy, "KerInitEddy", eddy_particle, 2, OPS_PARTICLE_ITERATE_ALL,
      range_parts, map,
      ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_rng, 6, "double", eddy_particle, map, OPS_READ));

  ops_printf("oSEM_2d_particles_grid_loop: %d eddies (decomposed by (y,z), "
             "positions frozen), grid %d x %d, %d steps\n",
             eddies, ny + 1, nz + 1, niter);
  ops_printf("compute_fluct driven %s\n",
             deposit ? "particle-outer (ops_par_particle_grid_loop, deposit)"
                     : "grid-outer (ops_par_loop, gather)");
  ops_printf("search stencil %d bins (%d x %d rectangle culled to the "
             "eddy_radius ellipse; radius %.3e, bin %.3e x %.3e)\n",
             n_search, 2 * nb_y + 1, 2 * nb_z + 1, eddy_radius, dbin_y, dbin_z);
  
  std::vector<Real> eddy_all(eddies * NCOMP);

  // time counters
  double c0, w0, c1, w1;
  double t_convect = 0, t_maps = 0, t_fluct = 0, t_rng = 0;

  for (int it = 1; it <= niter; it++) {

    ops_timers(&c0, &w0);
    ops_fill_random_uniform_particle(eddy_particle, eddy_particle_rng, eddy_particle_id, seed_gbl,
                                     (unsigned int)it + 1u, rng_method, true);
    ops_timers(&c1, &w1);
    t_rng += w1 - w0;

    ops_timers(&c0, &w0);

    ops_particle_par_loop(
        KerConvectEddies, "KerConvectEddies", eddy_particle, 2,
        OPS_PARTICLE_ITERATE_ALL, range_parts, map,
        ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_rng, 6, "double", eddy_particle, map, OPS_READ));

    ops_timers(&c1, &w1);
    t_convect += w1 - w0;

    // no re-binning, migration or ghost exchange: positions are frozen and the
    // loop above advanced the ghosts too

    ops_timers(&c0, &w0);
    ops_par_loop(zero_fluct, "zero_fluct", block, 2, grid_range,
                 ops_arg_dat(uprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(vprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(wprime, 1, S2D_00, "double", OPS_WRITE));

    if (deposit)
      // same kernel, outer over eddies. ITERATE_ALL is required- a ghost eddy
      // still deposits onto this rank's interior nodes near the boundary
      // ops_par_particle_grid_loop is used to write only in particle datsa
      ops_par_particle_grid_loop(compute_fluct, "compute_fluct_deposit",
                 eddy_particle, map, 2, OPS_PARTICLE_ITERATE_ALL, range_parts, S2D_SEARCH,
                 ops_arg_dat(uprime, 1, S2D_SEARCH, "double", OPS_RW),
                 ops_arg_dat(vprime, 1, S2D_SEARCH, "double", OPS_RW),
                 ops_arg_dat(wprime, 1, S2D_SEARCH, "double", OPS_RW),
                 ops_arg_dat(d_grid, 2, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a11, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a21, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a22, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a31, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a32, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat(a33, 1, S2D_SEARCH, "double", OPS_READ),
                 ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_READ));
    else
    // ops_par_loop is used to write from particle dats to normal dats (check seq_v2 header)
    ops_par_loop(compute_fluct, "compute_fluct", eddy_particle, map, S2D_SEARCH,
                 2, grid_range,
                 ops_arg_dat(uprime, 1, S2D_00, "double", OPS_RW),
                 ops_arg_dat(vprime, 1, S2D_00, "double", OPS_RW),
                 ops_arg_dat(wprime, 1, S2D_00, "double", OPS_RW),
                 ops_arg_dat(d_grid, 2, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a11, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a21, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a22, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a31, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a32, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a33, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_READ),
                 ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_READ));
    ops_timers(&c1, &w1);
    t_fluct += w1 - w0;

    if (nprint > 0 && it % nprint == 0) {
      collect_eddies(eddy_particle, eddy_particle_pos, eddy_particle_x,
                     eddy_particle_r, eddy_particle_eps, eddy_particle_id,
                     eddy_all);
      write_osem_step(block, d_grid, uprime, vprime, wprime, eddy_all, it);
      ops_printf("step %5d / %d\n", it, niter);
      report_ownership(eddy_particle, eddy_particle_pos, eddy_particle_id, map,
                       nid_print);
    }
  }

  // run with OPS_DIAGS=2 or this function will have no output
  ops_timing_output_stdout();

  const double ms = 1000.0 / (double)niter;
  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("rng fill            %9.3f\n", t_rng * ms);
  ops_printf("convect             %9.3f\n", t_convect * ms);
  ops_printf("compute_fluct       %9.3f\n", t_fluct * ms);

  ops_exit();
}
