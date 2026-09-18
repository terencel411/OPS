#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>

#define OPS_2D
#ifdef OPS_MPI
#include <mpi.h>
#include <ops_mpi_core.h>
#endif

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>
#include <ops_particle_insert_del.h>

#include "osem_constants.h"
#include "TBL_data.h"
#include "osem_common.h"
#include "particle_kernels.h"
#include "grid_kernels.h"
#include "ops_particle_random.h"
#include "osem_io.h"

typedef double Real;

// insert the eddies and give each one an id, deciding which rank owns it.
static void seed_eddies(ops_particle eddy_particle, ops_dat pos, ops_dat gid, int neddy) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)eddy_particle->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  if (neddy > (int)eddy_particle->Nmax) ops_particle_realloc_data(eddy_particle, neddy);

  Real *xp = (Real *)pos->data;
  int *ip = (int *)gid->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * ops_prandom_uniform(seed_gbl, i, 0u, 0);
    const Real z = eddy_z_min + (eddy_z_max - eddy_z_min) * ops_prandom_uniform(seed_gbl, i, 0u, 1);

    if (y < lo[0] || y >= hi[0] || z < lo[1] || z >= hi[1]) continue;

    xp[2 * n] = y;
    xp[2 * n + 1] = z;
    ip[n] = i;
    n++;
  }

  eddy_particle->no_particles = n;
}

// KerConvectEddies marks eddies that leave the eddy domain (x > x_max);
// gather the marked gids from every rank: O(exits), not O(eddies)
static void gather_exits(ops_particle p, ops_dat exit_flag, ops_dat gid, std::vector<int> &all) {

  const int *fl = (const int *)exit_flag->data;
  const int *ip = (const int *)gid->data;
  std::vector<int> mine;
  for (size_t i = 0; i < p->no_particles; i++)
    if (fl[i]) mine.push_back(ip[i]);

#ifdef OPS_MPI
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  int nmine = (int)mine.size();
  std::vector<int> cnt(nranks), disp(nranks);
  MPI_Allgather(&nmine, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int total = 0;
  for (int r = 0; r < nranks; r++) { disp[r] = total; total += cnt[r]; }
  all.resize(total);
  if (total > 0)
    MPI_Allgatherv(mine.data(), nmine, MPI_INT, all.data(), cnt.data(), disp.data(),
                   MPI_INT, MPI_COMM_WORLD);
#else
  all.swap(mine);
#endif
}

// re-create each exited gid at x_min on the rank whose box holds its new (y,z) (ops_particle_insert),
// delete the marked eddies (ops_particle_user_delete + ops_particle_remove_particles), rebuild bins and ghosts
static void reinsert_eddies(ops_particle p, ops_particle_mapping map, ops_dat x, ops_dat r,
                            ops_dat eps, ops_dat rng, ops_dat exit_flag, ops_dat gid,
                            const std::vector<int> &exited, unsigned int counter,
                            ops_dat *dats, int ndats) {

  ops_particle_reset_virtual_particles(p);

  // candidates: the same (seed, gid, counter) draws on every rank; KerDecideEddy keeps this rank's
  const int n = (int)exited.size();
  std::vector<double> xcrd(2 * n), draw(6 * n);
  for (int k = 0; k < n; k++) {
    ops_prandom_uniform_gid(seed_gbl, exited[k], counter, rng_method, 6, &draw[6 * k]);
    xcrd[2 * k] = eddy_y_min + (eddy_y_max - eddy_y_min) * draw[6 * k + 1];
    xcrd[2 * k + 1] = eddy_z_min + (eddy_z_max - eddy_z_min) * draw[6 * k + 2];
  }

  // insert first: ops_particle_insert clears every mark_deletion
  ops_particle_insert(KerInsertEddy, "KerInsertEddy", KerDecideEddy, p, 0, 2, n,
                      xcrd.data(), nullptr,
                      ops_arg_dat_particle(x, 1, "double", p, map, OPS_WRITE),
                      ops_arg_dat_particle(r, 1, "double", p, map, OPS_WRITE),
                      ops_arg_dat_particle(eps, 3, "double", p, map, OPS_WRITE),
                      ops_arg_dat_particle(rng, 6, "double", p, map, OPS_WRITE),
                      ops_arg_dat_particle(gid, 1, "int", p, map, OPS_WRITE),
                      ops_arg_dat_particle(exit_flag, 1, "int", p, map, OPS_WRITE),
                      ops_arg_gbl_particle(draw.data(), 6, "double", OPS_READ),
                      ops_arg_gbl_particle(exited.data(), 1, "int", OPS_READ));

  ops_particle_user_delete(KerDeleteEddy, "KerDeleteEddy", p, 2,
                           ops_arg_dat_particle(exit_flag, 1, "int", p, map, OPS_READ));
                           
  ops_particle_remove_particles(p, true);

  ops_particle_setup_maps_with_dats(p, dats, ndats);
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
  for (int a = 1; a < argc - 1; a++) {
    if (!strcmp(argv[a], "-niter"))  niter  = atoi(argv[a + 1]);
    if (!strcmp(argv[a], "-nprint")) nprint = atoi(argv[a + 1]);
  }
  rng_method = OPS_PRNG_MINSTD;

  // number of entries in TBL_data.h
  ntbl = 260;

  // 1 = uses instantiate_RST_TBL, 0 = uses instantiate_RST
  use_tbl = 1;

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

  ops_block block = ops_decl_block(2, "osem_block");

  int size[] = {ny + 1, nz + 1};
  int base[] = {0, 0};
  int d_m[] = {-1, -1};
  int d_p[] = {1, 1};
  Real *nd = NULL;
  int *ni = NULL;

  ops_dat d_grid = ops_decl_dat(block, 2, size, base, d_m, d_p, nd, "double", "d_grid");
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

  // reduction handles
  ops_reduction h_eddy = ops_decl_reduction_handle(eddies * NCOMP * sizeof(double), "double", "eddy_all");
  ops_reduction h_check = ops_decl_reduction_handle((eddies + 1) * sizeof(int), "int", "check");

  // declaring particle dats
  Real dxb[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, d_grid, 2, dxb);
  ops_particle eddy_particle = ops_decl_particle(block, "eddies", box);

  ops_dat eddy_particle_pos = ops_decl_particle_pos_dat(eddy_particle, 2, base, nd, "double", "position");
  ops_dat eddy_particle_x = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "x");
  ops_dat eddy_particle_r = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "radius");
  ops_dat eddy_particle_eps = ops_decl_particle_dat(eddy_particle, 3, base, nd, "double", "eps");
  ops_dat eddy_particle_id = ops_decl_particle_dat(eddy_particle, 1, base, ni, "int", "gid");
  ops_dat eddy_particle_exit = ops_decl_particle_dat(eddy_particle, 1, base, ni, "int", "exit");

  // rng field which holds 6 random vars for eddy_particle_xyz & eddy_particle_eps
  ops_dat eddy_particle_rng = ops_decl_particle_dat(eddy_particle, 6, base, nd, "double", "rnd");

  ops_particle_mapping map = ops_decl_mapping(
      eddy_particle, d_grid, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  ops_dat dat_border[] = {eddy_particle_pos, eddy_particle_x, eddy_particle_r,
                          eddy_particle_eps, eddy_particle_id};
  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);

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
  for (size_t i = 0; i < eddy_particle->no_particles; i++)
    ((int *)eddy_particle_exit->data)[i] = 0;
  ops_particle_setup_maps_with_dats(eddy_particle, dat_border, nborder);

  Real range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  // initialising eddies
  ops_fill_random_uniform_particle(eddy_particle, eddy_particle_rng, eddy_particle_id, seed_gbl, 1u, rng_method);

  ops_particle_par_loop(
      KerInitEddy, "KerInitEddy", eddy_particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
      range_parts, map,
      ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_WRITE),
      ops_arg_dat_particle(eddy_particle_rng, 6, "double", eddy_particle, map, OPS_READ));

  // ghosts were built before KerInitEddy set x, r and eps: refresh them once
  ops_particle_reset_virtual_particles(eddy_particle);
  ops_particle_setup_maps_with_dats(eddy_particle, dat_border, nborder);

  std::vector<Real> eddy_all(eddies * NCOMP);
  std::vector<int> exited;
  std::vector<int> check(eddies + 1);

  // time counters
  double c0, w0, c1, w1;
  double t_convect = 0, t_reinsert = 0, t_gather = 0, t_fluct = 0;
  long n_reinserted = 0;
  int n_rebuilds = 0;

  for (int it = 1; it <= niter; it++) {

    ops_timers(&c0, &w0);

    // ITERATE_ALL: ghosts advance too, so they stay equal to their owners between rebuilds
    ops_particle_par_loop(
        KerConvectEddies, "KerConvectEddies", eddy_particle, 2,
        OPS_PARTICLE_ITERATE_ALL, range_parts, map,
        ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_exit, 1, "int", eddy_particle, map, OPS_WRITE));

    ops_timers(&c1, &w1);
    t_convect += w1 - w0;

    ops_timers(&c0, &w0);

    gather_exits(eddy_particle, eddy_particle_exit, eddy_particle_id, exited);
    if (!exited.empty()) {   // same list on every rank, so every rank rebuilds together
      reinsert_eddies(eddy_particle, map, eddy_particle_x, eddy_particle_r,
                      eddy_particle_eps, eddy_particle_rng, eddy_particle_exit,
                      eddy_particle_id, exited, (unsigned int)it + 1u,
                      dat_border, nborder);
      n_reinserted += (long)exited.size();
      n_rebuilds++;
    }

    ops_timers(&c1, &w1);
    t_reinsert += w1 - w0;

    ops_timers(&c0, &w0);
    ops_particle_par_loop(
        KerGatherEddies, "KerGatherEddies", eddy_particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_READ),
        ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_READ),
        ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_READ),
        ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_READ),
        ops_arg_dat_particle(eddy_particle_id, 1, "int", eddy_particle, map, OPS_READ),
        ops_arg_reduce(h_eddy, eddies * NCOMP, "double", OPS_INC));
    ops_reduction_result(h_eddy, eddy_all.data());
    ops_timers(&c1, &w1);
    t_gather += w1 - w0;

    ops_timers(&c0, &w0);
    ops_par_loop(compute_fluct, "compute_fluct", block, 2, grid_range,
                 ops_arg_dat(uprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(vprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(wprime, 1, S2D_00, "double", OPS_WRITE),
                 ops_arg_dat(d_grid, 2, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a11, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a21, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a22, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a31, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a32, 1, S2D_00, "double", OPS_READ),
                 ops_arg_dat(a33, 1, S2D_00, "double", OPS_READ),
                 ops_arg_gbl(eddy_all.data(), eddies * NCOMP, "double",
                             OPS_READ));
    ops_timers(&c1, &w1);
    t_fluct += w1 - w0;

    if (nprint > 0 && it % nprint == 0) {
      // temp check
      const double bx[4] = {box->getLocalMin().x, box->getLocalMin().y,
                            box->getLocalMax().x, box->getLocalMax().y};
      ops_particle_par_loop(
          KerCheckEddies, "KerCheckEddies", eddy_particle, 2,
          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
          ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_READ),
          ops_arg_dat_particle(eddy_particle_id, 1, "int", eddy_particle, map, OPS_READ),
          ops_arg_gbl(bx, 4, "double", OPS_READ),
          ops_arg_reduce(h_check, eddies + 1, "int", OPS_INC));
      ops_reduction_result(h_check, check.data());
      int owned = 0, lost = 0, dup = 0;
      for (int g = 0; g < eddies; g++) {
        owned += check[g];
        lost += (check[g] == 0);
        dup += (check[g] > 1);
      }
      ops_printf("CHECK it %d owned %d lost %d dup %d outside_box %d reinserted %ld rebuilds %d\n",
                 it, owned, lost, dup, check[eddies], n_reinserted, n_rebuilds);
      // temp check

      write_osem_step(block, d_grid, uprime, vprime, wprime, eddy_all, it);
      ops_printf("step %5d / %d\n", it, niter);
    }
  }

  const double ms = 1000.0 / (double)niter;
  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("convect             %9.3f\n", t_convect * ms);
  ops_printf("reinsert            %9.3f\n", t_reinsert * ms);
  ops_printf("reduction op        %9.3f\n", t_gather * ms);
  ops_printf("compute_fluct       %9.3f\n", t_fluct * ms);

  ops_exit();
}
