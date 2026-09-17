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

#include "osem_constants.h"
#include "TBL_data.h"
#include "osem_common.h"
#include "particle_kernels.h"
#include "grid_kernels.h"
#include "ops_particle_random.h"
#include "osem_io.h"

typedef double Real;

// place the eddies and give each one an id, deciding which rank owns it. 
static void seed_eddies(ops_particle eddy_particle, ops_dat pos, ops_dat yz,
                        ops_dat gid, int neddy) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)eddy_particle->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  if (neddy > (int)eddy_particle->Nmax) ops_particle_realloc_data(eddy_particle, neddy);

  Real *xp = (Real *)pos->data;
  Real *yp = (Real *)yz->data;
  int *ip = (int *)gid->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * ops_prandom_uniform(seed_gbl, i, 0u, 0);
    const Real z = eddy_z_min + (eddy_z_max - eddy_z_min) * ops_prandom_uniform(seed_gbl, i, 0u, 1);

    if (y < lo[0] || y >= hi[0] || z < lo[1] || z >= hi[1]) continue;

    xp[2 * n] = y;
    xp[2 * n + 1] = z;
    yp[2 * n] = y;
    yp[2 * n + 1] = z;
    ip[n] = i;
    n++;
  }

  eddy_particle->no_particles = n;
}

// TEST: resets whose new owner is >= 2 ranks away along an axis, by gid
static void hop_diag(ops_particle p, ops_dat pos, ops_dat x, ops_dat gid, int it) {
#ifdef OPS_MPI
  static int nr = 0;
  static std::vector<double> bx;
  static std::vector<int> crd;
  sub_block *sb = OPS_sub_block_list[p->block->index];
  BoundingBox<Real> *box = (BoundingBox<Real> *)p->box_block;
  if (nr == 0) {
    MPI_Comm_size(MPI_COMM_WORLD, &nr);
    double mine[4] = {box->getLocalMin().x, box->getLocalMin().y, box->getLocalMax().x, box->getLocalMax().y};
    int mc[2] = {sb->coords[0], sb->coords[1]};
    bx.resize(4 * nr); crd.resize(2 * nr);
    MPI_Allgather(mine, 4, MPI_DOUBLE, bx.data(), 4, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(mc, 2, MPI_INT, crd.data(), 2, MPI_INT, MPI_COMM_WORLD);
  }
  const int me = ops_get_proc();
  const Real *xp = (const Real *)pos->data;
  const Real *xx = (const Real *)x->data;
  const int *ip = (const int *)gid->data;
  std::vector<int> far;
  for (size_t i = 0; i < p->no_particles; i++) {
    if (xx[i] != x_min) continue;
    int own = -1;
    for (int r = 0; r < nr; r++)
      if (xp[2*i] >= bx[4*r] && xp[2*i] < bx[4*r+2] && xp[2*i+1] >= bx[4*r+1] && xp[2*i+1] < bx[4*r+3]) own = r;
    if (own < 0 || abs(crd[2*own] - crd[2*me]) >= 2 || abs(crd[2*own+1] - crd[2*me+1]) >= 2) far.push_back(ip[i]);
  }
  int nl = (int)far.size(), ng;
  MPI_Allreduce(&nl, &ng, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ops_printf("HOP it %d two_hop_resets %d\n", it, ng);
  if (nl) { printf("HOPS it %d rank %d:", it, me); for (int g : far) printf(" %d", g); printf("\n"); }
  fflush(nullptr);   // not stdout: that macro crashes the ops-translator parser
#endif
}

// TEST: owned and still-flagged (mark 1) counts between migration and remove_delete_maps
static void pre_diag(ops_particle p, int it) {
  int loc[2] = {(int)p->no_particles, 0}, glob[2];
  for (size_t i = 0; i < p->no_particles; i++) if (p->mark_deletion[i] == 1) loc[1]++;
#ifdef OPS_MPI
  MPI_Allreduce(loc, glob, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  glob[0] = loc[0]; glob[1] = loc[1];
#endif
  ops_printf("PRE it %d owned_before_remove %d flagged1 %d\n", it, glob[0], glob[1]);
}

// TEST: owned / resets / stranded (owned but outside own box) / lost / duplicated
static void mig_diag(ops_particle p, ops_dat pos, ops_dat x, ops_dat gid, int it, int decide) {
  BoundingBox<Real> *box = (BoundingBox<Real> *)p->box_block;
  const Real *xp = (const Real *)pos->data;
  const Real *xx = (const Real *)x->data;
  const int *ip = (const int *)gid->data;
  const int n = (int)p->no_particles;
  int loc[3] = {n, 0, 0}, glob[3];
  std::vector<int> cnt(eddies, 0), gcnt(eddies, 0), sg;
  for (int i = 0; i < n; i++) {
    if (ip[i] >= 0 && ip[i] < eddies) cnt[ip[i]]++;
    if (xx[i] == x_min) loc[2]++;
    if (!box->isCoordinateInBoundingBox(xp + 2 * i)) { loc[1]++; sg.push_back(ip[i]); }
  }
#ifdef OPS_MPI
  MPI_Allreduce(loc, glob, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(cnt.data(), gcnt.data(), eddies, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int k = 0; k < 3; k++) glob[k] = loc[k];
  gcnt = cnt;
#endif
  int lost = 0, dup = 0;
  static std::vector<char> seen;
  if (seen.empty()) seen.assign(eddies, 0);
  std::string nl;
  for (int g = 0; g < eddies; g++) {
    if (gcnt[g] == 0) { lost++; if (!seen[g]) { seen[g] = 1; nl += " " + std::to_string(g); } }
    if (gcnt[g] > 1) dup++;
  }
  if (!nl.empty()) ops_printf("LOST it %d:%s\n", it, nl.c_str());
  ops_printf("MIG it %d decide %d owned %d resets %d stranded %d lost %d dup %d\n",
             it, decide, glob[0], glob[2], glob[1], lost, dup);
  if (!sg.empty()) {
    printf("MIGS it %d rank %d:", it, ops_get_proc());
    for (int g : sg) printf(" %d", g);
    printf("\n");
  }
  fflush(nullptr);   // not stdout: that macro crashes the ops-translator parser
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

  // declaring particle dats
  Real dxb[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, d_grid, 2, dxb);
  ops_particle eddy_particle = ops_decl_particle(block, "eddies", box);

  ops_dat eddy_particle_pos = ops_decl_particle_pos_dat(eddy_particle, 2, base, nd, "double", "position");
  ops_dat eddy_particle_yz = ops_decl_particle_dat(eddy_particle, 2, base, nd, "double", "yz");
  ops_dat eddy_particle_x = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "x");
  ops_dat eddy_particle_r = ops_decl_particle_dat(eddy_particle, 1, base, nd, "double", "radius");
  ops_dat eddy_particle_eps = ops_decl_particle_dat(eddy_particle, 3, base, nd, "double", "eps");
  ops_dat eddy_particle_id = ops_decl_particle_dat(eddy_particle, 1, base, ni, "int", "gid");
  
  // rng field which holds 6 random vars for eddy_particle_xyz & eddy_particle_eps
  ops_dat eddy_particle_rng = ops_decl_particle_dat(eddy_particle, 6, base, nd, "double", "rnd");

  ops_particle_mapping map = ops_decl_mapping(
      eddy_particle, d_grid, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  ops_dat dat_border[] = {eddy_particle_pos, eddy_particle_yz, eddy_particle_x,
                          eddy_particle_r, eddy_particle_eps, eddy_particle_id};
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
  seed_eddies(eddy_particle, eddy_particle_pos, eddy_particle_yz,
              eddy_particle_id, eddies);
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

  std::vector<Real> eddy_all(eddies * NCOMP);

  // time counters
  double c0, w0, c1, w1;
  double t_convect = 0, t_gather = 0, t_fluct = 0, t_rng = 0, t_map_update = 0;

  for (int it = 1; it <= niter; it++) {

    ops_timers(&c0, &w0);
    ops_fill_random_uniform_particle(eddy_particle, eddy_particle_rng, eddy_particle_id, seed_gbl,
                                     (unsigned int)it + 1u, rng_method);
    ops_timers(&c1, &w1);
    t_rng += w1 - w0;

    ops_timers(&c0, &w0);

    ops_particle_par_loop(
        KerConvectEddies, "KerConvectEddies", eddy_particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(eddy_particle_pos, 2, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_yz, 2, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_x, 1, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_r, 1, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_eps, 3, "double", eddy_particle, map, OPS_RW),
        ops_arg_dat_particle(eddy_particle_rng, 6, "double", eddy_particle, map, OPS_READ));

    ops_timers(&c1, &w1);
    t_convect += w1 - w0;

    ops_timers(&c0, &w0);

    // TEST: per-step map update, so a changed position dat moves the eddy
    {
      // hop_diag(eddy_particle, eddy_particle_pos, eddy_particle_x, eddy_particle_id, it);
      int decide = ops_particle_update_map_lists_actual_hybrid(eddy_particle);
      // pre_diag(eddy_particle, it);
      ops_particle_remove_delete_maps(eddy_particle, decide);
      if (decide) ops_particle_intrablock_border_map_update(eddy_particle, dat_border, nborder);
      else        ops_particle_intrablock_forward_map_update(eddy_particle, dat_border, nborder);
      ops_particle_reset_flags(eddy_particle, decide);
      // mig_diag(eddy_particle, eddy_particle_pos, eddy_particle_x, eddy_particle_id, it, decide);
    }

    ops_timers(&c1, &w1);
    t_map_update += w1 - w0;
    
    ops_timers(&c0, &w0);
    ops_particle_par_loop(
        KerGatherEddies, "KerGatherEddies", eddy_particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(eddy_particle_yz, 2, "double", eddy_particle, map, OPS_READ),
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
      write_osem_step(block, d_grid, uprime, vprime, wprime, eddy_all, it);
      ops_printf("step %5d / %d\n", it, niter);
    }
  }

  const double ms = 1000.0 / (double)niter;
  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("rng fill            %9.3f\n", t_rng * ms);
  ops_printf("convect             %9.3f\n", t_convect * ms);
  ops_printf("update map          %9.3f\n", t_map_update * ms);
  ops_printf("reduction op        %9.3f\n", t_gather * ms);
  ops_printf("compute_fluct       %9.3f\n", t_fluct * ms);

  ops_exit();
}
