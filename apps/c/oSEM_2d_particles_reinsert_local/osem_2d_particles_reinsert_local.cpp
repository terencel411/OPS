#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>

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

// insert the eddies and give each one an id, deciding which rank owns it.
static void seed_eddies(ops_particle eddy_particle, ops_dat pos, ops_dat x, ops_dat r,
                        ops_dat eps, ops_dat gid, ops_dat exit_flag, int neddy) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)eddy_particle->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  if (neddy > (int)eddy_particle->Nmax) ops_particle_realloc_data(eddy_particle, neddy);

  Real *yz_pos = (Real *)pos->data;
  Real *x_pos = (Real *)x->data;
  Real *e_rad = (Real *)r->data;
  Real *e_eps = (Real *)eps->data;
  int *e_fl = (int *)exit_flag->data;
  int *ip = (int *)gid->data;

  int n = 0;
  double v[6];
  for (int i = 0; i < neddy; i++) {
    ops_prandom_uniform_gid(seed_gbl, i, 1u, rng_method, 6, v);
    const Real y = eddy_y_min + (eddy_y_max - eddy_y_min) * v[1];
    const Real z = eddy_z_min + (eddy_z_max - eddy_z_min) * v[2];

    if (y < lo[0] || y >= hi[0] || z < lo[1] || z >= hi[1]) continue;

    yz_pos[2 * n] = y;
    yz_pos[2 * n + 1] = z;
    x_pos[n] = x_min + v[0] * (x_max - x_min);
    e_rad[n] = eddy_radius;
    e_eps[3 * n] = (v[3] < 0.5) ? -1.0 : 1.0;
    e_eps[3 * n + 1] = (v[4] < 0.5) ? -1.0 : 1.0;
    e_eps[3 * n + 2] = (v[5] < 0.5) ? -1.0 : 1.0;
    e_fl[n] = 0;
    ip[n] = i;
    n++;
  }

  eddy_particle->no_particles = n;
}

// How the re-inserted eddies are split across the ranks, as a running total:
// entry r covers ranks 0..r, so a uniform draw picks a rank by finding the
// first entry it falls below. An equal split in 2D, because x is not part of
// the position dat and a new eddy can land anywhere in (y,z) -- nothing to test.
//
// 3D, when it is written: x IS part of the position dat, so only the ranks
// whose box straddles x_min can receive a re-inserted eddy.
//
//   takes_part = (x_min >= box->getLocalMin().x && x_min < box->getLocalMax().x)
//   weight     = takes_part ? (this rank's y-z face area) : 0.0
//
// Allgather the weights over sb->comm and accumulate them the same way. A rank
// weighted 0 gets a zero-width slice and can never be dealt an eddy, so
// reinsert_local needs no change. Weighting by face area rather than equally
// keeps the eddy density uniform when the boxes differ.
static void reinsert_rank_splits(ops_particle p, std::vector<double> &rank_split) {

  int nranks = 1;
  (void)p;
#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  if (!sb->owned) { rank_split.clear(); return; }
  MPI_Comm_size(sb->comm, &nranks);
#endif

  rank_split.assign(nranks, 0.0);
  for (int r = 0; r < nranks; r++) rank_split[r] = (double)(r + 1) / (double)nranks;
}

// mark this rank's exits and return how many exited across every rank
static int count_exits(ops_particle p, ops_dat exit_flag) {

  const int *e_fl = (const int *)exit_flag->data;
  int curr_rank_exits = 0;
  for (size_t i = 0; i < p->no_particles; i++) {
    p->mark_deletion[i] = e_fl[i];
    curr_rank_exits += (e_fl[i] != 0);
  }

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  if (!sb->owned) return 0;
  int total = 0;
  MPI_Allreduce(&curr_rank_exits, &total, 1, MPI_INT, MPI_SUM, sb->comm);
  return total;
#else
  return curr_rank_exits;
#endif
}

// deal the np exited eddies over the ranks, then create this rank's split
// locally: y-z inside its own box, id counted on from the global maximum
static int reinsert_local(ops_particle p, ops_dat pos, ops_dat x, ops_dat r, ops_dat eps,
                          ops_dat exit_flag, ops_dat gid, int np, unsigned int step,
                          const std::vector<double> &rank_split, ops_dat *dats, int ndats) {

  ops_particle_reset_virtual_particles(p);

  BoundingBox<Real> *box = (BoundingBox<Real> *)p->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  int *ip = (int *)gid->data;
  int curr_rank_max_id = -1;
  for (size_t i = 0; i < p->no_particles; i++)
    if (ip[i] > curr_rank_max_id) curr_rank_max_id = ip[i];

  int curr_rank = 0, max_tag = curr_rank_max_id;
#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  MPI_Comm_rank(sb->comm, &curr_rank);
  MPI_Allreduce(&curr_rank_max_id, &max_tag, 1, MPI_INT, MPI_MAX, sb->comm);
#endif

  // every rank replays the same deal, so the split and the ids agree everywhere
  // without communication. seed_gbl + 1: seeding uses (seed_gbl, gid, 1) at step 1
  std::vector<int> curr_rank_j;
  double u[1];
  for (int j = 0; j < np; j++) {
    ops_prandom_uniform_gid(seed_gbl + 1u, j, step, rng_method, 1, u);
    int owner = 0;
    while (owner + 1 < (int)rank_split.size() && u[0] >= rank_split[owner]) owner++;
    if (owner == curr_rank) curr_rank_j.push_back(j);
  }

  const int new_eddies = (int)curr_rank_j.size();
  const size_t current_num_eddies = p->no_particles;
  if (current_num_eddies + new_eddies > p->Nmax)
    ops_particle_realloc_data(p, (int)(current_num_eddies + new_eddies));

  Real *yz_pos = (Real *)pos->data;
  Real *x_pos = (Real *)x->data;
  Real *e_rad = (Real *)r->data;
  Real *e_eps = (Real *)eps->data;
  int *e_fl = (int *)exit_flag->data;
  ip = (int *)gid->data;

  double v[6];
  for (int k = 0; k < new_eddies; k++) {
    const size_t i = current_num_eddies + k;
    const int new_gid = max_tag + 1 + curr_rank_j[k];
    ops_prandom_uniform_gid(seed_gbl, new_gid, 0u, rng_method, 6, v);
    yz_pos[2 * i] = lo[0] + (hi[0] - lo[0]) * v[1];
    yz_pos[2 * i + 1] = lo[1] + (hi[1] - lo[1]) * v[2];
    x_pos[i] = x_min;
    e_rad[i] = eddy_radius;
    e_eps[3 * i] = (v[3] < 0.5) ? -1.0 : 1.0;
    e_eps[3 * i + 1] = (v[4] < 0.5) ? -1.0 : 1.0;
    e_eps[3 * i + 2] = (v[5] < 0.5) ? -1.0 : 1.0;
    e_fl[i] = 0;
    p->mark_deletion[i] = 0;
    ip[i] = new_gid;
  }
  p->no_particles = current_num_eddies + new_eddies;

  ops_particle_remove_particles(p, true);
  ops_particle_setup_maps_with_dats(p, dats, ndats);
  return new_eddies;
}

// pack every rank's eddies into one array. ids grow without bound here, so the
// gid-indexed reduction the other reinsert apps use cannot be applied
static void gather_eddies(ops_particle p, ops_dat pos, ops_dat x, ops_dat r,
                          ops_dat eps, std::vector<Real> &all) {

  const Real *yz_pos = (const Real *)pos->data;
  const Real *x_pos = (const Real *)x->data;
  const Real *e_rad = (const Real *)r->data;
  const Real *e_eps = (const Real *)eps->data;

  const int n = (int)p->no_particles;
  std::vector<Real> curr_rank_eddies(NCOMP * n);
  for (int i = 0; i < n; i++) {
    Real *s = &curr_rank_eddies[NCOMP * i];
    s[E_X] = x_pos[i];
    s[E_Y] = yz_pos[2 * i];
    s[E_Z] = yz_pos[2 * i + 1];
    s[E_R] = e_rad[i];
    s[E_SX] = e_eps[3 * i];
    s[E_SY] = e_eps[3 * i + 1];
    s[E_SZ] = e_eps[3 * i + 2];
  }

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  if (!sb->owned) { all.assign(eddies * NCOMP, 0.0); return; }
  int nranks;
  MPI_Comm_size(sb->comm, &nranks);
  std::vector<int> cnt(nranks, 0), disp(nranks, 0);
  int num_curr_rank_vals = NCOMP * n;
  MPI_Allgather(&num_curr_rank_vals, 1, MPI_INT, cnt.data(), 1, MPI_INT, sb->comm);
  int total = 0;
  for (int rr = 0; rr < nranks; rr++) { disp[rr] = total; total += cnt[rr]; }
  if (total != eddies * NCOMP)
    throw OPSException(OPS_RUNTIME_ERROR, "eddy count drifted from the initial total");
  all.resize(total);
  MPI_Allgatherv(curr_rank_eddies.data(), num_curr_rank_vals, MPI_DOUBLE,
                 all.data(), cnt.data(), disp.data(), MPI_DOUBLE, sb->comm);
#else
  if (NCOMP * n != eddies * NCOMP)
    throw OPSException(OPS_RUNTIME_ERROR, "eddy count drifted from the initial total");
  all.swap(curr_rank_eddies);
#endif
}

// diagnostics: total owned, duplicate ids, and eddies outside their rank's box
static void check_eddies(ops_particle p, ops_dat pos, ops_dat gid,
                         int *owned, int *dup, int *outside, int *max_id) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)p->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};
  const Real *yz_pos = (const Real *)pos->data;
  const int *ip = (const int *)gid->data;

  const int n = (int)p->no_particles;
  int out = 0;
  for (int i = 0; i < n; i++)
    if (yz_pos[2 * i] < lo[0] || yz_pos[2 * i] >= hi[0] ||
        yz_pos[2 * i + 1] < lo[1] || yz_pos[2 * i + 1] >= hi[1]) out++;

  std::vector<int> ids(ip, ip + n);

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  if (!sb->owned) { *owned = *dup = *outside = *max_id = 0; return; }
  int nranks;
  MPI_Comm_size(sb->comm, &nranks);
  std::vector<int> cnt(nranks, 0), disp(nranks, 0);
  MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, sb->comm);
  int total = 0;
  for (int rr = 0; rr < nranks; rr++) { disp[rr] = total; total += cnt[rr]; }
  std::vector<int> allids(total);
  if (total > 0)
    MPI_Allgatherv(ids.data(), n, MPI_INT, allids.data(), cnt.data(), disp.data(),
                   MPI_INT, sb->comm);
  MPI_Allreduce(MPI_IN_PLACE, &out, 1, MPI_INT, MPI_SUM, sb->comm);
  ids.swap(allids);
  *owned = total;
#else
  *owned = n;
#endif

  std::sort(ids.begin(), ids.end());
  int d = 0;
  for (size_t i = 1; i < ids.size(); i++) if (ids[i] == ids[i - 1]) d++;
  *dup = d;
  *outside = out;
  *max_id = ids.empty() ? -1 : ids.back();
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

  // seed_eddies sets every field, so the first map build already has correct ghosts
  ops_particle_setup_partition();
  print_decomp(eddy_particle, d_grid);
  seed_eddies(eddy_particle, eddy_particle_pos, eddy_particle_x, eddy_particle_r,
              eddy_particle_eps, eddy_particle_id, eddy_particle_exit, eddies);
  ops_particle_setup_maps_with_dats(eddy_particle, dat_border, nborder);

  Real range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  std::vector<Real> eddy_all(eddies * NCOMP);
  std::vector<double> rank_split;
  reinsert_rank_splits(eddy_particle, rank_split);

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

    const int np = count_exits(eddy_particle, eddy_particle_exit);
    if (np > 0) {   // same np on every rank, so every rank rebuilds together
      reinsert_local(eddy_particle, eddy_particle_pos, eddy_particle_x, eddy_particle_r,
                     eddy_particle_eps, eddy_particle_exit, eddy_particle_id,
                     np, (unsigned int)it, rank_split, dat_border, nborder);
      n_reinserted += (long)np;
      n_rebuilds++;
    }

    ops_timers(&c1, &w1);
    t_reinsert += w1 - w0;

    ops_timers(&c0, &w0);
    gather_eddies(eddy_particle, eddy_particle_pos, eddy_particle_x,
                  eddy_particle_r, eddy_particle_eps, eddy_all);
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
      int owned = 0, dup = 0, outside = 0, max_id = 0;
      check_eddies(eddy_particle, eddy_particle_pos, eddy_particle_id,
                   &owned, &dup, &outside, &max_id);
      ops_printf("CHECK it %d owned %d dup %d outside_box %d max_id %d "
                 "reinserted %ld rebuilds %d\n",
                 it, owned, dup, outside, max_id, n_reinserted, n_rebuilds);

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
