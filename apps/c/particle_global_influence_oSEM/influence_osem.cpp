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

#include "osem_common.h"
#include "particle_kernels.h"
#include "grid_kernels.h"
#include "osem_io.h"

typedef double Real;

/* ---- oSEM's parameters, unchanged ---------------------------------- */

static const Real U0 = 823.6;
static const Real DT = 0.00000002;
static const Real DELTA = 0.007;
static const Real TI = 0.01;

static int NY = 100;
static int NZ = 150;
static int NITER = 2000;
static int NPRINT = 200;
static int NOUT = 0;   /* HDF5 frame interval; 0 = off */

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-niter") && i + 1 < argc) NITER = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-ny") && i + 1 < argc) NY = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nz") && i + 1 < argc) NZ = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nprint") && i + 1 < argc) NPRINT = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nout") && i + 1 < argc) NOUT = atoi(argv[++i]);
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
                        ops_dat rng, int neddy, const Real *prm) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

  if (neddy > (int)particle->Nmax) ops_particle_realloc_data(particle, neddy);

  Real *xp = (Real *)pos->data;
  int *ip = (int *)gid->data;
  int *sp = (int *)rng->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    /* Two draws per eddy from a stream seeded by its id. */
    unsigned int s = lcg_next(2463534242u + 2654435761u * (unsigned int)i);
    s = lcg_next(s);
    const Real y = prm[P_EYMIN] + (prm[P_EYMAX] - prm[P_EYMIN]) * lcg_unit(s);
    s = lcg_next(s);
    const Real z = prm[P_EZMIN] + (prm[P_EZMAX] - prm[P_EZMIN]) * lcg_unit(s);

    if (y < lo[0] || y >= hi[0] || z < lo[1] || z >= hi[1]) continue;

    xp[2 * n] = y;
    xp[2 * n + 1] = z;
    ip[n] = i;
    sp[n] = (int)s;
    n++;
  }

  particle->no_particles = n;
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
  parse_args(argc, argv);

  /* ---- 1. geometry, exactly oSEM's ------------------------------- */

  const Real r_max = 0.41 * DELTA;
  const Real x_min = -r_max, x_max = r_max, x_plane = 0.0;
  const Real y_min = 0.0, y_max = 0.009;
  const Real z_min = 0.0, z_max = 0.05;
  const Real eddy_y_min = y_min, eddy_y_max = y_max + r_max;
  const Real eddy_z_min = z_min - r_max, eddy_z_max = z_max + r_max;
  const Real rep_radius = 0.2 * DELTA;

  const Real vol = fabs((x_max - x_min) * (y_max - y_min + 2 * r_max) *
                        (z_max - z_min + 2 * r_max));
  const int NEDDY = (int)trunc(vol / pow(rep_radius, 3));

  Real prm[NPARAM];
  prm[P_XMIN] = x_min;
  prm[P_XMAX] = x_max;
  prm[P_XPLANE] = x_plane;
  prm[P_EYMIN] = eddy_y_min;
  prm[P_EYMAX] = eddy_y_max;
  prm[P_EZMIN] = eddy_z_min;
  prm[P_EZMAX] = eddy_z_max;
  prm[P_RADIUS] = rep_radius;
  prm[P_INCREMENT] = U0 * DT;
  prm[P_SHAPENORM] = 1.0 / 1.5829045;
  prm[P_U0TI] = U0 * TI;
  /* One flow-through takes (x_max - x_min) / (u0*dt) steps. Size the transverse
     drift so an eddy can cross the box in about that time -- fast enough to
     decorrelate, and still well under one cell per step (0.29 cells in y,
     0.43 in z at the default resolution), which is what keeps migration local. */
  {
    const Real nflow = (x_max - x_min) / (U0 * DT);
    prm[P_VTY] = (eddy_y_max - eddy_y_min) / nflow;
    prm[P_VTZ] = (eddy_z_max - eddy_z_min) / nflow;
  }

  /* ---- 2. block, grid dats, stencils ----------------------------- */

  ops_block block = ops_decl_block(2, "osem_block");

  int size[] = {NY + 1, NZ + 1};
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
      NEDDY * NCOMP * sizeof(double), "double", "all_eddies");
  ops_reduction h_cnt =
      ops_decl_reduction_handle(sizeof(int), "int", "eddy_count");
  ops_reduction h_stat =
      ops_decl_reduction_handle(4 * sizeof(double), "double", "fluct_stats");
  ops_reduction h_sync = ops_decl_reduction_handle(sizeof(int), "int", "sync");

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
  ops_dat p_rng = ops_decl_particle_dat(particle, 1, base, ni, "int", "rng");

  ops_particle_mapping map = ops_decl_mapping(
      particle, crd, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  ops_dat dat_border[] = {p_pos, p_x, p_r, p_eps, p_vt, p_gid, p_rng};
  ops_dat dat_forward[] = {p_pos, p_x, p_r, p_eps, p_vt, p_gid, p_rng};
  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);

  /* ---- 5. partition, coordinates, particle setup ----------------- */

  ops_partition("");

  const Real org[2] = {eddy_y_min, eddy_z_min};
  const Real span[2] = {eddy_y_max - eddy_y_min, eddy_z_max - eddy_z_min};
  const int nn[2] = {NY, NZ};
  int grid_range[] = {0, NY + 1, 0, NZ + 1};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(crd, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(org, 2, "double", OPS_READ),
               ops_arg_gbl(span, 2, "double", OPS_READ),
               ops_arg_gbl(nn, 2, "int", OPS_READ), ops_arg_idx());

  ops_par_loop(KerInitRST, "KerInitRST", block, 2, grid_range,
               ops_arg_dat(a11, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a21, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a22, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a31, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a32, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(a33, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(prm, NPARAM, "double", OPS_READ));

  ops_particle_setup_partition();
  seed_eddies(particle, p_pos, p_gid, p_rng, NEDDY, prm);
  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

  Real range_parts[] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};
  int neddy_gbl = NEDDY;

  ops_printf("\n=== oSEM with OPS particles (2-D inlet) ===\n");
  ops_printf("inlet %d x %d nodes over [%.4f, %.4f] x [%.4f, %.4f]\n", NY + 1,
             NZ + 1, eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max);
  ops_printf("eddies %d   radius %.5f   u0*dt %.3e   %d steps\n", NEDDY,
             rep_radius, prm[P_INCREMENT], NITER);
  ops_printf("gather buffer %d x %d doubles = %zu bytes/step\n", NEDDY, NCOMP,
             NEDDY * NCOMP * sizeof(double));

  /* ---- 6. initialise the eddies ---------------------------------- */

  ops_particle_par_loop(
      KerInitEddy, "KerInitEddy", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
      range_parts, map,
      ops_arg_dat_particle(p_x, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_r, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_eps, 3, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_vt, 2, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_rng, 1, "int", particle, map, OPS_RW),
      ops_arg_gbl(prm, NPARAM, "double", OPS_READ));

  std::vector<Real> all_eddies(NEDDY * NCOMP);

  /* ---- 7. time loop ---------------------------------------------- */

  double c0, w0, c1, w1;
  double t_convect = 0, t_gather = 0, t_fluct = 0;
  int lost_at = -1;

  /* Output is off by default: the timings quoted in the README were measured
     without it, and a frame write is not part of any of the three kernels. */
  osem_io_params io = {NY,
                       NZ,
                       NEDDY,
                       NITER,
                       NOUT,
                       DT,
                       prm[P_U0TI],
                       x_plane,
                       {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max},
                       {0.0, 0.0, 0.0}};
  if (NOUT > 0) {
    remove_stale_output("osem_output", NITER, NOUT, h_sync);
    ops_printf("writing %d HDF5 frames (osem_output_??????.h5)\n", NITER / NOUT);
  } else {
    ops_printf("hint: add -nout 10 to write frames for plot_osem_h5.py\n");
  }

  for (int it = 1; it <= NITER; it++) {

    /* -- convect: purely per-eddy, exactly like the advance kernel -- */
    ops_timers(&c0, &w0);
    ops_particle_par_loop(
        KerConvectEddies, "KerConvectEddies", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_x, 1, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_r, 1, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_eps, 3, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_vt, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_rng, 1, "int", particle, map, OPS_RW),
        ops_arg_gbl(prm, NPARAM, "double", OPS_READ));

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
        ops_arg_reduce(h_all, NEDDY * NCOMP, "double", OPS_INC));
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
    if (count != NEDDY && lost_at < 0) {
      lost_at = it;
      ops_printf("\n*** eddy population changed at step %d: %d of %d ***\n", it,
                 count, NEDDY);
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
                 ops_arg_gbl(all_eddies.data(), NEDDY * NCOMP, "double",
                             OPS_READ),
                 ops_arg_gbl(&neddy_gbl, 1, "int", OPS_READ),
                 ops_arg_gbl(prm, NPARAM, "double", OPS_READ));
    ops_timers(&c1, &w1);
    t_fluct += w1 - w0;

    /* -- output (not timed) ---------------------------------------- */
    if (NOUT > 0 && it % NOUT == 0) {
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

      write_osem_step(block, crd, uprime, vprime, wprime, all_eddies, io, it);
    }

    if (it % NPRINT == 0) ops_printf("step %5d / %d\n", it, NITER);
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
    for (int i = 0; i < NEDDY; i++)
      for (int k = 0; k < 3; k++) m[k] += all_eddies[NCOMP * i + E_SX + k];
    ops_printf("\nmean eps (x,y,z) over %d eddies: %+.4f %+.4f %+.4f"
               "   (0 = balanced)\n", NEDDY, m[0] / NEDDY, m[1] / NEDDY,
               m[2] / NEDDY);
  }

  const Real n = (st[3] > 0.0) ? st[3] : 1.0;
  ops_printf("\n--- final inlet plane --------------------------------\n");
  ops_printf("rms u' = %.4f   v' = %.4f   w' = %.4f   (u0*TI = %.4f)\n",
             sqrt(st[0] / n), sqrt(st[1] / n), sqrt(st[2] / n), prm[P_U0TI]);
  ops_printf("eddy population: %s\n",
             lost_at < 0 ? "conserved for the whole run"
                         : "CHANGED -- see the message above");

  const double ms = 1000.0 / (double)NITER;
  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("convect + migrate   %9.3f\n", t_convect * ms);
  ops_printf("gather (allgather)  %9.3f\n", t_gather * ms);
  ops_printf("compute_fluct       %9.3f\n", t_fluct * ms);

  ops_exit();
  return (lost_at < 0) ? 0 : 1;
}
