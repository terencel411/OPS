/*
 * OPS Particles -- particle-mesh vs. direct all-to-all, head to head
 * ==================================================================
 *
 * Companion to apps/c/particle_global_influence, which computes a per-particle
 * influence as an exact sum over EVERY other particle in the domain, using an
 * array-valued ops_reduction as an allgather. That is O(N^2) with a global
 * collective every step. This app implements the alternative -- a
 * particle-mesh (PM) pipeline -- and runs BOTH in the same executable so the
 * comparison is apples to apples.
 *
 * THE TWO METHODS
 *
 *   direct   gather every particle onto every rank (ops_arg_reduce, OPS_INC),
 *            then each particle sums over the whole population.
 *            Exact. O(N^2). One array-sized collective per step.
 *
 *   mesh     deposit particle charge onto the grid (CIC), solve lap(phi) = rho
 *            on the grid, interpolate phi back to the particles.
 *            Approximate. O(N + grid). Communication is the ordinary OPS halo
 *            exchange; with -bc zero there is no array-sized collective at all.
 *
 * WHY THE INTERACTION LAW CHANGED
 *
 *   A mesh can only reproduce a pairwise sum if the interaction kernel is the
 *   Green's function of a local differential operator. In 2-D that is
 *   (1/2pi) ln r, NOT 1/r -- the 1/r law of particle_global_influence has no
 *   local PDE behind it in two dimensions, so it has no PM equivalent. This
 *   app therefore uses the softened 2-D Coulomb potential
 *
 *       phi_i = sum_{j != i}  m_j * (1/4pi) ln(r_ij^2 + eps^2)
 *
 *   for BOTH methods, so the only thing that differs is how it is evaluated.
 *   That is the honest comparison; pretending PM could reproduce the 1/r sum
 *   would not be.
 *
 *   Charges are signed and sum to EXACTLY zero (particles are created in
 *   +w/-w pairs). That is not decoration either: a neutral distribution has a
 *   potential that decays, which is what makes phi = 0 on a finite boundary a
 *   usable far-field condition.
 *
 * THE THREE APPROXIMATIONS IN PM, AND WHAT THIS APP DOES ABOUT EACH
 *
 *   1. mesh resolution -- the mesh cannot represent structure below h.
 *      Handled by setting eps = h so the pairwise sum is smoothed at the same
 *      scale. Refine with -nx and the difference falls.
 *
 *   2. the boundary condition -- see -bc below. `direct` removes this error
 *      entirely at the cost of keeping one collective; `zero` removes the
 *      collective and accepts the image-charge error.
 *
 *   3. self-energy -- a particle deposits its own charge and then reads it
 *      back, so it feels itself; the direct sum excludes j == i. Corrected by
 *      three constants fitted at step 0 against the CIC shape factors (see
 *      KerFitSelf) and then FROZEN, so the agreement reported at the end of
 *      the run -- after the cloud has rotated and spread -- is a real test
 *      rather than a fit. Reported raw, with the 1-constant model, and with
 *      the 3-shape model.
 *
 * OPTIONS
 *
 *   -npart N     particle count (must be even)           default 2000
 *   -nx N        grid nodes per side                     default 129
 *   -nsteps N    timesteps                               default 500
 *   -sweeps N    red-black SOR sweeps per step           default 40
   -cold N      sweeps for the initial cold solve       default 4000
 *   -bc zero     phi = 0 on the boundary, no collective
 *   -bc direct   boundary values from the exact sum      default
 *   -method both|direct|mesh                             default both
 *   -check       also verify `direct` against a host reference
 *
 * Build:  make influence_pm_dev_seq   /   make influence_pm_dev_mpi
 * Run:    ./influence_pm_dev_seq -npart 2000
 *         OMP_NUM_THREADS=1 mpirun -np 4 ./influence_pm_dev_mpi -npart 20000
 *
 * SET OMP_NUM_THREADS=1 FOR MPI RUNS. The build links OpenMP; with the
 * variable unset every rank spawns one thread per core, and on a 12-core box
 * np=4 thrashes 48 threads. Measured on the SOR solve: 7.5 ms serial,
 * 2663 ms at np=2 unset, 5.8 ms at np=2 with it set. It looks exactly like a
 * hang or a deadlock in the solver, and it is neither.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define OPS_2D

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>
#include <ops_particle_grid_seq.h>   /* ops_par_particle_grid_loop  */
#include <ops_grid_part_seq_v2.h>    /* the grid-outer scatter loop */

#include "particle_kernels.h"        /* defines NCOMP and the C_* offsets */
#include "grid_kernels.h"            /* uses them, so include order matters */
#include "scatter_loop.h"

typedef double Real;

/* ---- run configuration, all settable from the command line ---------- */

static int NPART = 2000;
static int NX = 129;
static int NSTEPS = 500;
static int SWEEPS = 40;
static bool BC_DIRECT = true;
static bool DO_DIRECT = true;
static bool DO_MESH = true;
static bool DO_CHECK = false;
static int COLD = 4000;   /* sweeps for the initial cold solve */

/* ---- fixed physics -------------------------------------------------- */

const Real LENGTH = 1.0;
const Real CLOUD = 0.16;  /* half-width of the seeded cloud, about the centre */
const unsigned int SEED = 12345u;
const Real ACCEL[2] = {0.06, 0.04};
const Real OMEGA = 0.5;
const Real DT = 0.002;

const Real FOUR_PI = 4.0 * 3.14159265358979323846;

/* ================================================================== *
 *  Command line
 * ================================================================== */

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-npart") && i + 1 < argc) NPART = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nx") && i + 1 < argc) NX = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nsteps") && i + 1 < argc) NSTEPS = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-sweeps") && i + 1 < argc) SWEEPS = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-bc") && i + 1 < argc) BC_DIRECT = !strcmp(argv[++i], "direct");
    else if (!strcmp(argv[i], "-method") && i + 1 < argc) {
      const char *m = argv[++i];
      DO_DIRECT = strcmp(m, "mesh") != 0;
      DO_MESH = strcmp(m, "direct") != 0;
    }
    else if (!strcmp(argv[i], "-cold") && i + 1 < argc) COLD = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-check")) DO_CHECK = true;
  }

  if (NPART % 2 != 0) NPART++;      /* charges come in +/- pairs */
  if (NX % 2 == 0) NX++;            /* odd node count -> even cell count */
}

/* ================================================================== *
 *  Seeding
 * ================================================================== *
 *
 * As in particle_global_influence: every rank walks the whole global sequence
 * and keeps only the candidates inside its own subdomain, so the index into
 * that sequence is a global id -- unique, dense in [0,NPART), identical at any
 * rank count, and free.
 *
 * The one addition is that charges are created in +w / -w pairs, giving
 * sum(m) == 0 to the last bit.
 */
static void seed_particles(ops_particle particle, ops_dat pos, ops_dat mass,
                           ops_dat gid, std::vector<Real> &seed_x,
                           std::vector<Real> &seed_m) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};
  const Real c = 0.5 * LENGTH;

  if (NPART > (int)particle->Nmax) ops_particle_realloc_data(particle, NPART);

  Real *xp = (Real *)pos->data;
  Real *mp = (Real *)mass->data;
  int *ip = (int *)gid->data;

  std::mt19937 rng(SEED);
  std::uniform_real_distribution<Real> draw(c - CLOUD, c + CLOUD);

  int n = 0;
  for (int i = 0; i < NPART; i++) {
    Real x = draw(rng);
    Real y = draw(rng);

    /* Pair 2k with 2k+1: same magnitude, opposite sign. */
    const Real w = 1.0 + 0.25 * static_cast<Real>((i / 2) % 5);
    const Real m = (i % 2 == 0) ? w : -w;

    seed_x[2 * i] = x;
    seed_x[2 * i + 1] = y;
    seed_m[i] = m;

    if (x < lo[0] || x >= hi[0] || y < lo[1] || y >= hi[1]) continue;

    xp[2 * n] = x;
    xp[2 * n + 1] = y;
    mp[n] = m;
    ip[n] = i;
    n++;
  }

  particle->no_particles = n;
}

static void update_maps(ops_particle particle, ops_dat *dat_border, int nborder,
                        ops_dat *dat_forward, int nforward) {
  int decide = ops_particle_update_map_lists_actual_hybrid(particle);
  ops_particle_remove_delete_maps(particle, decide);
  if (decide)
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);
  else
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);
  ops_particle_reset_flags(particle, decide);
}

/* ================================================================== *
 *  Host reference for the direct method
 * ================================================================== *
 * Replays the same recurrence, then evaluates the same sum in the same order,
 * so a correct direct method matches to the last bit.
 */
static void reference_direct(const std::vector<Real> &seed_x,
                             const std::vector<Real> &seed_m, int nsteps,
                             Real eps2, std::vector<Real> &phi_ref) {

  std::vector<Real> x(2 * NPART), v(2 * NPART, 0.0);
  for (int i = 0; i < 2 * NPART; i++) x[i] = seed_x[i];

  const Real cx = 0.5 * LENGTH, cy = 0.5 * LENGTH;
  for (int i = 0; i < NPART; i++) {
    v[2 * i + 0] = -OMEGA * (x[2 * i + 1] - cy);
    v[2 * i + 1] = OMEGA * (x[2 * i + 0] - cx);
  }

  for (int s = 0; s < nsteps; s++)
    for (int i = 0; i < NPART; i++)
      for (int d = 0; d < 2; d++) {
        v[2 * i + d] += ACCEL[d] * DT;
        x[2 * i + d] += v[2 * i + d] * DT;
      }

  const Real k = 1.0 / FOUR_PI;
  for (int i = 0; i < NPART; i++) {
    Real sum = 0.0;
    for (int j = 0; j < NPART; j++) {
      if (j == i) continue;
      const Real dx = x[2 * i] - x[2 * j];
      const Real dy = x[2 * i + 1] - x[2 * j + 1];
      sum += seed_m[j] * log(dx * dx + dy * dy + eps2);
    }
    phi_ref[i] = k * sum;
  }
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);
  parse_args(argc, argv);

  const int NY = NX;

  /* ---- 1. block, grid dats, stencils ------------------------------- */

  ops_block block = ops_decl_block(2, "pmesh_block");

  int size[] = {NX, NY};
  int base[] = {0, 0};
  int d_m[] = {-1, -1};
  int d_p[] = {1, 1};

  Real *null_dbl = NULL;
  int *null_int = NULL;

  ops_dat x_grid = ops_decl_dat(block, 2, size, base, d_m, d_p, null_dbl,
                                "double", "x_grid");
  ops_dat g_q = ops_decl_dat(block, 1, size, base, d_m, d_p, null_dbl,
                             "double", "charge");
  ops_dat g_qabs = ops_decl_dat(block, 1, size, base, d_m, d_p, null_dbl,
                                "double", "charge_abs");
  ops_dat g_phi = ops_decl_dat(block, 1, size, base, d_m, d_p, null_dbl,
                               "double", "potential");

  int s00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s00, "0,0");

  int s5[] = {0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
  ops_stencil S2D_5pt = ops_decl_stencil(2, 5, s5, "5pt");

  int s9[] = {-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1};
  ops_stencil S2D_9pt = ops_decl_stencil(2, 9, s9, "9pt");

  /* ---- 2. reduction handles ---------------------------------------- */

  ops_reduction h_all = ops_decl_reduction_handle(
      NPART * NCOMP * sizeof(double), "double", "all_state");
  ops_reduction h_scalar =
      ops_decl_reduction_handle(NPART * sizeof(double), "double", "all_scalar");
  ops_reduction h_r1 = ops_decl_reduction_handle(sizeof(double), "double", "r1");
  ops_reduction h_r2 = ops_decl_reduction_handle(sizeof(double), "double", "r2");
  ops_reduction h_r3 = ops_decl_reduction_handle(sizeof(double), "double", "r3");
  ops_reduction h_max = ops_decl_reduction_handle(sizeof(double), "double", "mx");
  ops_reduction h_cnt = ops_decl_reduction_handle(sizeof(int), "int", "cnt");
  ops_reduction h_fit =
      ops_decl_reduction_handle(11 * sizeof(double), "double", "self_fit");

  /* ---- 3. particles ------------------------------------------------- */

  Real dx_box[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, x_grid, 2, dx_box);

  ops_particle particle = ops_decl_particle(block, "charges", box);

  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 2, base, null_dbl,
                                            "double", "position");
  ops_dat p_vel = ops_decl_particle_dat(particle, 2, base, null_dbl, "double",
                                        "velocity");
  ops_dat p_mass = ops_decl_particle_dat(particle, 1, base, null_dbl, "double",
                                         "charge");
  ops_dat p_phid = ops_decl_particle_dat(particle, 1, base, null_dbl, "double",
                                         "phi_direct");
  ops_dat p_phim = ops_decl_particle_dat(particle, 1, base, null_dbl, "double",
                                         "phi_mesh");
  ops_dat p_gid = ops_decl_particle_dat(particle, 1, base, null_int, "int",
                                        "gid");

  /* Halo depth 1. Not a tuning choice: the particle ghost band is only
     correct to one cell under MPI (particle_tutorial_3_deposit README 3a),
     and CIC needs exactly one. */
  ops_particle_mapping map = ops_decl_mapping(
      particle, x_grid, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  ops_dat dat_border[] = {p_pos, p_vel, p_mass, p_phid, p_phim, p_gid};
  ops_dat dat_forward[] = {p_pos, p_vel, p_mass, p_gid};
  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);

  /* ---- 4. partition, coordinates, particle setup ------------------- */

  ops_partition("");

  const Real h = LENGTH / static_cast<Real>(NX - 1);
  Real hh = h;
  int grid_range[] = {0, NX, 0, NY};
  int interior[] = {1, NX - 1, 1, NY - 1};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(&hh, 1, "double", OPS_READ), ops_arg_idx());

  ops_particle_setup_partition();

  std::vector<Real> seed_x(2 * NPART), seed_m(NPART);
  seed_particles(particle, p_pos, p_mass, p_gid, seed_x, seed_m);

  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

  /* Softening matched to the mesh: below h the mesh cannot represent anything,
     so smoothing the pairwise sum at the same scale is what makes the two
     methods comparable at all. */
  Real eps2 = h * h;
  Real kcoef = 1.0 / FOUR_PI;
  Real range_parts[] = {0.0, LENGTH, 0.0, LENGTH};
  int npart_gbl = NPART;

  const Real centre[2] = {0.5 * LENGTH, 0.5 * LENGTH};
  Real omega_swirl = OMEGA;
  Real dt = DT;

  ops_printf("\n=== particle-mesh vs direct all-to-all ===\n");
  ops_printf("particles %d   grid %dx%d (h = %.5f)   steps %d\n", NPART, NX, NY,
             h, NSTEPS);
  ops_printf("kernel  phi_i = sum_j m_j ln(r^2 + eps^2)/4pi,  eps = h\n");
  ops_printf("methods %s%s   bc %s   sor sweeps/step %d\n",
             DO_DIRECT ? "direct " : "", DO_MESH ? "mesh" : "",
             BC_DIRECT ? "direct-sum" : "zero", SWEEPS);

  /* ---- 5. initial state -------------------------------------------- */

  ops_particle_par_loop(
      KerInitState, "KerInitState", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
      range_parts, map,
      ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
      ops_arg_gbl(&omega_swirl, 1, "double", OPS_READ),
      ops_arg_gbl(centre, 2, "double", OPS_READ));

  ops_par_loop(KerZeroScalar, "KerZeroPhiG", block, 2, grid_range,
               ops_arg_dat(g_phi, 1, S2D_00, "double", OPS_WRITE));

  /* ---- timing accumulators ---------------------------------------- */

  double t_gather = 0, t_direct = 0, t_deposit = 0, t_bc = 0, t_solve = 0,
         t_interp = 0, t_move = 0, c0, w0, c1, w1;

  std::vector<Real> all_state(NPART * NCOMP);

  /* SOR over-relaxation factor: the classic optimum for a 5-point Laplacian on
     an n x n mesh. Turns O(n^2) Jacobi iterations into O(n). */
  const Real omega_sor = 2.0 / (1.0 + sin(M_PI / static_cast<Real>(NX - 1)));
  Real om = omega_sor;

  /* ---- the two evaluators ----------------------------------------- */

  auto gather_state = [&]() {
    ops_particle_par_loop(
        KerPublishState, "KerPublishState", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_reduce(h_all, NPART * NCOMP, "double", OPS_INC));
    ops_reduction_result(h_all, all_state.data());
  };

  auto direct_influence = [&]() {
    ops_particle_par_loop(
        KerInfluenceDirect, "KerInfluenceDirect", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_WRITE),
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_gbl(all_state.data(), NPART * NCOMP, "double", OPS_READ),
        ops_arg_gbl(&npart_gbl, 1, "int", OPS_READ),
        ops_arg_gbl(&eps2, 1, "double", OPS_READ),
        ops_arg_gbl(&kcoef, 1, "double", OPS_READ));
  };

  auto deposit = [&]() {
    ops_par_loop(KerZeroScalar, "KerZeroQ", block, 2, grid_range,
                 ops_arg_dat(g_q, 1, S2D_00, "double", OPS_WRITE));
    ops_par_loop(KerZeroScalar, "KerZeroQabs", block, 2, grid_range,
                 ops_arg_dat(g_qabs, 1, S2D_00, "double", OPS_WRITE));

    ops_par_scatter_loop(
        KerDepositCIC, "KerDepositCIC", particle, map, S2D_9pt, 2, grid_range,
        ops_arg_dat(g_q, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(g_qabs, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_READ),
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
        ops_arg_gbl(&hh, 1, "double", OPS_READ));
  };

  auto set_boundary = [&]() {
    if (BC_DIRECT) {
      /* Four thin ranges: the box edge only. O(N * N_boundary). */
      int edges[4][4] = {{0, 1, 0, NY},
                         {NX - 1, NX, 0, NY},
                         {1, NX - 1, 0, 1},
                         {1, NX - 1, NY - 1, NY}};
      for (int e = 0; e < 4; e++)
        ops_par_loop(KerBoundaryDirect, "KerBoundaryDirect", block, 2, edges[e],
                     ops_arg_dat(g_phi, 1, S2D_00, "double", OPS_WRITE),
                     ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_READ),
                     ops_arg_gbl(all_state.data(), NPART * NCOMP, "double",
                                 OPS_READ),
                     ops_arg_gbl(&npart_gbl, 1, "int", OPS_READ),
                     ops_arg_gbl(&eps2, 1, "double", OPS_READ),
                     ops_arg_gbl(&kcoef, 1, "double", OPS_READ));
    }
    /* -bc zero: g_phi's boundary was zeroed at startup and the SOR only
       touches the interior, so there is nothing to do here -- and nothing to
       communicate globally. */
  };

  auto sor_sweeps = [&](int nsweeps) {
    for (int s = 0; s < nsweeps; s++)
      for (int colour = 0; colour < 2; colour++) {
        int col = colour;
        ops_par_loop(KerSOR, "KerSOR", block, 2, interior,
                     ops_arg_dat(g_phi, 1, S2D_5pt, "double", OPS_RW),
                     ops_arg_dat(g_q, 1, S2D_00, "double", OPS_READ),
                     ops_arg_idx(), ops_arg_gbl(&om, 1, "double", OPS_READ),
                     ops_arg_gbl(&col, 1, "int", OPS_READ));
      }
  };

  auto residual = [&]() {
    Real r = 0.0;
    ops_par_loop(KerResidual, "KerResidual", block, 2, interior,
                 ops_arg_dat(g_phi, 1, S2D_5pt, "double", OPS_READ),
                 ops_arg_dat(g_q, 1, S2D_00, "double", OPS_READ),
                 ops_arg_reduce(h_r1, 1, "double", OPS_INC));
    ops_reduction_result(h_r1, &r);
    return sqrt(r);
  };

  auto interpolate = [&]() {
    ops_particle_par_loop(
        KerZeroPhiP, "KerZeroPhiP", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
        range_parts, map,
        ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_WRITE));

    ops_par_particle_grid_loop(
        KerInterpCIC, "KerInterpCIC", particle, map, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, S2D_9pt,
        ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat(g_phi, 1, S2D_9pt, "double", OPS_READ),
        ops_arg_dat(x_grid, 2, S2D_9pt, "double", OPS_READ),
        ops_arg_gbl(&hh, 1, "double", OPS_READ));
  };

  /* ---- 6. deposit conservation check ------------------------------- */

  if (DO_MESH) {
    if (BC_DIRECT) gather_state();
    deposit();

    Real qsum = 0.0, qabssum = 0.0;
    ops_par_loop(KerSumScalar, "KerSumQ", block, 2, grid_range,
                 ops_arg_dat(g_q, 1, S2D_00, "double", OPS_READ),
                 ops_arg_reduce(h_r2, 1, "double", OPS_INC));
    ops_reduction_result(h_r2, &qsum);
    ops_par_loop(KerSumScalar, "KerSumQabs", block, 2, grid_range,
                 ops_arg_dat(g_qabs, 1, S2D_00, "double", OPS_READ),
                 ops_arg_reduce(h_r3, 1, "double", OPS_INC));
    ops_reduction_result(h_r3, &qabssum);

    Real want = 0.0;
    for (int i = 0; i < NPART; i++) want += fabs(seed_m[i]);

    ops_printf("\ndeposit conservation: sum|q| on mesh = %.10f, sum|m| = %.10f,"
               " rel err %.2e\n", qabssum, want, fabs(qabssum - want) / want);
    ops_printf("                     sum q on mesh  = %.3e (neutral cloud)\n",
               qsum);

    /* Cold solve: no previous field to warm start from. */
    set_boundary();
    Real r_before = residual();
    sor_sweeps(COLD);
    ops_printf("cold Poisson solve: residual %.3e -> %.3e (%d sweeps,"
               " omega = %.4f)\n", r_before, residual(), COLD, omega_sor);
  }

  /* ---- 7. self-energy calibration at step 0 ------------------------ */

  Real coef1[3] = {0, 0, 0};   /* one-constant model  */
  Real coef3[3] = {0, 0, 0};   /* three-shape model   */

  if (DO_MESH && DO_DIRECT) {
    gather_state();
    direct_influence();
    interpolate();

    Real f[11] = {};
    ops_particle_par_loop(
        KerFitSelf, "KerFitSelf", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
        range_parts, map,
        ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_gbl(&hh, 1, "double", OPS_READ),
        ops_arg_reduce(h_fit, 11, "double", OPS_INC));
    ops_reduction_result(h_fit, f);

    /* One constant: c = sum(m d) / sum(m^2). */
    const Real c1 = (f[10] > 0.0) ? f[9] / f[10] : 0.0;
    coef1[0] = coef1[1] = coef1[2] = c1;

    /* Three shapes: solve the symmetric 3x3 normal equations by Cramer. */
    const Real M[3][3] = {{f[0], f[1], f[2]},
                          {f[1], f[3], f[4]},
                          {f[2], f[4], f[5]}};
    const Real r[3] = {f[6], f[7], f[8]};
    const Real det =
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
        M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
        M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    if (fabs(det) > 1e-30) {
      for (int k = 0; k < 3; k++) {
        Real Mk[3][3];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) Mk[a][b] = (b == k) ? r[a] : M[a][b];
        coef3[k] = (Mk[0][0] * (Mk[1][1] * Mk[2][2] - Mk[1][2] * Mk[2][1]) -
                    Mk[0][1] * (Mk[1][0] * Mk[2][2] - Mk[1][2] * Mk[2][0]) +
                    Mk[0][2] * (Mk[1][0] * Mk[2][1] - Mk[1][1] * Mk[2][0])) / det;
      }
    } else {
      coef3[0] = coef3[1] = coef3[2] = c1;
    }

    ops_printf("self-energy fitted at step 0 (then frozen):\n");
    ops_printf("   one constant   c  = %+.6f\n", c1);
    ops_printf("   three shapes  (a,b,c) = %+.6f %+.6f %+.6f\n", coef3[0],
               coef3[1], coef3[2]);
  }

  /* ---- 8. time loop ----------------------------------------------- */

  for (int step = 1; step <= NSTEPS; step++) {

    ops_timers(&c0, &w0);
    ops_particle_par_loop(
        KerAdvance, "KerAdvance", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
        range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_RW),
        ops_arg_gbl(ACCEL, 2, "double", OPS_READ),
        ops_arg_gbl(&dt, 1, "double", OPS_READ));
    update_maps(particle, dat_border, nborder, dat_forward, nforward);
    ops_timers(&c1, &w1);
    t_move += w1 - w0;

    /* ---- direct: gather + O(N^2) sum ---- */
    if (DO_DIRECT || BC_DIRECT) {
      ops_timers(&c0, &w0);
      gather_state();
      ops_timers(&c1, &w1);
      t_gather += w1 - w0;
    }
    if (DO_DIRECT) {
      ops_timers(&c0, &w0);
      direct_influence();
      ops_timers(&c1, &w1);
      t_direct += w1 - w0;
    }

    /* ---- mesh: deposit -> boundary -> solve -> interpolate ---- */
    if (DO_MESH) {
      ops_timers(&c0, &w0);
      deposit();
      ops_timers(&c1, &w1);
      t_deposit += w1 - w0;

      ops_timers(&c0, &w0);
      set_boundary();
      ops_timers(&c1, &w1);
      t_bc += w1 - w0;

      ops_timers(&c0, &w0);
      sor_sweeps(SWEEPS);
      ops_timers(&c1, &w1);
      t_solve += w1 - w0;

      ops_timers(&c0, &w0);
      interpolate();
      ops_timers(&c1, &w1);
      t_interp += w1 - w0;
    }
  }

  /* ---- 9. accuracy report ----------------------------------------- */

  ops_printf("\n--- accuracy at the final step ------------------------\n");

  if (DO_MESH) ops_printf("Poisson residual after the run: %.3e\n", residual());

  if (DO_MESH && DO_DIRECT) {
    Real mx = 0, d2 = 0, r2 = 0;
    int cnt = 0;

    auto compare = [&](const char *label) {
      ops_particle_par_loop(
          KerCompare, "KerCompare", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
          range_parts, map,
          ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_READ),
          ops_arg_reduce(h_max, 1, "double", OPS_MAX),
          ops_arg_reduce(h_r2, 1, "double", OPS_INC),
          ops_arg_reduce(h_r3, 1, "double", OPS_INC),
          ops_arg_reduce(h_cnt, 1, "int", OPS_INC));
      ops_reduction_result(h_max, &mx);
      ops_reduction_result(h_r2, &d2);
      ops_reduction_result(h_r3, &r2);
      ops_reduction_result(h_cnt, &cnt);
      ops_printf("%-22s max |dphi| %.4e   rms |dphi| %.4e"
                 "   rms(phi) %.4e   -> %6.2f %%\n",
                 label, mx, sqrt(d2 / cnt), sqrt(r2 / cnt),
                 100.0 * sqrt(d2 / r2));
    };

    auto subtract = [&](const Real *coef) {
      ops_particle_par_loop(
          KerSubtractSelf, "KerSubtractSelf", particle, 2,
          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
          ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_RW),
          ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
          ops_arg_gbl(&hh, 1, "double", OPS_READ),
          ops_arg_gbl(coef, 3, "double", OPS_READ));
    };

    compare("raw (self-energy in)");

    subtract(coef1);
    compare("- 1-constant self");

    /* Already corrected by coef1, so apply only the difference. */
    Real delta[3] = {coef3[0] - coef1[0], coef3[1] - coef1[1],
                     coef3[2] - coef1[2]};
    subtract(delta);
    compare("- 3-shape self");

    ops_printf("particles compared: %d of %d\n", cnt, NPART);

    /* ---- error budget: where does the residual live? ---- */
    {
      ops_reduction h_sh = ops_decl_reduction_handle(
          3 * NNBIN * sizeof(double), "double", "err_by_shape");
      std::vector<Real> sh(3 * NNBIN, 0.0);
      ops_particle_par_loop(
          KerErrorByShape, "KerErrorByShape", particle, 2,
          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
          ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
          ops_arg_gbl(&hh, 1, "double", OPS_READ),
          ops_arg_reduce(h_sh, 3 * NNBIN, "double", OPS_INC));
      ops_reduction_result(h_sh, sh.data());
      ops_printf("\nresidual/m vs CIC shape factor A (self-energy test):\n");
      ops_printf("     A        count     mean(d/m)    rms(d/m)\n");
      for (int b = 0; b < NNBIN; b++) {
        if (sh[3 * b + 2] < 0.5) continue;
        const Real c = sh[3 * b + 2];
        ops_printf("  %.3f-%.3f %7.0f   %+10.4f   %10.4f\n",
                   0.25 + 0.75 * b / NNBIN, 0.25 + 0.75 * (b + 1) / NNBIN, c,
                   sh[3 * b] / c, sqrt(sh[3 * b + 1] / c));
      }

      ops_reduction h_nn = ops_decl_reduction_handle(
          2 * NNBIN * sizeof(double), "double", "err_by_nn");
      std::vector<Real> nn(2 * NNBIN, 0.0);

      gather_state();
      ops_particle_par_loop(
          KerErrorByNN, "KerErrorByNN", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
          range_parts, map,
          ops_arg_dat_particle(p_phim, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
          ops_arg_gbl(all_state.data(), NPART * NCOMP, "double", OPS_READ),
          ops_arg_gbl(&npart_gbl, 1, "int", OPS_READ),
          ops_arg_gbl(&hh, 1, "double", OPS_READ),
          ops_arg_reduce(h_nn, 2 * NNBIN, "double", OPS_INC));
      ops_reduction_result(h_nn, nn.data());

      ops_printf("\nerror vs nearest-neighbour distance"
                 " (rms |dphi| per bin, rms(phi) = %.3e):\n", sqrt(r2 / cnt));
      ops_printf("  r_nn/h      count    rms|dphi|   share of total err^2\n");
      Real tot = 0.0;
      for (int b = 0; b < NNBIN; b++) tot += nn[2 * b];
      for (int b = 0; b < NNBIN; b++) {
        if (nn[2 * b + 1] < 0.5) continue;
        ops_printf("  %.3f-%.3f %7.0f   %10.3e   %6.1f %%\n", b / 8.0,
                   (b + 1) / 8.0, nn[2 * b + 1],
                   sqrt(nn[2 * b] / nn[2 * b + 1]),
                   100.0 * nn[2 * b] / (tot > 0 ? tot : 1.0));
      }
    }
  }

  if (DO_CHECK && DO_DIRECT) {
    std::vector<Real> phi_ref(NPART), phi_got(NPART);
    reference_direct(seed_x, seed_m, NSTEPS, eps2, phi_ref);

    ops_particle_par_loop(
        KerPublishScalar, "KerPublishPhiD", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_phid, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_reduce(h_scalar, NPART, "double", OPS_INC));
    ops_reduction_result(h_scalar, phi_got.data());

    Real worst = 0.0;
    for (int i = 0; i < NPART; i++)
      worst = fmax(worst, fabs(phi_got[i] - phi_ref[i]));
    ops_printf("direct vs host reference: max |dphi| = %.3e  -> %s\n", worst,
               worst < 1e-9 ? "PASS" : "FAIL");
  }

  /* ---- 10. timing report ------------------------------------------ */

  const double n = static_cast<double>(NSTEPS);
  const double ms = 1000.0 / n;

  ops_printf("\n--- cost per timestep (ms, wall) ---------------------\n");
  ops_printf("motion + migration      %9.3f\n", t_move * ms);
  if (DO_DIRECT || BC_DIRECT)
    ops_printf("gather (allgather)      %9.3f\n", t_gather * ms);
  if (DO_DIRECT)
    ops_printf("DIRECT  O(N^2) sum      %9.3f\n", t_direct * ms);
  if (DO_MESH) {
    ops_printf("MESH    deposit         %9.3f\n", t_deposit * ms);
    ops_printf("MESH    boundary        %9.3f\n", t_bc * ms);
    ops_printf("MESH    %4d SOR sweeps %9.3f\n", SWEEPS, t_solve * ms);
    ops_printf("MESH    interpolate     %9.3f\n", t_interp * ms);
  }

  const double direct_total = (t_direct + t_gather) * ms;
  const double mesh_total =
      (t_deposit + t_bc + t_solve + t_interp + (BC_DIRECT ? t_gather : 0.0)) * ms;

  if (DO_DIRECT) ops_printf("        DIRECT total    %9.3f\n", direct_total);
  if (DO_MESH) ops_printf("        MESH   total    %9.3f\n", mesh_total);
  if (DO_DIRECT && DO_MESH && mesh_total > 0.0)
    ops_printf("\nmesh is %.2fx %s than direct at N = %d\n",
               direct_total > mesh_total ? direct_total / mesh_total
                                         : mesh_total / direct_total,
               direct_total > mesh_total ? "FASTER" : "slower", NPART);

  ops_exit();
  return 0;
}
