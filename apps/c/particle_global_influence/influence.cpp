/*
 * OPS Particles -- all-to-all particle interaction with no application MPI
 * =======================================================================
 *
 * THE PROBLEM
 *
 *   Particles carry position and velocity. A kernel advances them; that part
 *   is embarrassingly parallel, because a particle's new state depends only
 *   on its own dats and OPS has already split the population across ranks.
 *
 *   Then you want one more per-particle quantity -- call it the INFLUENCE --
 *   defined as a sum over EVERY OTHER PARTICLE IN THE DOMAIN:
 *
 *       phi_i = sum_{j != i}  m_j / (eps + |r_i - r_j|)
 *
 *   Nearest neighbour contributes most, the farthest one still contributes.
 *   Nothing is truncated, so this is a genuine all-to-all: rank 0 needs the
 *   positions of particles living on rank 7 and vice versa.
 *
 *   The particle library's own communication does NOT give you that. Border
 *   exchange and virtual (ghost) particles reach only as far as the halo band
 *   sized by the stencil passed to ops_decl_mapping -- a cell or two.
 *   ops_particle_inter_loop is likewise a neighbour search through that map.
 *   Both are built for short-range, cut-off interactions. There is no
 *   ops_particle_allgather.
 *
 * THE MECHANISM
 *
 *   An ops_reduction is already a global collective: under MPI,
 *   ops_reduction_result() ends in an MPI_Allreduce over OPS_MPI_GLOBAL and
 *   every rank gets the answer back (ops/c/src/mpi/ops_mpi_rt_support.cpp:1281).
 *   Nothing says a reduction has to be one scalar -- the handle is sized in
 *   BYTES and reduced element-wise:
 *
 *       ops_decl_reduction_handle(NPART * NCOMP * sizeof(double), "double", ...)
 *
 *   So: give every particle a globally unique id in [0,NPART), have each rank
 *   INC its own particles' state into their own slots of that array, and take
 *   the result. Every slot has exactly one non-zero contributor, so SUM
 *   returns the value itself, bit for bit. A sum-reduction over a disjointly
 *   filled array is an allgather -- expressed entirely in OPS.
 *
 *   Kernel C therefore becomes two loops with a reduction result between them:
 *
 *       C1  ops_particle_par_loop(KerPublishState, ... ops_arg_reduce(OPS_INC))
 *           ops_reduction_result(h_all, all_state)     <-- the Allreduce
 *       C2  ops_particle_par_loop(KerInfluence,    ... ops_arg_gbl (OPS_READ))
 *
 *   C2 sees the whole population as a flat read-only array and sums over it.
 *   The application contains no MPI call of any kind -- not in the physics,
 *   not in the verification, not in the particle count.
 *
 * WHAT THIS COSTS (read before scaling it up)
 *
 *   - one global collective per timestep: a hard synchronisation point
 *   - NPART*NCOMP*8 bytes moved per rank per step
 *   - O(NPART) work per particle, so O(NPART^2) per step overall
 *   - the MPI backend allocates dim*nranks elements of scratch inside the
 *     reduce (ops_mpi_rt_support.cpp:1282), so the buffer costs
 *     NPART*NCOMP*8*nranks bytes momentarily
 *
 *   Fine for thousands of particles. For millions, the answer is not a better
 *   gather, it is a better algorithm -- project onto the grid with
 *   ops_par_loop's particle overload, solve/smooth there, and interpolate back
 *   with ops_par_particle_grid_loop. That is O(N) and uses only the halo
 *   exchange OPS already does. See the README.
 *
 * THE RULES THIS DEPENDS ON
 *
 *   1. gids are globally unique and dense in [0,NPART). Here every rank walks
 *      the same seeding sequence, so the sequence index IS the gid, with no
 *      communication needed to assign it.
 *   2. C1 iterates OPS_PARTICLE_ITERATE_LOCAL. A ghost copy carries its
 *      owner's gid; iterating ALL would contribute it twice and double every
 *      gathered position.
 *   3. Every rank reaches ops_reduction_result() the same number of times.
 *      It is collective. Note that ops_particle_par_loop returns early when a
 *      rank owns no particles -- that is safe here, because ops_arg_reduce()
 *      is evaluated at the call site (which is what zeroes and registers the
 *      handle), not inside the loop body.
 *   4. The population is fixed. Insertion/deletion would move gids; see the
 *      README for what changes.
 *
 * Build:  make dev_seq      (fastest: no translator)
 *         make dev_mpi      (MPI, no translator)
 * Run:    ./influence_dev_seq
 *         mpirun -np 4 ./influence_dev_mpi
 *
 * The run is self-checking: the influence is recomputed on the host from the
 * known exact trajectory of all NPART particles and compared. A correct run
 * prints PASS, and prints the SAME numbers at any rank count.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define OPS_2D

#include <ops_seq_v2.h>       /* grid ops_par_loop                          */
#include <ops_particle_seq.h> /* ops_particle_par_loop, particle library    */

#include "grid_kernels.h"
#include "particle_kernels.h" /* also defines NCOMP and the C_* offsets     */
#include "influence_io.h"     /* per-step HDF5 frames, MPI-free             */

typedef double Real;

/* ------------------------------------------------------------------ *
 *  Simulation parameters
 * ------------------------------------------------------------------ */

const int NX = 41;   /* grid nodes in x                                  */
const int NY = 41;   /* grid nodes in y                                  */
const Real LENGTH = 1.0;

const int NPART = 200;            /* fixed population                     */
const unsigned int SEED = 12345u; /* fixes the seeding pattern            */

/* Seed inside a margin so the cloud never reaches the domain edge: there is
   no boundary condition in this app, and OPS deletes particles that leave
   the bounding box -- which would silently break the gather (a missing
   particle leaves its slot at zero, i.e. a phantom particle at the origin
   with zero strength). The margin is generous because the swirl below makes
   the cloud spread as well as drift; the run checks the count, so if you
   change these numbers and particles escape, it says FAIL rather than
   quietly giving a wrong answer.                                          */
const Real MARGIN = 0.30;

const Real ACCEL[2] = {0.06, 0.04}; /* constant acceleration               */
const Real OMEGA = 0.5;             /* initial swirl rate, see KerInitState */
const Real DT = 0.002;
const int NSTEPS = 750;
const int NPRINT = 15;    /* -> 50 HDF5 frames                            */

const Real SOFTEN = 0.01; /* eps in the influence law                     */

/* ================================================================== *
 *  Seeding
 * ================================================================== *
 *
 * Standard OPS Particles seeding (see particle_tutorial_1_drift for the full
 * discussion), with one extra requirement that the gather depends on:
 *
 *   EVERY RANK DRAWS THE WHOLE GLOBAL SEQUENCE and keeps only the candidates
 *   inside its own subdomain. The index into that sequence is then a global
 *   id that is unique, dense in [0,NPART), identical at any rank count, and
 *   costs no communication to assign.
 *
 * That id is what indexes the reduction buffer later. If you instead seed
 * only "your share" per rank, you need a prefix sum over per-rank counts to
 * build the same numbering -- and an OPS int reduction of dim = nranks does
 * that without MPI too (rank r INCs its count into slot r, then every rank
 * sums slots < r).
 *
 * Do not snap particles onto grid nodes: subdomain boundaries are cuts in
 * index space and land exactly on nodes, and a particle seeded on one is lost
 * in the first migration.
 */
void seed_particles(ops_particle particle, ops_dat pos, ops_dat vel,
                    ops_dat gid, ops_dat mass, std::vector<Real> &seed_x,
                    std::vector<Real> &seed_m) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};
  const Real glo[2] = {box->getGlobalMin().x, box->getGlobalMin().y};
  const Real ghi[2] = {box->getGlobalMax().x, box->getGlobalMax().y};

  if (NPART > (int)particle->Nmax)
    ops_particle_realloc_data(particle, NPART);

  Real *xp = (Real *)pos->data;
  Real *up = (Real *)vel->data;
  Real *mp = (Real *)mass->data;
  int *ip = (int *)gid->data;

  std::mt19937 rng(SEED);
  std::uniform_real_distribution<Real> draw_x(glo[0] + MARGIN, ghi[0] - MARGIN);
  std::uniform_real_distribution<Real> draw_y(glo[1] + MARGIN, ghi[1] - MARGIN);

  int n = 0;
  for (int i = 0; i < NPART; i++) {

    /* Draw for every candidate before the ownership test, so all ranks
       consume the generator identically.                                 */
    Real x = draw_x(rng);
    Real y = draw_y(rng);

    /* A per-particle strength, so the buffer carries something other than
       geometry and the gather is shown to be general.                     */
    Real m = 1.0 + 0.25 * static_cast<Real>(i % 5);

    /* The host reference needs the same starting state on every rank. */
    seed_x[2 * i] = x;
    seed_x[2 * i + 1] = y;
    seed_m[i] = m;

    if (x < lo[0] || x >= hi[0] || y < lo[1] || y >= hi[1]) continue;

    xp[2 * n] = x;
    xp[2 * n + 1] = y;
    up[2 * n] = 0.0;
    up[2 * n + 1] = 0.0;
    mp[n] = m;
    ip[n] = i; /* <-- the global id the gather is indexed by */

    n++;
  }

  particle->no_particles = n;
}

/* ================================================================== *
 *  Per-step map / migration cycle -- unchanged from tutorials 1 and 2
 * ================================================================== */
void update_maps(ops_particle particle, ops_dat *dat_border, int nborder,
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
 *  Host reference
 * ================================================================== *
 *
 * Every rank knows the whole seeding sequence, and the motion is a closed
 * form, so the exact answer is computable locally with no communication.
 * Replicating the same forward-Euler recurrence (rather than the analytic
 * n(n+1)/2 formula) keeps the arithmetic identical to the kernel's, so the
 * comparison is tight enough to catch a single misplaced particle.
 *
 * The summation order here matches KerInfluence exactly: ascending j,
 * skipping self.
 */
void reference_influence(const std::vector<Real> &seed_x,
                         const std::vector<Real> &seed_m, int nsteps,
                         std::vector<Real> &phi_ref) {

  std::vector<Real> x(2 * NPART), v(2 * NPART, 0.0);
  for (int i = 0; i < 2 * NPART; i++) x[i] = seed_x[i];

  /* Same swirl KerInitState applies, about the domain centre. */
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

  for (int i = 0; i < NPART; i++) {
    Real sum = 0.0;
    for (int j = 0; j < NPART; j++) {
      if (j == i) continue;
      const Real dx = x[2 * i] - x[2 * j];
      const Real dy = x[2 * i + 1] - x[2 * j + 1];
      sum += seed_m[j] / (SOFTEN + sqrt(dx * dx + dy * dy));
    }
    phi_ref[i] = sum;
  }
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  /* ---------------------------------------------------------------- *
   * 1. Block and grid
   * ---------------------------------------------------------------- */

  ops_block block = ops_decl_block(2, "influence_block");

  int size[] = {NX, NY};
  int base[] = {0, 0};
  int d_m[] = {-1, -1};
  int d_p[] = {1, 1};

  Real *null_dbl = NULL;
  int *null_int = NULL;

  ops_dat x_grid = ops_decl_dat(block, 2, size, base, d_m, d_p, null_dbl,
                                "double", "x_grid");

  /* ---------------------------------------------------------------- *
   * 2. Stencils
   * ---------------------------------------------------------------- */

  int s2d_00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2d_00, "0,0");

  int s2d_9pt[] = {-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1};
  ops_stencil S2D_9pt = ops_decl_stencil(2, 9, s2d_9pt, "9pt");

  /* ---------------------------------------------------------------- *
   * 3. Reduction handles -- the global "bus"
   * ---------------------------------------------------------------- *
   * h_all is the gather buffer: NPART particles x NCOMP doubles, reduced
   * element-wise. Size is in BYTES.
   *
   * h_worst / h_count exist so that even the verification is done through
   * OPS collectives rather than MPI_Allreduce.
   *
   * A reduction handle must be declared AFTER at least one ops_block exists
   * (the MPI build sizes its storage per block and throws otherwise).
   */

  ops_reduction h_all = ops_decl_reduction_handle(
      NPART * NCOMP * sizeof(double), "double", "all_particle_state");
  ops_reduction h_worst =
      ops_decl_reduction_handle(sizeof(double), "double", "worst_error");
  ops_reduction h_count =
      ops_decl_reduction_handle(sizeof(int), "int", "particles_seen");

  /* h_phi gathers the computed influence for output -- the same allgather
     again, one component per particle. h_sync is used only as a collective
     barrier by the HDF5 writer (see influence_io.h).                      */
  ops_reduction h_phi =
      ops_decl_reduction_handle(NPART * sizeof(double), "double", "all_phi");
  ops_reduction h_sync =
      ops_decl_reduction_handle(sizeof(int), "int", "sync");

  /* ---------------------------------------------------------------- *
   * 4. Bounding box, particle set, particle dats
   * ---------------------------------------------------------------- */

  Real dx_box[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, x_grid, 2, dx_box);

  ops_particle particle = ops_decl_particle(block, "influencers", box);

  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 2, base, null_dbl,
                                            "double", "position");

  ops_dat p_vel =
      ops_decl_particle_dat(particle, 2, base, null_dbl, "double", "velocity");
  ops_dat p_mass =
      ops_decl_particle_dat(particle, 1, base, null_dbl, "double", "strength");
  ops_dat p_phi =
      ops_decl_particle_dat(particle, 1, base, null_dbl, "double", "influence");
  ops_dat p_gid =
      ops_decl_particle_dat(particle, 1, base, null_int, "int", "gid");

  /* ---------------------------------------------------------------- *
   * 5. Mapping
   * ---------------------------------------------------------------- *
   * S2D_9pt here only widens the bin array's halo, i.e. "track particles up
   * to one cell outside my subdomain". Note what it does NOT do: it does not
   * bring distant particles within reach. That is precisely why the influence
   * sum needs the reduction bus and not the ghost layer.
   */

  ops_particle_mapping map = ops_decl_mapping(
      particle, x_grid, S2D_9pt, OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  /* ---------------------------------------------------------------- *
   * 6. Communication dat lists
   * ---------------------------------------------------------------- *
   * Ownership migration moves every dat declared with assign = true
   * automatically. The border list is what populates the GHOST layer, so a
   * dat belongs here if a kernel reads it on ghosts. Nothing in this app
   * iterates ITERATE_ALL, but the list is kept complete anyway: gid and
   * strength are exactly the fields whose loss would corrupt the gather in a
   * way that is hard to spot.
   */

  ops_dat dat_border[] = {p_pos, p_vel, p_mass, p_phi, p_gid};
  ops_dat dat_forward[] = {p_pos, p_vel};
  ops_dat dat_output[] = {p_gid, p_pos, p_vel, p_mass, p_phi};

  const int nborder = sizeof(dat_border) / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);
  const int noutput = sizeof(dat_output) / sizeof(dat_output[0]);

  /* ---------------------------------------------------------------- *
   * 7. Partition, fill coordinates, set up the particle side
   * ---------------------------------------------------------------- */

  ops_partition("");

  Real dx = LENGTH / static_cast<Real>(NX - 1);
  int grid_range[] = {0, NX, 0, NY};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(&dx, 1, "double", OPS_READ), ops_arg_idx());

  ops_particle_setup_partition();

  /* ---------------------------------------------------------------- *
   * 8. Seed and build the maps
   * ---------------------------------------------------------------- */

  std::vector<Real> seed_x(2 * NPART), seed_m(NPART);
  seed_particles(particle, p_pos, p_vel, p_gid, p_mass, seed_x, seed_m);

  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

  ops_printf("OPS Particles: all-to-all influence via an OPS reduction\n");
  ops_printf("grid %dx%d, %d particles, a = (%g, %g), swirl = %g, dt = %g,"
             " %d steps\n",
             NX, NY, NPART, ACCEL[0], ACCEL[1], OMEGA, DT, NSTEPS);
  ops_printf("gather buffer: %d x %d doubles = %zu bytes per step\n", NPART,
             NCOMP, NPART * NCOMP * sizeof(double));

  /* ---------------------------------------------------------------- *
   * 9. Kernel A -- initial state
   * ---------------------------------------------------------------- */

  Real range_parts[] = {0.0, LENGTH, 0.0, LENGTH};

  const Real centre[2] = {0.5 * LENGTH, 0.5 * LENGTH};
  Real omega = OMEGA;

  ops_particle_par_loop(
      KerInitState, "KerInitState", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
      range_parts, map,
      ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_phi, 1, "double", particle, map, OPS_WRITE),
      ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
      ops_arg_gbl(&omega, 1, "double", OPS_READ),
      ops_arg_gbl(centre, 2, "double", OPS_READ));

  /* ---------------------------------------------------------------- *
   * 10. Time loop
   * ---------------------------------------------------------------- */

  Real dt = DT;
  int npart_gbl = NPART;
  Real soften = SOFTEN;

  /* Where ops_reduction_result() delivers the gathered state. */
  std::vector<Real> all_state(NPART * NCOMP);
  std::vector<Real> all_phi(NPART);

  /* Everything a plot script needs about the run, travelling with the data. */
  influence_io_params io_params = {NX,
                                  NY,
                                  NPART,
                                  (int)SEED,
                                  NSTEPS,
                                  NPRINT,
                                  LENGTH,
                                  DT,
                                  {ACCEL[0], ACCEL[1]},
                                  OMEGA,
                                  SOFTEN,
                                  {box->getGlobalMin().x + MARGIN,
                                   box->getGlobalMax().x - MARGIN,
                                   box->getGlobalMin().y + MARGIN,
                                   box->getGlobalMax().y - MARGIN}};

  remove_stale_output("influence_output", NSTEPS, NPRINT, h_sync);

  for (int step = 1; step <= NSTEPS; step++) {

    /* -- Kernel B: purely local, no communication ------------------ */

    ops_particle_par_loop(
        KerAdvance, "KerAdvance", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
        range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_RW),
        ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_RW),
        ops_arg_gbl(ACCEL, 2, "double", OPS_READ),
        ops_arg_gbl(&dt, 1, "double", OPS_READ));

    /* Positions changed: repair the spatial index, migrate owners. */
    update_maps(particle, dat_border, nborder, dat_forward, nforward);

    /* -- Kernel C: the all-to-all --------------------------------- *
     *
     * C1 scatter. ITERATE_LOCAL is mandatory: a ghost carries its owner's
     * gid and would contribute to the same slot a second time.
     */

    ops_particle_par_loop(
        KerPublishState, "KerPublishState", particle, 2,
        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_vel, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_mass, 1, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_reduce(h_all, NPART * NCOMP, "double", OPS_INC));

    /* THE COLLECTIVE. On return, all_state holds every particle in the
       domain, on every rank. In serial this is a memcpy.               */
    ops_reduction_result(h_all, all_state.data());

    /* C2 apply: each owned particle sums over the whole population. */

    ops_particle_par_loop(
        KerInfluence, "KerInfluence", particle, 2, OPS_PARTICLE_ITERATE_LOCAL,
        range_parts, map,
        ops_arg_dat_particle(p_phi, 1, "double", particle, map, OPS_WRITE),
        ops_arg_dat_particle(p_pos, 2, "double", particle, map, OPS_READ),
        ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
        ops_arg_gbl(all_state.data(), NPART * NCOMP, "double", OPS_READ),
        ops_arg_gbl(&npart_gbl, 1, "int", OPS_READ),
        ops_arg_gbl(&soften, 1, "double", OPS_READ));

    /* -- Output ---------------------------------------------------- *
     *
     * The particle API has no HDF5 writer, so the app provides one. It needs
     * every particle on the writing rank -- which the gather has already
     * done. all_state is current; phi has just been computed, so gather that
     * too and the frame is complete, with no MPI in the writer.
     */

    if (step % NPRINT == 0) {

      ops_particle_par_loop(
          KerPublishInfluence, "KerPublishInfluence", particle, 2,
          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
          ops_arg_dat_particle(p_phi, 1, "double", particle, map, OPS_READ),
          ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
          ops_arg_reduce(h_phi, NPART, "double", OPS_INC));

      ops_reduction_result(h_phi, all_phi.data());

      write_influence_step(block, x_grid, all_state, all_phi, io_params, step);

      if (step % (10 * NPRINT) == 0)
        ops_printf("step %5d / %d\n", step, NSTEPS);
    }
  }

  /* ---------------------------------------------------------------- *
   * 11. Check
   * ---------------------------------------------------------------- *
   * Reference computed on the host from the known trajectory, compared
   * inside a kernel, reduced with OPS. No MPI anywhere in this app.
   */

  std::vector<Real> phi_ref(NPART);
  reference_influence(seed_x, seed_m, NSTEPS, phi_ref);

  ops_particle_par_loop(
      KerCheckInfluence, "KerCheckInfluence", particle, 2,
      OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
      ops_arg_dat_particle(p_phi, 1, "double", particle, map, OPS_READ),
      ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
      ops_arg_gbl(phi_ref.data(), NPART, "double", OPS_READ),
      ops_arg_reduce(h_worst, 1, "double", OPS_MAX),
      ops_arg_reduce(h_count, 1, "int", OPS_INC));

  Real worst = 0.0;
  int found = 0;
  ops_reduction_result(h_worst, &worst);
  ops_reduction_result(h_count, &found);

  /* A couple of gathered values, to show the buffer really is global. */
  ops_printf("\ngathered state, first 3 particles (x, y, strength):\n");
  for (int i = 0; i < 3; i++)
    ops_printf("  gid %3d : %12.9f %12.9f %6.3f   phi = %12.6f\n", i,
               all_state[NCOMP * i + C_X], all_state[NCOMP * i + C_Y],
               all_state[NCOMP * i + C_M], phi_ref[i]);

  const Real tol = 1e-10;
  int ok = (worst < tol) && (found == NPART);

  ops_printf("\n---------------------------------------------\n");
  ops_printf("particles expected  : %d\n", NPART);
  ops_printf("particles found     : %d\n", found);
  ops_printf("max influence error : %.3e  (tol %.1e)\n", worst, tol);
  ops_printf("RESULT              : %s\n", ok ? "PASS" : "FAIL");
  ops_printf("---------------------------------------------\n");

  write_influence_final(block, x_grid, all_state, all_phi, io_params, NSTEPS);

  ops_particle_print_dats_to_txtfile(particle, dat_output, noutput,
                                     "particles_final.txt");

  ops_printf("\nwrote %d HDF5 frames: influence_output_??????.h5"
             " (+ influence_output.h5)\n",
             NSTEPS / NPRINT);
  ops_printf("plot with:  python3 plot_influence_h5.py\n");

  ops_exit();
  return ok ? 0 : 1;
}
