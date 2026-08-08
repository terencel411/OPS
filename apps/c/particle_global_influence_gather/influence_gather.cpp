/*
 * OPS Particles -- all-to-all influence with an MPI_Allgatherv gather
 * ==================================================================
 *
 * Same problem and same physics as particle_global_influence: every particle
 * needs a quantity summed over EVERY other particle in the domain, so the whole
 * population has to be on every rank once per step.
 *
 * That app does the gather with an array-valued ops_reduction -- each rank INCs
 * its particles into slots picked by global id, and the MPI_Allreduce inside
 * ops_reduction_result hands the array back. Elegant, and it needs no MPI in
 * the application at all.
 *
 * THIS app does the same gather with MPI_Allgatherv, which is the primitive
 * that actually matches the operation. An Allreduce ships the full N*NCOMP
 * array from every rank and sums it; the data is disjoint, so a concatenation
 * is what is wanted. Each rank sends only its own particles.
 *
 * Measured against the reduction version, at N = 5000: 4.4x faster at np = 4
 * and np = 8, 2.4x at np = 1, with the gap widening as ranks are added --
 * the reduction's cost grows roughly proportionally to rank count. Both
 * produce bit-identical buffers. See the README for the tables.
 *
 * WHAT ALLGATHERV HAS TO DO THAT THE REDUCTION DID NOT
 *
 *   1. exchange the counts. Allgatherv needs to know how many doubles each
 *      rank is sending, so a small MPI_Allgather comes first. Particles
 *      migrate between ranks, so this cannot be cached.
 *   2. carry identities. The received data arrives in RANK order, so every
 *      particle must carry its global id and the buffer must be permuted into
 *      gid order afterwards. The reduction needed neither: the slot IS the id.
 *   3. pack a send buffer. The state lives in four separate dats; Allgatherv
 *      wants one contiguous block.
 *
 *   All three are inside the timed region. Anything else would flatter it.
 *
 * WHY THE PACK IS HOST CODE
 *
 *   The OPS-native way to fill a per-particle buffer would be
 *   ops_arg_gbl_particle, which exists for exactly this. It does not work:
 *   in ops_particle_seq.h:681-689 the shift is
 *
 *       static void shift_arg(const ops_arg &arg, char *p, const int offs, ...)
 *         else if (arg.argtype == OPS_ARG_GBL_PARTICLE) p += offs * arg.elem_size;
 *
 *   with `p` taken BY VALUE, so the increment is dead code and the pointer
 *   never advances -- every particle would write to slot 0. (Compare the IDP
 *   branch just above it, which mutates instance->arg_idp[0], and the ACCP
 *   handler, which calls next() on the pointed-to accessor. Those work.)
 *
 *   So the pack reads the particle dats' contiguous AoS storage directly, the
 *   same way seeding and the HDF5 writers in these apps already do.
 *
 * EVERYTHING ELSE IS OPS
 *
 *   Raw MPI appears in exactly two functions, gather_state_mpi() and
 *   gather_scalar_mpi(). The loops, kernels, migration cycle, the max/count
 *   reductions in the verification and the collective barrier in the HDF5
 *   writer are all ordinary OPS.
 *
 * OPTIONS
 *   -npart N     particle count                       default 2000
 *   -nsteps N    timesteps                            default 200
 *   -nprint N    write an HDF5 frame every N steps    default 0 (off)
 *   -check       verify against a host reference (O(N^2) on the host)
 *
 * Build:  make influence_gather_dev_seq / _dev_mpi
 * Run:    OMP_NUM_THREADS=1 mpirun -np 4 ./influence_gather_dev_mpi -npart 5000
 *
 * SET OMP_NUM_THREADS=1 FOR MPI RUNS -- see the pmesh app's README.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <string>
#include <vector>

#define OPS_2D

#include <ops_seq_v2.h>       /* grid ops_par_loop                          */
#include <ops_particle_seq.h> /* ops_particle_par_loop, particle library    */

#include "grid_kernels.h"
#include "particle_kernels.h" /* also defines NCOMP and the C_* offsets     */
#include "gather_io.h"        /* per-step HDF5 frames                       */

typedef double Real;

/* ------------------------------------------------------------------ *
 *  Simulation parameters
 * ------------------------------------------------------------------ */

const int NX = 41;   /* grid nodes in x                                  */
const int NY = 41;   /* grid nodes in y                                  */
const Real LENGTH = 1.0;

static int NPART = 2000;          /* set with -npart                      */
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
static int NSTEPS = 200;      /* set with -nsteps  */
static bool DO_CHECK = false;
static int NPRINT = 0;        /* HDF5 frame interval; 0 = off             */

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-npart") && i + 1 < argc) NPART = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-nsteps") && i + 1 < argc) NSTEPS = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-check")) DO_CHECK = true;
    else if (!strcmp(argv[i], "-nprint") && i + 1 < argc) NPRINT = atoi(argv[++i]);
  }
}

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
 *  The MPI_Allgatherv route
 * ================================================================== *
 *
 * Produces exactly what the reduction produces: all_state[NPART*NCOMP] in
 * GLOBAL ID order, identical on every rank. Everything here is timed as part
 * of the method -- the count exchange, the pack, the collective, the permute.
 *
 * Three things the Allreduce gets for free and this does not:
 *
 *   1. the counts. Allgatherv needs to know how many doubles each rank is
 *      sending, so a small MPI_Allgather has to come first. Particles migrate,
 *      so it cannot be cached.
 *   2. the identities. The received data arrives in rank order, so each
 *      particle must carry its gid and be permuted afterwards. The reduction
 *      needs neither: the slot IS the id.
 *   3. the packing. The state lives in four separate dats; Allgatherv wants one
 *      contiguous buffer.
 *
 * In a serial build this degenerates to pack + permute with no communication,
 * which is the honest serial cost of the same code path.
 */
static void gather_state_mpi(ops_particle particle, ops_dat pos, ops_dat vel,
                             ops_dat mass, ops_dat gid,
                             std::vector<Real> &send, std::vector<Real> &recv,
                             std::vector<Real> &all_state) {

  const int nlocal = (int)particle->no_particles;
  const int W = NCOMP + 1; /* state, plus the gid so the permute can undo the
                              rank ordering */

  /* ---- pack ---- */
  send.resize((size_t)nlocal * W);
  {
    const Real *xp = (const Real *)pos->data;
    const Real *up = (const Real *)vel->data;
    const Real *mp = (const Real *)mass->data;
    const int *ip = (const int *)gid->data;
    for (int i = 0; i < nlocal; i++) {
      Real *d = &send[(size_t)i * W];
      d[C_X] = xp[2 * i];
      d[C_Y] = xp[2 * i + 1];
      d[C_VX] = up[2 * i];
      d[C_VY] = up[2 * i + 1];
      d[C_M] = mp[i];
      d[NCOMP] = (Real)ip[i];
    }
  }

  int ntotal = nlocal;

#ifdef OPS_MPI
  MPI_Comm comm = OPS_sub_block_list[particle->block->index]->comm;
  int nranks = 1;
  MPI_Comm_size(comm, &nranks);

  /* ---- (1) how much is everyone sending? ---- */
  std::vector<int> counts(nranks), displs(nranks);
  int nsend = nlocal * W;
  MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  ntotal = 0;
  for (int r = 0; r < nranks; r++) {
    displs[r] = ntotal;
    ntotal += counts[r];
  }
  ntotal /= W;

  /* ---- (2) the collective ---- */
  recv.resize((size_t)ntotal * W);
  MPI_Allgatherv(send.data(), nsend, MPI_DOUBLE, recv.data(), counts.data(),
                 displs.data(), MPI_DOUBLE, comm);
#else
  recv.swap(send);
#endif

  /* ---- permute rank order -> gid order ---- */
  for (int k = 0; k < ntotal; k++) {
    const Real *sblk = &recv[(size_t)k * W];
    const int g = (int)sblk[NCOMP];
    memcpy(&all_state[(size_t)g * NCOMP], sblk, NCOMP * sizeof(Real));
  }

#ifndef OPS_MPI
  send.swap(recv); /* give the pack buffer back for the next step */
#endif
}

/* Gather one value per particle into gid order -- used for HDF5 output, where
 * the writer needs the whole population. Same shape as gather_state_mpi(), and
 * kept separate because the quantity it gathers (the influence) does not exist
 * until after the state gather has already been consumed.
 */
static void gather_scalar_mpi(ops_particle particle, ops_dat val, ops_dat gid,
                              std::vector<Real> &send, std::vector<Real> &recv,
                              std::vector<Real> &out) {

  const int nlocal = (int)particle->no_particles;

  send.resize((size_t)nlocal * 2);
  {
    const Real *v = (const Real *)val->data;
    const int *ip = (const int *)gid->data;
    for (int i = 0; i < nlocal; i++) {
      send[2 * i] = v[i];
      send[2 * i + 1] = (Real)ip[i];
    }
  }

  int ntotal = nlocal;

#ifdef OPS_MPI
  MPI_Comm comm = OPS_sub_block_list[particle->block->index]->comm;
  int nranks = 1;
  MPI_Comm_size(comm, &nranks);

  std::vector<int> counts(nranks), displs(nranks);
  int nsend = nlocal * 2;
  MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  ntotal = 0;
  for (int r = 0; r < nranks; r++) {
    displs[r] = ntotal;
    ntotal += counts[r];
  }
  ntotal /= 2;

  recv.resize((size_t)ntotal * 2);
  MPI_Allgatherv(send.data(), nsend, MPI_DOUBLE, recv.data(), counts.data(),
                 displs.data(), MPI_DOUBLE, comm);
#else
  recv.swap(send);
#endif

  for (int k = 0; k < ntotal; k++) out[(int)recv[2 * k + 1]] = recv[2 * k];

#ifndef OPS_MPI
  send.swap(recv);
#endif
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
  parse_args(argc, argv);

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
   * These are genuine reductions -- a max, two sums and a barrier. The gather
   * itself is MPI_Allgatherv; nothing here stands in for it.
   *
   * A reduction handle must be declared AFTER at least one ops_block exists
   * (the MPI build sizes its storage per block and throws otherwise).
   */

  ops_reduction h_worst =
      ops_decl_reduction_handle(sizeof(double), "double", "worst_error");
  ops_reduction h_count =
      ops_decl_reduction_handle(sizeof(int), "int", "particles_seen");
  /* Used only as a collective barrier by the HDF5 writer. */
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

  /* Where either gather delivers its result. */
  std::vector<Real> all_state(NPART * NCOMP);
  std::vector<Real> send_buf, recv_buf;

  double t_gather = 0, c0, w0, c1, w1;
  double step_gather = 0;   /* this step only, for the frames */

  /* Output needs the influence gathered by gid as well; same allgather trick,
     one component per particle. Off by default so timing runs pay nothing. */
  std::vector<Real> all_phi(NPART);
  gather_io_params io_params = {NX,
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
                                 box->getGlobalMax().y - MARGIN},
                                0.0};

  if (NPRINT > 0) {
    remove_stale_output("gather_output", NSTEPS, NPRINT, h_sync);
    ops_printf("writing %d HDF5 frames (gather_output_??????.h5)\n",
               NSTEPS / NPRINT);
  }

  ops_printf("\n=== MPI_Allgatherv vs the ops_reduction allgather ===\n");
  ops_printf("particles %d   steps %d   buffer %d x %d doubles = %zu bytes\n",
             NPART, NSTEPS, NPART, NCOMP, NPART * NCOMP * sizeof(double));
  ops_printf("gather: MPI_Allgatherv (+ count exchange, pack, permute)\n");
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

    /* -- The gather ------------------------------------------------ *
     *
     * On return all_state holds every particle in the domain, on every rank,
     * in global-id order. Count exchange, pack, collective and permute are all
     * inside the timer.
     */

    ops_timers(&c0, &w0);
    gather_state_mpi(particle, p_pos, p_vel, p_mass, p_gid, send_buf, recv_buf,
                     all_state);
    ops_timers(&c1, &w1);
    step_gather = w1 - w0;
    t_gather += step_gather;

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

    /* -- Output (not timed: not part of either gather) ------------- */

    if (NPRINT > 0 && step % NPRINT == 0) {
      gather_scalar_mpi(particle, p_phi, p_gid, send_buf, recv_buf, all_phi);

      io_params.t_gather_ms = 1000.0 * step_gather;

      write_gather_step(block, x_grid, all_state, all_phi, io_params, step);
    }

    if (step % 50 == 0) ops_printf("step %5d / %d\n", step, NSTEPS);
  }

  /* ---------------------------------------------------------------- *
   * 11. Check
   * ---------------------------------------------------------------- *
   * Reference computed on the host from the known trajectory, compared
   * inside a kernel, reduced with OPS. No MPI anywhere in this app.
   */

  std::vector<Real> phi_ref(NPART);
  Real worst = 0.0;
  int found = 0;

  if (DO_CHECK) {
  reference_influence(seed_x, seed_m, NSTEPS, phi_ref);

  ops_particle_par_loop(
      KerCheckInfluence, "KerCheckInfluence", particle, 2,
      OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
      ops_arg_dat_particle(p_phi, 1, "double", particle, map, OPS_READ),
      ops_arg_dat_particle(p_gid, 1, "int", particle, map, OPS_READ),
      ops_arg_gbl(phi_ref.data(), NPART, "double", OPS_READ),
      ops_arg_reduce(h_worst, 1, "double", OPS_MAX),
      ops_arg_reduce(h_count, 1, "int", OPS_INC));

  ops_reduction_result(h_worst, &worst);
  ops_reduction_result(h_count, &found);
  }

  /* ---------------------------------------------------------------- *
   * 12. The comparison
   * ---------------------------------------------------------------- */

  if (NPRINT > 0) {
    io_params.t_gather_ms = 1000.0 * step_gather;
    write_gather_final(block, x_grid, all_state, all_phi, io_params, NSTEPS);
    ops_printf("\nwrote %d HDF5 frames; plot with:"
               "  python3 plot_gather_h5.py\n", NSTEPS / NPRINT);
  }

  const double ms = 1000.0 / static_cast<double>(NSTEPS);

  ops_printf("\n--- gather cost per timestep (ms, wall) --------------\n");
  ops_printf("MPI_Allgatherv (+counts, pack, permute)  %9.4f\n",
             t_gather * ms);

  if (DO_CHECK) {
    const Real tol = 1e-10;
    int ok = (worst < tol) && (found == NPART);
    ops_printf("\n---------------------------------------------\n");
    ops_printf("particles expected  : %d\n", NPART);
    ops_printf("particles found     : %d\n", found);
    ops_printf("max influence error : %.3e  (tol %.1e)\n", worst, tol);
    ops_printf("RESULT              : %s\n", ok ? "PASS" : "FAIL");
    ops_printf("---------------------------------------------\n");
    ops_exit();
    return ok ? 0 : 1;
  }

  ops_exit();
  return 0;
}
