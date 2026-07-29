/*
 * OPS Particles -- Tutorial 1: constant-velocity drift
 * =====================================================
 *
 * The smallest OPS Particles application that is still correct under MPI.
 *
 * Particles are seeded as a uniform lattice over the whole domain and drift at
 * a constant, prescribed velocity along x. There is no fluid solver and no
 * interpolation -- the point is to show the *structure* of a particle
 * application with nothing else in the way.
 *
 * The domain is periodic in x and walled in y, so the population never
 * changes: a particle leaving the right-hand edge re-enters on the left and
 * exactly fills the gap it left. The result is a steady stream that looks the
 * same at every instant, however long you run.
 *
 * The exact answer is still known -- x0 + v*t, folded back into the domain by
 * those boundary conditions -- and the program checks it itself, so a correct
 * run is unambiguous.
 *
 * What this tutorial introduces
 *   1. the declaration order that OPS Particles requires
 *   2. the seeding contract (you allocate, you write, you set the count)
 *   3. ops_particle_par_loop and ACCP<T>& kernels
 *   4. the per-step map/migration cycle
 *   5. why some particle dats must be listed for border exchange
 *   6. periodic boundaries via particle halo groups, and why a wrap written
 *      by hand in a kernel is wrong under MPI (see section 6b)
 *   7. reflecting walls in a kernel, and why they must run before the map
 *      update (see KerApplyWalls)
 *
 * What it deliberately leaves out (see tutorial 2)
 *   - reading anything from the grid
 *   - interpolation, ops_par_particle_grid_loop
 *
 * Build:  make dev_seq   (fastest: no translator)
 *         make dev_mpi   (MPI, no translator)
 *         make mpi       (MPI, via the OPS translator)
 * Run:    ./tutorial1_dev_seq
 *         mpirun -np 4 ./tutorial1_dev_mpi
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define OPS_2D

#include <ops_seq_v2.h>          /* grid ops_par_loop                       */
#include <ops_particle_seq.h>    /* ops_particle_par_loop, particle library */

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include "grid_kernels.h"
#include "particle_kernels.h"

/* Periodic HDF5 output -- one self-contained .h5 file per output step, holding
   the block, the grid dat, the particle dats and the run constants.  Same
   pattern as apps/c/oSEM and apps/c/LBM-Particle_tracking.                  */
#include "drift_io.h"

typedef double Real;

/* ------------------------------------------------------------------ *
 *  Simulation parameters -- all in one place so you can experiment
 * ------------------------------------------------------------------ */

const int  NX      = 41;        /* grid nodes in x                        */
const int  NY      = 41;        /* grid nodes in y                        */
const Real LENGTH  = 1.0;       /* domain is [0,LENGTH] x [0,LENGTH]      */

const int  NPX     = 10;        /* particles seeded in x                  */
const int  NPY     = 10;        /* particles seeded in y                  */

/* The particles are seeded as a uniform lattice over the WHOLE domain, not
   into a sub-box: with a periodic x boundary that makes the stream steady --
   every column leaving the right edge re-enters on the left and exactly fills
   the gap it left, so any snapshot looks like any other. See
   seed_particles().                                                      */

/* Drift is along x only. The top and bottom walls are still enforced, so
   giving VEL[1] a non-zero value makes the particles bounce between them
   instead of streaming out of the domain.                              */
const Real VEL[2]  = {0.5, 0.0};    /* the constant drift velocity        */
const Real DT      = 0.001;
const int  NSTEPS  = 1000;          /* 0.5 domain widths per 1000 steps   */
const int  NPRINT  = 100;

/* ================================================================== *
 *  Seeding
 * ================================================================== *
 *
 * Particles are placed by writing straight into the dats' raw storage.
 * There is no kernel for this: it happens before any map exists.
 *
 * The contract you must satisfy:
 *   1. grow the storage first if you need more than particle->Nmax (1000)
 *   2. write every per-particle field you care about
 *   3. set particle->no_particles yourself -- nothing infers it
 *   4. under MPI, seed only inside THIS rank's part of the domain, and
 *      make the ids globally unique
 */
void seed_particles(ops_particle particle, ops_dat pos, ops_dat vel,
                    ops_dat ids, ops_dat x0) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  /* The bounding box knows two extents:
   *   getGlobalMin/Max -- the whole domain, same on every rank
   *   getLocalMin/Max  -- the part owned by THIS rank
   * We want a globally-defined seeding pattern, but each rank may only
   * place the particles that fall inside its own subdomain.            */
  const Real lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const Real hi[2] = {box->getLocalMax().x, box->getLocalMax().y};
  const Real glo[2] = {box->getGlobalMin().x, box->getGlobalMin().y};
  const Real ghi[2] = {box->getGlobalMax().x, box->getGlobalMax().y};

  /* Uniform lattice over the WHOLE domain, defined globally.
   *
   * Dividing by NPX and not NPX-1 is what makes the stream steady. The x
   * boundary is periodic, so the lattice has to TILE: with NPX-1 spacing there
   * would be a particle on both edges, and since those are the same point
   * under periodicity the result is a double-density column at the seam and a
   * gap of one dx beside it. The stream would then pulse once per lap instead
   * of looking the same at every instant.
   *
   * The +0.5 puts particles at cell centres so none is ever seeded exactly on
   * a boundary. In y that is a correctness matter, not tidiness: KerApplyWalls
   * tests with strict inequalities, so a particle sitting exactly on y = 0
   * would never be pushed back inside, and the half-open [lo,hi) binning
   * convention can read one sitting exactly on y = LENGTH as outside.       */
  const Real dx = (ghi[0] - glo[0]) / static_cast<Real>(NPX);
  const Real dy = (ghi[1] - glo[1]) / static_cast<Real>(NPY);

  /* Then shift the whole lattice half a GRID cell.
   *
   * Subdomain boundaries are cuts in index space, so they always land on grid
   * nodes. A particle seeded exactly on a node can therefore sit exactly on a
   * rank boundary, and one that does is lost in the first migration -- it
   * survives seeding (the ownership filter below picks a single owner) but
   * disappears from the count by the next output. Measured at np = 8: the
   * column at x = 0.25, which is node 10, took all 10 of its particles with
   * it while every other column was fine.
   *
   * Offsetting by half a cell puts every particle strictly inside a cell, so
   * it cannot coincide with a boundary at ANY rank count. Shifting the whole
   * lattice rigidly keeps the spacing uniform, so the pattern still tiles
   * across the periodic seam and the stream stays steady.                  */
  const Real half_cell_x = 0.5 * (ghi[0] - glo[0]) / static_cast<Real>(NX - 1);
  const Real half_cell_y = 0.5 * (ghi[1] - glo[1]) / static_cast<Real>(NY - 1);

  /* Worst case every candidate lands on this rank, so make room for them. */
  if (NPX * NPY > (int)particle->Nmax)
    ops_particle_realloc_data(particle, NPX * NPY);

  Real *xp   = (Real *)pos->data;
  Real *up   = (Real *)vel->data;
  Real *xp0  = (Real *)x0->data;
  int  *idp  = (int  *)ids->data;

  int n = 0;
  for (int i = 0; i < NPX; i++) {
    for (int j = 0; j < NPY; j++) {
      Real x = glo[0] + dx * (static_cast<Real>(i) + 0.5) + half_cell_x;
      Real y = glo[1] + dy * (static_cast<Real>(j) + 0.5) + half_cell_y;

      /* Skip candidates owned by another rank. In serial lo/hi span the
         whole domain, so nothing is skipped.                            */
      if (x < lo[0] || x >= hi[0] || y < lo[1] || y >= hi[1]) continue;

      /* Particle dats are AoS: data[dim * particle_index + component] */
      xp[2 * n]     = x;
      xp[2 * n + 1] = y;

      /* x0 records where this particle started. We never touch it again.
         It rides along with the particle when it migrates between ranks,
         which is what makes the final check below possible.             */
      xp0[2 * n]     = x;
      xp0[2 * n + 1] = y;

      up[2 * n]     = 0.0;
      up[2 * n + 1] = 0.0;

      /* (4) Ids must be unique across ALL ranks. Here the seeding pattern
             is a known global lattice, so the cleanest id is the global
             lattice index: it is unique by construction, identical no
             matter how many ranks you run on, and it lets you follow one
             particle across runs.

             When you CANNOT derive a global index -- random seeding, say
             -- the general pattern is a prefix sum over per-rank counts:

               int n_local = n, offset = 0;
               int *counts = (int *)malloc(nprocs * sizeof(int));
               MPI_Allgather(&n_local, 1, MPI_INT, counts, 1, MPI_INT,
                             MPI_COMM_WORLD);
               for (int r = 0; r < myrank; r++) offset += counts[r];
               for (int p = 0; p < n; p++) idp[p] += offset;

             That yields unique ids, but a given id then labels a
             different particle at a different rank count.               */
      idp[n] = i * NPY + j;

      n++;
    }
  }

  /* (3) Tell OPS how many particles this rank now owns. */
  particle->no_particles = n;
}

/* ================================================================== *
 *  The per-step map / migration cycle
 * ================================================================== *
 *
 * Call this once per timestep, after positions change.
 *
 * Step 1 walks the particles and asks "did anything move far enough to
 * matter?".  Particles that barely moved are skipped entirely; particles
 * that changed cell are re-linked in place.  Only if a particle crossed
 * into or out of the halo band -- or left the domain -- does it return
 * true, and that decision is then made collectively across all ranks.
 *
 * Step 3 is the important fork:
 *   decide == true  -> expensive: migrate ownership, rebuild the ghost
 *                      layer and all maps.  Communicates dat_border.
 *   decide == false -> cheap: just refresh values on the ghost layer
 *                      that already exists.  Communicates dat_forward.
 *
 * In serial every one of these communication calls is an empty function
 * body in the OPS library, so this costs nothing -- write it anyway, and
 * the same source runs correctly under MPI.
 */
void update_maps(ops_particle particle,
                 ops_dat *dat_border,  int nborder,
                 ops_dat *dat_forward, int nforward) {

  int decide = ops_particle_update_map_lists_actual_hybrid(particle);

#ifdef OPS_MPI
  /* Carry particles across the periodic seam BEFORE the deletion pass below,
     or the ones that just left through the right-hand edge get removed as
     out-of-domain instead of re-appearing on the left.                    */
  ops_particle_halo_transfer_group_map(OPS_HALO_GRP_EXCHANGE, decide);
#endif

  ops_particle_remove_delete_maps(particle, decide);

  /* Each branch of the migration cycle has a matching halo transfer: the
     ghost band across the seam is maintained on exactly the same schedule as
     the ghost band between ranks.                                          */
  if (decide) {
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);
#ifdef OPS_MPI
    ops_particle_halo_transfer_group_map(OPS_HALO_GRP_BORDER, decide);
#endif
  } else {
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);
#ifdef OPS_MPI
    ops_particle_halo_transfer_group_map(OPS_HALO_GRP_FORWARD, decide);
#endif
  }

  ops_particle_reset_flags(particle, decide);
}

/* ================================================================== *
 *  Verification
 * ================================================================== *
 *
 * Constant velocity and forward Euler still means the exact answer is known,
 * the boundary conditions just fold it back into the domain:
 *
 *   x : re-injection makes the motion periodic, so the exact position is
 *       x0 + v*t reduced modulo the domain width.
 *   y : reflecting walls turn the motion into a triangle wave of period 2*Ly.
 *       With VEL[1] = 0 this collapses to y == y0.
 *
 * Any deviation beyond round-off is a real bug, not discretisation error.
 */
int check_result(ops_particle particle, ops_dat pos, ops_dat x0, Real t,
                 const Real *lo, const Real *hi) {

  Real *xp  = (Real *)pos->data;
  Real *xp0 = (Real *)x0->data;

  const Real Lx = hi[0] - lo[0];
  const Real Ly = hi[1] - lo[1];

  Real worst = 0.0;
  for (size_t p = 0; p < particle->no_particles; p++) {

    /* Periodic in x. fmod can return a negative remainder, hence the fixup. */
    Real sx = fmod(xp0[2 * p] - lo[0] + VEL[0] * t, Lx);
    if (sx < 0.0) sx += Lx;
    Real ex = lo[0] + sx;

    /* Measure the x error the short way round the loop. A particle sitting
       within round-off of the seam is at hi in the simulation and at lo in
       the formula (or the reverse); those are the same physical point, and
       only the modular distance says so.                                   */
    Real dx = fabs(xp[2 * p] - ex);
    dx = fmin(dx, Lx - dx);

    /* Reflected in y: fold [0,2Ly) back onto [0,Ly] to get the triangle. */
    Real sy = fmod(xp0[2 * p + 1] - lo[1] + VEL[1] * t, 2.0 * Ly);
    if (sy < 0.0) sy += 2.0 * Ly;
    Real ey = lo[1] + (sy <= Ly ? sy : 2.0 * Ly - sy);

    worst = fmax(worst, dx);
    worst = fmax(worst, fabs(xp[2 * p + 1] - ey));
  }

  int nlocal = (int)particle->no_particles;
  int ntotal = nlocal;

#ifdef OPS_MPI
  Real worst_global;
  MPI_Allreduce(&worst,  &worst_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nlocal, &ntotal,       1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
  worst = worst_global;
#endif

  /* Accumulating v*dt NSTEPS times loses a little precision, so scale the
     tolerance with the number of steps rather than demanding exactness. */
  const Real tol = 1e-12 * NSTEPS;
  int ok = (worst < tol) && (ntotal == NPX * NPY);

  ops_printf("\n---------------------------------------------\n");
  ops_printf("particles expected : %d\n", NPX * NPY);
  ops_printf("particles found    : %d\n", ntotal);
  ops_printf("max position error : %.3e  (tol %.1e)\n", worst, tol);
  ops_printf("RESULT             : %s\n", ok ? "PASS" : "FAIL");
  ops_printf("---------------------------------------------\n");

  return ok ? 0 : 1;
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  /* ---------------------------------------------------------------- *
   * 1. Block and grid
   * ---------------------------------------------------------------- *
   * The grid is here only so that the bounding box and the particle
   * mapping have a coordinate reference. It carries no physics in this
   * tutorial -- x_grid just holds node coordinates.                    */

  ops_block block = ops_decl_block(2, "tutorial_block");

  int size[] = {NX, NY};
  int base[] = {0, 0};
  int d_m[]  = {-1, -1};
  int d_p[]  = { 1,  1};

  Real *null_dbl = NULL;
  int  *null_int = NULL;

  ops_dat x_grid = ops_decl_dat(block, 2, size, base, d_m, d_p, null_dbl,
                                "double", "x_grid");

  /* ---------------------------------------------------------------- *
   * 2. Stencils
   * ---------------------------------------------------------------- */

  int s2d_00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2d_00, "0,0");

  int s2d_9pt[] = {-1,-1, -1,0, -1,1, 0,-1, 0,0, 0,1, 1,-1, 1,0, 1,1};
  ops_stencil S2D_9pt = ops_decl_stencil(2, 9, s2d_9pt, "9pt");

  /* ---------------------------------------------------------------- *
   * 3. Bounding box  -- the bridge from index space to physical space
   * ---------------------------------------------------------------- *
   * Particles are located by coordinates, not indices, so the library
   * needs to know the physical extent of the block. Deriving it from the
   * coordinate dat means it is automatically consistent with the grid.  */

  Real dx_box[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, x_grid, 2, dx_box);

  /* ---------------------------------------------------------------- *
   * 4. The particle set and its dats
   * ---------------------------------------------------------------- */

  ops_particle particle = ops_decl_particle(block, "drifters", box);

  /* The position dat is privileged: exactly one per particle set, its
     dim must equal the block dimension, and every piece of machinery in
     the library (binning, migration, deletion) reads positions from it.
     Declare it BEFORE any mapping.                                     */
  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 2, base, null_dbl,
                                            "double", "position");

  /* Ordinary particle dats: length Nmax, `dim` components each, AoS. */
  ops_dat p_vel = ops_decl_particle_dat(particle, 2, base, null_dbl,
                                        "double", "velocity");
  ops_dat p_x0  = ops_decl_particle_dat(particle, 2, base, null_dbl,
                                        "double", "start_position");
  ops_dat p_ids = ops_decl_particle_dat(particle, 1, base, null_int,
                                        "int", "id");

  /* ---------------------------------------------------------------- *
   * 5. The mapping -- a cell-linked list binning particles onto cells
   * ---------------------------------------------------------------- *
   * NOTE: S2D_9pt here is NOT an interpolation stencil. It only widens
   * the halo depth of the bin array, i.e. it declares "track particles up
   * to one cell outside my subdomain". That is what sizes the ghost
   * particle band used during migration.                               */

  ops_particle_mapping map = ops_decl_mapping(particle, x_grid, S2D_9pt,
                                              OPS_WITH_VIRTUAL,
                                              OPS_UNIFORM_STAG, 1);

  /* ---------------------------------------------------------------- *
   * 6. Dat lists for particle communication
   * ---------------------------------------------------------------- *
   * border  : exchanged when a particle changes rank owner. EVERY dat
   *           whose value must survive migration belongs here -- a dat
   *           left out silently holds garbage after a particle moves.
   *           p_x0 is in this list: that is what lets the final check
   *           work under MPI.
   * forward : the cheap per-step refresh of the existing ghost layer.
   *
   * The counts are computed from the arrays, never typed by hand -- the
   * two must agree and nothing checks that for you.                    */

  ops_dat dat_border[]  = {p_pos, p_vel, p_ids, p_x0};
  ops_dat dat_forward[] = {p_pos, p_vel};
  ops_dat dat_output[]  = {p_ids, p_pos, p_vel};

  const int nborder  = sizeof(dat_border)  / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);
  const int noutput  = sizeof(dat_output)  / sizeof(dat_output[0]);

  /* ---------------------------------------------------------------- *
   * 6b. Periodic halo in x  -- how particles get re-injected
   * ---------------------------------------------------------------- *
   * Particles that reach the right-hand edge must re-appear on the left.
   *
   * The obvious implementation -- a kernel that does `if (x >= x_max) x -=
   * LENGTH` -- works in serial and fails under MPI. That assignment teleports
   * a particle the full width of the domain, and the migration cycle only
   * moves particles to a NEIGHBOURING rank. Whenever the decomposition puts
   * more than one rank between the two edges the particle is silently lost
   * (np = 8) or the run segfaults inside the map rebuild (np = 3). Measured:
   * a manual wrap passes at np = 1,2,4 and fails at np = 3,5,6,8.
   *
   * The supported mechanism is a particle halo carrying a translation, the
   * same construction apps/c/testing_virtual uses. Two halos are needed, one
   * per direction: +LENGTH sends a particle off the right edge back to the
   * left, -LENGTH does the reverse.
   *
   * MPI BUILDS ONLY. The single-node library rejects this setup outright
   * ("Particle Halo exchange must be set after block boxes are set",
   * ops/c/src/sequential/ops_particle_host_single_node.cpp:1908), so the
   * serial builds keep the direct wrap in KerWrapX -- which is safe there
   * precisely because there are no ranks to migrate between.
   *
   * OPS_PART_ORIENT_ON marks the dat that must be SHIFTED by `translate` as
   * it crosses -- that is the position, and only the position. Everything
   * else (velocity, id, seed position) rides across unchanged, so it is
   * declared ORIENT_OFF. Leaving p_x0 out of this list entirely would break
   * the final check the same way leaving it out of dat_border would.        */

#ifdef OPS_MPI
  ops_particle_halo_data h_pos = ops_particle_decl_data_halo(p_pos, p_pos,
                                                             OPS_PART_ORIENT_ON);
  ops_particle_halo_data h_vel = ops_particle_decl_data_halo(p_vel, p_vel,
                                                             OPS_PART_ORIENT_OFF);
  ops_particle_halo_data h_ids = ops_particle_decl_data_halo(p_ids, p_ids,
                                                             OPS_PART_ORIENT_OFF);
  ops_particle_halo_data h_x0  = ops_particle_decl_data_halo(p_x0, p_x0,
                                                             OPS_PART_ORIENT_OFF);

  ops_particle_halo_data halo_dats[] = {h_pos, h_vel, h_ids, h_x0};
  const int nhalo_dats = sizeof(halo_dats) / sizeof(halo_dats[0]);

  /* The band either side of the seam that participates in the exchange. One
     cell is enough: a particle cannot cross more than a cell in a step.     */
  Real dx_halo[]      = {LENGTH / static_cast<Real>(NX - 1), 0.0};
  Real translate_p[]  = { LENGTH, 0.0};   /* right edge -> left edge         */
  Real translate_m[]  = {-LENGTH, 0.0};   /* left edge  -> right edge        */
  int  halo_dir[]     = {0, 1};

  ops_particle_halo exch_x1 = ops_particle_decl_halo(particle, particle,
      halo_dats, nhalo_dats, dx_halo, halo_dir, halo_dir, translate_p);
  ops_particle_halo exch_x2 = ops_particle_decl_halo(particle, particle,
      halo_dats, nhalo_dats, dx_halo, halo_dir, halo_dir, translate_m);
  ops_particle_halo bord_x1 = ops_particle_decl_halo(particle, particle,
      halo_dats, nhalo_dats, dx_halo, halo_dir, halo_dir, translate_p);
  ops_particle_halo bord_x2 = ops_particle_decl_halo(particle, particle,
      halo_dats, nhalo_dats, dx_halo, halo_dir, halo_dir, translate_m);

  /* EXCHANGE moves particles across the seam, BORDER keeps the ghost band on
     either side of it populated. They mirror the two branches of the
     migration cycle in update_maps().                                       */
  ops_particle_halo exch_grp[] = {exch_x1, exch_x2};
  ops_particle_halo bord_grp[] = {bord_x1, bord_x2};

  ops_particle_halo_group group_exch = ops_particle_decl_halo_group(
      exch_grp, 2, OPS_HALO_GRP_EXCHANGE, OPS_WITH_VIRTUAL);
  ops_particle_halo_group group_bord = ops_particle_decl_halo_group(
      bord_grp, 2, OPS_HALO_GRP_DEFAULT, OPS_WITH_VIRTUAL);
#endif /* OPS_MPI */

  /* ---------------------------------------------------------------- *
   * 7. Partition
   * ---------------------------------------------------------------- */

  ops_partition("");

  /* ---------------------------------------------------------------- *
   * 8. Fill the coordinate grid
   * ---------------------------------------------------------------- *
   * ORDER MATTERS. ops_create_bounding_box() above only remembered a
   * pointer to x_grid; the actual physical bounds are computed later, by
   * ops_particle_setup_partition(). So the grid must hold real
   * coordinates BEFORE that call, or the box is derived from all-zeros
   * and OPS throws "bounding box of non-positive volume".              */

  Real dx = LENGTH / static_cast<Real>(NX - 1);
  int grid_range[] = {0, NX, 0, NY};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(&dx, 1, "double", OPS_READ),
               ops_arg_idx());

  /* Now the coordinates exist, the particle side can be set up:
     builds per-rank bounding boxes, initialises maps, sets up comms.   */
  ops_particle_setup_partition();

#ifdef OPS_MPI
  /* Halo groups can only be registered once the partition exists, because that
     is when each rank learns which ranks it shares the seam with.           */
  ops_particle_set_halo_group(group_exch);
  ops_particle_set_halo_group(group_bord);
#endif

  /* The physical extent OPS will test against when it decides whether a
     particle has left the domain. Take it from the box rather than assuming
     [0,LENGTH]: the box is derived from the coordinate dat and is sized by the
     bin count, so it need not agree with the grid range exactly. The boundary
     kernel has to use the same numbers OPS uses, or particles get deleted on
     the step they are re-injected.                                         */
  const Real dom_lo[2] = {box->getGlobalMin().x, box->getGlobalMin().y};
  const Real dom_hi[2] = {box->getGlobalMax().x, box->getGlobalMax().y};

  /* ---------------------------------------------------------------- *
   * 9. Seed particles, then build the maps for the first time
   * ---------------------------------------------------------------- */

  seed_particles(particle, p_pos, p_vel, p_ids, p_x0);

  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

#ifdef OPS_MPI
  /* Populate the ghost band across the periodic seam for the first time. */
  ops_particle_halo_transfer_group(OPS_HALO_GRP_BORDER, true);
#endif

  ops_printf("OPS Particles tutorial 1: constant-velocity drift\n");
  ops_printf("grid %dx%d, %d particles, v = (%g, %g), dt = %g, %d steps\n",
             NX, NY, NPX * NPY, VEL[0], VEL[1], DT, NSTEPS);
  ops_printf("domain [%g,%g] x [%g,%g]: re-injecting in x, walls in y\n",
             dom_lo[0], dom_hi[0], dom_lo[1], dom_hi[1]);

  ops_particle_print_dats_to_txtfile(particle, dat_output, noutput,
                                     "particles_step_0.txt");

  /* Everything a plot script needs about the run, travelling with the data. */
  drift_io_params io_params = {NX, NY, NPX, NPY, NSTEPS, NPRINT, LENGTH, DT,
                               {VEL[0], VEL[1]},
                               {dom_lo[0], dom_hi[0], dom_lo[1], dom_hi[1]}};

  /* Initial state -> drift_output_000000.h5 */
  HDF5_IO_Write_drift_block_dynamic(block, -1, x_grid, particle, dat_output,
                                    noutput, io_params);

  /* ---------------------------------------------------------------- *
   * 10. Set the (constant) velocity once
   * ---------------------------------------------------------------- *
   * range_parts is the physical iteration region. It is only consulted
   * for OPS_PARTICLE_ITERATE_RANDOM, but the argument is always required.
   *
   * OPS_PARTICLE_ITERATE_LOCAL means "owned particles only". Use it for
   * anything that advances state: iterating ghosts as well would advance
   * particles that another rank owns.                                   */

  Real range_parts[] = {0.0, LENGTH, 0.0, LENGTH};

  ops_particle_par_loop(KerSetVelocity, "KerSetVelocity", particle, 2,
                        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                        ops_arg_dat_particle(p_vel, 2, "double", particle, map,
                                             OPS_WRITE),
                        ops_arg_gbl(VEL, 2, "double", OPS_READ));

  /* ---------------------------------------------------------------- *
   * 11. Time loop
   * ---------------------------------------------------------------- */

  Real dt = DT;

  for (int step = 1; step <= NSTEPS; step++) {

    ops_particle_par_loop(KerUpdatePosition, "KerUpdatePosition", particle, 2,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 2, "double", particle,
                                               map, OPS_WRITE),
                          ops_arg_dat_particle(p_vel, 2, "double", particle,
                                               map, OPS_READ),
                          ops_arg_gbl(&dt, 1, "double", OPS_READ));

#ifndef OPS_MPI
    /* Serial: re-inject by hand. Under MPI this is the halo groups' job and
       doing it here as well would hide the crossing from them entirely.    */
    ops_particle_par_loop(KerWrapX, "KerWrapX", particle, 2,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 2, "double", particle,
                                               map, OPS_RW),
                          ops_arg_gbl(dom_lo, 2, "double", OPS_READ),
                          ops_arg_gbl(dom_hi, 2, "double", OPS_READ));
#endif

    /* Bounce anything that hit a wall. This MUST happen before update_maps():
       that is where OPS deletes out-of-domain particles, and by then it is
       too late. The x edges are handled above / by the halo groups.        */
    ops_particle_par_loop(KerApplyWalls, "KerApplyWalls", particle, 2,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 2, "double", particle,
                                               map, OPS_RW),
                          ops_arg_dat_particle(p_vel, 2, "double", particle,
                                               map, OPS_RW),
                          ops_arg_gbl(dom_lo, 2, "double", OPS_READ),
                          ops_arg_gbl(dom_hi, 2, "double", OPS_READ));

    /* Positions changed, so the spatial index may need repairing and
       particles may need to move to another rank.                      */
    update_maps(particle, dat_border, nborder, dat_forward, nforward);

    if (step % NPRINT == 0) {
      std::string fname = "particles_step_" + std::to_string(step) + ".txt";
      ops_particle_print_dats_to_txtfile(particle, dat_output, noutput,
                                         fname.c_str());

      /* Same state as the .txt dump, but as drift_output_<step>.h5 */
      HDF5_IO_Write_drift_block_dynamic(block, step - 1, x_grid, particle,
                                        dat_output, noutput, io_params);

      ops_printf("step %5d / %d\n", step, NSTEPS);
    }
  }

  /* Final state -> drift_output.h5 */
  HDF5_IO_Write_drift_block(block, NSTEPS - 1, x_grid, particle, dat_output,
                            noutput, io_params);

  /* ---------------------------------------------------------------- *
   * 12. Check against the exact answer
   * ---------------------------------------------------------------- */

  int status = check_result(particle, p_pos, p_x0, DT * NSTEPS, dom_lo, dom_hi);

  ops_exit();
  return status;
}
