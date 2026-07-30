/*
 * OPS Particles -- Tutorial 1 (3D): constant-velocity drift
 * =========================================================
 *
 * The 3D counterpart of apps/c/particle_tutorial_1_drift. Read that one first:
 * the structure is identical and its comments explain the *why*. This file
 * notes only what changes when you go from 2D to 3D.
 *
 * Particles are seeded as a uniform lattice through the whole box and drift at
 * a constant, prescribed velocity along x. There is no fluid solver and no
 * interpolation -- the point is to show the *structure* of a particle
 * application with nothing else in the way.
 *
 * The box is periodic in x, so the population never changes: a particle
 * leaving the downstream face re-enters upstream and exactly fills the gap it
 * left. The result is a steady stream that looks the same at every instant,
 * however long you run.
 *
 * There is no y or z motion anywhere in this app, and no boundary condition on
 * the four side faces either -- none is needed, because a particle's y and z
 * never change.
 *
 * The exact answer is still known -- x0 + v*t, folded back into the box by the
 * periodic boundary -- and the program checks it itself, so a correct run is
 * unambiguous.
 *
 * What changes from the 2D app
 *   1. #define OPS_3D, block dims 3, every geometry array gains a third entry
 *   2. the grid accessor is xf(component, i, j, k)
 *   3. the mapping stencil is the 27-point cube, not the 9-point square
 *   4. the seeding lattice is a triple loop, and the half-grid-cell offset is
 *      applied in all three directions -- see seed_particles()
 *   5. translate/dx_halo for the periodic halo are 3-vectors
 *
 * Build:  make tutorial1_3d_dev_seq   (fastest: no translator)
 *         make tutorial1_3d_dev_mpi   (MPI, no translator)
 *         make tutorial1_3d_mpi       (MPI, via the OPS translator)
 * Run:    ./tutorial1_3d_dev_seq
 *         mpirun -np 4 ./tutorial1_3d_dev_mpi
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define OPS_3D

#include <ops_seq_v2.h>          /* grid ops_par_loop                       */
#include <ops_particle_seq.h>    /* ops_particle_par_loop, particle library */

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include "grid_kernels.h"
#include "particle_kernels.h"

/* Periodic HDF5 output -- one self-contained .h5 file per output step, holding
   the block, the grid dat, the particle dats and the run constants.        */
#include "drift_io.h"

typedef double Real;

/* ------------------------------------------------------------------ *
 *  Simulation parameters -- all in one place so you can experiment
 * ------------------------------------------------------------------ */

const int  NX      = 41;        /* grid nodes in x                        */
const int  NY      = 21;        /* grid nodes in y                        */
const int  NZ      = 21;        /* grid nodes in z                        */

/* An elongated box: longer along the drift direction, as in a channel.
   These lengths and node counts give cubic cells, dx = dy = dz = 0.025. */
const Real LX      = 1.0;
const Real LY      = 0.5;
const Real LZ      = 0.5;

const int  NPX     = 10;        /* particles seeded along x               */
const int  NPY     = 5;         /* particles seeded along y               */
const int  NPZ     = 5;         /* particles seeded along z               */

/* The particles are seeded as a uniform lattice through the WHOLE box, not
   into a sub-region: with a periodic x boundary that makes the stream steady
   -- every plane of particles leaving the downstream face re-enters upstream
   and exactly fills the gap it left, so any snapshot looks like any other.  */

/* Drift is along x only, and the whole app assumes it: nothing here confines
   a particle in y or z, because with VEL[1] = VEL[2] = 0 nothing ever moves
   in those directions. Giving either a non-zero value would let particles
   stream out through a side face, where OPS deletes them and the final check
   fails -- and would also expose the library defect recorded in
   documentation/ops-particles-defects.md section 7.                      */
const Real VEL[3]  = {0.5, 0.0, 0.0};   /* the constant drift velocity    */
const Real DT      = 0.001;
const int  NSTEPS  = 1000;      /* 0.5 box lengths per 1000 steps         */
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
 *   4. under MPI, seed only inside THIS rank's part of the box, and make the
 *      ids globally unique
 */
void seed_particles(ops_particle particle, ops_dat pos, ops_dat vel,
                    ops_dat ids, ops_dat x0) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  /* The bounding box knows two extents:
   *   getGlobalMin/Max -- the whole box, same on every rank
   *   getLocalMin/Max  -- the part owned by THIS rank
   * We want a globally-defined seeding pattern, but each rank may only
   * place the particles that fall inside its own subdomain.            */
  const Real lo[3]  = {box->getLocalMin().x,  box->getLocalMin().y,
                       box->getLocalMin().z};
  const Real hi[3]  = {box->getLocalMax().x,  box->getLocalMax().y,
                       box->getLocalMax().z};
  const Real glo[3] = {box->getGlobalMin().x, box->getGlobalMin().y,
                       box->getGlobalMin().z};
  const Real ghi[3] = {box->getGlobalMax().x, box->getGlobalMax().y,
                       box->getGlobalMax().z};

  /* Uniform lattice through the WHOLE box, defined globally.
   *
   * Dividing by NPX and not NPX-1 is what makes the stream steady. The x
   * boundary is periodic, so the lattice has to TILE: with NPX-1 spacing there
   * would be a plane of particles on both end faces, and since those are the
   * same place under periodicity the result is a double-density plane at the
   * seam and a gap of one dx beside it. The stream would then pulse once per
   * lap instead of looking the same at every instant.
   *
   * The +0.5 puts particles at cell centres so none is ever seeded exactly on
   * a face. That matters because OPS bins on a half-open [lo,hi) convention,
   * which reads a particle sitting exactly on hi as outside.               */
  const Real dx = (ghi[0] - glo[0]) / static_cast<Real>(NPX);
  const Real dy = (ghi[1] - glo[1]) / static_cast<Real>(NPY);
  const Real dz = (ghi[2] - glo[2]) / static_cast<Real>(NPZ);

  /* Then shift the whole lattice half a GRID cell, in ALL THREE directions.
   *
   * Subdomain boundaries are cuts in index space, so they always land on grid
   * nodes. A particle seeded exactly on a node can therefore sit exactly on a
   * rank boundary, and one that does is lost in the first migration -- it
   * survives seeding (the ownership filter below picks a single owner) but
   * disappears from the count by the next output. Measured in the 2D app at
   * np = 8: the column at x = 0.25, which is node 10, took all 10 of its
   * particles with it while every other column was fine.
   *
   * This matters MORE in 3D than in 2D. MPI_Dims_create splits y and z as well
   * as x -- np = 8 gives a 2x2x2 decomposition -- so there are node-aligned
   * rank boundaries in every direction, not just along the drift.
   *
   * Offsetting by half a cell puts every particle strictly inside a cell, so
   * it cannot coincide with a boundary at ANY rank count. Shifting the whole
   * lattice rigidly keeps the spacing uniform, so the pattern still tiles
   * across the periodic seam and the stream stays steady.                  */
  const Real half_cell_x = 0.5 * (ghi[0] - glo[0]) / static_cast<Real>(NX - 1);
  const Real half_cell_y = 0.5 * (ghi[1] - glo[1]) / static_cast<Real>(NY - 1);
  const Real half_cell_z = 0.5 * (ghi[2] - glo[2]) / static_cast<Real>(NZ - 1);

  /* Worst case every candidate lands on this rank, so make room for them. */
  if (NPX * NPY * NPZ > (int)particle->Nmax)
    ops_particle_realloc_data(particle, NPX * NPY * NPZ);

  Real *xp   = (Real *)pos->data;
  Real *up   = (Real *)vel->data;
  Real *xp0  = (Real *)x0->data;
  int  *idp  = (int  *)ids->data;

  int n = 0;
  for (int i = 0; i < NPX; i++) {
    for (int j = 0; j < NPY; j++) {
      for (int k = 0; k < NPZ; k++) {
        Real x = glo[0] + dx * (static_cast<Real>(i) + 0.5) + half_cell_x;
        Real y = glo[1] + dy * (static_cast<Real>(j) + 0.5) + half_cell_y;
        Real z = glo[2] + dz * (static_cast<Real>(k) + 0.5) + half_cell_z;

        /* Skip candidates owned by another rank. In serial lo/hi span the
           whole box, so nothing is skipped.                              */
        if (x < lo[0] || x >= hi[0] ||
            y < lo[1] || y >= hi[1] ||
            z < lo[2] || z >= hi[2]) continue;

        /* Particle dats are AoS: data[dim * particle_index + component] */
        xp[3 * n]     = x;
        xp[3 * n + 1] = y;
        xp[3 * n + 2] = z;

        /* x0 records where this particle started. We never touch it again.
           It rides along with the particle when it migrates between ranks,
           which is what makes the final check below possible.             */
        xp0[3 * n]     = x;
        xp0[3 * n + 1] = y;
        xp0[3 * n + 2] = z;

        up[3 * n]     = 0.0;
        up[3 * n + 1] = 0.0;
        up[3 * n + 2] = 0.0;

        /* (4) Ids must be unique across ALL ranks. The seeding pattern is a
               known global lattice, so the cleanest id is the global lattice
               index: unique by construction, identical no matter how many
               ranks you run on, and it lets you follow one particle across
               runs.                                                        */
        idp[n] = (i * NPY + j) * NPZ + k;

        n++;
      }
    }
  }

  /* (3) Tell OPS how many particles this rank now owns. */
  particle->no_particles = n;
}

/* ================================================================== *
 *  The per-step map / migration cycle
 * ================================================================== *
 *
 * Call this once per timestep, after positions change. Identical to the 2D
 * app -- none of this is dimension-dependent.
 *
 * Step 1 walks the particles and asks "did anything move far enough to
 * matter?".  Only if a particle crossed into or out of the halo band -- or
 * left the box -- does it return true, and that decision is then made
 * collectively across all ranks.
 *
 * The fork:
 *   decide == true  -> expensive: migrate ownership, rebuild the ghost
 *                      layer and all maps.  Communicates dat_border.
 *   decide == false -> cheap: just refresh values on the ghost layer
 *                      that already exists.  Communicates dat_forward.
 */
void update_maps(ops_particle particle,
                 ops_dat *dat_border,  int nborder,
                 ops_dat *dat_forward, int nforward) {

  int decide = ops_particle_update_map_lists_actual_hybrid(particle);

#ifdef OPS_MPI
  /* Carry particles across the periodic seam BEFORE the deletion pass below,
     or the ones that just left through the downstream face get removed as
     out-of-box instead of re-appearing upstream.                          */
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
 * Constant velocity and forward Euler means the exact answer is known, the
 * periodic boundary just folds it back into the box:
 *
 *   x    : re-injection makes the motion periodic, so the exact position is
 *          x0 + v*t reduced modulo the box length.
 *   y, z : the drift has no y or z component, so the exact answer is y == y0
 *          and z == z0 for every particle, for all time. Checking them is not
 *          redundant -- it is what catches a particle whose data was mangled
 *          by a migration.
 *
 * Any deviation beyond round-off is a real bug, not discretisation error.
 */
int check_result(ops_particle particle, ops_dat pos, ops_dat x0, Real t,
                 const Real *lo, const Real *hi) {

  Real *xp  = (Real *)pos->data;
  Real *xp0 = (Real *)x0->data;

  const Real Lx = hi[0] - lo[0];

  Real worst = 0.0;
  for (size_t p = 0; p < particle->no_particles; p++) {

    /* Periodic in x. fmod can return a negative remainder, hence the fixup. */
    Real sx = fmod(xp0[3 * p] - lo[0] + VEL[0] * t, Lx);
    if (sx < 0.0) sx += Lx;
    Real ex = lo[0] + sx;

    /* Measure the x error the short way round the loop. A particle sitting
       within round-off of the seam is at hi in the simulation and at lo in
       the formula (or the reverse); those are the same physical point, and
       only the modular distance says so.                                   */
    Real dx = fabs(xp[3 * p] - ex);
    dx = fmin(dx, Lx - dx);

    worst = fmax(worst, dx);
    worst = fmax(worst, fabs(xp[3 * p + 1] - xp0[3 * p + 1]));
    worst = fmax(worst, fabs(xp[3 * p + 2] - xp0[3 * p + 2]));
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
  int ok = (worst < tol) && (ntotal == NPX * NPY * NPZ);

  ops_printf("\n---------------------------------------------\n");
  ops_printf("particles expected : %d\n", NPX * NPY * NPZ);
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

  ops_block block = ops_decl_block(3, "tutorial_block_3d");

  int size[] = {NX, NY, NZ};
  int base[] = {0, 0, 0};
  int d_m[]  = {-1, -1, -1};
  int d_p[]  = { 1,  1,  1};

  Real *null_dbl = NULL;
  int  *null_int = NULL;

  /* dim 3: each node stores its (x,y,z) coordinate. */
  ops_dat x_grid = ops_decl_dat(block, 3, size, base, d_m, d_p, null_dbl,
                                "double", "x_grid");

  /* ---------------------------------------------------------------- *
   * 2. Stencils
   * ---------------------------------------------------------------- */

  int s3d_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3d_000, "0,0,0");

  /* The 27-point cube: the 3D analogue of the 2D app's 9-point square. */
  int s3d_27pt[] = {
    -1,-1,-1,  -1,-1, 0,  -1,-1, 1,
    -1, 0,-1,  -1, 0, 0,  -1, 0, 1,
    -1, 1,-1,  -1, 1, 0,  -1, 1, 1,
     0,-1,-1,   0,-1, 0,   0,-1, 1,
     0, 0,-1,   0, 0, 0,   0, 0, 1,
     0, 1,-1,   0, 1, 0,   0, 1, 1,
     1,-1,-1,   1,-1, 0,   1,-1, 1,
     1, 0,-1,   1, 0, 0,   1, 0, 1,
     1, 1,-1,   1, 1, 0,   1, 1, 1
  };
  ops_stencil S3D_27pt = ops_decl_stencil(3, 27, s3d_27pt, "27pt");

  /* ---------------------------------------------------------------- *
   * 3. Bounding box  -- the bridge from index space to physical space
   * ---------------------------------------------------------------- *
   * Particles are located by coordinates, not indices, so the library
   * needs to know the physical extent of the block.
   *
   * Derived from the coordinate dat, exactly as in the 2D app, so that it
   * cannot disagree with the grid.
   *
   * !! THIS APP DOES NOT RUN UNDER MPI YET. !!  Every rank dies in
   * ops_particle_setup_partition() with "Defined bounding box of non-positive
   * volume". The cause is a library defect, not this app:
   * _ops_construct_local_box_from_dat()
   * (ops/c/include/ops_particle_box_host_funcs.h:45) has a 3D array-of-structs
   * branch that assigns xmin[0..2], xmax[0] and xmax[1] but *never assigns
   * xmax[2]*. The MPI caller zero-initialises its xmin/xmax
   * (ops_particle_box_mpi_funcs.h:139-141), so xmax[2] stays equal to xmin[2]
   * and the volume check rejects it. The 2D branch and the 3D SoA branch of
   * the same function are both complete -- only 3D AoS, the default, is hit.
   *
   * Serial only appears to work: its caller
   * (ops_particle_box_host_funcs.h:128) leaves xmin/xmax uninitialised, so
   * xmax[2] is read from stack garbage. The z bounds in a serial run are
   * therefore not trustworthy even though the run reports PASS.
   *
   * Declaring the region explicitly instead --
   *   Real region[] = {0,0,0, LX,LY,LZ};
   *   ops_create_bounding_box(block, 3, region);
   * -- does reach the correct branch, but that branch needs the mapping's
   * cell size, and a grid+stencil mapping derives that from the coordinate
   * dat through the box, so map->dx comes out {0,0,0} and it fails a step
   * later. There is no app-side workaround; the library needs the one missing
   * assignment. See documentation/ops-particles-defects.md.             */

  Real dx_box[] = {0.0, 0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, x_grid, 3, dx_box);

  /* ---------------------------------------------------------------- *
   * 4. The particle set and its dats
   * ---------------------------------------------------------------- */

  ops_particle particle = ops_decl_particle(block, "drifters", box);

  /* The position dat is privileged: exactly one per particle set, its
     dim must equal the block dimension (3 here), and every piece of
     machinery in the library (binning, migration, deletion) reads
     positions from it. Declare it BEFORE any mapping.                  */
  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 3, base, null_dbl,
                                            "double", "position");

  /* Ordinary particle dats: length Nmax, `dim` components each, AoS. */
  ops_dat p_vel = ops_decl_particle_dat(particle, 3, base, null_dbl,
                                        "double", "velocity");
  ops_dat p_x0  = ops_decl_particle_dat(particle, 3, base, null_dbl,
                                        "double", "start_position");
  ops_dat p_ids = ops_decl_particle_dat(particle, 1, base, null_int,
                                        "int", "id");

  /* ---------------------------------------------------------------- *
   * 5. The mapping -- a cell-linked list binning particles onto cells
   * ---------------------------------------------------------------- *
   * NOTE: S3D_27pt here is NOT an interpolation stencil. It only widens
   * the halo depth of the bin array, i.e. it declares "track particles up
   * to one cell outside my subdomain". That is what sizes the ghost
   * particle band used during migration.                               */

  ops_particle_mapping map = ops_decl_mapping(particle, x_grid, S3D_27pt,
                                              OPS_WITH_VIRTUAL,
                                              OPS_UNIFORM_STAG, 1);

  /* ---------------------------------------------------------------- *
   * 6. Dat lists for particle communication
   * ---------------------------------------------------------------- *
   * border  : exchanged when a particle changes rank owner. EVERY dat
   *           whose value must survive migration belongs here.
   * forward : the cheap per-step refresh of the existing ghost layer.  */

  ops_dat dat_border[]  = {p_pos, p_vel, p_ids, p_x0};
  ops_dat dat_forward[] = {p_pos, p_vel};
  ops_dat dat_output[]  = {p_ids, p_pos, p_vel};

  const int nborder  = sizeof(dat_border)  / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);
  const int noutput  = sizeof(dat_output)  / sizeof(dat_output[0]);

  /* ---------------------------------------------------------------- *
   * 6b. Periodic halo in x  -- how particles get re-injected
   * ---------------------------------------------------------------- *
   * A kernel that does `if (x >= x_max) x -= LX` works in serial and fails
   * under MPI: it teleports a particle the full length of the box, and the
   * migration cycle only moves particles to a NEIGHBOURING rank. The
   * supported mechanism is a particle halo carrying a translation, two of
   * them, one per direction.
   *
   * MPI BUILDS ONLY. The single-node library rejects this setup outright
   * ("Particle Halo exchange must be set after block boxes are set",
   * ops/c/src/sequential/ops_particle_host_single_node.cpp:1908), so the
   * serial builds keep the direct wrap in KerWrapX.
   *
   * OPS_PART_ORIENT_ON marks the dat that must be SHIFTED by `translate` as
   * it crosses -- the position, and only the position.
   *
   * In 3D the only change is that translate, dx_halo and the direction list
   * are 3-vectors. The translation is still purely along x.               */

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
     cell is enough: a particle cannot cross more than a cell in a step.   */
  Real dx_halo[]     = {LX / static_cast<Real>(NX - 1), 0.0, 0.0};
  Real translate_p[] = { LX, 0.0, 0.0};   /* downstream face -> upstream   */
  Real translate_m[] = {-LX, 0.0, 0.0};   /* upstream face -> downstream   */
  int  halo_dir[]    = {0, 1, 2};

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
     migration cycle in update_maps().                                     */
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
   * and OPS throws "bounding box of non-positive volume". That throw is
   * exactly what blocks the oSEM_3D_particles port -- see
   * git show HEAD:apps/c/oSEM_3D_particles/PORT_NOTES.md               */

  Real dx3[] = {LX / static_cast<Real>(NX - 1),
                LY / static_cast<Real>(NY - 1),
                LZ / static_cast<Real>(NZ - 1)};
  int grid_range[] = {0, NX, 0, NY, 0, NZ};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 3, grid_range,
               ops_arg_dat(x_grid, 3, S3D_000, "double", OPS_WRITE),
               ops_arg_gbl(dx3, 3, "double", OPS_READ),
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
     particle has left the box. Take it from the box rather than assuming
     [0,LX] x [0,LY] x [0,LZ]: the box is derived from the coordinate dat and
     is sized by the bin count, so it need not agree with the grid range
     exactly. The boundary kernel has to use the same numbers OPS uses, or
     particles get deleted on the step they are re-injected.               */
  const Real dom_lo[3] = {box->getGlobalMin().x, box->getGlobalMin().y,
                          box->getGlobalMin().z};
  const Real dom_hi[3] = {box->getGlobalMax().x, box->getGlobalMax().y,
                          box->getGlobalMax().z};

  /* ---------------------------------------------------------------- *
   * 9. Seed particles, then build the maps for the first time
   * ---------------------------------------------------------------- */

  seed_particles(particle, p_pos, p_vel, p_ids, p_x0);

  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

#ifdef OPS_MPI
  /* Populate the ghost band across the periodic seam for the first time. */
  ops_particle_halo_transfer_group(OPS_HALO_GRP_BORDER, true);
#endif

  ops_printf("OPS Particles tutorial 1 (3D): constant-velocity drift\n");
  ops_printf("grid %dx%dx%d, %d particles, v = (%g, %g, %g), dt = %g, %d steps\n",
             NX, NY, NZ, NPX * NPY * NPZ, VEL[0], VEL[1], VEL[2], DT, NSTEPS);
  ops_printf("box [%g,%g] x [%g,%g] x [%g,%g]: periodic in x, no motion in y/z\n",
             dom_lo[0], dom_hi[0], dom_lo[1], dom_hi[1], dom_lo[2], dom_hi[2]);

  ops_particle_print_dats_to_txtfile(particle, dat_output, noutput,
                                     "particles_step_0.txt");

  /* Everything a plot script needs about the run, travelling with the data. */
  drift_io_params io_params = {NX, NY, NZ, NPX, NPY, NPZ, NSTEPS, NPRINT,
                               LX, LY, LZ, DT,
                               {VEL[0], VEL[1], VEL[2]},
                               {dom_lo[0], dom_hi[0], dom_lo[1], dom_hi[1],
                                dom_lo[2], dom_hi[2]}};

  /* Initial state -> drift3d_output_000000.h5 */
  HDF5_IO_Write_drift_block_dynamic(block, -1, x_grid, particle, dat_output,
                                    noutput, io_params);

  /* ---------------------------------------------------------------- *
   * 10. Set the (constant) velocity once
   * ---------------------------------------------------------------- *
   * range_parts is the physical iteration region -- 2 entries per
   * dimension, so 6 in 3D. It is only consulted for
   * OPS_PARTICLE_ITERATE_RANDOM, but the argument is always required.
   *
   * OPS_PARTICLE_ITERATE_LOCAL means "owned particles only".            */

  Real range_parts[] = {0.0, LX, 0.0, LY, 0.0, LZ};

  ops_particle_par_loop(KerSetVelocity, "KerSetVelocity", particle, 3,
                        OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                        ops_arg_dat_particle(p_vel, 3, "double", particle, map,
                                             OPS_WRITE),
                        ops_arg_gbl(VEL, 3, "double", OPS_READ));

  /* ---------------------------------------------------------------- *
   * 11. Time loop
   * ---------------------------------------------------------------- */

  Real dt = DT;

  for (int step = 1; step <= NSTEPS; step++) {

    ops_particle_par_loop(KerUpdatePosition, "KerUpdatePosition", particle, 3,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 3, "double", particle,
                                               map, OPS_WRITE),
                          ops_arg_dat_particle(p_vel, 3, "double", particle,
                                               map, OPS_READ),
                          ops_arg_gbl(&dt, 1, "double", OPS_READ));

#ifndef OPS_MPI
    /* Serial: re-inject by hand. This MUST happen before update_maps(): that
       is where OPS deletes out-of-box particles, and by then it is too late.
       Under MPI it is the halo groups' job, and doing it here as well would
       hide the crossing from them entirely.                                */
    ops_particle_par_loop(KerWrapX, "KerWrapX", particle, 3,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 3, "double", particle,
                                               map, OPS_RW),
                          ops_arg_gbl(dom_lo, 3, "double", OPS_READ),
                          ops_arg_gbl(dom_hi, 3, "double", OPS_READ));
#endif

    /* Positions changed, so the spatial index may need repairing and
       particles may need to move to another rank.                      */
    update_maps(particle, dat_border, nborder, dat_forward, nforward);

    if (step % NPRINT == 0) {
      std::string fname = "particles_step_" + std::to_string(step) + ".txt";
      ops_particle_print_dats_to_txtfile(particle, dat_output, noutput,
                                         fname.c_str());

      /* Same state as the .txt dump, but as drift3d_output_<step>.h5 */
      HDF5_IO_Write_drift_block_dynamic(block, step - 1, x_grid, particle,
                                        dat_output, noutput, io_params);

      ops_printf("step %5d / %d\n", step, NSTEPS);
    }
  }

  /* Final state -> drift3d_output.h5 */
  HDF5_IO_Write_drift_block(block, NSTEPS - 1, x_grid, particle, dat_output,
                            noutput, io_params);

  /* ---------------------------------------------------------------- *
   * 12. Check against the exact answer
   * ---------------------------------------------------------------- */

  int status = check_result(particle, p_pos, p_x0, DT * NSTEPS, dom_lo, dom_hi);

  ops_exit();
  return status;
}
