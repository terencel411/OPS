/*
 * OPS Particles -- Tutorial 3: scattering particle data onto the grid
 * ===================================================================
 *
 * Tutorials 1 and 2 only ever moved data GRID -> PARTICLE. This one goes the
 * other way: particles accumulate onto a grid dat. That is the direction a
 * synthetic-eddy method, a particle-in-cell deposition or any source-term
 * coupling needs, and it uses a different loop form from either earlier
 * tutorial.
 *
 * There is no physics here. Particles carry a weight and drift at a constant
 * velocity; each step their weight is scattered onto a density grid. The
 * program checks itself against an exact integer identity.
 *
 * What this tutorial introduces
 *   1. the grid-outer / particle-inner ops_par_loop overload
 *      (ops_grid_part_seq_v2.h:562) -- the only form that may WRITE grid dats
 *   2. why the accumulator must be zeroed in a separate loop
 *   3. a search stencil wide enough to cover a physical interaction radius
 *
 * WHY THIS APP EXISTS
 * -------------------
 * This loop form is the one piece of the particle API that no working app in
 * this tree exercises (LBM-PSM is its only user and does not compile against
 * the current library). It is also the only form whose name collides with the
 * ordinary grid ops_par_loop, so the translator tries to code-generate it.
 * Isolating it in ~250 lines means any defect is found here rather than inside
 * an application.
 *
 * THE CHECK
 * ---------
 *   -mode all      every particle deposits at every stencil point it is
 *                  reachable from, with no geometric test, so
 *                      sum(rho) == N_particles * stencil_points
 *                  exactly, at any rank count. One integer, and it covers bin
 *                  traversal, ghost particles and the halo.
 *
 *   -mode radius   only nodes within RADIUS receive weight. No closed form, so
 *                  this is checked by requiring serial and MPI to agree.
 *
 * Build:  make tutorial3_dev_seq    (fastest: no translator)
 *         make tutorial3_dev_mpi    (MPI, no translator)
 * Run:    ./tutorial3_dev_seq
 *         mpirun -np 4 ./tutorial3_dev_mpi
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define OPS_2D

#include <ops_seq_v2.h>              /* grid ops_par_loop                   */
#include <ops_particle_seq.h>        /* ops_particle_par_loop               */
#include <ops_grid_part_seq_v2.h>    /* grid-outer particle ops_par_loop    */

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include "scatter_loop.h"
#include "grid_kernels.h"
#include "particle_kernels.h"

typedef double Real;

/* ------------------------------------------------------------------ *
 *  Parameters
 * ------------------------------------------------------------------ */

const int  NX     = 41;
const int  NY     = 41;
const Real LENGTH = 1.0;

int NPX = 10;   /* overridable with -npart, to probe OPS_MAX_PART */
int NPY = 10;

/* Seeded well inside the domain.
 *
 * The margin matters for the exact check. A particle in bin b is visited once
 * per stencil offset s, by the grid node b-s. The identity
 * sum(rho) == N*stencil_points therefore only holds if every one of those
 * source nodes is inside the iteration range -- i.e. if particles stay at
 * least HALO cells away from the grid boundary. This box keeps them near the
 * centre so the identity survives even a half-width of 12 cells. */
const Real SEED_BOX[4] = {0.40, 0.60, 0.40, 0.60};

const Real VEL[2] = {0.5, 0.25};
const Real DT     = 0.001;
const int  NSTEPS = 200;

const Real WEIGHT = 1.0;    /* what each particle deposits in "all" mode */

/* Set by -mode identity. Declared at file scope because seed_particles() needs
   it and it must be known before the particles are written. */
int identity_mode = 0;

/* Interaction radius, in cells. HALO cells either side gives a
   (2*HALO+1)^2 search stencil. Overridable with -halo to probe how the
   stencil width interacts with the MPI halo depth.                     */
int HALO = 1;

/* ================================================================== *
 *  Seeding -- identical in structure to tutorial 1
 * ================================================================== */
void seed_particles(ops_particle particle, ops_dat pos, ops_dat vel,
                    ops_dat wgt) {

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;

  if (NPX * NPY > (int)particle->Nmax)
    ops_particle_realloc_data(particle, NPX * NPY);

  Real *xp = (Real *)pos->data;
  Real *up = (Real *)vel->data;
  Real *wp = (Real *)wgt->data;

  const Real dx = (SEED_BOX[1] - SEED_BOX[0]) / static_cast<Real>(NPX - 1);
  const Real dy = (SEED_BOX[3] - SEED_BOX[2]) / static_cast<Real>(NPY - 1);

  int n = 0;
  for (int i = 0; i < NPX; i++) {
    for (int j = 0; j < NPY; j++) {
      const Real x = SEED_BOX[0] + dx * static_cast<Real>(i);
      const Real y = SEED_BOX[2] + dy * static_cast<Real>(j);

      /* Use the library's own ownership predicate rather than comparing
         against getLocalMin/Max by hand, so this cannot disagree with what
         the migration machinery believes.                                */
      const Real xy[2] = {x, y};
      if (!box->isCoordinateInBoundingBox(xy)) continue;

      xp[2 * n]     = x;
      xp[2 * n + 1] = y;
      up[2 * n]     = VEL[0];
      up[2 * n + 1] = VEL[1];
      /* IDENTITY MODE: a DISTINCT weight per particle -- the global lattice
         index plus one. With every particle carrying 1.0 (the original
         WEIGHT), reading the wrong particle's data is invisible, which is
         exactly the blind spot this mode exists to close. */
      wp[n]         = identity_mode ? (Real)(i * NPY + j + 1) : WEIGHT;
      n++;
    }
  }

  particle->no_particles = n;
}

/* ================================================================== *
 *  The per-step map / migration cycle -- unchanged from tutorial 1
 * ================================================================== */
void update_maps(ops_particle particle,
                 ops_dat *dat_border,  int nborder,
                 ops_dat *dat_forward, int nforward) {

  int decide = ops_particle_update_map_lists_actual_hybrid(particle);
  ops_particle_remove_delete_maps(particle, decide);

  if (decide)
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);
  else
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);

  ops_particle_reset_flags(particle, decide);
}

/* Build a square stencil of half-width h: (2h+1)^2 points. */
int *build_box_stencil(int h, int &npoints) {
  npoints = (2 * h + 1) * (2 * h + 1);
  int *s = (int *)malloc(sizeof(int) * 2 * npoints);
  int k = 0;
  for (int i = -h; i <= h; i++)
    for (int j = -h; j <= h; j++) {
      s[2 * k] = i; s[2 * k + 1] = j; k++;
    }
  return s;
}

int total_particles(ops_particle particle) {
  int n = (int)particle->no_particles;
#ifdef OPS_MPI
  int tot = 0;
  MPI_Allreduce(&n, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return tot;
#else
  return n;
#endif
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  int radius_mode = 0;
  int nsteps = NSTEPS;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc)
      { const char *m = argv[++i];
        radius_mode   = (strcmp(m, "radius") == 0);
        identity_mode = (strcmp(m, "identity") == 0); }
    else if (strcmp(argv[i], "-halo") == 0 && i + 1 < argc)
      HALO = atoi(argv[++i]);
    else if (strcmp(argv[i], "-nsteps") == 0 && i + 1 < argc)
      nsteps = atoi(argv[++i]);
    else if (strcmp(argv[i], "-npart") == 0 && i + 1 < argc)
      { NPX = atoi(argv[++i]); NPY = NPX; }
  }

  /* ---- 1. Block and grid ------------------------------------------ */

  ops_block block = ops_decl_block(2, "tutorial3_block");

  int size[] = {NX, NY};
  int base[] = {0, 0};
  int d_m[]  = {-1, -1};
  int d_p[]  = { 1,  1};

  Real *null_dbl = NULL;

  ops_dat x_grid = ops_decl_dat(block, 2, size, base, d_m, d_p, null_dbl,
                                "double", "x_grid");
  ops_dat rho    = ops_decl_dat(block, 1, size, base, d_m, d_p, null_dbl,
                                "double", "density");

  /* ---- 2. Stencils ------------------------------------------------- */

  int s2d_00[] = {0, 0};
  ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2d_00, "0,0");

  /* One stencil, two jobs -- they are NOT the same mechanism:
       - as the deposit loop's map_stencil it decides which bins each grid
         node searches;
       - passed to ops_decl_mapping it sets the bin array's halo depth, i.e.
         how far outside its subdomain a rank tracks ghost particles.
     Both need the same reach, so one object serves both.               */
  int nsten = 0;
  int *s2d_box = build_box_stencil(HALO, nsten);
  ops_stencil S2D_BOX = ops_decl_stencil(2, nsten, s2d_box, "search_box");

  /* ---- 3. Bounding box -------------------------------------------- */

  Real dx_box[] = {0.0, 0.0};
  BoundingBox<Real> *box = ops_create_bounding_box(block, x_grid, 2, dx_box);

  /* ---- 4. Particle set and dats ------------------------------------ */

  ops_particle particle = ops_decl_particle(block, "depositors", box);

  ops_dat p_pos = ops_decl_particle_pos_dat(particle, 2, base, null_dbl,
                                            "double", "position");
  ops_dat p_vel = ops_decl_particle_dat(particle, 2, base, null_dbl,
                                        "double", "velocity");
  ops_dat p_wgt = ops_decl_particle_dat(particle, 1, base, null_dbl,
                                        "double", "weight");

  /* ---- 5. Mapping -------------------------------------------------- */

  ops_particle_mapping map = ops_decl_mapping(particle, x_grid, S2D_BOX,
                                              OPS_WITH_VIRTUAL,
                                              OPS_UNIFORM_STAG, 1);

  /* p_wgt is in the border list because its value must survive a change of
     owning rank -- it is what gets deposited. */
  ops_dat dat_border[]  = {p_pos, p_vel, p_wgt};
  ops_dat dat_forward[] = {p_pos, p_vel, p_wgt};

  const int nborder  = sizeof(dat_border)  / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);

  /* ---- 6. Partition, fill grid, set up particles -------------------- */

  ops_partition("");

  Real dx = LENGTH / static_cast<Real>(NX - 1);
  int grid_range[] = {0, NX, 0, NY};

  ops_par_loop(KerInitGrid, "KerInitGrid", block, 2, grid_range,
               ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(&dx, 1, "double", OPS_READ),
               ops_arg_idx());

  /* Only now are the coordinates real, so only now can the box be derived. */
  ops_particle_setup_partition();

  seed_particles(particle, p_pos, p_vel, p_wgt);
  ops_particle_setup_maps_with_dats(particle, dat_border, nborder);

  const int npart = total_particles(particle);
  Real radius = HALO * dx;

  ops_printf("OPS Particles tutorial 3: scatter onto the grid\n");
  ops_printf("grid %dx%d, %d particles, mode=%s, halo=%d, stencil=%dx%d=%d\n",
             NX, NY, npart,
             identity_mode ? "identity" : (radius_mode ? "radius" : "all"),
             HALO, 2 * HALO + 1, 2 * HALO + 1, nsten);

  /* ---- 7. Time loop ------------------------------------------------ */

  Real range_parts[] = {0.0, LENGTH, 0.0, LENGTH};
  Real dt = DT;

  for (int step = 1; step <= nsteps; step++) {

    ops_particle_par_loop(KerUpdatePosition, "KerUpdatePosition", particle, 2,
                          OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                          ops_arg_dat_particle(p_pos, 2, "double", particle,
                                               map, OPS_WRITE),
                          ops_arg_dat_particle(p_vel, 2, "double", particle,
                                               map, OPS_READ),
                          ops_arg_gbl(&dt, 1, "double", OPS_READ));

    update_maps(particle, dat_border, nborder, dat_forward, nforward);

    ops_par_loop(KerZeroDensity, "KerZeroDensity", block, 2, grid_range,
                 ops_arg_dat(rho, 1, S2D_00, "double", OPS_WRITE));

    /* THE LOOP THIS APP EXISTS TO TEST. */
    if (radius_mode)
      ops_par_scatter_loop(KerDepositRadius, "KerDepositRadius", particle, map,
                   S2D_BOX, 2, grid_range,
                   ops_arg_dat(rho, 1, S2D_00, "double", OPS_INC),
                   ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_READ),
                   ops_arg_dat_particle(p_pos, 2, "double", particle, map,
                                        OPS_READ),
                   ops_arg_dat_particle(p_wgt, 1, "double", particle, map,
                                        OPS_READ),
                   ops_arg_gbl(&radius, 1, "double", OPS_READ));
    else
      ops_par_scatter_loop(KerDepositAll, "KerDepositAll", particle, map,
                   S2D_BOX, 2, grid_range,
                   ops_arg_dat(rho, 1, S2D_00, "double", OPS_INC),
                   ops_arg_dat_particle(p_wgt, 1, "double", particle, map,
                                        OPS_READ));
  }

  /* ---- 8. Check ---------------------------------------------------- */

  Real h_sum = 0.0;
  ops_reduction r_sum = ops_decl_reduction_handle(sizeof(Real), "double",
                                                  "rho_sum");
  ops_par_loop(KerSumDensity, "KerSumDensity", block, 2, grid_range,
               ops_arg_dat(rho, 1, S2D_00, "double", OPS_READ),
               ops_arg_reduce(r_sum, 1, "double", OPS_INC));
  ops_reduction_result(r_sum, &h_sum);

  const int nfinal = total_particles(particle);
  int ok = (nfinal == NPX * NPY);

  ops_printf("\n---------------------------------------------\n");
  ops_printf("particles expected : %d\n", NPX * NPY);
  ops_printf("particles found    : %d\n", nfinal);
  ops_printf("sum(rho)           : %.10g\n", h_sum);

  if (!radius_mode) {
    /* Exact integer identity -- see the header comment.
       In identity mode the weights are 1..N, so the expected total is
       stencil_points * sum(1..N) = nsten * N*(N+1)/2. Getting this right
       requires each visit to read the RIGHT particle's weight, not merely to
       visit the right NUMBER of particles. */
    const Real N = (Real)(NPX * NPY);
    const Real expect = identity_mode
                      ? (Real)nsten * N * (N + 1.0) / 2.0
                      : N * (Real)nsten * WEIGHT;
    const Real err = fabs(h_sum - expect);
    ops_printf("sum(rho) expected  : %.10g   (%s)\n", expect,
               identity_mode ? "stencil * sum(1..N)" : "N * stencil");
    ops_printf("error              : %.3e\n", err);
    ok = ok && (err < 1e-9);
  } else {
    ops_printf("(radius mode: compare this number between np=1 and np=4)\n");
  }

  ops_printf("RESULT             : %s\n", ok ? "PASS" : "FAIL");
  ops_printf("---------------------------------------------\n");

  free(s2d_box);
  ops_exit();
  return ok ? 0 : 1;
}
