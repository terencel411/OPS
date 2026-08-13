/*
 * oSEM_3d_particles_v1 -- ../oSEM_3d with the eddies as OPS particles and the
 * MPI_Allgatherv replaced by an OPS reduction. Everything else is ../oSEM_3d's.
 * The README explains the workflow, the constraints and the deviations.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "constants.h"
#include "Pir_data.h"
#include "TBL_data.h"

#define OPS_3D
#define OPS_API 2
#include <ops_seq_v2.h>
#include <ops_particle_seq.h>

#include "osem3d_common.h"
#include "opensbliblock00_kernels.h"
#include "eddy_kernels.h"
#include "ops_particle_random.h"
#include "io.h"

typedef double Real;

/* Diagnostic only (-degenerate-rng): u -> 0.5 + 0.5u reproduces ../oSEM_3d's
   broken draw -- upper half of every range, and every sign +1. */
static int opt_degen = 0;

static void degrade_rnd(ops_particle particle, ops_dat rnd) {
  double *d = (double *)rnd->data;
  const long n = (long)particle->no_particles * rnd->dim;
  for (long i = 0; i < n; i++) d[i] = 0.5 + 0.5 * d[i];
}

/* The reduction dims below are bare literals because the translator's
   parseIntLiteral wants a single token. These keep them in step. */
static_assert(UINTERP_CAP == 4096, "literal in the ops_arg_reduce call must match");
static_assert(RST_CAP == 600, "literal in the ops_arg_reduce call must match");

/*
 * Place the eddies, give each a global id, and decide who owns it. Host code:
 * a kernel cannot set the particle count, and ownership is decided by position.
 * `pos` is the position clamped into the block -- an ownership handle only; the
 * eddy's real coordinates go to `ecrd`. Half-open test, since boxes share faces.
 */
static void seed_eddies(ops_particle particle, ops_dat pos, ops_dat ecrd,
                        ops_dat gid, int neddy, unsigned int seed) {
  Real lo[OPS_MAX_DIM], hi[OPS_MAX_DIM];
  for (int d = 0; d < OPS_MAX_DIM; d++) { lo[d] = 1.0e30; hi[d] = -1.0e30; }

  BoundingBox<Real> *box = (BoundingBox<Real> *)particle->box_block;
  box->getLocalMaxMin(lo, hi);          /* untouched on a rank owning nothing */

  const ops_point<Real> gmin = box->getGlobalMin();
  const ops_point<Real> gmax = box->getGlobalMax();
  const Real glo[3] = {gmin.x, gmin.y, gmin.z};
  const Real ghi[3] = {gmax.x, gmax.y, gmax.z};

  if (neddy > (int)particle->Nmax) ops_particle_realloc_data(particle, neddy);

  Real *xp = (Real *)pos->data;
  Real *ep = (Real *)ecrd->data;
  int *ip = (int *)gid->data;

  int n = 0;
  for (int i = 0; i < neddy; i++) {
    Real u[3] = {ops_prandom_uniform(seed, i, 0u, 0),
                 ops_prandom_uniform(seed, i, 0u, 1),
                 ops_prandom_uniform(seed, i, 0u, 2)};
    if (opt_degen) for (int d = 0; d < 3; d++) u[d] = 0.5 + 0.5 * u[d];

    Real e[3];
    e[0] = eddy_x_min + (eddy_x_max - eddy_x_min) * u[0];
    e[1] = eddy_y_min + (eddy_y_max - eddy_y_min) * u[1];
    e[2] = eddy_z_min + (eddy_z_max - eddy_z_min) * u[2];

    /* The ownership handle: nearest point of the block to the eddy. */
    Real q[3];
    int outside = 0;
    for (int d = 0; d < 3; d++) {
      q[d] = e[d] < glo[d] ? glo[d] : (e[d] > ghi[d] ? ghi[d] : e[d]);
      if (q[d] < lo[d] || q[d] >= hi[d]) outside = 1;
    }
    /* The global top face belongs to the rank that ends there, or no rank
       would claim an eddy clamped exactly onto it. */
    if (outside) {
      outside = 0;
      for (int d = 0; d < 3; d++)
        if (q[d] < lo[d] || (q[d] > hi[d]) ||
            (q[d] == hi[d] && hi[d] != ghi[d])) outside = 1;
    }
    if (outside) continue;

    xp[3*n+0] = q[0]; xp[3*n+1] = q[1]; xp[3*n+2] = q[2];
    ep[3*n+0] = e[0]; ep[3*n+1] = e[1]; ep[3*n+2] = e[2];
    ip[n] = i;
    n++;
  }
  particle->no_particles = n;
}

int main(int argc, char **argv)
{
// Initializing OPS
ops_init(argc,argv,1);
// Set restart to 1 to restart the simulation from HDF5 file
restart = 0;
// User defined constant values
Lx1 = 100.0;

// ../oSEM_3d hardwires 750 x 250 x 150. The default here is small enough to
// run on a laptop; -ngrid puts the full size one flag away.
block0np0 = 150;
block0np1 = 50;
block0np2 = 30;
int    opt_niter = 100;
// -neddies overrides the count the eddy box implies. 0 runs the solver with no
// synthetic turbulence at all, which is the control for "is a blow-up the eddy
// path or the grid".
int    opt_neddies = -1;
// -pad-below-wall restores ../oSEM_3d's eddy_y_min = -radius, which seeds
// eddies under a solid wall. Default here is 0.0, as in the 2-D reference.
int    opt_padwall = 0;
fluct_scale[0] = fluct_scale[1] = fluct_scale[2] = 1.0;
int    opt_nout  = 10;
seed_gbl = 182383739u;

for (int i = 1; i < argc; i++) {
  if (!strcmp(argv[i], "-ngrid") && i + 3 < argc) {
    block0np0 = atoi(argv[i+1]);
    block0np1 = atoi(argv[i+2]);
    block0np2 = atoi(argv[i+3]);
    i += 3;
  } else if (!strcmp(argv[i], "-niter") && i + 1 < argc) {
    opt_niter = atoi(argv[++i]);
  } else if (!strcmp(argv[i], "-nout") && i + 1 < argc) {
    opt_nout = atoi(argv[++i]);
  } else if (!strcmp(argv[i], "-fluct") && i + 3 < argc) {
    fluct_scale[0] = atof(argv[i+1]);
    fluct_scale[1] = atof(argv[i+2]);
    fluct_scale[2] = atof(argv[i+3]);
    i += 3;
  } else if (!strcmp(argv[i], "-pad-below-wall")) {
    opt_padwall = 1;
  } else if (!strcmp(argv[i], "-degenerate-rng")) {
    opt_degen = 1;
  } else if (!strcmp(argv[i], "-neddies") && i + 1 < argc) {
    opt_neddies = atoi(argv[++i]);
  } else if (!strcmp(argv[i], "-seed") && i + 1 < argc) {
    seed_gbl = (unsigned int)strtoul(argv[++i], NULL, 10);
  }
}

Delta0block0 = 375.0/(block0np0-1);
Delta1block0 = 100.0/(block0np1-1);
Delta2block0 = 40.0/(block0np2);
niter = opt_niter;
double rkB[] = {(1.0/3.0), (15.0/16.0), (8.0/15.0)};
double rkA[] = {0, (-5.0/9.0), (-153.0/128.0)};
dt = 0.025;
write_output_file = opt_nout;
HDF5_timing = 0;
Pr = 0.72;
Minf = 2.0;
r = pow(Pr, 0.33333333);
Re = 873.4;
gama = 1.4;
SuthT = 110.4;
RefT = 202.17;
Twall = 1 + 0.5 * r * (gama-1) * Minf*Minf;
by = 4.0;
inv2Delta0block0 = 1.0/(Delta0block0*Delta0block0);
inv2Delta1block0 = 1.0/(Delta1block0*Delta1block0);
inv2Delta2block0 = 1.0/(Delta2block0*Delta2block0);
inv2Minf = 1.0/(Minf*Minf);
invDelta0block0 = 1.0/(Delta0block0);
invDelta1block0 = 1.0/(Delta1block0);
invDelta2block0 = 1.0/(Delta2block0);
invLx1 = 1.0/(Lx1);
invPr = 1.0/(Pr);
invRe = 1.0/(Re);
invRefT = 1.0/(RefT);
inv_gamma_m1 = 1.0/((-1 + gama));
start_averaging = (niter > 100) ? 0.5*niter : 50;
invniter = 1.0/(niter - start_averaging);

ny = (int)trunc(block0np1 * 0.6);
uinterp = (double*)malloc(ny * sizeof(double));

//------------------- eddy variables---------------------------
// ../oSEM_3d hardwires y_cutoff = 150, which is both the length of the a11..a33
// profiles and the y extent of the interp_RST loop. At a reduced grid that
// range would run off the block, so it is capped.
y_cutoff = (block0np1 < 150) ? block0np1 : 150;
ndata = 121;
a11 = (double*)malloc(y_cutoff * sizeof(double));
a21 = (double*)malloc(y_cutoff * sizeof(double));
a22 = (double*)malloc(y_cutoff * sizeof(double));
a33 = (double*)malloc(y_cutoff * sizeof(double));
delta = 11.6973525411;
radius = 0.2 * delta;
eddy_x_min = -radius;
eddy_x_max = radius;
eddy_y_min = opt_padwall ? (0.0 - radius) : 0.0;   // see opt_padwall above
eddy_y_max = delta + radius;
eddy_z_min = -radius;
eddy_z_max = 40.0 + radius;
eddy_vol = std::abs((eddy_x_max - eddy_x_min) * (eddy_y_max - eddy_y_min) * (eddy_z_max - eddy_z_min));
eddies = trunc(eddy_vol/(radius*radius*radius));
if (opt_neddies >= 0) eddies = opt_neddies;
// The reduction always carries at least one slot, so -neddies 0 does not have
// to be special-cased through the gather.
const int nred = (eddies > 0 ? eddies : 1) * NCOMP;
increment = 1.0 * dt;

// The gather buffer, in place of ../oSEM_3d's seven eddy_*_gbl host arrays.
eddy_all = (double*)malloc(nred * sizeof(double));

if (ny > UINTERP_CAP) {
  ops_printf("ny = %d exceeds UINTERP_CAP = %d; raise it in osem3d_common.h\n", ny, UINTERP_CAP);
  ops_exit(); exit(1);
}

ops_printf("\neddies = %d\n", eddies);
//-------------------------------------------------------------

ops_decl_const("Delta0block0" , 1, "double", &Delta0block0);
ops_decl_const("Delta1block0" , 1, "double", &Delta1block0);
ops_decl_const("Delta2block0" , 1, "double", &Delta2block0);
ops_decl_const("HDF5_timing" , 1, "int", &HDF5_timing);
ops_decl_const("Lx1" , 1, "double", &Lx1);
ops_decl_const("Minf" , 1, "double", &Minf);
ops_decl_const("Pr" , 1, "double", &Pr);
ops_decl_const("Re" , 1, "double", &Re);
ops_decl_const("RefT" , 1, "double", &RefT);
ops_decl_const("SuthT" , 1, "double", &SuthT);
ops_decl_const("Twall" , 1, "double", &Twall);
ops_decl_const("block0np0" , 1, "int", &block0np0);
ops_decl_const("block0np1" , 1, "int", &block0np1);
ops_decl_const("block0np2" , 1, "int", &block0np2);
ops_decl_const("by" , 1, "double", &by);
ops_decl_const("dt" , 1, "double", &dt);
ops_decl_const("gama" , 1, "double", &gama);
ops_decl_const("inv2Delta0block0" , 1, "double", &inv2Delta0block0);
ops_decl_const("inv2Delta1block0" , 1, "double", &inv2Delta1block0);
ops_decl_const("inv2Delta2block0" , 1, "double", &inv2Delta2block0);
ops_decl_const("inv2Minf" , 1, "double", &inv2Minf);
ops_decl_const("invDelta0block0" , 1, "double", &invDelta0block0);
ops_decl_const("invDelta1block0" , 1, "double", &invDelta1block0);
ops_decl_const("invDelta2block0" , 1, "double", &invDelta2block0);
ops_decl_const("invLx1" , 1, "double", &invLx1);
ops_decl_const("invPr" , 1, "double", &invPr);
ops_decl_const("invRe" , 1, "double", &invRe);
ops_decl_const("invRefT" , 1, "double", &invRefT);
ops_decl_const("inv_gamma_m1" , 1, "double", &inv_gamma_m1);
ops_decl_const("invniter" , 1, "double", &invniter);
ops_decl_const("niter" , 1, "int", &niter);
ops_decl_const("simulation_time" , 1, "double", &simulation_time);
ops_decl_const("start_iter" , 1, "int", &start_iter);
ops_decl_const("write_output_file" , 1, "int", &write_output_file);
ops_decl_const("ny", 1, "int", &ny);
ops_decl_const("uinterp", ny, "double", uinterp);
ops_decl_const("yprofdata", 121, "double", yprofdata);
ops_decl_const("uprofdata", 121, "double", uprofdata);


// ---------------------- eddy global constants-------------------------------
ops_decl_const("y_cutoff", 1, "int", &y_cutoff);
ops_decl_const("ndata", 1, "int", &ndata);
// ops_decl_const("ydata", ndata, "double", ydata);
// ops_decl_const("uudata", ndata, "double", uudata);
// ops_decl_const("uvdata", ndata, "double", uvdata);
// ops_decl_const("vvdata", ndata, "double", vvdata);
// ops_decl_const("wwdata", ndata, "double", wwdata);
ops_decl_const("delta", 1, "double", &delta);
ops_decl_const("radius", 1, "double", &radius);
ops_decl_const("eddy_vol", 1, "double", &eddy_vol);
ops_decl_const("eddies", 1, "int", &eddies);
ops_decl_const("increment", 1, "double", &increment);
ops_decl_const("fluct_scale", 3, "double", fluct_scale);

ops_decl_const("eddy_x_min", 1, "double", &eddy_x_min);
ops_decl_const("eddy_x_max", 1, "double", &eddy_x_max);
ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);


//----------------------------------------------------------------------------


// Define and Declare OPS Block
ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
#include "defdec_data_set.h"
// Define and declare stencils
#include "stencils.h"
#include "bc_exchanges.h"

// ----------------------------- the eddy particles ---------------------------
// On the fluid block, not one of their own: OPS partitions the RANKS between
// blocks, so a second block would halve the solver's parallelism (README).
// d_coords is the grid packed into one dim-3 dat, which is what the bounding
// box and the mapping require.
ops_dat d_coords;
{
int halo_p[] = {1, 1, 1};
int halo_m[] = {-1, -1, -1};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
d_coords = ops_decl_dat(opensbliblock00, 3, size, base, halo_m, halo_p, value, "double", "d_coords");
}

int s3d_27[81];
{
int n = 0;
for (int k = -1; k <= 1; k++)
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) { s3d_27[n++] = i; s3d_27[n++] = j; s3d_27[n++] = k; }
}
ops_stencil S3D_27pt = ops_decl_stencil(3, 27, s3d_27, "eddy_27pt");

// h_eddy is the allgather that replaces the MPI_Allgatherv; h_uinterp and h_rst
// do the same for the two inlet profiles, which ../oSEM_3d also gets wrong.
ops_reduction h_eddy    = ops_decl_reduction_handle(nred * sizeof(double), "double", "eddy_all");
ops_reduction h_cnt     = ops_decl_reduction_handle(sizeof(int), "int", "eddy_count");
ops_reduction h_rhomin  = ops_decl_reduction_handle(sizeof(double), "double", "rho_min");
ops_reduction h_uinterp = ops_decl_reduction_handle(UINTERP_CAP * sizeof(double), "double", "uinterp_all");
ops_reduction h_rst     = ops_decl_reduction_handle(RST_CAP * sizeof(double), "double", "rst_all");

double dx_box[] = {0.0, 0.0, 0.0};
BoundingBox<Real> *box = ops_create_bounding_box(opensbliblock00, d_coords, 3, dx_box);
ops_particle eddy_parts = ops_decl_particle(opensbliblock00, "eddies", box);

int p_base[] = {0, 0, 0};
double *null_dbl = NULL;
int *null_int = NULL;

// p_pos is the ownership handle only; p_e carries the eddy's coordinates.
ops_dat p_pos = ops_decl_particle_pos_dat(eddy_parts, 3, p_base, null_dbl, "double", "position");
ops_dat p_e   = ops_decl_particle_dat(eddy_parts, 3, p_base, null_dbl, "double", "eddy_xyz");
ops_dat p_r   = ops_decl_particle_dat(eddy_parts, 1, p_base, null_dbl, "double", "radius");
ops_dat p_eps = ops_decl_particle_dat(eddy_parts, 3, p_base, null_dbl, "double", "eps");
ops_dat p_id  = ops_decl_particle_dat(eddy_parts, 1, p_base, null_int, "int", "eddy_id");
// Six independent uniforms per eddy per step, refilled before each kernel --
// ../oSEM_3d's fill-then-read structure, but keyed on the global eddy id so an
// eddy's stream belongs to the eddy and not to whichever rank holds it.
ops_dat p_rnd = ops_decl_particle_dat(eddy_parts, 6, p_base, null_dbl, "double", "rnd");

ops_particle_mapping map = ops_decl_mapping(eddy_parts, d_coords, S3D_27pt,
                                            OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

ops_dat dat_border[] = {p_pos, p_e, p_r, p_eps, p_id};
const int nborder = sizeof(dat_border)/sizeof(dat_border[0]);

// Not a loop bound: only ITERATE_RANDOM reads this, and every loop here is
// ITERATE_LOCAL, which takes its count from particle->no_particles.
double eddy_region[] = {eddy_x_min, eddy_x_max, eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};
// Init OPS partition
double partition_start0, elapsed_partition_start0, partition_end0, elapsed_partition_end0;
ops_timers(&partition_start0, &elapsed_partition_start0);
ops_partition("");
printf("Rank %d passed partition\n", ops_get_proc());
// exit(-1);
ops_timers(&partition_end0, &elapsed_partition_end0);
ops_printf("-----------------------------------------\n MPI partition and reading input file time: %lf\n -----------------------------------------\n", elapsed_partition_end0-elapsed_partition_start0);
// Restart procedure
ops_printf("\033[1;32m");
if (restart == 1){
ops_printf("OpenSBLI is restarting from the input file: restart.h5\n");
}
else {
ops_printf("OpenSBLI is starting from the initial condition.\n");
}
ops_printf("\033[0m");
// Constants from HDF5 restart file
if (restart == 1){
ops_get_const_hdf5("simulation_time", 1, "double", (char*)&simulation_time, "restart.h5");
ops_get_const_hdf5("iter", 1, "int", (char*)&start_iter, "restart.h5");
}
else {
simulation_time = 0.0;
start_iter = 0;
}
tstart = simulation_time;

if (restart == 0){
int iteration_range_36_block0[] = {-5, block0np0 + 5, -5, block0np1 + 5, -5, block0np2 + 5};
ops_par_loop(opensbliblock00Kernel036, "Grid_based_initialisation0", opensbliblock00, 3, iteration_range_36_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(x0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_idx());
}

int iteration_range_38_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel038, "MetricsEquation evaluation", opensbliblock00, 3, iteration_range_38_block0,
ops_arg_dat(x1_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(D11_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(detJ_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_39_block0[] = {0, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel039, "Metric_copy_block0 boundary dir0 side0", opensbliblock00, 3, iteration_range_39_block0,
ops_arg_dat(D11_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(detJ_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_40_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel040, "Metric_copy_block0 boundary dir0 side1", opensbliblock00, 3, iteration_range_40_block0,
ops_arg_dat(D11_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW),
ops_arg_dat(detJ_B0, 1, stencil_0_22_00_00_8, "double", OPS_RW));

int iteration_range_41_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel041, "Metric_copy_block0 boundary dir1 side0", opensbliblock00, 3, iteration_range_41_block0,
ops_arg_dat(D11_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(detJ_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

int iteration_range_42_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel042, "Metric_copy_block0 boundary dir1 side1", opensbliblock00, 3, iteration_range_42_block0,
ops_arg_dat(D11_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW),
ops_arg_dat(detJ_B0, 1, stencil_0_00_22_00_8, "double", OPS_RW));

ops_halo_transfer(periodicBC_direction2_side0_43_block0);
ops_halo_transfer(periodicBC_direction2_side1_44_block0);
int iteration_range_46_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel046, "MetricsEquation evaluation", opensbliblock00, 3, iteration_range_46_block0,
ops_arg_dat(D11_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(SD111_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

// velocity profile initialisation
int iteration_range_uinterp[] = {0, 1, 0, ny, 0, 1};
ops_par_loop(uinterp_kernel, "uinterp_kernel", opensbliblock00, 3, iteration_range_uinterp,
ops_arg_dat(d_uinterp, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_reduce(h_uinterp, 4096, "double", OPS_INC),   /* == UINTERP_CAP */
ops_arg_idx());

// ../oSEM_3d uses ops_dat_fetch_data here, which under MPI copies only this
// rank's slice and writes it at offset 0.
{
std::vector<double> buf(UINTERP_CAP, 0.0);
ops_reduction_result(h_uinterp, buf.data());
for (int j = 0; j < ny; j++) uinterp[j] = buf[j];
}
ops_update_const("uinterp", ny, "double", uinterp);

for(int i{0}; i < ny; i++){
  ops_printf("u: %f \n", uinterp[i]);
}

// -------------------------eddy initialisation-----------------------

// Pack the fluid coordinates for the bounding box, then partition the eddies.
{
int coord_range[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(KerPackCoords, "KerPackCoords", opensbliblock00, 3, coord_range,
ops_arg_dat(d_coords, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(x0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ));
}

ops_particle_setup_partition();
seed_eddies(eddy_parts, p_pos, p_e, p_id, eddies, seed_gbl);
ops_particle_setup_maps_with_dats(eddy_parts, dat_border, nborder);

// The position is already placed by seed_eddies; this writes the rest.
ops_fill_random_uniform_particle(eddy_parts, p_rnd, p_id, seed_gbl, 1u, OPS_PRNG_MINSTD);
if (opt_degen) degrade_rnd(eddy_parts, p_rnd);

ops_particle_par_loop(KerInitEddy, "instantiate_eddies", eddy_parts, 3,
OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
ops_arg_dat_particle(p_r, 1, "double", eddy_parts, map, OPS_WRITE),
ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_WRITE),
ops_arg_dat_particle(p_rnd, 6, "double", eddy_parts, map, OPS_READ));

// Is every eddy owned exactly once? Seeding is a half-open box test, so this
// is the check that the per-rank boxes really do tile the eddy box.
{
int count = 0;
ops_particle_par_loop(KerCountEddies, "count_eddies", eddy_parts, 3,
OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
ops_arg_dat_particle(p_id, 1, "int", eddy_parts, map, OPS_READ),
ops_arg_reduce(h_cnt, 1, "int", OPS_INC));
ops_reduction_result(h_cnt, &count);
ops_printf("eddy particles owned: %d of %d%s\n", count, eddies,
           count == eddies ? "" : "   <-- EDDIES LOST AT SEEDING");
}

int interp_iter_range[] = {0, 1, 0, y_cutoff, 0, 1};
ops_par_loop(interp_RST, "interp_RST", opensbliblock00, 3, interp_iter_range,
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(d_a11, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(d_a21, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(d_a22, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(d_a33, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_gbl(ydata, ndata, "double", OPS_READ),
ops_arg_gbl(uudata, ndata, "double", OPS_READ),
ops_arg_gbl(uvdata, ndata, "double", OPS_READ),
ops_arg_gbl(vvdata, ndata, "double", OPS_READ),
ops_arg_gbl(wwdata, ndata, "double", OPS_READ),
ops_arg_reduce(h_rst, 600, "double", OPS_INC),   /* == RST_CAP */
ops_arg_idx());

// The allgather, in place of ../oSEM_3d's four ops_dat_fetch_data calls.
{
std::vector<double> rst(RST_CAP, 0.0);
ops_reduction_result(h_rst, rst.data());
for (int j = 0; j < y_cutoff; j++) {
  a11[j] = rst[4*j+0]; a21[j] = rst[4*j+1];
  a22[j] = rst[4*j+2]; a33[j] = rst[4*j+3];
}
}

ops_decl_const("a11", y_cutoff, "double", a11);
ops_decl_const("a21", y_cutoff, "double", a21);
ops_decl_const("a22", y_cutoff, "double", a22);
ops_decl_const("a33", y_cutoff, "double", a33);

ops_printf("eddies: %i. Eddy volume: %f \n", eddies, eddy_vol);
ops_printf("xmax: %f, xmin: %f \n", eddy_x_max, eddy_x_min);
ops_printf("ymax: %f, ymin: %f %s\n", eddy_y_max, eddy_y_min,
           opt_padwall ? "(-pad-below-wall: ../oSEM_3d's box, extends under the wall)"
                       : "(no padding below the wall)");
ops_printf("zmax: %f, zmin: %f \n", eddy_z_max, eddy_z_min);
ops_printf("gather buffer: %d x %d doubles per step\n", eddies, NCOMP);
ops_printf("fluctuation scale (u,v,w): %.2f %.2f %.2f\n", fluct_scale[0], fluct_scale[1], fluct_scale[2]);
if (opt_degen)
  ops_printf("*** -degenerate-rng: emulating ../oSEM_3d's broken draw ***\n");

for (int i{0}; i < y_cutoff; i++){
  ops_printf("a11: %f, a21: %f, a22: %f, a33: %f \n", a11[i], a21[i], a22[i], a33[i]);
}


//--------------------------------------------------------------------

// Initialize loop timers
double cpu_start0, elapsed_start0, cpu_end0, elapsed_end0;
ops_timers(&cpu_start0, &elapsed_start0);
double inner_start, elapsed_inner_start;
double inner_end, elapsed_inner_end;
ops_timers(&inner_start, &elapsed_inner_start);
for(iter=start_iter; iter<=start_iter+niter - 1; iter++)
{
simulation_time = tstart + dt*((iter - start_iter)+1);
ops_update_const("simulation_time", 1, "double", &simulation_time);
if(fmod(iter+1, 1) == 0){
        ops_timers(&inner_end, &elapsed_inner_end);
        ops_printf("Iteration: %d. Time-step: %.3e. Simulation time: %.5f. Time/iteration: %lf.\n", iter+1, dt, simulation_time, (elapsed_inner_end - elapsed_inner_start)/1);
        ops_NaNcheck(rho_B0);
        {
        int rmin_range[] = {0, block0np0, 0, block0np1, 0, block0np2};
        double rmin = 0.0;
        ops_par_loop(KerRhoMin, "KerRhoMin", opensbliblock00, 3, rmin_range,
        ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
        ops_arg_reduce(h_rhomin, 1, "double", OPS_MIN));
        ops_reduction_result(h_rhomin, &rmin);
        ops_printf("   rho_min %.6f%s\n", rmin, rmin <= 0.0 ? "   <-- NON-PHYSICAL" : "");
        }
        ops_timers(&inner_start, &elapsed_inner_start);
}

// ------------------------ eddy convection -----------------------------------
// ../oSEM_3d's two lines per step -- refill the random dat, run convect_eddies
// -- then the gather. No migration: ownership is by eddy id, fixed at seeding.
ops_fill_random_uniform_particle(eddy_parts, p_rnd, p_id, seed_gbl,
                                 (unsigned int)(iter - start_iter) + 2u, OPS_PRNG_MINSTD);
if (opt_degen) degrade_rnd(eddy_parts, p_rnd);

ops_particle_par_loop(KerConvectEddies, "convect_eddies", eddy_parts, 3,
OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
ops_arg_dat_particle(p_e, 3, "double", eddy_parts, map, OPS_RW),
ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_RW),
ops_arg_dat_particle(p_rnd, 6, "double", eddy_parts, map, OPS_READ));

// The gather: OPS_INC by global id, so the MPI_Allreduce inside
// ops_reduction_result hands every rank the complete list in id order.
ops_particle_par_loop(KerPublishEddy, "publish_eddies", eddy_parts, 3,
OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
ops_arg_dat_particle(p_e, 3, "double", eddy_parts, map, OPS_READ),
ops_arg_dat_particle(p_r, 1, "double", eddy_parts, map, OPS_READ),
ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_READ),
ops_arg_dat_particle(p_id, 1, "int", eddy_parts, map, OPS_READ),
ops_arg_reduce(h_eddy, nred, "double", OPS_INC));
ops_reduction_result(h_eddy, eddy_all);

// A probe on the gather alone: sums over the whole list in id order, so they
// must not depend on rank count. Zero radius means a slot no rank wrote.
if (fmod(1 + iter, write_output_file) == 0 || iter == 0) {
  double sx = 0.0, sy = 0.0, sz = 0.0, seps = 0.0;
  int missing = 0;
  for (int i = 0; i < eddies; i++) {
    const double *e = eddy_all + NCOMP*i;
    sx += e[E_X]; sy += e[E_Y]; sz += e[E_Z];
    seps += e[E_SX] + e[E_SY] + e[E_SZ];
    if (e[E_R] <= 0.0) missing++;
  }
  ops_printf("eddy gather [iter %d]: sum x %.12e y %.12e z %.12e  eps %+.0f  missing %d\n",
             iter, sx, sy, sz, seps, missing);
}

// Eddies recycle; they are never created or destroyed.
{
int count = 0;
ops_particle_par_loop(KerCountEddies, "count_eddies", eddy_parts, 3,
OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
ops_arg_dat_particle(p_id, 1, "int", eddy_parts, map, OPS_READ),
ops_arg_reduce(h_cnt, 1, "int", OPS_INC));
ops_reduction_result(h_cnt, &count);
if (count != eddies)
  ops_printf("*** eddy population %d != %d at iter %d ***\n", count, eddies, iter);
}

//-----------------------------------------------------------------------------------

int iteration_range_30_block0[] = {-2, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel030, "Dirichlet boundary dir0 side0", opensbliblock00, 3, iteration_range_30_block0,
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_gbl(eddy_all, eddies * NCOMP, "double", OPS_READ),
ops_arg_idx());

int iteration_range_31_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel031, "Extrapolation boundary dir0 side1", opensbliblock00, 3, iteration_range_31_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW));

int iteration_range_32_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel032, "IsothermalWall boundary dir1 side0", opensbliblock00, 3, iteration_range_32_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_21_00_9, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW));

int iteration_range_33_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel033, "Dirichlet boundary dir1 side1", opensbliblock00, 3, iteration_range_33_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

ops_halo_transfer(periodicBC_direction2_side0_34_block0);
ops_halo_transfer(periodicBC_direction2_side1_35_block0);
for(stage=0; stage<=2; stage++)
{
int iteration_range_1_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel001, "CRu0_B0", opensbliblock00, 3, iteration_range_1_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_3_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel003, "CRu1_B0", opensbliblock00, 3, iteration_range_3_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_5_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel005, "CRu2_B0", opensbliblock00, 3, iteration_range_5_block0,
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_16_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel016, "CRp_B0", opensbliblock00, 3, iteration_range_16_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_7_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel007, "CRT_B0", opensbliblock00, 3, iteration_range_7_block0,
ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(T_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_22_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel022, "CRmu_B0", opensbliblock00, 3, iteration_range_22_block0,
ops_arg_dat(T_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(mu_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_0_block0[] = {0, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel000, "Derivative evaluation CD u0_B0 xi0 ", opensbliblock00, 3, iteration_range_0_block0,
ops_arg_dat(u0_B0, 1, stencil_0_44_00_00_19, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_2_block0[] = {0, block0np0, -2, block0np1 + 2, 0, block0np2};
ops_par_loop(opensbliblock00Kernel002, "Derivative evaluation CD u1_B0 xi0 ", opensbliblock00, 3, iteration_range_2_block0,
ops_arg_dat(u1_B0, 1, stencil_0_44_00_00_19, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_4_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel004, "Derivative evaluation CD u2_B0 xi0 ", opensbliblock00, 3, iteration_range_4_block0,
ops_arg_dat(u2_B0, 1, stencil_0_44_00_00_19, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_6_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel006, "Derivative evaluation CD T_B0 xi0 ", opensbliblock00, 3, iteration_range_6_block0,
ops_arg_dat(T_B0, 1, stencil_0_44_00_00_19, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_8_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel008, "Derivative evaluation CD u0_B0 xi1 ", opensbliblock00, 3, iteration_range_8_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_9_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel009, "Derivative evaluation CD u1_B0 xi1 ", opensbliblock00, 3, iteration_range_9_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_10_block0[] = {0, block0np0, 0, block0np1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel010, "Derivative evaluation CD u2_B0 xi1 ", opensbliblock00, 3, iteration_range_10_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_11_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel011, "Derivative evaluation CD T_B0 xi1 ", opensbliblock00, 3, iteration_range_11_block0,
ops_arg_dat(T_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_12_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel012, "Derivative evaluation CD u0_B0 xi2 ", opensbliblock00, 3, iteration_range_12_block0,
ops_arg_dat(u0_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_13_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel013, "Derivative evaluation CD u1_B0 xi2 ", opensbliblock00, 3, iteration_range_13_block0,
ops_arg_dat(u1_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_14_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel014, "Derivative evaluation CD u2_B0 xi2 ", opensbliblock00, 3, iteration_range_14_block0,
ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_15_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel015, "Derivative evaluation CD T_B0 xi2 ", opensbliblock00, 3, iteration_range_15_block0,
ops_arg_dat(T_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

int iteration_range_28_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel028, "Convective terms", opensbliblock00, 3, iteration_range_28_block0,
ops_arg_dat(D11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(p_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_44_00_00_19, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_idx());

int iteration_range_29_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel029, "Viscous terms", opensbliblock00, 3, iteration_range_29_block0,
ops_arg_dat(D11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(SD111_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(T_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(mu_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(u0_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(u1_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(u2_B0, 1, stencil_0_44_44_22_43, "double", OPS_READ),
ops_arg_dat(wk0_B0, 1, stencil_0_00_44_22_27, "double", OPS_READ),
ops_arg_dat(wk10_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk1_B0, 1, stencil_0_00_44_00_19, "double", OPS_READ),
ops_arg_dat(wk2_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk5_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk6_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(wk9_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_idx());

int iteration_range_49_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel049, "Temporal solution advancement", opensbliblock00, 3, iteration_range_49_block0,
ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_gbl(&rkA[stage], 1, "double", OPS_READ),
ops_arg_gbl(&rkB[stage], 1, "double", OPS_READ));

int iteration_range_30_block0[] = {-2, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel030, "Dirichlet boundary dir0 side0", opensbliblock00, 3, iteration_range_30_block0,
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_gbl(eddy_all, eddies * NCOMP, "double", OPS_READ),
ops_arg_idx());

int iteration_range_31_block0[] = {block0np0 - 1, block0np0, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel031, "Extrapolation boundary dir0 side1", opensbliblock00, 3, iteration_range_31_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_12_00_00_9, "double", OPS_RW));

int iteration_range_32_block0[] = {-2, block0np0 + 2, 0, 1, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel032, "IsothermalWall boundary dir1 side0", opensbliblock00, 3, iteration_range_32_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_21_00_9, "double", OPS_RW),
ops_arg_dat(rho_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_22_00_11, "double", OPS_RW));

int iteration_range_33_block0[] = {-2, block0np0 + 2, block0np1 - 1, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel033, "Dirichlet boundary dir1 side1", opensbliblock00, 3, iteration_range_33_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

ops_halo_transfer(periodicBC_direction2_side0_34_block0);
ops_halo_transfer(periodicBC_direction2_side1_35_block0);
}
if(iter > start_averaging){
int iteration_range_47_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel047, "user kernel InTheSimulation", opensbliblock00, 3, iteration_range_47_block0,
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhorhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

int iteration_range_new01_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel_new01, "opensbliblock00Kernel_new01", opensbliblock00, 3, iteration_range_new01_block0,
ops_arg_dat(mu_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_44_00_16, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_44_00_16, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(D11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(taux0x1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(l_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(mu_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(du0dx1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(utau_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_idx());
}

if (fmod(1 + iter,write_output_file) == 0 || iter == 0){
HDF5_IO_Write_0_opensbliblock00_dynamic(opensbliblock00, iter, rho_B0, rhou0_B0, rhou1_B0, rhou2_B0, rhoE_B0, x0_B0, x1_B0, x2_B0, D11_B0, T_B0, mu_B0, p_B0, HDF5_timing);
}

}
ops_timers(&cpu_end0, &elapsed_end0);
ops_printf("\nTimings are:\n");
ops_printf("-----------------------------------------\n");
ops_printf("Total Wall time %lf\n",elapsed_end0-elapsed_start0);

int iteration_range_48_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel048, "user kernel AfterSimulationEnds", opensbliblock00, 3, iteration_range_48_block0,
ops_arg_dat(rhoE_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhorhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(taux0x1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(l_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(du0dx1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(mu_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(utau_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

HDF5_IO_Write_0_opensbliblock00(opensbliblock00, rho_B0, rhou0_B0, rhou1_B0, rhou2_B0, rhoE_B0, x0_B0, x1_B0, x2_B0, D11_B0, T_B0, mu_B0, p_B0, HDF5_timing);
HDF5_IO_Write_1_opensbliblock00(opensbliblock00, rho_mean_B0, rhou0_mean_B0, rhou1_mean_B0, rhou2_mean_B0, rhoE_mean_B0, rhou0u0_mean_B0, rhou1u1_mean_B0, rhou2u2_mean_B0, rhou0u1_mean_B0, rhou1u2_mean_B0, rhou0u2_mean_B0, rhou0u0_mean_B0, taux0x1_mean_B0, l_mean_B0, du0dx1_mean_B0, mu_mean_B0, u0_mean_B0, u1_mean_B0, u2_mean_B0, u0u0_mean_B0, u1u1_mean_B0, u2u2_mean_B0, u0u1_mean_B0, utau_mean_B0, HDF5_timing);


ops_exit();
//Main program end 
}
