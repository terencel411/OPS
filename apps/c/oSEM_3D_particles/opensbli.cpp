#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include "constants.h"
#include "Pir_data.h"
#include "TBL_data.h"
#define OPS_3D
#define OPS_API 2
#include "ops_seq_v2.h"
#include <ops_particle_seq.h>
#include <ops_grid_part_seq_v2.h>
#ifdef OPS_MPI
#include <mpi.h>
#endif
#include "scatter_loop.h"
#include "opensbliblock00_kernels.h"
#include "io.h"
#include "sem_particles.h"

// -----------------------------------------------------------------------------

int main(int argc, char **argv) 
{
// Initializing OPS 
ops_init(argc,argv,1);
// Set restart to 1 to restart the simulation from HDF5 file
restart = 0;
// User defined constant values
Lx1 = 100.0;
/* Development sizes. Production is 750 x 250 x 150, which needs ~18 GB.
   Override with -np0/-np1/-np2. All mapping arithmetic below is derived, so it
   recomputes automatically. */
block0np0 = 150;
block0np1 = 100;
block0np2 = 60;
for (int ia = 1; ia < argc; ia++) {
  if (strcmp(argv[ia], "-np0") == 0 && ia+1 < argc) block0np0 = atoi(argv[++ia]);
  else if (strcmp(argv[ia], "-np1") == 0 && ia+1 < argc) block0np1 = atoi(argv[++ia]);
  else if (strcmp(argv[ia], "-np2") == 0 && ia+1 < argc) block0np2 = atoi(argv[++ia]);
}
Delta0block0 = 375.0/(block0np0-1);
Delta1block0 = 100.0/(block0np1-1);
Delta2block0 = 40.0/(block0np2);
niter = 55000;
niter = 100;
double rkB[] = {(1.0/3.0), (15.0/16.0), (8.0/15.0)};
double rkA[] = {0, (-5.0/9.0), (-153.0/128.0)};
dt = 0.025;
write_output_file = 5000;
write_output_file = 10;
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
// start_averaging = 25000;
start_averaging = 50;
invniter = 1.0/(niter - start_averaging);

ny = (int)trunc(block0np1 * 0.6);
uinterp = (double*)malloc(ny * sizeof(double));

//------------------- eddy variables---------------------------
y_cutoff = 150;
ndata = 121;
a11 = (double*)malloc(y_cutoff * sizeof(double));
a21 = (double*)malloc(y_cutoff * sizeof(double));
a22 = (double*)malloc(y_cutoff * sizeof(double));
a33 = (double*)malloc(y_cutoff * sizeof(double));
delta = 11.6973525411;
radius = 0.2 * delta;
eddy_x_min = -radius;
eddy_x_max = radius;
eddy_y_min = 0.0 - radius;
eddy_y_max = delta + radius;
eddy_z_min = -radius;
eddy_z_max = 40.0 + radius;
eddy_vol = std::abs((eddy_x_max - eddy_x_min) * (eddy_y_max - eddy_y_min) * (eddy_z_max - eddy_z_min));
eddies = trunc(eddy_vol/(radius*radius*radius));
// eddies = 380;
eddiesm2 = 2 * eddies;
eddy_x_gbl = (double*)malloc(eddiesm2 * sizeof(double));
eddy_y_gbl = (double*)malloc(eddiesm2 * sizeof(double));
eddy_z_gbl = (double*)malloc(eddiesm2 * sizeof(double));
eddy_r_gbl = (double*)malloc(eddiesm2 * sizeof(double));
eddy_increment_gbl = (double*)malloc(eddiesm2 * sizeof(double));
eddy_eps_x_gbl = (int*)malloc(eddiesm2 * sizeof(int));
eddy_eps_y_gbl = (int*)malloc(eddiesm2 * sizeof(int));
eddy_eps_z_gbl = (int*)malloc(eddiesm2 * sizeof(int));
a = 5;
c = 3;
m = pow(2, 29);
seed_gbl = 182383739;

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
ops_decl_const("uinterp", ny, "double", &uinterp[0]);
ops_decl_const("yprofdata", 121, "double", &yprofdata[0]);
ops_decl_const("uprofdata", 121, "double", &uprofdata[0]);


// ---------------------- eddy global constants-------------------------------
ops_decl_const("y_cutoff", 1, "int", &y_cutoff);
ops_decl_const("ndata", 1, "int", &ndata);
// ops_decl_const("ydata", ndata, "double", &ydata[0]);
// ops_decl_const("uudata", ndata, "double", &uudata[0]);
// ops_decl_const("uvdata", ndata, "double", &uvdata[0]);
// ops_decl_const("vvdata", ndata, "double", &vvdata[0]);
// ops_decl_const("wwdata", ndata, "double", &wwdata[0]);
ops_decl_const("delta", 1, "double", &delta);
ops_decl_const("radius", 1, "double", &radius);
ops_decl_const("eddy_vol", 1, "double", &eddy_vol);
ops_decl_const("eddies", 1, "int", &eddies);
ops_decl_const("eddiesm2", 1, "int", &eddiesm2);

ops_decl_const("eddy_x_min", 1, "double", &eddy_x_min);
ops_decl_const("eddy_x_max", 1, "double", &eddy_x_max);
ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);


//----------------------------------------------------------------------------


// Define and Declare OPS Block
ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
// ops_block eddy_block = ops_decl_block(3, "eddy_block");
#include "defdec_data_set.h"
// Define and declare stencils
#include "stencils.h"
#include "bc_exchanges.h"
// Init OPS partition
double partition_start0, elapsed_partition_start0, partition_end0, elapsed_partition_end0;
ops_timers(&partition_start0, &elapsed_partition_start0);

/* ================= OPS Particles: eddies ================= */
double sem_x_fixed = 0.0;                       /* eddies sit on a fixed x plane */
double sem_increment = 0.0;                     /* set with the flow parameters   */

int sem_reach[3], sem_ngrid[3] = {block0np0, block0np1, block0np2};
sem_reach[0] = 4;                                                  /* x: not a search dim */
sem_reach[1] = (int)ceil(radius / Delta1block0);
sem_reach[2] = (int)ceil(radius / Delta2block0);
sem_map_geom sem_g = sem_make_geom(sem_reach, sem_ngrid);

ops_printf("SEM particles: %d eddies, reach %d x %d x %d cells\n",
           eddies, sem_reach[0], sem_reach[1], sem_reach[2]);
ops_printf("  bins %d x %d x %d (stride %d x %d %d), coords dat %d x %d x %d\n",
           sem_g.nbin[0], sem_g.nbin[1], sem_g.nbin[2],
           sem_g.stride[0], sem_g.stride[1], sem_g.stride[2],
           sem_g.nodes[0], sem_g.nodes[1], sem_g.nodes[2]);

ops_dat sem_coords;
{
  int sz[] = {sem_g.nodes[0], sem_g.nodes[1], sem_g.nodes[2]};
  int bs[] = {0, 0, 0};
  int hm[] = {0, 0, 0}, hp[] = {0, 0, 0};
  double *v = NULL;
  sem_coords = ops_decl_dat(opensbliblock00, 3, sz, bs, hm, hp, v,
                            "double", "sem_coords");
}

double sem_dx_box[] = {0.0, 0.0, 0.0};
BoundingBox<double> *sem_box =
    ops_create_bounding_box(opensbliblock00, sem_coords, 3, sem_dx_box);

ops_particle sem_parts = ops_decl_particle(opensbliblock00, "eddies", sem_box);

int sem_base[] = {0, 0, 0};
double *sem_nd = NULL; int *sem_ni = NULL;
ops_dat sem_pos  = ops_decl_particle_pos_dat(sem_parts, 3, sem_base, sem_nd, "double", "eddy_pos");
ops_dat sem_px   = ops_decl_particle_dat(sem_parts, 1, sem_base, sem_nd, "double", "eddy_px");
ops_dat sem_pr   = ops_decl_particle_dat(sem_parts, 1, sem_base, sem_nd, "double", "eddy_pr");
ops_dat sem_pinc = ops_decl_particle_dat(sem_parts, 1, sem_base, sem_nd, "double", "eddy_pinc");
ops_dat sem_peps = ops_decl_particle_dat(sem_parts, 3, sem_base, sem_ni, "int", "eddy_peps");
ops_dat sem_pseed= ops_decl_particle_dat(sem_parts, 1, sem_base, sem_ni, "int", "eddy_pseed");

int sem_nsten = 0;
int *sem_sten_arr = sem_box_stencil(1, sem_nsten);   /* 3x3x3 BINS */
int sem_stride[] = {sem_g.stride[0], sem_g.stride[1], sem_g.stride[2]};
ops_stencil SEM_STEN = ops_decl_prolong_stencil(3, sem_nsten, sem_sten_arr,
                                                sem_stride, "eddy_reach");

ops_particle_mapping sem_map =
    ops_decl_mapping(sem_parts, sem_coords, SEM_STEN, sem_stride,
                     OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

ops_dat sem_border[]  = {sem_pos, sem_px, sem_pr, sem_pinc, sem_peps, sem_pseed};
ops_dat sem_forward[] = {sem_pos, sem_px, sem_pr, sem_peps};
const int sem_nborder  = sizeof(sem_border) / sizeof(sem_border[0]);
const int sem_nforward = sizeof(sem_forward) / sizeof(sem_forward[0]);
/* ========================================================= */

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
ops_arg_idx());

ops_dat_fetch_data(d_uinterp, 0, (char*)uinterp);

/* ---- SEM particles: fill coords, derive the box, seed ---- */
{
  double sem_d[3] = {Delta0block0, Delta1block0, Delta2block0};
  int sem_rng[] = {0, sem_g.nodes[0], 0, sem_g.nodes[1], 0, sem_g.nodes[2]};
  ops_par_loop(SemInitCoords, "SemInitCoords", opensbliblock00, 3, sem_rng,
               ops_arg_dat(sem_coords, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
               ops_arg_gbl(sem_d, 3, "double", OPS_READ),
               ops_arg_idx());

  /* Only now do real coordinates exist, so only now can the box be derived. */
  ops_particle_setup_partition();

  sem_increment = 1.0 * Delta0block0;   /* placeholder: matched to convect later */

  /* CLAMPED to the grid's coordinate range. The eddy box nominally extends one
     radius outside the domain in y and z, but the bounding box cannot start
     below the grid origin, so those centres are dropped -- see PORT_NOTES.md. */
  double sem_lo[3], sem_hi[3];
  sem_lo[0] = eddy_x_min;                       sem_hi[0] = eddy_x_max;
  sem_lo[1] = fmax(eddy_y_min, 0.0);            sem_hi[1] = eddy_y_max;
  sem_lo[2] = fmax(eddy_z_min, 0.0);            sem_hi[2] = eddy_z_max;

  sem_seed_eddies(sem_parts, sem_pos, sem_px, sem_pr, sem_pinc, sem_peps,
                  sem_pseed, eddies, (unsigned int)seed_gbl,
                  sem_x_fixed, sem_increment, radius, sem_lo, sem_hi);

  ops_particle_setup_maps_with_dats(sem_parts, sem_border, sem_nborder);

  ops_printf("SEM STAGE A: seeded %d of %d eddies (clamped y>=%.3f z>=%.3f)\n",
             sem_total_particles(sem_parts), eddies, sem_lo[1], sem_lo[2]);
}

ops_update_const("uinterp", ny, "double", &uinterp[0]);

for(int i{0}; i < ny; i++){
  ops_printf("u: %f \n", uinterp[i]);
}

// -------------------------eddy initialisation-----------------------\

seed_gbl = (a*seed_gbl + c) % m;
ops_randomgen_init(seed_gbl, 0);
ops_fill_random_uniform(eddy_x_rng);
seed_gbl = (a*seed_gbl + c) % m;
//ops_randomgen_init(seed_gbl, 0);
ops_fill_random_uniform(eddy_bulk_rng);
int eddy_iter_range[] = {0, eddies, 0, 1, 0, 1};
ops_par_loop(instantiate_eddies, "instantiate_eddies", opensbliblock00, 3, eddy_iter_range,
ops_arg_dat(eddy_x, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_y, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_z, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_increment, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_eps_x, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_eps_y, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_eps_z, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_x_rng, 1, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_dat(eddy_bulk_rng, 5, stencil_0_00_00_00_3, "int", OPS_READ));

ops_dat_fetch_data(eddy_x, 0, (char*)eddy_x_gbl);
ops_dat_fetch_data(eddy_y, 0, (char*)eddy_y_gbl);
ops_dat_fetch_data(eddy_z, 0, (char*)eddy_z_gbl);
ops_dat_fetch_data(eddy_r, 0, (char*)eddy_r_gbl);
ops_dat_fetch_data(eddy_eps_x, 0, (char*)eddy_eps_x_gbl);
ops_dat_fetch_data(eddy_eps_y, 0, (char*)eddy_eps_y_gbl);
ops_dat_fetch_data(eddy_eps_z, 0, (char*)eddy_eps_z_gbl);

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
ops_arg_gbl(wwdata, ndata, "double", OPS_READ));

// a11, a12, a22, a33 not used in the code after this
ops_dat_fetch_data(d_a11, 0, (char*)a11);
ops_dat_fetch_data(d_a21, 0, (char*)a21);
ops_dat_fetch_data(d_a22, 0, (char*)a22);
ops_dat_fetch_data(d_a33, 0, (char*)a33);

ops_decl_const("a11", ndata, "double", &a11[0]);
ops_decl_const("a21", ndata, "double", &a21[0]);
ops_decl_const("a22", ndata, "double", &a22[0]);
ops_decl_const("a33", ndata, "double", &a33[0]);

ops_printf("eddies: %i. Eddy volume: %f \n", eddies, eddy_vol);
ops_printf("xmax: %f, xmin: %f \n", eddy_x_max, eddy_x_min);
ops_printf("ymax: %f, ymin: %f \n", eddy_y_max, eddy_y_min);
ops_printf("zmax: %f, zmin: %f \n", eddy_z_max, eddy_z_min);

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
        ops_timers(&inner_start, &elapsed_inner_start);
}

// ------------------------ eddy convection -----------------------------------------------

seed_gbl = (a*seed_gbl + c) % m;
//ops_randomgen_init(seed_gbl, 0);
ops_fill_random_uniform(eddy_bulk_rng);
// int eddy_iter_range[] = {0, eddies, 0, 1, 0, 1};
ops_par_loop(convect_eddies, "convect_eddies", opensbliblock00, 3, eddy_iter_range,
ops_arg_dat(eddy_x, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(eddy_y, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_z, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_increment, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(eddy_eps_x, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_eps_y, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_eps_z, 1, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(eddy_bulk_rng, 5, stencil_0_00_00_00_3, "int", OPS_READ));

ops_dat_fetch_data(eddy_x, 0, (char*) eddy_x_gbl);
ops_dat_fetch_data(eddy_y, 0, (char*) eddy_y_gbl);
ops_dat_fetch_data(eddy_z, 0, (char*) eddy_z_gbl);
ops_dat_fetch_data(eddy_r, 0, (char*) eddy_r_gbl);
ops_dat_fetch_data(eddy_eps_x, 0, (char*) eddy_eps_x_gbl);
ops_dat_fetch_data(eddy_eps_y, 0, (char*) eddy_eps_y_gbl);
ops_dat_fetch_data(eddy_eps_z, 0, (char*) eddy_eps_z_gbl);

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
ops_arg_gbl(eddy_x_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_y_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_z_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_r_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_eps_x_gbl, eddies, "int", OPS_READ),
ops_arg_gbl(eddy_eps_y_gbl, eddies, "int", OPS_READ),
ops_arg_gbl(eddy_eps_z_gbl, eddies, "int", OPS_READ),
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
ops_arg_gbl(eddy_x_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_y_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_z_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_r_gbl, eddies, "double", OPS_READ),
ops_arg_gbl(eddy_eps_x_gbl, eddies, "int", OPS_READ),
ops_arg_gbl(eddy_eps_y_gbl, eddies, "int", OPS_READ),
ops_arg_gbl(eddy_eps_z_gbl, eddies, "int", OPS_READ),
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
