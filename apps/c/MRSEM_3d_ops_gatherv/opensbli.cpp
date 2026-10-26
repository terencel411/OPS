#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include "constants.h"
#include "Pir_data.h"
#include "TBL_data.h"
#define OPS_3D
#define OPS_API 2
#include "ops_seq.h"
#include "opensbliblock00_kernels.h"
#include "io.h"
int main(int argc, char **argv) 
{
// Initializing OPS 
ops_init(argc,argv,1);
// Set restart to 1 to restart the simulation from HDF5 file
restart = 0;
// User defined constant values
Lx1 = 100.0;
block0np0 = 750;
block0np1 = 250;
block0np2 = 150;
niter = 55000;
start_averaging = 25000;
double rkB[] = {(1.0/3.0), (15.0/16.0), (8.0/15.0)};
double rkA[] = {0, (-5.0/9.0), (-153.0/128.0)};
dt = 0.025;
write_output_file = 5000;

// grid, iterations and output interval from the command line; the physical domain is
// fixed at 375 x 100 x 40, so -ngrid changes the resolution, not the box
for (int a = 1; a < argc; a++) {
  if (!strcmp(argv[a], "-ngrid") && a + 3 < argc) {
    block0np0 = atoi(argv[a + 1]); block0np1 = atoi(argv[a + 2]); block0np2 = atoi(argv[a + 3]);
  }
  if (!strcmp(argv[a], "-niter") && a + 1 < argc) niter = atoi(argv[a + 1]);
  if (!strcmp(argv[a], "-nout") && a + 1 < argc) write_output_file = atoi(argv[a + 1]);
}
// the default 25000 is fixed, not a fraction of niter, so a short run would leave
// start_averaging past the end and invniter negative
if (start_averaging >= niter) start_averaging = niter / 2;

// derived from the grid, so these must follow the parser
Delta0block0 = 375.0/(block0np0-1);
Delta1block0 = 100.0/(block0np1-1);
Delta2block0 = 40.0/(block0np2);
HDF5_timing = 0;
gama = 1.4;
Re = 873.4;
Minf = 2.0;
Pr = 0.72;
r = pow(Pr, 0.3333333);
Twall = 1.0 + 0.5 * r * (gama-1) * Minf*Minf;
RefT = 202.17;
SuthT = 110.4;
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
invniter = 1.0/(niter - start_averaging);

// velocity profile
ny = (int)trunc(block0np1 * 0.6);
uinterp = (double*)malloc(ny * sizeof(double));


//------------------- eddy variables---------------------------
y_cutoff = 150;
ndata = 121;
regnum = 0;
a11 = (double*)malloc(y_cutoff * sizeof(double));
a21 = (double*)malloc(y_cutoff * sizeof(double));
a22 = (double*)malloc(y_cutoff * sizeof(double));
a33 = (double*)malloc(y_cutoff * sizeof(double));
delta = 11.6973525411;
nregions = 3;
radii = (double*)malloc(sizeof(double) * 3 * nregions);
double lplus = 0.05711318346685964;
radii[0] = 100.0 * lplus;
radii[1] = 20.0 * lplus;
radii[2] = 60.0 * lplus;
radii[3] = 60.0 * lplus;
radii[4] = 30.0 * lplus;
radii[5] = 30.0 * lplus;
radii[6] = 0.20 * delta;
radii[7] = 0.20 * delta;
radii[8] = 0.20 * delta;
eddy_vels = (double*)malloc(sizeof(double) * nregions);
eddy_vels[0] = 0.62;//1.0;//0.62;
eddy_vels[1] = 0.62;//1.0;//0.62;
eddy_vels[2] = 0.80;//1.0;//0.80;
eddy_y_min = (double*)malloc(sizeof(double) * nregions);
eddy_y_max = (double*)malloc(sizeof(double) * nregions);
eddy_y_min[0] = 5.0 * lplus;
eddy_y_max[0] = 60.0 * lplus;
eddy_y_min[1] = 60.0 * lplus;
eddy_y_max[1] = 0.6 * delta;
eddy_y_min[2] = 0.6 * delta;
eddy_y_max[2] = 1.0 * delta + radii[7];
eddy_z_min = (double*)malloc(sizeof(double) * nregions);
eddy_z_max = (double*)malloc(sizeof(double) * nregions);
for(int i{0}; i < nregions; i++){
  eddy_z_min[i] = 0.0 - radii[3*i+2];
  eddy_z_max[i] = 40.0 + radii[3*i+2];
}
eddy_vols = (double*)malloc(sizeof(double) * nregions);
eddies = (int*)malloc(sizeof(int) * nregions);
eddiesm3 = (int*)malloc(sizeof(int) * nregions);
for (int i{0}; i < nregions; i++){
  eddy_vols[i] = std::abs(2 * radii[3*i] * (eddy_y_max[i] - eddy_y_min[i]) * (eddy_z_max[i] - eddy_z_min[i]));
  eddies[i] = trunc(eddy_vols[i]/(radii[3*i]*radii[3*i+1]*radii[3*i+2]));
  eddiesm3[i] = 3 * eddies[i];
  radii[3*i] = radii[3*i] / eddy_vels[i];// convert to time scale instead of x length
}
eddy_pos1_gbl = (double*)malloc(sizeof(double) * eddiesm3[0]);
eddy_r1_gbl = (double*)malloc(sizeof(double) * eddiesm3[0]);
eddy_eps1_gbl = (int*)malloc(sizeof(int) * eddiesm3[0]);
eddy_pos2_gbl = (double*)malloc(sizeof(double) * eddiesm3[1]);
eddy_r2_gbl = (double*)malloc(sizeof(double) * eddiesm3[1]);
eddy_eps2_gbl = (int*)malloc(sizeof(int) * eddiesm3[1]);
eddy_pos3_gbl = (double*)malloc(sizeof(double) * eddiesm3[2]);
eddy_r3_gbl = (double*)malloc(sizeof(double) * eddiesm3[2]);
eddy_eps3_gbl = (int*)malloc(sizeof(double) * eddiesm3[2]);

a = 5;
c = 3;
m = pow(2, 29);
seed_gbl = 182383739;

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
//ops_decl_const("regnum", 1, "int", &regnum);
// literal 121, not ndata: a variable dim makes the translator declare these as
// `extern double *ydata` while TBL_data.h defines arrays -- see util.py:555-561
ops_decl_const("ydata", 121, "double", &ydata[0]);
ops_decl_const("uudata", 121, "double", &uudata[0]);
ops_decl_const("uvdata", 121, "double", &uvdata[0]);
ops_decl_const("vvdata", 121, "double", &vvdata[0]);
ops_decl_const("wwdata", 121, "double", &wwdata[0]);
ops_decl_const("delta", 1, "double", &delta);
ops_decl_const("radii", 3*nregions, "double", &radii[0]);
ops_decl_const("eddy_vols", nregions, "double", &eddy_vols[0]);
ops_decl_const("eddies", nregions, "int", &eddies[0]);
ops_decl_const("eddiesm3", nregions, "int", &eddiesm3[0]);
ops_decl_const("eddy_y_min", nregions, "double", &eddy_y_min[0]);
ops_decl_const("eddy_y_max", nregions, "double", &eddy_y_max[0]);
ops_decl_const("eddy_z_min", nregions, "double", &eddy_z_min[0]);
ops_decl_const("eddy_z_max", nregions, "double", &eddy_z_max[0]);

//----------------------------------------------------------------------------



// Define and Declare OPS Block
ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
#include "defdec_data_set.h"
// Define and declare stencils
#include "stencils.h"
#include "bc_exchanges.h"
// Init OPS partition
double partition_start0, elapsed_partition_start0, partition_end0, elapsed_partition_end0;
ops_timers(&partition_start0, &elapsed_partition_start0);
ops_partition("");
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

ops_dat_global_fetch_data(d_uinterp, 0, (char*)uinterp);
ops_update_const("uinterp", ny, "double", &uinterp[0]);

for(int i{0}; i < ny; i++){
  ops_printf("u: %f \n", uinterp[i]);
}

// -------------------------eddy initialisation-----------------------\

seed_gbl = (a*seed_gbl + c) % m;
ops_randomgen_init(seed_gbl, 0);
ops_fill_random_uniform(bulk_rng1);
ops_fill_random_uniform(bulk_rng2);
ops_fill_random_uniform(bulk_rng3);

int eddy_iter_range1[] = {0, eddies[0], 0, 1, 0, 1};
int eddy_iter_range2[] = {0, eddies[1], 0, 1, 0, 1};
int eddy_iter_range3[] = {0, eddies[2], 0, 1, 0, 1};

regnum = 0;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(instantiate_eddies, "instantiate_eddies", opensbliblock00, 3, eddy_iter_range1,
ops_arg_dat(eddy_pos1, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r1, 3, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(eddy_eps1, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng1, 6, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

regnum = 1;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(instantiate_eddies, "instantiate_eddies", opensbliblock00, 3, eddy_iter_range2,
ops_arg_dat(eddy_pos2, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r2, 3, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(eddy_eps2, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng2, 6, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

regnum = 2;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(instantiate_eddies, "instantiate_eddies", opensbliblock00, 3, eddy_iter_range3,
ops_arg_dat(eddy_pos3, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r3, 3, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(eddy_eps3, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng3, 6, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

ops_dat_global_fetch_data(eddy_pos1, 0, (char*)eddy_pos1_gbl);
ops_dat_global_fetch_data(eddy_r1, 0, (char*)eddy_r1_gbl);
ops_dat_global_fetch_data(eddy_eps1, 0, (char*)eddy_eps1_gbl);
ops_dat_global_fetch_data(eddy_pos2, 0, (char*)eddy_pos2_gbl);
ops_dat_global_fetch_data(eddy_r2, 0, (char*)eddy_r2_gbl);
ops_dat_global_fetch_data(eddy_eps2, 0, (char*)eddy_eps2_gbl);
ops_dat_global_fetch_data(eddy_pos3, 0, (char*)eddy_pos3_gbl);
ops_dat_global_fetch_data(eddy_r3, 0, (char*)eddy_r3_gbl);
ops_dat_global_fetch_data(eddy_eps3, 0, (char*)eddy_eps3_gbl);


int interp_iter_range[] = {0, 1, 0, y_cutoff, 0, 1};
ops_par_loop(interp_RST, "interp_RST", opensbliblock00, 3, interp_iter_range,
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(d_a11, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(d_a21, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(d_a22, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(d_a33, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

ops_dat_global_fetch_data(d_a11, 0, (char*)a11);
ops_dat_global_fetch_data(d_a21, 0, (char*)a21);
ops_dat_global_fetch_data(d_a22, 0, (char*)a22);
ops_dat_global_fetch_data(d_a33, 0, (char*)a33);

ops_decl_const("a11", y_cutoff, "double", &a11[0]);
ops_decl_const("a21", y_cutoff, "double", &a21[0]);
ops_decl_const("a22", y_cutoff, "double", &a22[0]);
ops_decl_const("a33", y_cutoff, "double", &a33[0]);

for (int i{0}; i < nregions; i++){
  ops_printf("region: %i \n", i);
  ops_printf("eddies: %i. Eddy volume: %f \n", eddies[i], eddy_vols[i]);
  ops_printf("ymax: %f, ymin: %f \n", eddy_y_max[i], eddy_y_min[i]);
  ops_printf("zmax: %f, zmin: %f \n", eddy_z_max[i], eddy_z_min[i]);
}

for (int i{0}; i < y_cutoff; i++){
  ops_printf("a11: %f, a21: %f, a22: %f, a33: %f \n", a11[i], a21[i], a22[i], a33[i]);
}


//--------------------------------------------------------------------




ops_reduction h_rhomin = ops_decl_reduction_handle(sizeof(double), "double", "rho_min");

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
if(fmod(iter+1, 100) == 0){
        ops_timers(&inner_end, &elapsed_inner_end);
        ops_printf("Iteration: %d. Time-step: %.3e. Simulation time: %.5f. Time/iteration: %lf.\n", iter+1, dt, simulation_time, (elapsed_inner_end - elapsed_inner_start)/100);
        ops_NaNcheck(rho_B0);

        {
        int rmin_range[] = {0, block0np0, 0, block0np1, 0, block0np2};
        double rmin = 0.0;
        ops_par_loop(KerRhoMin, "KerRhoMin", opensbliblock00, 3, rmin_range,
        ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
        ops_arg_reduce(h_rhomin, 1, "double", OPS_MIN));
        ops_reduction_result(h_rhomin, &rmin);
        ops_printf("   rho_min %.6f\n", rmin);
        }
        ops_timers(&inner_start, &elapsed_inner_start);
}

// ------------------------ eddy convection -----------------------------------------------

ops_fill_random_uniform(bulk_rng1);
ops_fill_random_uniform(bulk_rng2);
ops_fill_random_uniform(bulk_rng3);

regnum=0;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(convect_eddies, "convect_eddies", opensbliblock00, 3, eddy_iter_range1,
ops_arg_dat(eddy_pos1, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r1, 3, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(eddy_eps1, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng1, 3, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

regnum=1;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(convect_eddies, "convect_eddies", opensbliblock00, 3, eddy_iter_range2,
ops_arg_dat(eddy_pos2, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r2, 3, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(eddy_eps2, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng2, 3, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

regnum=2;
//ops_update_const("regnum", 1, "int", &regnum);
ops_par_loop(convect_eddies, "convect_eddies", opensbliblock00, 3, eddy_iter_range3,
ops_arg_dat(eddy_pos3, 3, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(eddy_r3, 3, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(eddy_eps3, 3, stencil_0_00_00_00_3, "int", OPS_WRITE),
ops_arg_dat(bulk_rng3, 3, stencil_0_00_00_00_3, "int", OPS_READ),
ops_arg_gbl(&regnum, 1, "int", OPS_READ));

ops_dat_global_fetch_data(eddy_pos1, 0, (char*)eddy_pos1_gbl);
ops_dat_global_fetch_data(eddy_eps1, 0, (char*)eddy_eps1_gbl);
ops_dat_global_fetch_data(eddy_pos2, 0, (char*)eddy_pos2_gbl);
ops_dat_global_fetch_data(eddy_eps2, 0, (char*)eddy_eps2_gbl);
ops_dat_global_fetch_data(eddy_pos3, 0, (char*)eddy_pos3_gbl);
ops_dat_global_fetch_data(eddy_eps3, 0, (char*)eddy_eps3_gbl);


//-----------------------------------------------------------------------------------



int iteration_range_30_block0[] = {-2, 1, -2, block0np1 + 2, -2, block0np2 + 2};
ops_par_loop(opensbliblock00Kernel030, "Dirichlet boundary dir0 side0", opensbliblock00, 3, iteration_range_30_block0,
ops_arg_dat(x1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_gbl(eddy_pos1_gbl, eddiesm3[0], "double", OPS_READ),
ops_arg_gbl(eddy_r1_gbl, eddiesm3[0], "double", OPS_READ),
ops_arg_gbl(eddy_eps1_gbl, eddiesm3[0], "int", OPS_READ),
ops_arg_gbl(eddy_pos2_gbl, eddiesm3[1], "double", OPS_READ),
ops_arg_gbl(eddy_r2_gbl, eddiesm3[1], "double", OPS_READ),
ops_arg_gbl(eddy_eps2_gbl, eddiesm3[1], "int", OPS_READ),
ops_arg_gbl(eddy_pos3_gbl, eddiesm3[2], "double", OPS_READ),
ops_arg_gbl(eddy_r3_gbl, eddiesm3[2], "double", OPS_READ),
ops_arg_gbl(eddy_eps3_gbl, eddiesm3[2], "int", OPS_READ),
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
ops_arg_dat(x2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),   
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_gbl(eddy_pos1_gbl, eddiesm3[0], "double", OPS_READ),
ops_arg_gbl(eddy_r1_gbl, eddiesm3[0], "double", OPS_READ),
ops_arg_gbl(eddy_eps1_gbl, eddiesm3[0], "int", OPS_READ),
ops_arg_gbl(eddy_pos2_gbl, eddiesm3[1], "double", OPS_READ),
ops_arg_gbl(eddy_r2_gbl, eddiesm3[1], "double", OPS_READ),
ops_arg_gbl(eddy_eps2_gbl, eddiesm3[1], "int", OPS_READ),
ops_arg_gbl(eddy_pos3_gbl, eddiesm3[2], "double", OPS_READ),
ops_arg_gbl(eddy_r3_gbl, eddiesm3[2], "double", OPS_READ),
ops_arg_gbl(eddy_eps3_gbl, eddiesm3[2], "int", OPS_READ),
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

if(iter >= start_averaging){
int iteration_range_47_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel047, "user kernel InTheSimulation", opensbliblock00, 3, iteration_range_47_block0,
ops_arg_dat(mu_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(mu_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhoE_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhohou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

int iteration_range_new01_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
ops_par_loop(opensbliblock00Kernel_new01, "opensbliblock00Kernel_new01", opensbliblock00, 3, iteration_range_new01_block0,
ops_arg_dat(mu_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(rho_B0, 1, stencil_0_00_44_00_16, "double", OPS_READ),
ops_arg_dat(rhou0_B0, 1, stencil_0_00_44_00_16, "double", OPS_READ),
ops_arg_dat(D11_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
ops_arg_dat(taux0x1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(l_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(du0dx1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
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
ops_arg_dat(mu_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhoE_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rho_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhohou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou0u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou1u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(rhou2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u0_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u0u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u1u1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(u2u2_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(taux0x1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(l_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(du0dx1_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
ops_arg_dat(utau_mean_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

HDF5_IO_Write_0_opensbliblock00(opensbliblock00, rho_B0, rhou0_B0, rhou1_B0, rhou2_B0, rhoE_B0, x0_B0, x1_B0, x2_B0, D11_B0, T_B0, mu_B0, p_B0, HDF5_timing);
HDF5_IO_Write_1_opensbliblock00(opensbliblock00, rho_mean_B0, rhou0_mean_B0, rhou1_mean_B0, rhou2_mean_B0, rhoE_mean_B0, rhou0u0_mean_B0, rhou1u1_mean_B0, rhou2u2_mean_B0, rhou0u1_mean_B0, rhou1u2_mean_B0, rhou0u2_mean_B0, rhohou0u0_mean_B0, mu_mean_B0, u0u0_mean_B0, u0u1_mean_B0, u1u1_mean_B0, u2u2_mean_B0, u0_mean_B0, u1_mean_B0, u2_mean_B0, taux0x1_mean_B0, l_mean_B0, du0dx1_mean_B0, utau_mean_B0, HDF5_timing);
ops_exit();
//Main program end 
}