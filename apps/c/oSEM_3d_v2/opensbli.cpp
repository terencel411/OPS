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
#include "eddy_gather.h"

static inline double host_x1_of_j(int j){
  return Lx1 * sinh(by * invLx1 * Delta1block0 * (double)j) / sinh(by);
}

// a*seed_gbl overflows int whenever seed_gbl > (INT_MAX-c)/a = 429496728, which
// is ~20% of the m = 2^29 modulus range.  In int arithmetic that made half of
// all draws negative, so the uniforms spanned (-1,1) and the eddies landed up to
// twice outside the box.  Do the multiply in 64 bits.
static inline double host_rng_uniform(){
  seed_gbl = (int)(((long long)a*seed_gbl + c) % m);
  return ((double)seed_gbl) / ((double)m);
}
// The low bit of this LCG is degenerate: 5*s+3 == s+1 (mod 2), so seed_gbl%2
// strictly alternates.  With six draws per eddy that gave every eddy the
// identical sign triple.  Take a high bit, where the generator actually mixes.
static inline int host_rng_sign(){
  seed_gbl = (int)(((long long)a*seed_gbl + c) % m);
  return (seed_gbl >= (m >> 1)) ? 1 : -1;
}

static void host_instantiate_eddies(){
  for (int i = 0; i < eddies; i++){
    eddy_x_gbl[i] = eddy_x_min + host_rng_uniform() * (eddy_x_max - eddy_x_min);
    eddy_y_gbl[i] = eddy_y_min + host_rng_uniform() * (eddy_y_max - eddy_y_min);
    eddy_z_gbl[i] = eddy_z_min + host_rng_uniform() * (eddy_z_max - eddy_z_min);
    eddy_eps_x_gbl[i] = host_rng_sign();
    eddy_eps_y_gbl[i] = host_rng_sign();
    eddy_eps_z_gbl[i] = host_rng_sign();
    eddy_r_gbl[i] = radius;
    eddy_increment_gbl[i] = 1.0 * dt;
  }
}

static void host_convect_eddies(){
  for (int i = 0; i < eddies; i++){
    eddy_x_gbl[i] = eddy_x_gbl[i] + eddy_increment_gbl[i];
    if (eddy_x_gbl[i] > eddy_x_max){
      eddy_x_gbl[i] = eddy_x_min;
      eddy_y_gbl[i] = eddy_y_min + host_rng_uniform() * (eddy_y_max - eddy_y_min);
      eddy_z_gbl[i] = eddy_z_min + host_rng_uniform() * (eddy_z_max - eddy_z_min);
      eddy_eps_x_gbl[i] = host_rng_sign();
      eddy_eps_y_gbl[i] = host_rng_sign();
      eddy_eps_z_gbl[i] = host_rng_sign();
    }
  }
}
// -----------------------------------------------------------------------------

int main(int argc, char **argv) 
{
// Initializing OPS 
ops_init(argc,argv,1);
// Set restart to 1 to restart the simulation from HDF5 file
restart = 0;
// User defined constant values
Lx1 = 100.0;
block0np0 = 750;//1100;
block0np1 = 250;
block0np2 = 100;
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
ops_update_const("uinterp", ny, "double", &uinterp[0]);

for(int i{0}; i < ny; i++){
  ops_printf("u: %f \n", uinterp[i]);
}

// -------------------------eddy initialisation-----------------------\

int eddy_iter_range[] = {0, eddies, 0, 1, 0, 1};
eddy_check_init(eddies);

if (eddy_mode == EDDY_MODE_HOST_BCAST) {
  // Reference path: rank 0 fills all `eddies` entries with the plain host loop,
  // then hands the arrays to everyone else.  Nothing is decomposed, so there is
  // no slab to reassemble and no question about which rank holds what.
  if (eddy_is_root()) host_instantiate_eddies();
  eddy_bcast(eddy_x_gbl, eddy_y_gbl, eddy_z_gbl, eddy_r_gbl, eddy_increment_gbl,
             eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl, eddies);
} else {
  // The eddy dats are decomposed in x and are only written on the ranks owning
  // the (y=0, z=0) pencil, so the host-side eddy_*_gbl arrays that Kernel030
  // reads as ops_arg_gbl have to be rebuilt in full on every rank.  rho_B0 is
  // passed only as a full-size reference dat for the block decomposition.
  eddy_gather_init(eddy_x, rho_B0, eddies);

  seed_gbl = (int)(((long long)a*seed_gbl + c) % m);  // 64-bit: a*seed_gbl overflows int above 429496728
  ops_randomgen_init(seed_gbl, 0);
  ops_fill_random_uniform(eddy_x_rng);
  seed_gbl = (int)(((long long)a*seed_gbl + c) % m);  // 64-bit: a*seed_gbl overflows int above 429496728
  //ops_randomgen_init(seed_gbl, 0);
  ops_fill_random_uniform(eddy_bulk_rng);
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

  eddy_gather(eddy_x, eddy_y, eddy_z, eddy_r, eddy_increment,
              eddy_eps_x, eddy_eps_y, eddy_eps_z,
              eddy_x_gbl, eddy_y_gbl, eddy_z_gbl, eddy_r_gbl, eddy_increment_gbl,
              eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl);
}

eddy_gather_check("instantiate_eddies", eddy_x_gbl, eddy_y_gbl, eddy_z_gbl,
                  eddy_r_gbl, eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl);
eddy_report("instantiate_eddies", eddies, eddy_x_gbl, eddy_y_gbl, eddy_z_gbl,
            eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl,
            eddy_x_min, eddy_x_max, eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max);

char fname[64];
sprintf(fname, "convect_eddies_rank%d.txt", ops_get_proc());
FILE* eddy_f = fopen(fname, "w");
fprintf(eddy_f, "rank %d  eddies = %d\n\n", ops_get_proc(), eddies);
fprintf(eddy_f, "%4s  %14s %14s %14s %14s %14s %6s %6s %6s\n",
        "iter", "x", "y", "z", "r", "incr", "epx", "epy", "epz");

ops_printf("instantiate eddies [rank %d]: %g %g %g %g %g %g %g %g %g %g\n",
  ops_get_proc(), eddy_x_gbl[0], eddy_x_gbl[5], eddy_x_gbl[10], eddy_x_gbl[15], eddy_x_gbl[20],
  eddy_x_gbl[eddies-1], eddy_x_gbl[eddies-1-5], eddy_x_gbl[eddies-1-10], eddy_x_gbl[eddies-1-15], eddy_x_gbl[eddies-1-20]);

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


//-----------------------------------------------------------------------------------

if (eddy_mode == EDDY_MODE_HOST_BCAST) {
  // Reference path: rank 0 advances its own copy of the whole eddy field, then
  // broadcasts it.  Every rank leaves this block holding the same `eddies`
  // entries, which is what Kernel030's ops_arg_gbl reads need.
  if (eddy_is_root()) host_convect_eddies();
  eddy_bcast(eddy_x_gbl, eddy_y_gbl, eddy_z_gbl, eddy_r_gbl, eddy_increment_gbl,
             eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl, eddies);
} else {
  seed_gbl = (int)(((long long)a*seed_gbl + c) % m);  // 64-bit: a*seed_gbl overflows int above 429496728
  //ops_randomgen_init(seed_gbl, 0);
  ops_fill_random_uniform(eddy_bulk_rng);
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

  // Rebuild the full eddy arrays on every rank before Kernel030 consumes them as
  // ops_arg_gbl.  Must happen after every convect_eddies: the kernel only ran on
  // the ranks owning the (y=0, z=0) pencil, and each of those holds one x-slab.
  eddy_gather(eddy_x, eddy_y, eddy_z, eddy_r, eddy_increment,
              eddy_eps_x, eddy_eps_y, eddy_eps_z,
              eddy_x_gbl, eddy_y_gbl, eddy_z_gbl, eddy_r_gbl, eddy_increment_gbl,
              eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl);
}

// Set eddy_gather_verify = 1 (eddy_gather.h) to assert every iteration that all
// ranks hold bit-identical arrays.  Collective, so it stays outside any rank test.
if (eddy_gather_verify)
  eddy_gather_check("convect_eddies", eddy_x_gbl, eddy_y_gbl, eddy_z_gbl,
                    eddy_r_gbl, eddy_eps_x_gbl, eddy_eps_y_gbl, eddy_eps_z_gbl);


fprintf(eddy_f, "--- (iter %d) first 5 ---\n", iter);
for (int j = 0; j < 5; j++) {
    fprintf(eddy_f, "%4d  %14.6e %14.6e %14.6e %14.6e %14.6e %6d %6d %6d\n",
            j, eddy_x_gbl[j], eddy_y_gbl[j], eddy_z_gbl[j],
            eddy_r_gbl[j], eddy_increment_gbl[j],
            eddy_eps_x_gbl[j], eddy_eps_y_gbl[j], eddy_eps_z_gbl[j]);
}

fprintf(eddy_f, "\n--- last 5 ---\n");
for (int j = eddies - 5; j < eddies; j++) {
    fprintf(eddy_f, "%4d  %14.6e %14.6e %14.6e %14.6e %14.6e %6d %6d %6d\n",
            j, eddy_x_gbl[j], eddy_y_gbl[j], eddy_z_gbl[j],
            eddy_r_gbl[j], eddy_increment_gbl[j],
            eddy_eps_x_gbl[j], eddy_eps_y_gbl[j], eddy_eps_z_gbl[j]);
}

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

fclose(eddy_f);
eddy_gather_free();

ops_exit();
//Main program end 
}
