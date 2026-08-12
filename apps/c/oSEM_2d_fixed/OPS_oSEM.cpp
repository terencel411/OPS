#include <cstdlib>
#include <cmath>
#include <random>
#include <map>
#include <stdio.h>
#include <fstream>
#include <string>

std::mt19937 urng;
std::uniform_real_distribution<double> y_rand;
std::uniform_real_distribution<double> z_rand;
std::uniform_int_distribution<int> eps_rand;
std::map<int, int> eps_map;

#define OPS_2D
#include "ops_seq.h"
#include "ops_lib_core.h"
#include "TBL_data.h"
#include "OPS_oSEM_constants.h"
#include "OPS_oSEM_eddy_functions.h"
#include "OPS_oSEM_kernels.h"
#include "OPS_oSEM_gather.h"
#include "OPS_oSEM_random.h"
#include "io.h"

int main(int argc, char** argv){
    ops_init(argc, argv, 1);


    // ------------------------------- Declare constant variable values ------------------------------------------
    u0 = 823.6;
    dt = 0.00000002;
    delta = 0.007;
    r_max = 0.41 * delta;
    ny = 100;
    nz = 150;
    niter = 2000;
    write_output_file = 200;
    TI = 0.01;
    x_min = -r_max;
    x_max = r_max;
    x_plane = 0.0;
    y_min = 0.0;
    y_max = 0.009; //1.5 * delta + r_max;
    eddy_y_min = y_min;
    eddy_y_max = y_max + r_max;
    z_min = 0.0; // - r_max;
    z_max = 0.05; // + r_max;
    eddy_z_min = z_min - r_max;
    eddy_z_max = z_max + r_max;
    vol = std::abs((x_max - x_min) * (y_max - y_min + 2 * r_max) * (z_max - z_min + 2 * r_max));
    rep_radius = 0.2 * delta; 
    calc_eddies(eddies, vol, rep_radius);

    printf("eddies: %i", eddies);

    // LCG values (change the seed init using LCG)
    a = 5;
    c = 3;
    m = pow(2, 29);
    seed_gbl = 2893328493; // reminder that this is declared or updated as an OPS constant
    
    // ------------------------------------ Allocate memory for eddy variables -----------------------------
    x_gbl = (double*)malloc(sizeof(double) * eddies);
    y_gbl = (double*)malloc(sizeof(double) * eddies);
    z_gbl = (double*)malloc(sizeof(double) * eddies);
    r_gbl = (double*)malloc(sizeof(double) * eddies);
    increment_gbl = (double*)malloc(sizeof(double) * eddies);
    eps_x_gbl = (int*)malloc(sizeof(int) * eddies);
    eps_y_gbl = (int*)malloc(sizeof(int) * eddies);
    eps_z_gbl = (int*)malloc(sizeof(int) * eddies);
    
    yinterp = (double*)(y_inp);
    r11interp = (double*)(uu_inp);
    r21interp = (double*)(uv_inp);
    r22interp = (double*)(vv_inp);
    r33interp = (double*)(ww_inp);

    //y_rand = std::uniform_real_distribution<double>(y_min - r_max, y_max + r_max); // may have issues
    //z_rand = std::uniform_real_distribution<double>(z_min - r_max, z_max + r_max); // may have issues
    //eps_rand = std::uniform_int_distribution<int>(0, 1);
    //eps_map = std::map<int, int>{{0, -1}, {1, 1}};


    // --------------------------- convert variables to ops const ---------------------------------------------------
    ops_decl_const("eddies", 1, "int", &eddies);
    ops_decl_const("u0", 1, "double", &u0);
    ops_decl_const("dt", 1, "double", &dt);
    ops_decl_const("x_min", 1, "double", &x_min);
    ops_decl_const("x_max", 1, "double", &x_max);
    ops_decl_const("y_min", 1, "double", &y_min);
    ops_decl_const("y_max", 1, "double", &y_max);
    ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
    ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
    ops_decl_const("z_min", 1, "double", &z_min);
    ops_decl_const("z_max", 1, "double", &z_max);
    ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
    ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);
    ops_decl_const("x_plane", 1, "double", &x_plane);
    ops_decl_const("TI", 1, "double", &TI);
    ops_decl_const("ny", 1, "int", &ny);
    ops_decl_const("nz", 1, "int", &nz);
    ops_decl_const("delta", 1, "double", &delta);
    ops_decl_const("r_max", 1, "double", &r_max);

    
    int eddy_size[] = {eddies, 1};
    int eddy_base[] = {0, 0};
    int eddy_pad_minus[] = {0, 0};
    int eddy_pad_plus[] = {0, 0};

    int inlet_size[] = {ny, nz};
    int inlet_base[] = {0, 0};
    int inlet_pad_minus[] = {0, 0};
    int inlet_pad_plus[] = {0, 0};
    

    // inlet plane variables
    double* y_inlet = NULL;
    double* z_inlet = NULL;
    double* uprime = NULL;
    double* vprime = NULL;
    double* wprime = NULL;
    double* a11 = NULL;
    double* a21 = NULL;
    double* a22 = NULL;
    double* a31 = NULL;
    double* a32 = NULL;
    double* a33 = NULL;

    // Random number arrays
    // consider uinsg 1 ops_dat to store all 5 sets of  random numbers: 1x y position, 1x z position, 3x random direction (x, y, z) for each eddy
    // DOUBLE, not int: ops_fill_random_uniform gives uniform_real[0,1) on a
    // double dat and uniform_int[0, INT_MAX] on an int one. See the note above
    // instantiate_eddies -- the int form is what made every eddy sign +1 and
    // confined every position to the upper half of its range.
    double* x_rng = NULL; // only necessary when instantiating eddies
    double* y_rng = NULL;
    double* z_rng = NULL;
    double* eps_x_rng = NULL;
    double* eps_y_rng = NULL;
    double* eps_z_rng = NULL;
    

    ops_block inlet_block = ops_decl_block(2, "inlet");
    ops_block eddy_block = ops_decl_block(2, "eddies");

    // OPS version of eddy variables
    //
    // NULL, not the malloc'd host array. ops_decl_dat takes ownership of a
    // non-NULL pointer (ops_lib_core.cpp:637-641 reallocs it) and MPI
    // partitioning allocates only when dat->data == NULL
    // (ops_mpi_partition.cpp:811). Passing x_gbl here, as the reference does,
    // leaves the dat and the app's "global" array pointing at the SAME memory:
    // gather_check reports them aliasing at np>1. The gather would then write
    // 1718 global values over the ~430-entry local slab the next
    // convect_eddies reads. With NULL, OPS owns the distributed storage and
    // x_gbl & co. are purely the host destinations for the gather.
    
    ops_dat d_x_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (double*)NULL, "double", "x_gbl"); 
    ops_dat d_y_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (double*)NULL, "double", "y_gbl"); 
    ops_dat d_z_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (double*)NULL, "double", "z_gbl");
    ops_dat d_r_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (double*)NULL, "double", "r_gbl");
    ops_dat d_increment_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (double*)NULL, "double", "increment_gbl");   
    ops_dat d_eps_x_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (int*)NULL, "int", "eps_x_gbl");
    ops_dat d_eps_y_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (int*)NULL, "int", "eps_y_gbl");
    ops_dat d_eps_z_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (int*)NULL, "int", "eps_z_gbl"); 

    // One int per eddy holding its GLOBAL index, filled once by
    // instantiate_gid. This is what the particle app gets from its p_gid dat;
    // with it the random fill needs nothing but dat->data.
    ops_dat d_gid = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, (int*)NULL, "int", "gid");

    ops_dat d_x_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, x_rng, "double", "x_rng");
    ops_dat d_y_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, y_rng, "double", "y_rng");
    ops_dat d_z_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, z_rng, "double", "z_rng");
    ops_dat d_eps_x_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_x_rng, "double", "eps_x_rng");
    ops_dat d_eps_y_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_y_rng, "double", "eps_y_rng");
    ops_dat d_eps_z_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_z_rng, "double", "eps_z_rng");


    // OPS_version of inlet plane variables
    ops_dat d_y_inlet = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, y_inlet, "double", "y_inlet");
    ops_dat d_z_inlet = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, z_inlet, "double", "z_inlet");
    ops_dat d_uprime = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, uprime, "double", "uprime");
    ops_dat d_vprime = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, vprime, "double", "vprime");
    ops_dat d_wprime = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, wprime, "double", "wprime");
    ops_dat d_a11 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a11, "double", "a11");
    ops_dat d_a21 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a21, "double", "a21");
    ops_dat d_a22 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a22, "double", "a22");
    ops_dat d_a31 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a31, "double", "a31");
    ops_dat d_a32 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a32, "double", "a32");
    ops_dat d_a33 = ops_decl_dat(inlet_block, 1, inlet_size, inlet_base, inlet_pad_minus, inlet_pad_plus, a33, "double", "a33");

    

    // define stencils
    int s1d_00[] = {0};
    ops_stencil S1D_00 = ops_decl_stencil(1, 1, s1d_00, "self1d");
    int eddy_iter_range[] = {0, eddies, 0, 1};

    int s2d_00[] = {0, 0};
    ops_stencil S2D_00 = ops_decl_stencil(1, 1, s2d_00, "self");
    int iter_range[] = {0, ny, 0, nz};

    /* Pin the eddy block to a 1-D decomposition.
     *
     * The eddy data is 1-D but has to be declared as a 2-D block {eddies, 1},
     * because OPS compiles for ONE dimensionality per translation unit: ACC's
     * accessors are macro-gated, operator()(int) under OPS_1D and
     * operator()(int,int) under OPS_2D (ops_lib_core.h:1458 and :1621), and
     * the inlet plane genuinely needs 2-D. ops_decl_block(1, ...) here does
     * not compile. So the degenerate second dimension is forced, not sloppy.
     *
     * Left alone, MPI_Dims_create splits BOTH dimensions once there are enough
     * ranks. At np = 8 it picks 4x2 and, since dimension 1 has global extent 1
     * and cannot be halved, both ranks of each pair report extent 1 -- the
     * same eddies held twice. Measured: slices summing to 3436 for 1718
     * eddies. The gather can de-duplicate that, but the two copies do not stay
     * in step on `x`, the one quantity convect_eddies carries across steps, so
     * np = 8 diverged from np = 1 on x_gbl and the three velocities while
     * everything redrawn each step stayed bit-identical.
     *
     * force_decomp is indexed [block_index * dims + d] and 0 means "let
     * MPI_Dims_create choose". Block 0 is the inlet (free), block 1 is the
     * eddies, pinned to {nranks, 1}. MPI_Dims_create honours nonzero entries,
     * so the eddy block is split along dimension 0 only and never duplicated.
     */
    /* NOT ops_num_procs(). With no processes_per_block given, OPS splits the
       ranks BETWEEN the blocks: processes_per_block[i] = nproc / nblocks
       (ops_mpi_partition.cpp:154-155). Two blocks here, so the eddy block only
       ever sees half the ranks -- which is why the banner reports a 2 x 1 grid
       at np = 4, and why asking for {np, 1} throws "force_decomp requested
       more processes than available for the block". */
    const int nblocks_decomp = 2;                       /* inlet + eddies */
    int procs_eddy = ops_num_procs() / nblocks_decomp;
    if (procs_eddy < 1) procs_eddy = 1;
    int force_decomp[4] = {0, 0, procs_eddy, 1};
    std::map<std::string, void *> partition_opts;
    partition_opts["force_decomp"] = (void *)force_decomp;
    ops_partition_opts("2D_block_DECOMPSE", partition_opts);

    // The gather assumes the eddy block is split along dimension 0 only, and
    // that the slices tile [0, eddies) with no gap or overlap. Verify once
    // rather than assume it.
    gather_check(d_x_gbl, eddies, (void*)x_gbl);

    ops_par_loop(instantiate_gid, "instantiate_gid", eddy_block, 2, eddy_iter_range,
    ops_arg_dat(d_gid, 1, S2D_00, "int", OPS_WRITE),
    ops_arg_idx());

    seed_gbl = (a * seed_gbl + c) % m;
    // Rank-invariant fill, keyed on the global eddy id -- see
    // OPS_oSEM_random.h. ops_randomgen_init/ops_fill_random_uniform are gone:
    // they seed per RANK and walk local storage, so an eddy's randomness
    // belonged to a (rank, slot) pair and every rank count was a different
    // realisation. Counter 1 is the seeding pass; the timestep loop uses i+2.
    osem_fill_random_uniform(d_x_rng, d_gid, seed_gbl, 1u, OSEM_RNG_X);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    osem_fill_random_uniform(d_y_rng, d_gid, seed_gbl, 1u, OSEM_RNG_Y);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    osem_fill_random_uniform(d_z_rng, d_gid, seed_gbl, 1u, OSEM_RNG_Z);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    osem_fill_random_uniform(d_eps_x_rng, d_gid, seed_gbl, 1u, OSEM_RNG_EPS_X);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    osem_fill_random_uniform(d_eps_y_rng, d_gid, seed_gbl, 1u, OSEM_RNG_EPS_Y);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    osem_fill_random_uniform(d_eps_z_rng, d_gid, seed_gbl, 1u, OSEM_RNG_EPS_Z);

    // instantiate eddy values
    ops_par_loop(instantiate_eddies, "instantiate_eddies", eddy_block, 2, eddy_iter_range,
    ops_arg_dat(d_x_gbl, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_y_gbl, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_z_gbl, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_r_gbl, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_increment_gbl, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_eps_x_gbl, 1, S2D_00, "int", OPS_WRITE),
    ops_arg_dat(d_eps_y_gbl, 1, S2D_00, "int", OPS_WRITE),
    ops_arg_dat(d_eps_z_gbl, 1, S2D_00, "int", OPS_WRITE),
    ops_arg_dat(d_x_rng, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_y_rng, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_z_rng, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_eps_x_rng, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_eps_y_rng, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_eps_z_rng, 1, S2D_00, "double", OPS_READ));


    //ops_decl_const("x_gbl", eddies, "double", &x_gbl[0]);
    //ops_decl_const("y_gbl", eddies, "double", &y_gbl[0]);
    //ops_decl_const("z_gbl", eddies, "double", &z_gbl[0]);
    //ops_decl_const("r_gbl", eddies, "double", &r_gbl[0]);
    //ops_decl_const("eps_x_gbl", eddies, "int", &eps_x_gbl[0]);
    //ops_decl_const("eps_y_gbl", eddies, "int", &eps_y_gbl[0]);
    //ops_decl_const("eps_z_gbl", eddies, "int", &eps_z_gbl[0]);

    ops_par_loop(instantiate_grid, "instantiate_grid", inlet_block, 2, iter_range,
    ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_idx());    

    ops_par_loop(instantiate_RST_TBL, "instantiate_RST_TBL", inlet_block, 2, iter_range,
    ops_arg_dat(d_a11, 1, S2D_00, "double", OPS_RW),
    ops_arg_dat(d_a21, 1, S2D_00, "double", OPS_RW),
    ops_arg_dat(d_a22, 1, S2D_00, "double", OPS_RW),
    ops_arg_dat(d_a31, 1, S2D_00, "double", OPS_RW),
    ops_arg_dat(d_a32, 1, S2D_00, "double", OPS_RW),
    ops_arg_dat(d_a33, 1, S2D_00, "double", OPS_RW), 
    ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_READ),
    ops_arg_gbl(yinterp, 260, "double", OPS_READ),
    ops_arg_gbl(r11interp, 260, "double", OPS_READ),
    ops_arg_gbl(r21interp, 260, "double", OPS_READ),
    ops_arg_gbl(r22interp, 260, "double", OPS_READ),
    ops_arg_gbl(r33interp, 260, "double", OPS_READ));

    std::string filename;
    /*
    filename = "a11.dat";
    ops_print_dat_to_txtfile(d_a11, filename.c_str());
    filename = "a21.dat";
    ops_print_dat_to_txtfile(d_a21, filename.c_str());
    filename = "a22.dat";
    ops_print_dat_to_txtfile(d_a22, filename.c_str());
    filename = "a31.dat";
    ops_print_dat_to_txtfile(d_a31, filename.c_str());
    filename = "a32.dat";
    ops_print_dat_to_txtfile(d_a32, filename.c_str());
    filename = "a33.dat";
    ops_print_dat_to_txtfile(d_a33, filename.c_str());
*/
    double ct0, et0;
    double ct1, et1;

    ops_timers(&ct0, &et0);

    filename = "y_inlet.txt";
    //ops_print_dat_to_txtfile(d_y_inlet, filename.c_str());
    filename = "z_inlet.txt";
    //ops_print_dat_to_txtfile(d_z_inlet, filename.c_str());
    printf("%s \n", "------------------------------");

    printf("======================================\n");

    for(i=0; i < niter; i++){
        if(fmod(i+1, write_output_file) == 0){
	        ops_printf("Reached iteration %d\n", i+1);
        }

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        osem_fill_random_uniform(d_y_rng, d_gid, seed_gbl, (unsigned int)i + 2u, OSEM_RNG_Y);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        osem_fill_random_uniform(d_z_rng, d_gid, seed_gbl, (unsigned int)i + 2u, OSEM_RNG_Z);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        osem_fill_random_uniform(d_eps_x_rng, d_gid, seed_gbl, (unsigned int)i + 2u, OSEM_RNG_EPS_X);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        osem_fill_random_uniform(d_eps_y_rng, d_gid, seed_gbl, (unsigned int)i + 2u, OSEM_RNG_EPS_Y);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        osem_fill_random_uniform(d_eps_z_rng, d_gid, seed_gbl, (unsigned int)i + 2u, OSEM_RNG_EPS_Z);


        ops_par_loop(convect_eddies, "convect_eddies", eddy_block, 2, eddy_iter_range,
        ops_arg_dat(d_x_gbl, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(d_y_gbl, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_z_gbl, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_r_gbl, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(d_increment_gbl, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_eps_x_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_eps_y_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_eps_z_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_y_rng, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_z_rng, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_eps_x_rng, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_eps_y_rng, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_eps_z_rng, 1, S2D_00, "double", OPS_READ));

        // THE FIX. These were seven ops_dat_fetch_data calls, which under MPI
        // give every rank its own slice written at offset 0 while
        // compute_fluct below loops over the global `eddies`. See the header
        // for the library line that drops the displacement.
        gather_eddy_dat(d_x_gbl, x_gbl, eddies);
        gather_eddy_dat(d_y_gbl, y_gbl, eddies);
        gather_eddy_dat(d_z_gbl, z_gbl, eddies);
        gather_eddy_dat(d_r_gbl, r_gbl, eddies);
        gather_eddy_dat(d_eps_x_gbl, eps_x_gbl, eddies);
        gather_eddy_dat(d_eps_y_gbl, eps_y_gbl, eddies);
        gather_eddy_dat(d_eps_z_gbl, eps_z_gbl, eddies);

        //ops_update_const("x_gbl", eddies, "double", x_gbl);
        //ops_update_const("y_gbl", eddies, "double", y_gbl);
        //ops_update_const("z_gbl", eddies, "double", z_gbl);
        //ops_update_const("r_gbl", eddies, "double", r_gbl);
        //ops_update_const("eps_x_gbl", eddies, "int", eps_x_gbl);
        //ops_update_const("eps_y_gbl", eddies, "int", eps_y_gbl);
        //ops_update_const("eps_z_gbl", eddies, "int", eps_z_gbl);

        ops_par_loop(compute_fluct, "compute_fluct", inlet_block, 2, iter_range,
        ops_arg_dat(d_y_inlet, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_z_inlet, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a11, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a21, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a22, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a31, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a32, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_a33, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_uprime, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_vprime, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_wprime, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_gbl(x_gbl, eddies, "double", OPS_READ),
        ops_arg_gbl(y_gbl, eddies, "double", OPS_READ),
        ops_arg_gbl(z_gbl, eddies, "double", OPS_READ),
        ops_arg_gbl(r_gbl, eddies, "double", OPS_READ),
        ops_arg_gbl(eps_x_gbl, eddies, "int", OPS_READ),
        ops_arg_gbl(eps_y_gbl, eddies, "int", OPS_READ),
        ops_arg_gbl(eps_z_gbl, eddies, "int", OPS_READ));

        /*
        filename = std::string("u_test" + std::to_string(i) + ".dat");
        ops_print_dat_to_txtfile(d_uprime, filename.c_str());
        filename = std::string("v_test" + std::to_string(i) + ".dat");
        ops_print_dat_to_txtfile(d_vprime, filename.c_str());
        filename = std::string("w_test" + std::to_string(i) + ".dat");
        ops_print_dat_to_txtfile(d_wprime, filename.c_str());*/

        if(fmod(i+1, write_output_file) == 0){
	        HDF5_IO_Write_inlet_block_dynamic(
                inlet_block, 
                i, 
                d_y_inlet, d_z_inlet, 
                d_a11, d_a21, d_a22, d_a31, d_a32, d_a33, 
                d_uprime, d_vprime, d_wprime, 
                x_gbl, y_gbl, z_gbl, r_gbl, 
                eps_x_gbl, eps_y_gbl, eps_z_gbl
            );
        }
    }

    ops_timers(&ct1, &et1);

    ops_printf("time elapsed: %f \n", et1 - et0);

    ops_printf("%s \n", "--------------------");

    HDF5_IO_Write_inlet_block(
        inlet_block, 
        i, 
        d_y_inlet, d_z_inlet, 
        d_a11, d_a21, d_a22, d_a31, d_a32, d_a33, 
        d_uprime, d_vprime, d_wprime, 
        x_gbl, y_gbl, z_gbl, r_gbl, 
        eps_x_gbl, eps_y_gbl, eps_z_gbl
    );

    ops_exit();
}
