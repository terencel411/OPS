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
#include "OPS_oSEM_io.h"

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
    int* x_rng = NULL; // only necessary when instantiating eddies
    int* y_rng = NULL;
    int* z_rng = NULL;
    int* eps_x_rng = NULL;
    int* eps_y_rng = NULL;
    int* eps_z_rng = NULL;
    

    ops_block inlet_block = ops_decl_block(2, "inlet");
    ops_block eddy_block = ops_decl_block(2, "eddies");

    // OPS version of eddy variables
    
    ops_dat d_x_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, x_gbl, "double", "x_gbl"); 
    ops_dat d_y_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, y_gbl, "double", "y_gbl"); 
    ops_dat d_z_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, z_gbl, "double", "z_gbl");
    ops_dat d_r_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, r_gbl, "double", "r_gbl");
    ops_dat d_increment_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, increment_gbl, "double", "increment_gbl");   
    ops_dat d_eps_x_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_x_gbl, "int", "eps_x_gbl");
    ops_dat d_eps_y_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_y_gbl, "int", "eps_y_gbl");
    ops_dat d_eps_z_gbl = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_z_gbl, "int", "eps_z_gbl"); 

    ops_dat d_x_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, x_rng, "int", "x_rng");
    ops_dat d_y_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, y_rng, "int", "y_rng");
    ops_dat d_z_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, z_rng, "int", "z_rng");
    ops_dat d_eps_x_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_x_rng, "int", "eps_x_rng");
    ops_dat d_eps_y_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_y_rng, "int", "eps_y_rng");
    ops_dat d_eps_z_rng = ops_decl_dat(eddy_block, 1, eddy_size, eddy_base, eddy_pad_minus, eddy_pad_plus, eps_z_rng, "int", "eps_z_rng");


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

    ops_partition("2D_block_DECOMPSE");

    seed_gbl = (a * seed_gbl + c) % m;
    ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_x_rng);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_y_rng);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_z_rng);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_eps_x_rng);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_eps_y_rng);
    //seed_gbl = (a * seed_gbl + c) % m;
    //ops_randomgen_init(seed_gbl, 0);
    ops_fill_random_uniform(d_eps_z_rng);

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
    ops_arg_dat(d_x_rng, 1, S2D_00, "int", OPS_READ),
    ops_arg_dat(d_y_rng, 1, S2D_00, "int", OPS_READ),
    ops_arg_dat(d_z_rng, 1, S2D_00, "int", OPS_READ),
    ops_arg_dat(d_eps_x_rng, 1, S2D_00, "int", OPS_READ),
    ops_arg_dat(d_eps_y_rng, 1, S2D_00, "int", OPS_READ),
    ops_arg_dat(d_eps_z_rng, 1, S2D_00, "int", OPS_READ));


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
        ops_fill_random_uniform(d_y_rng);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        ops_fill_random_uniform(d_z_rng);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        ops_fill_random_uniform(d_eps_x_rng);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        ops_fill_random_uniform(d_eps_y_rng);

        //seed_gbl = (a * seed_gbl + c) % m;
        //ops_randomgen_init(seed_gbl, 0);
        ops_fill_random_uniform(d_eps_z_rng);


        ops_par_loop(convect_eddies, "convect_eddies", eddy_block, 2, eddy_iter_range,
        ops_arg_dat(d_x_gbl, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(d_y_gbl, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_z_gbl, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(d_r_gbl, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(d_increment_gbl, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(d_eps_x_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_eps_y_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_eps_z_gbl, 1, S2D_00, "int", OPS_WRITE),
        ops_arg_dat(d_y_rng, 1, S2D_00, "int", OPS_READ),
        ops_arg_dat(d_z_rng, 1, S2D_00, "int", OPS_READ),
        ops_arg_dat(d_eps_x_rng, 1, S2D_00, "int", OPS_READ),
        ops_arg_dat(d_eps_y_rng, 1, S2D_00, "int", OPS_READ),
        ops_arg_dat(d_eps_z_rng, 1, S2D_00, "int", OPS_READ));

        ops_dat_fetch_data(d_x_gbl, 0, (char*)x_gbl);
        ops_dat_fetch_data(d_y_gbl, 0, (char*)y_gbl);
        ops_dat_fetch_data(d_z_gbl, 0, (char*)z_gbl);
        ops_dat_fetch_data(d_r_gbl, 0, (char*)r_gbl);
        ops_dat_fetch_data(d_eps_x_gbl, 0, (char*)eps_x_gbl);
        ops_dat_fetch_data(d_eps_y_gbl, 0, (char*)eps_y_gbl);
        ops_dat_fetch_data(d_eps_z_gbl, 0, (char*)eps_z_gbl);

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
        
        // Write to HDF5 file periodically
        if(fmod(i+1, write_output_file) == 0){
	        HDF5_IO_Write_inlet_block_dynamic(
                inlet_block, 
                i, 
                d_y_inlet, d_z_inlet, 
                d_a11, d_a21, d_a22, d_a31, d_a32, d_a33, 
                d_uprime, d_vprime, d_wprime
            );
        }
    }

    ops_timers(&ct1, &et1);

    ops_printf("time elapsed: %f \n", et1 - et0);

    ops_printf("%s \n", "--------------------");

    // Write final output to HDF5 file
    HDF5_IO_Write_inlet_block(
        inlet_block, 
        i, 
        d_y_inlet, d_z_inlet, 
        d_a11, d_a21, d_a22, d_a31, d_a32, d_a33, 
        d_uprime, d_vprime, d_wprime
    );

    ops_exit();
}
