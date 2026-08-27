// #include <hdf5_hl.h>

void write_constants(const char* filename){
    ops_write_const_hdf5("eddies", 1, "int", (char*)&eddies, filename);
    ops_write_const_hdf5("u0", 1, "double", (char*)&u0, filename);
    ops_write_const_hdf5("dt", 1, "double", (char*)&dt, filename);
    ops_write_const_hdf5("x_min", 1, "double", (char*)&x_min, filename);
    ops_write_const_hdf5("x_max", 1, "double", (char*)&x_max, filename);
    ops_write_const_hdf5("y_min", 1, "double", (char*)&y_min, filename);
    ops_write_const_hdf5("y_max", 1, "double", (char*)&y_max, filename);
    ops_write_const_hdf5("eddy_y_min", 1, "double", (char*)&eddy_y_min, filename);
    ops_write_const_hdf5("eddy_y_max", 1, "double", (char*)&eddy_y_max, filename);
    ops_write_const_hdf5("z_min", 1, "double", (char*)&z_min, filename);
    ops_write_const_hdf5("z_max", 1, "double", (char*)&z_max, filename);
    ops_write_const_hdf5("eddy_z_min", 1, "double", (char*)&eddy_z_min, filename);
    ops_write_const_hdf5("eddy_z_max", 1, "double", (char*)&eddy_z_max, filename);
    ops_write_const_hdf5("x_plane", 1, "double", (char*)&x_plane, filename);
    ops_write_const_hdf5("TI", 1, "double", (char*)&TI, filename);
    ops_write_const_hdf5("ny", 1, "int", (char*)&ny, filename);
    ops_write_const_hdf5("nz", 1, "int", (char*)&nz, filename);
    ops_write_const_hdf5("delta", 1, "double", (char*)&delta, filename);
    ops_write_const_hdf5("r_max", 1, "double", (char*)&r_max, filename);
    ops_write_const_hdf5("write_output_file", 1, "int", (char*)&write_output_file, filename);
    ops_write_const_hdf5("niter", 1, "int", (char*)&niter, filename);
    ops_write_const_hdf5("i", 1, "int", (char*)&i, filename);
    ops_write_const_hdf5("x_gbl", eddies, "double", (char*)&x_gbl, filename);
    ops_write_const_hdf5("y_gbl", eddies, "double", (char*)&y_gbl, filename);
    ops_write_const_hdf5("z_gbl", eddies, "double", (char*)&z_gbl, filename);
    ops_write_const_hdf5("eps_x_gbl", eddies, "int", (char*)&eps_x_gbl, filename);
    ops_write_const_hdf5("eps_y_gbl", eddies, "int", (char*)&eps_y_gbl, filename);
    ops_write_const_hdf5("eps_z_gbl", eddies, "int", (char*)&eps_z_gbl, filename);
}

void HDF5_IO_Write_inlet_block_dynamic(
    ops_block& inlet_block, 
    int i, 
    ops_dat& d_y_inlet, ops_dat& d_z_inlet, 
    ops_dat& d_a11, ops_dat& d_a21, ops_dat& d_a22, ops_dat& d_a31, ops_dat& d_a32, ops_dat& d_a33, 
    ops_dat& d_uprime, ops_dat& d_vprime, ops_dat& d_wprime, 
    double* x_gbl, double* y_gbl, double* z_gbl, double* r_gbl, 
    int* eps_x_gbl, int* eps_y_gbl, int* eps_z_gbl
){
    double cpu_start0, elapsed_start0;

    ops_timers(&cpu_start0, &elapsed_start0);

    // Writing OPS datasets
    char name0[80];
    sprintf(name0, "oSEM_output_%06d.h5", i + 1);
    ops_fetch_block_hdf5_file(inlet_block, name0);
    ops_fetch_dat_hdf5_file(d_y_inlet, name0);
    ops_fetch_dat_hdf5_file(d_z_inlet, name0);
    ops_fetch_dat_hdf5_file(d_a11, name0);
    ops_fetch_dat_hdf5_file(d_a21, name0);
    ops_fetch_dat_hdf5_file(d_a22, name0);
    ops_fetch_dat_hdf5_file(d_a31, name0);
    ops_fetch_dat_hdf5_file(d_a32, name0);
    ops_fetch_dat_hdf5_file(d_a33, name0);
    ops_fetch_dat_hdf5_file(d_uprime, name0);
    ops_fetch_dat_hdf5_file(d_vprime, name0);
    ops_fetch_dat_hdf5_file(d_wprime, name0);

    // Writing simulation constants
    write_constants(name0);

    double cpu_end0, elapsed_end0;
    ops_timers(&cpu_end0, &elapsed_end0);
    ops_printf("-----------------------------------------\n");
    ops_printf("Time to write HDF5 file: %s: %lf\n", name0, elapsed_end0-elapsed_start0);
    ops_printf("-----------------------------------------\n");

}

void HDF5_IO_Write_inlet_block(
    ops_block& inlet_block, 
    int i, 
    ops_dat& d_y_inlet, ops_dat& d_z_inlet, 
    ops_dat& d_a11, ops_dat& d_a21, ops_dat& d_a22, ops_dat& d_a31, ops_dat& d_a32, ops_dat& d_a33, 
    ops_dat& d_uprime, ops_dat& d_vprime, ops_dat& d_wprime, 
    double* x_gbl, double* y_gbl, double* z_gbl, double* r_gbl, 
    int* eps_x_gbl, int* eps_y_gbl, int* eps_z_gbl
){
    double cpu_start0, elapsed_start0;
    ops_timers(&cpu_start0, &elapsed_start0);

    // Writing OPS datasets
    char name0[80];
    sprintf(name0, "oSEM_output.h5", i + 1);
    ops_fetch_block_hdf5_file(inlet_block, name0);
    ops_fetch_dat_hdf5_file(d_y_inlet, name0);
    ops_fetch_dat_hdf5_file(d_z_inlet, name0);
    ops_fetch_dat_hdf5_file(d_a11, name0);
    ops_fetch_dat_hdf5_file(d_a21, name0);
    ops_fetch_dat_hdf5_file(d_a22, name0);
    ops_fetch_dat_hdf5_file(d_a31, name0);
    ops_fetch_dat_hdf5_file(d_a32, name0);
    ops_fetch_dat_hdf5_file(d_a33, name0);
    ops_fetch_dat_hdf5_file(d_uprime, name0);
    ops_fetch_dat_hdf5_file(d_vprime, name0);
    ops_fetch_dat_hdf5_file(d_wprime, name0);

    // Writing simulation constants
    write_constants(name0);

    double cpu_end0, elapsed_end0;
    ops_timers(&cpu_end0, &elapsed_end0);
    ops_printf("-----------------------------------------\n");
    ops_printf("Time to write HDF5 file: %s: %lf\n", name0, elapsed_end0-elapsed_start0);
    ops_printf("-----------------------------------------\n");
}
