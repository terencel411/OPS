void write_constants(const char* filename){
ops_write_const_hdf5("Delta0block0", 1, "double", (char*)&Delta0block0, filename);
ops_write_const_hdf5("Delta1block0", 1, "double", (char*)&Delta1block0, filename);
ops_write_const_hdf5("Delta2block0", 1, "double", (char*)&Delta2block0, filename);
ops_write_const_hdf5("HDF5_timing", 1, "int", (char*)&HDF5_timing, filename);
ops_write_const_hdf5("Lx1", 1, "double", (char*)&Lx1, filename);
ops_write_const_hdf5("Minf", 1, "double", (char*)&Minf, filename);
ops_write_const_hdf5("Pr", 1, "double", (char*)&Pr, filename);
ops_write_const_hdf5("Re", 1, "double", (char*)&Re, filename);
ops_write_const_hdf5("RefT", 1, "double", (char*)&RefT, filename);
ops_write_const_hdf5("SuthT", 1, "double", (char*)&SuthT, filename);
ops_write_const_hdf5("Twall", 1, "double", (char*)&Twall, filename);
ops_write_const_hdf5("block0np0", 1, "int", (char*)&block0np0, filename);
ops_write_const_hdf5("block0np1", 1, "int", (char*)&block0np1, filename);
ops_write_const_hdf5("block0np2", 1, "int", (char*)&block0np2, filename);
ops_write_const_hdf5("by", 1, "double", (char*)&by, filename);
ops_write_const_hdf5("dt", 1, "double", (char*)&dt, filename);
ops_write_const_hdf5("gama", 1, "double", (char*)&gama, filename);
ops_write_const_hdf5("niter", 1, "int", (char*)&niter, filename);
ops_write_const_hdf5("simulation_time", 1, "double", (char*)&simulation_time, filename);
ops_write_const_hdf5("start_iter", 1, "int", (char*)&start_iter, filename);
ops_write_const_hdf5("write_output_file", 1, "int", (char*)&write_output_file, filename);
ops_write_const_hdf5("iter", 1, "int", (char*)&iter, filename);
}

void HDF5_IO_Write_0_opensbliblock00_dynamic(ops_block& opensbliblock00, int iter, ops_dat& rho_B0, ops_dat& rhou0_B0, ops_dat& rhou1_B0, ops_dat& rhou2_B0, ops_dat& rhoE_B0, ops_dat& x0_B0, ops_dat& x1_B0, ops_dat& x2_B0, ops_dat& D11_B0, ops_dat& T_B0, ops_dat& mu_B0, ops_dat& p_B0, int HDF5_timing){
double cpu_start0, elapsed_start0;
if (HDF5_timing == 1){
ops_timers(&cpu_start0, &elapsed_start0);
}
// Writing OPS datasets
char name0[80];
sprintf(name0, "opensbli_output_%06d.h5", iter + 1);
ops_fetch_block_hdf5_file(opensbliblock00, name0);
ops_fetch_dat_hdf5_file(rho_B0, name0);
ops_fetch_dat_hdf5_file(rhou0_B0, name0);
ops_fetch_dat_hdf5_file(rhou1_B0, name0);
ops_fetch_dat_hdf5_file(rhou2_B0, name0);
ops_fetch_dat_hdf5_file(rhoE_B0, name0);
ops_fetch_dat_hdf5_file(x0_B0, name0);
ops_fetch_dat_hdf5_file(x1_B0, name0);
ops_fetch_dat_hdf5_file(x2_B0, name0);
ops_fetch_dat_hdf5_file(D11_B0, name0);
ops_fetch_dat_hdf5_file(T_B0, name0);
ops_fetch_dat_hdf5_file(mu_B0, name0);
ops_fetch_dat_hdf5_file(p_B0, name0);
// Writing simulation constants
write_constants(name0);
if (HDF5_timing == 1){
double cpu_end0, elapsed_end0;
ops_timers(&cpu_end0, &elapsed_end0);
ops_printf("-----------------------------------------\n");
ops_printf("Time to write HDF5 file: %s: %lf\n", name0, elapsed_end0-elapsed_start0);
ops_printf("-----------------------------------------\n");

}
}

void HDF5_IO_Write_0_opensbliblock00(ops_block& opensbliblock00, ops_dat& rho_B0, ops_dat& rhou0_B0, ops_dat& rhou1_B0, ops_dat& rhou2_B0, ops_dat& rhoE_B0, ops_dat& x0_B0, ops_dat& x1_B0, ops_dat& x2_B0, ops_dat& D11_B0, ops_dat& T_B0, ops_dat& mu_B0, ops_dat& p_B0, int HDF5_timing){
double cpu_start0, elapsed_start0;
if (HDF5_timing == 1){
ops_timers(&cpu_start0, &elapsed_start0);
}
// Writing OPS datasets
char name0[80];
sprintf(name0, "opensbli_output.h5");
ops_fetch_block_hdf5_file(opensbliblock00, name0);
ops_fetch_dat_hdf5_file(rho_B0, name0);
ops_fetch_dat_hdf5_file(rhou0_B0, name0);
ops_fetch_dat_hdf5_file(rhou1_B0, name0);
ops_fetch_dat_hdf5_file(rhou2_B0, name0);
ops_fetch_dat_hdf5_file(rhoE_B0, name0);
ops_fetch_dat_hdf5_file(x0_B0, name0);
ops_fetch_dat_hdf5_file(x1_B0, name0);
ops_fetch_dat_hdf5_file(x2_B0, name0);
ops_fetch_dat_hdf5_file(D11_B0, name0);
ops_fetch_dat_hdf5_file(T_B0, name0);
ops_fetch_dat_hdf5_file(mu_B0, name0);
ops_fetch_dat_hdf5_file(p_B0, name0);
// Writing simulation constants
write_constants(name0);
if (HDF5_timing == 1){
double cpu_end0, elapsed_end0;
ops_timers(&cpu_end0, &elapsed_end0);
ops_printf("-----------------------------------------\n");
ops_printf("Time to write HDF5 file: %s: %lf\n", name0, elapsed_end0-elapsed_start0);
ops_printf("-----------------------------------------\n");

}
}

void HDF5_IO_Write_1_opensbliblock00(ops_block& opensbliblock00, ops_dat& rho_mean_B0, ops_dat& rhou0_mean_B0, ops_dat& rhou1_mean_B0, ops_dat& rhou2_mean_B0, ops_dat& rhoE_mean_B0, ops_dat& rhou0u0_mean_B0, ops_dat& rhou1u1_mean_B0, ops_dat& rhou2u2_mean_B0, ops_dat& rhou0u1_mean_B0, ops_dat& rhou1u2_mean_B0, ops_dat& rhou0u2_mean_B0, ops_dat& rhorhou0u0_mean_B0, ops_dat& taux0x1_mean_B0, ops_dat& l_mean_B0, ops_dat& du0dx1_mean_B0, ops_dat& mu_mean_B0, ops_dat& u0_mean_B0, ops_dat& u1_mean_B0, ops_dat& u2_mean_B0, ops_dat& u0u0_mean_B0, ops_dat& u1u1_mean_B0, ops_dat& u2u2_mean_B0, ops_dat& u0u1_mean_B0, ops_dat& utau_mean_B0, int HDF5_timing){
double cpu_start0, elapsed_start0;
if (HDF5_timing == 1){
ops_timers(&cpu_start0, &elapsed_start0);
}
// Writing OPS datasets
ops_fetch_block_hdf5_file(opensbliblock00, "stats_output.h5");
ops_fetch_dat_hdf5_file(rho_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou0_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhoE_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou0u0_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou1u1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou2u2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou0u1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou1u2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhou0u2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(rhorhou0u0_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(taux0x1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(l_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(du0dx1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(mu_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u0_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u0u0_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u1u1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u2u2_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(u0u1_mean_B0, "stats_output.h5");
ops_fetch_dat_hdf5_file(utau_mean_B0, "stats_output.h5");
// Writing simulation constants
write_constants("stats_output.h5");
if (HDF5_timing == 1){
double cpu_end0, elapsed_end0;
ops_timers(&cpu_end0, &elapsed_end0);
ops_printf("-----------------------------------------\n");
ops_printf("Time to write HDF5 file: %s: %lf\n", "stats_output.h5", elapsed_end0-elapsed_start0);
ops_printf("-----------------------------------------\n");

}
}

