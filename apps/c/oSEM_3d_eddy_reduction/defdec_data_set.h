ops_dat D11_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
D11_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "D11_B0");
}
ops_dat detJ_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
detJ_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "detJ_B0");
}
ops_dat mu_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
mu_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "mu_B0");
}
ops_dat p_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
p_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "p_B0");
}
ops_dat Residual0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual0_B0");
}
ops_dat Residual1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual1_B0");
}
ops_dat Residual2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual2_B0");
}
ops_dat Residual3_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual3_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual3_B0");
}
ops_dat Residual4_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual4_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual4_B0");
}
ops_dat rho_B0;
{
if (restart == 1){
rho_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "rho_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rho_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rho_B0");
}
}
ops_dat rho_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rho_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rho_mean_B0");
}
ops_dat rho_RKold_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rho_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rho_RKold_B0");
}
ops_dat rhoE_B0;
{
if (restart == 1){
rhoE_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "rhoE_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhoE_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhoE_B0");
}
}
ops_dat rhoE_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhoE_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhoE_mean_B0");
}
ops_dat rhoE_RKold_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhoE_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhoE_RKold_B0");
}
ops_dat rhou0_B0;
{
if (restart == 1){
rhou0_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "rhou0_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0_B0");
}
}
ops_dat rhou0_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0_mean_B0");
}
ops_dat rhou0_RKold_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0_RKold_B0");
}
ops_dat rhou0u0_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0u0_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0u0_mean_B0");
}
ops_dat rhou0u1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0u1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0u1_mean_B0");
}
ops_dat rhou0u2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0u2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0u2_mean_B0");
}
ops_dat rhou1_B0;
{
if (restart == 1){
rhou1_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "rhou1_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1_B0");
}
}
ops_dat rhou1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1_mean_B0");
}
ops_dat rhou1_RKold_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1_RKold_B0");
}
ops_dat rhou1u1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1u1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1u1_mean_B0");
}
ops_dat rhou1u2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1u2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1u2_mean_B0");
}
ops_dat rhou2_B0;
{
if (restart == 1){
rhou2_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "rhou2_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2_B0");
}
}
ops_dat rhou2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2_mean_B0");
}
ops_dat rhou2_RKold_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2_RKold_B0");
}
ops_dat rhou2u2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2u2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2u2_mean_B0");
}
ops_dat SD111_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
SD111_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "SD111_B0");
}
ops_dat T_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
T_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "T_B0");
}
ops_dat u0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u0_B0");
}
ops_dat u1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u1_B0");
}
ops_dat u2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u2_B0");
}
ops_dat wk0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk0_B0");
}
ops_dat wk10_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk10_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk10_B0");
}
ops_dat wk11_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk11_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk11_B0");
}
ops_dat wk1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk1_B0");
}
ops_dat wk2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk2_B0");
}
ops_dat wk3_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk3_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk3_B0");
}
ops_dat wk4_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk4_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk4_B0");
}
ops_dat wk5_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk5_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk5_B0");
}
ops_dat wk6_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk6_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk6_B0");
}
ops_dat wk7_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk7_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk7_B0");
}
ops_dat wk8_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk8_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk8_B0");
}
ops_dat wk9_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk9_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk9_B0");
}
ops_dat x0_B0;
{
if (restart == 1){
x0_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "x0_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
x0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "x0_B0");
}
}
ops_dat x1_B0;
{
if (restart == 1){
x1_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "x1_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
x1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "x1_B0");
}
}
ops_dat x2_B0;
{
if (restart == 1){
x2_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "x2_B0", "restart.h5");
}
else {
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
x2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "x2_B0");
}
}
// new dat
ops_dat taux0x1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
taux0x1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "taux0x1_mean_B0");
}

ops_dat rhorhou0u0_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhorhou0u0_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhorhou0u0_mean_B0");
}


ops_dat du0dx1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
du0dx1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "du0dx1_mean_B0");
}

ops_dat l_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
l_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "l_mean_B0");
}

ops_dat mu_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
mu_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "mu_mean_B0");
}
ops_dat u0_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u0_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u0_mean_B0");
}
ops_dat u1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u1_mean_B0");
}
ops_dat u2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u2_mean_B0");
}
ops_dat u0u0_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u0u0_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u0u0_mean_B0");
}
ops_dat u1u1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u1u1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u1u1_mean_B0");
}
ops_dat u2u2_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u2u2_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u2u2_mean_B0");
}
ops_dat u0u1_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u0u1_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u0u1_mean_B0");
}
ops_dat utau_mean_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
utau_mean_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "utau_mean_B0");
}

ops_dat d_uinterp;
{
int halo_p[] = {0, 1, 0};
int halo_m[] = {0, -1, 0};
int size[] = {1, ny, 1};
int base[] = {0, 0, 0};
double* value = NULL;
d_uinterp = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "d_uinterp");
}

//--------------------------eddy dats------------------------
ops_dat eddy_x;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
double* value = NULL;
eddy_x = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "eddy_x");
}
ops_dat eddy_y;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
double* value = NULL;
eddy_y = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "eddy_y");
}
ops_dat eddy_z;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
double* value = NULL;
eddy_z = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "eddy_z");
}
ops_dat eddy_r;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
double* value = NULL;
eddy_r = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "eddy_r");
}
ops_dat eddy_increment;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
double* value = NULL;
eddy_increment = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "eddy_increment");
}
ops_dat eddy_eps_x;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
int* value = NULL;
eddy_eps_x = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "int", "eddy_eps_x");
}
ops_dat eddy_eps_y;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
int* value = NULL;
eddy_eps_y = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "int", "eddy_eps_y");
}
ops_dat eddy_eps_z;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
int* value = NULL;
eddy_eps_z = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "int", "eddy_eps_z");
}
ops_dat eddy_x_rng;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
int* value = NULL;
eddy_x_rng = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "int", "eddy_x_rng");
}
ops_dat eddy_bulk_rng;
{
int halo_p[] = {1, 0, 0};
int halo_m[] = {-1, 0, 0};
int size[] = {eddies, 1, 1};
int base[] = {0, 0, 0};
int* value = NULL;
eddy_bulk_rng = ops_decl_dat(opensbliblock00, 5, size, base, halo_m, halo_p, value, "int", "eddy_bulk_rng");
}

ops_dat d_a11;
{
int halo_p[] = {0, 1, 0};
int halo_m[] = {0, -1, 0};
int size[] = {1, y_cutoff, 1};
int base[] = {0, 0, 0};
double* value = NULL;
d_a11 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "d_a11");
}
ops_dat d_a21;
{
int halo_p[] = {0, 1, 0};
int halo_m[] = {0, -1, 0};
int size[] = {1, y_cutoff, 1};
int base[] = {0, 0, 0};
double* value = NULL;
d_a21 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "d_a21");
}
ops_dat d_a22;
{
int halo_p[] = {0, 1, 0};
int halo_m[] = {0, -1, 0};
int size[] = {1, y_cutoff, 1};
int base[] = {0, 0, 0};
double* value = NULL;
d_a22 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "d_a22");
}
ops_dat d_a33;
{
int halo_p[] = {0, 1, 0};
int halo_m[] = {0, -1, 0};
int size[] = {1, y_cutoff, 1};
int base[] = {0, 0, 0};
double* value = NULL;
d_a33 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "d_a33");
}