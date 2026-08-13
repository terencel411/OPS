// Declaration of global constants
int restart;
int iter;
int stage;
double tstart;
double Delta0block0;
double Delta1block0;
double Delta2block0;
int HDF5_timing;
double Lx1;
double Minf;
double Pr;
double r;
double Re;
double RefT;
double SuthT;
double Twall;
int block0np0;
int block0np1;
int block0np2;
double by;
double dt;
double gama;
double inv2Delta0block0;
double inv2Delta1block0;
double inv2Delta2block0;
double inv2Minf;
double invDelta0block0;
double invDelta1block0;
double invDelta2block0;
double invLx1;
double invPr;
double invRe;
double invRefT;
double inv_gamma_m1;
double start_averaging;
double invniter;
int niter;
double simulation_time;
int start_iter;
int write_output_file;

int ny; // interpolation values
//double* yinterp;
double* uinterp;

// eddy constants
int y_cutoff;
int ndata;
double* a11;
double* a21;
double* a22;
double* a33;
double delta;
double radius;
double eddy_x_min;
double eddy_x_max;
double eddy_y_min;
double eddy_y_max;
double eddy_z_min;
double eddy_z_max;
double eddy_vol;
int eddies;

// Per-step convection distance. A per-eddy dat in oSEM_3d; identical for all.
double increment;

unsigned int seed_gbl;

// The gather buffer replacing oSEM_3d's seven eddy_*_gbl arrays. Layout in
// osem3d_common.h.
double* eddy_all;

// Diagnostic: per-component scale on the inlet fluctuation (-fluct A B C).
double fluct_scale[3];
