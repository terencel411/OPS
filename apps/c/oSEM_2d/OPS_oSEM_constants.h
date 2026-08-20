double u0;
double dt;
double delta;
double r_max;
int ny;
int nz;
int niter;
int i;
int write_output_file;
double TI;
double x_min;
double x_max;
double x_plane;
double y_min;
double y_max;
double eddy_y_min;
double eddy_y_max;
double z_min; 
double z_max;
double eddy_z_min;
double eddy_z_max;
double vol;
double rep_radius;
//constexpr int eddies = trunc(vol / (rep_radius*rep_radius*rep_radius));
int eddies;
int a;
int c;
int m;
int seed_gbl;

double* x_gbl = nullptr;
double* y_gbl = nullptr;
double* z_gbl = nullptr;
double* r_gbl = nullptr;
double* increment_gbl = nullptr;
int* eps_x_gbl = nullptr;
int* eps_y_gbl = nullptr;
int* eps_z_gbl = nullptr;

double* yinterp = nullptr;
double* r11interp = nullptr;
double* r21interp = nullptr;
double* r22interp = nullptr;
double* r33interp = nullptr;



