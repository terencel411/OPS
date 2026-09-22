#ifndef _OSEM_CONSTANTS_H_
#define _OSEM_CONSTANTS_H_

double u0;
double dt;
double delta;
double ti;

double y_min;
double y_max;
double z_min;
double z_max;
double x_plane;

int ny;
int nz;

double r_max;

double x_min;
double x_max;

double eddy_y_min;
double eddy_y_max;
double eddy_z_min;
double eddy_z_max;

double eddy_radius;
double increment;

double vol;
int eddies;
double u0ti;

unsigned int seed_gbl;
int niter;
int nprint;
int rng_method;
int ntbl;
int use_tbl;

#endif /* _OSEM_CONSTANTS_H_ */
