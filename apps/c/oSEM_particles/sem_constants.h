/*
 * sem_constants.h -- oSEM on OPS Particles
 *
 * Geometry and flow parameters, kept identical to
 * OPS/apps/c/oSEM/OPS_oSEM.cpp so the two can be compared directly.
 *
 * These are plain globals here and are also registered with ops_decl_const()
 * in main(), which is what makes them visible inside kernels (the
 * translator-generated kernel files see ops_decl_const variables and nothing
 * else from the application headers).
 */

#ifndef _SEM_CONSTANTS_H_
#define _SEM_CONSTANTS_H_

/* ---- flow ---- */
double u0;          /* convection velocity of the eddies                    */
double dt;          /* timestep                                             */
double delta;       /* boundary layer thickness                             */
double r_max;       /* 0.41 * delta                                         */
double TI;          /* turbulence intensity (unused once TBL data is read)  */

/* ---- inlet plane resolution ---- */
int ny;             /* nodes in y (wall normal)                             */
int nz;             /* nodes in z (spanwise)                                */

/* ---- the plane itself ---- */
double y_min, y_max;
double z_min, z_max;
double x_plane;     /* the streamwise station the plane sits at             */

/* ---- the eddy box ----
 * Larger than the plane by r_max, so eddies just outside can still influence
 * it. This box is what the particle BoundingBox must span. */
double x_min, x_max;
double eddy_y_min, eddy_y_max;
double eddy_z_min, eddy_z_max;

/* ---- eddy population ---- */
double vol;
double rep_radius;  /* representative eddy length scale, 0.2 * delta        */
int    eddies;      /* trunc(vol / rep_radius^3) -- 1718 for the defaults   */

/* ---- run control ---- */
int niter;
int write_output_file;

/* Number of eddies that fit in the box, as oSEM computes it. */
inline void calc_eddies(int &n, const double &v, const double &rep) {
  n = (int)trunc(v / (rep * rep * rep));
}

#endif /* _SEM_CONSTANTS_H_ */
