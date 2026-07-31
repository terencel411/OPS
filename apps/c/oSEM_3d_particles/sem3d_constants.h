/*
 * sem3d_constants.h -- oSEM_3d eddies as OPS particles
 *
 * Geometry and eddy population, taken from ../oSEM_3d/opensbli.cpp:73, 112-121
 * so the two eddy fields describe the same physical problem.
 *
 * Plain globals. They are also registered with ops_decl_const() in main(),
 * because ops_decl_const variables are the only application state a
 * translator-generated kernel file can see -- Stage A reads none of them from a
 * kernel, but the convection kernel will.
 */

#ifndef _SEM3D_CONSTANTS_H_
#define _SEM3D_CONSTANTS_H_

/* ---- flow ----
 * oSEM_3d is non-dimensional. Eddies convect at the freestream velocity, which
 * is 1.0, so an eddy advances 1.0*dt per step
 * (opensbliblock00_kernels.h:51 -- eddy_increment = 1.0 * dt).
 */
double u0;          /* eddy convection velocity, 1.0                        */
double dt;          /* timestep, 0.025                                      */
double delta;       /* boundary layer thickness, 11.6973525411              */
double radius;      /* eddy length scale, 0.2 * delta -- note delta == 5*r   */

/* ---- the eddy box ----
 * A slab straddling the inlet plane x = 0, one eddy radius either side, and
 * overhanging the fluid domain by one radius in y and z as well so that eddies
 * just outside can still reach inlet nodes.
 */
double eddy_x_min, eddy_x_max;
double eddy_y_min, eddy_y_max;
double eddy_z_min, eddy_z_max;

/* ---- eddy population ---- */
double eddy_vol;
int    eddies;      /* trunc(eddy_vol / radius^3) -- 267 for the defaults    */

/* Base seed of the counter-based stream. Read inside KerInstantiateEddies, so
   unlike the rest of these it has to be an ops_decl_const in earnest. Signed
   because ops_decl_const has no unsigned int type; the kernel casts it back. */
int eddy_seed;

/* ---- the fluid inlet plane the eddies have to cover ----
 * z spans block0np2 * Delta2block0 = 150 * (40/150) and is periodic; the
 * wall-normal direction is resolved to delta. These bound the region Kernel030
 * evaluates, which is what the coverage diagnostic samples.
 */
double span_z;

/* Number of eddies that fit in the box, as oSEM_3d computes it
 * (opensbli.cpp:121). */
inline void calc_eddies(int &n, const double &v, const double &rep) {
  n = (int)trunc(v / (rep * rep * rep));
}

#endif /* _SEM3D_CONSTANTS_H_ */
