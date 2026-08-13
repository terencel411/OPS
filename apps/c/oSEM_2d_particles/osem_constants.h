/* osem_constants.h -- every constant and run option, in the oSEM style 
   DEFINITIONS, not declarations. Include this from the driver translation 
   unit ONLY. */

#ifndef _OSEM_CONSTANTS_H_
#define _OSEM_CONSTANTS_H_

#include <vector>       /* for targ, group 5 below                          */

/* ---- 1. flow ------------------------------------------------------ */

double u0;              /* convection speed; eddies are swept past at this  */
double dt;
double delta;           /* boundary-layer thickness -- sets every length    */
double ti;              /* turbulence intensity                             */

/* ---- 2. inlet plane ----------------------------------------------- *
 * The 2-D (y,z) face at x = x_plane where the fluctuations are wanted.
 * These are the PHYSICAL bounds, before the eddy box pads them by r_max.
 */

double y_min;
double y_max;
double z_min;
double z_max;
double x_plane;         /* where the plane sits in x; 0                     */

int ny;                 /* grid INTERVALS, not a physical size              */
int nz;

/* ---- 3. eddy box --------------------------------------------------- * 
   The 3-D slab the eddies live in. */

double r_max;           /* 0.41 * delta -- the eddy search radius           */

double x_min;           /* = -r_max                                         */
double x_max;           /* = +r_max                                         */

double eddy_y_min;      /* = y_min                                          */
double eddy_y_max;      /* = y_max + r_max                                  */
double eddy_z_min;      /* = z_min - r_max                                  */
double eddy_z_max;      /* = z_max + r_max                                  */

/* ---- 4. eddy properties -------------------------------------------- */

double eddy_radius;     /* 0.2 * delta                                      */
double increment;       /* u0 * dt -- streamwise distance moved per step     */
double vt_y;            /* transverse drift scale, see KerConvectEddies     */
double vt_z;

/* ---- 5. derived ----------------------------------------------------- */

double vol;             /* eddy box volume                                  */
int eddies;             /* = vol / eddy_radius^3, one per cube of that side */
double u0ti;            /* u0 * ti -- the diagonal of the Cholesky RST       */
double shape_norm;      /* 1/1.5829045, so the raw signal has unit variance */

/* The rms each component should have at each of the ny+1 wall-normal 
   stations, interleaved u,v,w -- sqrt of the tabulated diagonal Reynolds 
   stresses, zero under -rst iso. Sized and filled in main once ny is known, 
   then never written again, which is what lets it live here: it is derived 
   from the table, but it is as fixed for a run as anything else in this 
   file. */
std::vector<double> targ;

/* ---- 6. run options -------------------------------------------------- * 
   Driver only -- no kernel reads these, so none is registered with 
   ops_decl_const. */

unsigned int seed_gbl;  /* changing it changes the whole realisation        */
int niter;
int nprint;             /* report + HDF5 frame interval; <= 0 = both off    */
int rng_method;         /* one of ops_particle_rng_method                    */
int ntbl;               /* points in the tabulated RST profile (TBL_data.h) */
int use_tbl;            /* 1 = boundary-layer profile, 0 = isotropic        */

#endif /* _OSEM_CONSTANTS_H_ */
