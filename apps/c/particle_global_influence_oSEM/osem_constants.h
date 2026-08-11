/*
 * osem_constants.h  --  every constant and run option, in the oSEM style
 *
 * DEFINITIONS, not declarations. Include this from the driver translation unit
 * ONLY. The kernels reference these as bare globals; for the grid kernels the
 * OPS translator emits matching `extern` declarations into its generated files
 * (see apps/c/oSEM/mpi_openmp/mpi_openmp_kernels.cpp, which carries
 * `extern double u0;`). Particle kernels are resolved by C++ templates in the
 * driver's own TU, so they see these directly.
 *
 * Nothing here has a value. Values are assigned at the top of main, so the
 * whole configuration of a run reads top-to-bottom in one place -- oSEM's
 * arrangement, and the reason this file is only declarations.
 *
 * Not everything below is registered with ops_decl_const. Only the subset the
 * KERNELS read needs to be, and that is what makes those visible to
 * translator-generated and device code. The rest are here for locality: one
 * place to look for anything that configures a run.
 *
 * Groups, in the order a run uses them:
 *   1. flow            physical inputs
 *   2. inlet plane     the 2-D face where u',v',w' are wanted
 *   3. eddy box        the 3-D slab around it, derived from r_max
 *   4. eddy properties per-eddy quantities
 *   5. derived         computed in main from the above
 *   6. run options     driver only, no kernel reads these
 */

#ifndef _OSEM_CONSTANTS_H_
#define _OSEM_CONSTANTS_H_

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
 * The 3-D slab the eddies live in. Every bound is set by r_max, because an
 * eddy further than that from the plane contributes exactly zero: x spans
 * +/- r_max about the plane, and y/z are the inlet padded by r_max so eddies
 * just outside still reach nodes near the edge.
 */

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

/* ---- 6. run options -------------------------------------------------- *
 * Driver only -- no kernel reads these, so none is registered with
 * ops_decl_const. oSEM keeps niter, write_output_file and seed_gbl alongside
 * the physical constants too.
 */

unsigned int seed_gbl;  /* changing it changes the whole realisation        */
int niter;
int nprint;             /* report + HDF5 frame interval; <= 0 = both off    */
int rng_method;         /* one of ops_particle_rng_method                    */
int ntbl;               /* points in the tabulated RST profile (TBL_data.h) */
int use_tbl;            /* 1 = boundary-layer profile, 0 = isotropic        */

#endif /* _OSEM_CONSTANTS_H_ */
