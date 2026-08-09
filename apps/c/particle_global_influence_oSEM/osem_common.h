/*
 * osem_common.h  --  shared definitions for the particle-based oSEM
 *
 * Included by both kernel headers and the driver, so it must come first.
 */

#ifndef _OSEM_COMMON_H_
#define _OSEM_COMMON_H_

#include <cmath>

/* ------------------------------------------------------------------ *
 * The gather buffer
 * ------------------------------------------------------------------ *
 * NCOMP doubles per eddy, indexed by GLOBAL eddy id. This is the array that
 * replaces oSEM's seven separate x_gbl/y_gbl/z_gbl/r_gbl/eps_*_gbl host
 * arrays. One buffer instead of seven, because one reduction is cheaper than
 * seven and the layout keeps a single eddy's data together.
 *
 * The signs are +1/-1 integers in oSEM; they ride here as doubles, which is
 * exact for values that small and saves a second collective of a different
 * type.
 */
#define NCOMP 7
#define E_X 0   /* streamwise position -- NOT part of the particle position */
#define E_Y 1
#define E_Z 2
#define E_R 3   /* eddy radius                                             */
#define E_SX 4  /* eps_x                                                   */
#define E_SY 5  /* eps_y                                                   */
#define E_SZ 6  /* eps_z                                                   */

/* ------------------------------------------------------------------ *
 * Run parameters, passed into kernels as one ops_arg_gbl array
 * ------------------------------------------------------------------ *
 * oSEM puts these in ops_decl_const globals. A parameter block keeps the
 * kernel signatures short and, more usefully, keeps every kernel reading the
 * same numbers the driver computed -- no second copy to drift out of step.
 */
#define P_XMIN 0
#define P_XMAX 1
#define P_XPLANE 2
#define P_EYMIN 3
#define P_EYMAX 4
#define P_EZMIN 5
#define P_EZMAX 6
#define P_RADIUS 7   /* 0.2 * delta, the eddy radius                       */
#define P_INCREMENT 8 /* u0 * dt, the per-step convection distance         */
#define P_SHAPENORM 9 /* the 1/1.5829045 normalisation from oSEM           */
#define P_U0TI 10     /* u0 * TI, the diagonal of the Cholesky RST         */
#define P_VTY 11      /* transverse speed scale in y, see KerConvectEddies */
#define P_VTZ 12      /* transverse speed scale in z                       */
#define NPARAM 13

/* ------------------------------------------------------------------ *
 * Randoms
 * ------------------------------------------------------------------ *
 * See ops_particle_random.h. The randoms are filled into a particle dat by the
 * driver before each kernel, the same shape of call as oSEM's
 * ops_fill_random_uniform, and keyed on the particle's GLOBAL ID so an eddy's
 * stream belongs to the eddy rather than to whichever rank owns it.
 *
 * This file used to carry a per-eddy LCG advanced inside the kernels. That is
 * gone: the counter-based generator needs no state at all, so there is no
 * stream ordering to get wrong -- which is what caused the position/sign
 * correlation documented in the README.
 */

#endif /* _OSEM_COMMON_H_ */
