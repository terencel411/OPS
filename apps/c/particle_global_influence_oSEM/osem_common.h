/*
 * osem_common.h  --  shared definitions for the particle-based oSEM
 *
 * Included by both kernel headers and the driver, so it must come first.
 */

#ifndef _OSEM_COMMON_H_
#define _OSEM_COMMON_H_

/* The kernel headers use sqrt/fabs/exp and include nothing themselves, so this
   is where they get it -- both in the driver TU and in the translator's
   generated kernel files. */
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

/* Run parameters are NOT here. An earlier version passed them into the kernels
   as one ops_arg_gbl `prm[NPARAM]` array; they are ops_decl_const globals now,
   declared in osem_constants.h and given their values at the top of main, the
   same arrangement oSEM uses. The P_* index block that went with the old
   scheme is gone. */

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
