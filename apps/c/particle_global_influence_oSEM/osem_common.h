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
 * Per-eddy random stream
 * ------------------------------------------------------------------ *
 * oSEM draws its randoms with ops_fill_random_uniform into int dats on the
 * eddy block, once per quantity per timestep. That is not portable here:
 * eddies are particles, they migrate between ranks, and a rank-indexed random
 * dat would hand a migrating eddy somebody else's stream.
 *
 * Instead each eddy carries its own LCG state as a particle dat. The stream
 * then belongs to the eddy, travels with it across ranks, and is identical at
 * any rank count -- which the grid-dat version cannot be.
 *
 * It also sidesteps a known defect: ops_fill_random_uniform on an int dat
 * never returns a negative value, so oSEM's `(eps_rng < 0) ? -1 : 1` sign
 * draws are ALWAYS +1, and the eddy positions only ever fill the positive
 * octant. Here the sign comes from the top bit of the LCG state, which is
 * genuinely two-valued.
 *
 * Constants are the Numerical Recipes LCG rather than oSEM's a=5, c=3,
 * m=2^29: that generator has a very short period in its low bits and would
 * make the recycled eddies visibly correlated.
 */
static inline unsigned int lcg_next(unsigned int s) {
  return 1664525u * s + 1013904223u;
}

/* Output mixer (the murmur3-style finalizer).
 *
 * THIS IS LOAD-BEARING, not hygiene. A bare LCG state is a deterministic
 * function of the previous one, so consecutive draws are strongly related. The
 * kernels here draw an eddy's position and then its signs from consecutive
 * states, which without mixing makes eps_x a deterministic function of x -- and
 * because compute_fluct selects eddies by x (the |x| < r test), it then selects
 * a BIASED set of eps_x.
 *
 * Measured before this was added: rms u' came out 8.85-9.79 across realisations
 * while v' and w' sat at 4.8-6.7, when all three must be statistically equal
 * (a11 = a22 = a33, and each component is one sign times one shape). The signs
 * were individually unbiased -- mean eps was +0.007, -0.005, 0.000 -- so the
 * defect was purely the correlation with position, which no test of the signs
 * alone would have caught.
 *
 * oSEM does not need this because it draws each quantity from a separate
 * ops_fill_random_uniform call rather than from one running stream.
 */
static inline unsigned int lcg_mix(unsigned int z) {
  z ^= z >> 16;
  z *= 0x7feb352du;
  z ^= z >> 15;
  z *= 0x846ca68bu;
  z ^= z >> 16;
  return z;
}

/* Uniform in [0,1). */
static inline double lcg_unit(unsigned int s) {
  return (double)lcg_mix(s) * (1.0 / 4294967296.0);
}

/* +1 or -1, from the top bit of the mixed output. */
static inline int lcg_sign(unsigned int s) {
  return (lcg_mix(s) & 0x80000000u) ? -1 : 1;
}

#endif /* _OSEM_COMMON_H_ */
