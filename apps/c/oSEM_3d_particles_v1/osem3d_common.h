/*
 * osem3d_common.h -- the eddy gather buffer layout
 *
 * Included by the driver and by opensbliblock00_kernels.h, so it must come
 * first.
 */

#ifndef _OSEM3D_COMMON_H_
#define _OSEM3D_COMMON_H_

/* NCOMP doubles per eddy, indexed by GLOBAL eddy id. This one buffer replaces
   oSEM_3d's seven eddy_x_gbl/eddy_y_gbl/.../eddy_eps_z_gbl host arrays: one
   reduction is cheaper than seven, and it keeps an eddy's data together.

   The signs are +1/-1 ints in oSEM_3d and ride here as doubles, which is exact
   at that magnitude and saves a second collective of a different type. */
#define NCOMP 7
#define E_X 0  /* streamwise position, measured from the inlet plane */
#define E_Y 1
#define E_Z 2
#define E_R 3  /* eddy radius */
#define E_SX 4 /* eps_x */
#define E_SY 5 /* eps_y */
#define E_SZ 6 /* eps_z */

/* ------------------------------------------------------------------ *
 * Capacities for the two inlet-profile gathers
 * ------------------------------------------------------------------ *
 * The profiles are gathered by the same OPS_INC reduction the eddies use, but
 * from grid ops_par_loops, which the translator parses -- and it requires the
 * reduction dimension to be an integer LITERAL (ops-translator/cpp/parser.py:
 * 188-207, parseIntLiteral accepts only INTEGER_LITERAL). ny and y_cutoff are
 * runtime values, so the reductions are declared at a fixed capacity and only
 * the first ny / 4*y_cutoff slots are used; the rest stay zero. main() checks
 * the run fits.
 *
 * y_cutoff is capped at 150 by main(), so 4*150 is exact rather than generous.
 */
#define UINTERP_CAP 4096
#define RST_CAP 600

#endif /* _OSEM3D_COMMON_H_ */
