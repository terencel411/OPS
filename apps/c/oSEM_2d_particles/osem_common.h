/*
 * osem_common.h -- the eddy gather buffer layout. Included by both kernel
 * headers and the driver, so it comes first.
 */

#ifndef _OSEM_COMMON_H_
#define _OSEM_COMMON_H_

// NCOMP doubles per eddy, indexed by global eddy id. One buffer in place of
// oSEM's seven x_gbl/y_gbl/... host arrays. Signs ride as doubles: exact here.
#define NCOMP 7
#define E_X 0   /* streamwise position -- NOT part of the particle position */
#define E_Y 1
#define E_Z 2
#define E_R 3   /* eddy radius                                             */
#define E_SX 4  /* eps_x                                                   */
#define E_SY 5  /* eps_y                                                   */
#define E_SZ 6  /* eps_z                                                   */

#endif /* _OSEM_COMMON_H_ */
