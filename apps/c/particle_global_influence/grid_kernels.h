/*
 * grid_kernels.h  --  all-to-all particle influence
 *
 * Ordinary OPS grid kernels. The grid exists only to give the bounding box
 * and the particle mapping a coordinate reference; it carries no physics.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

void KerInitGrid(ACC<double> &xf, const double *dx, const int *idx) {
  xf(0, 0, 0) = (*dx) * static_cast<double>(idx[0]);
  xf(1, 0, 0) = (*dx) * static_cast<double>(idx[1]);
}

#endif /* _GRID_KERNELS_H_ */
