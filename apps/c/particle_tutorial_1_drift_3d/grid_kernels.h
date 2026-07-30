/*
 * grid_kernels.h  --  OPS Particles tutorial 1, 3D
 *
 * Ordinary OPS grid kernels. Nothing particle-specific here.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

/*
 * Fill the coordinate dat: node (i,j,k) sits at (i*dx, j*dy, k*dz).
 *
 * This is the ONLY grid computation in this tutorial. We need it because
 * ops_create_bounding_box() derives the physical extent of the domain from a
 * coordinate dat, and ops_decl_mapping() bins particles relative to that same
 * grid. The grid carries no physics in this app.
 *
 * ACC<double>& is the ordinary OPS grid accessor; in 3D it is
 *   xf(component, i_offset, j_offset, k_offset)
 * idx[] comes from ops_arg_idx() and holds the global index of this point.
 *
 * dx is a 3-vector rather than a scalar: the box need not be a cube, and
 * deriving each direction separately keeps the grid consistent with LX/LY/LZ
 * whatever they are set to.
 */
void KerInitGrid(ACC<double> &xf, const double *dx, const int *idx) {
  xf(0, 0, 0, 0) = dx[0] * static_cast<double>(idx[0]);
  xf(1, 0, 0, 0) = dx[1] * static_cast<double>(idx[1]);
  xf(2, 0, 0, 0) = dx[2] * static_cast<double>(idx[2]);
}

#endif /* _GRID_KERNELS_H_ */
