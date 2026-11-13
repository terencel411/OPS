/*
 * grid_kernels.h
 *
 *  Created on: Oct 19, 2026
 *      Author: valantis
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

#include <cmath>

void KernelInitGrid(ACC<double>& xgrid, double *dx, int *idx) {

  xgrid(0, 0, 0) = idx[0] * dx[0];
  xgrid(1, 0, 0) = idx[1] * dx[1];
}

void KernelInitNonUnf(ACC<double>& xgrid, double *dx, int *idx) {

  xgrid(1, 0, 0) = idx[1] * dx[1];

  xgrid(0, 0, 0) = sinh(4. * (dx[0] * idx[0]))/ sinh(4.);
}

void KerPartGridInter(ACCP<double> &xp, ACC<double> &xf) {

  printf(" xp [%12.9e %12.9e] interacts with grid [%12.9e %12.9e]\n",
         xp(0), xp(1), xf(0, 0, 0), xf(1, 0, 0));
}


void KerGridPartInter(ACC<double>&xf, ACCP<double>& xp) {

  printf("R %d xp [%12.9e %12.9e] interacts with grid [%12.9e %12.9e]\n",
         ops_get_proc(),
         xp(0), xp(1), xf(0, 0, 0), xf(1, 0, 0));
}

#endif /* APPS_C_TESTING_MULTIGRID_MAPS_GRID_KERNELS_H_ */
