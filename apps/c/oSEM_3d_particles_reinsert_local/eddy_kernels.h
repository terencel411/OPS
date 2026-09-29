#ifndef _EDDY_KERNELS_H_
#define _EDDY_KERNELS_H_

// initialise the 3d grid for eddies
void KerInitGrid(ACC<double> &grid, const ACC<double> &x0, const ACC<double> &x1,
                   const ACC<double> &x2) {
  grid(0, 0, 0, 0) = x0(0, 0, 0);
  grid(1, 0, 0, 0) = x1(0, 0, 0);
  grid(2, 0, 0, 0) = x2(0, 0, 0);
}

// increments eddy in x direction, marks the eddy for deletion when it crosses eddy_x_max
void KerConvectEddies(ACCP<double> &eddy, ACCP<int> &eddy_exit) {

  eddy(0) = eddy(0) + increment;
  eddy_exit(0) = (eddy(0) > eddy_x_max) ? 1 : 0;
}

// counts the number of eddies that crosses eddy_x_max and finds the largest particle id (i.e. last particle inserted)
void KerCountExits(const ACCP<int> &eddy_exit, const ACCP<int> &eddy_id,
                   int *np, int *max_id) {

  *np += eddy_exit(0);
  if (eddy_id(0) > *max_id) *max_id = eddy_id(0);
}

#endif /* _EDDY_KERNELS_H_ */
