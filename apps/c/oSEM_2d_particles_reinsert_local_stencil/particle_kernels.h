#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

// increments eddy in x direction, marks the eddy for deletion when it crosses x_max
void KerConvectEddies(ACCP<double> &eddy_x, ACCP<int> &eddy_exit) {

  eddy_x(0) += increment;
  eddy_exit(0) = (eddy_x(0) > x_max) ? 1 : 0;
}

// counts the number of eddies that crosses x_max and finds the largest particle id (i.e. last particle inserted)
void KerCountExits(const ACCP<int> &eddy_exit, const ACCP<int> &eddy_id,
                   int *np, int *max_id) {

  *np += eddy_exit(0);
  if (eddy_id(0) > *max_id) *max_id = eddy_id(0);
}

#endif /* _PARTICLE_KERNELS_H_ */
