#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

// increment x: eddy past x_max is marked for deletion
void KerConvectEddies(ACCP<double> &eddy_x, ACCP<int> &eddy_exit) {

  eddy_x(0) += increment;
  eddy_exit(0) = (eddy_x(0) > x_max) ? 1 : 0;
}

#endif /* _PARTICLE_KERNELS_H_ */
