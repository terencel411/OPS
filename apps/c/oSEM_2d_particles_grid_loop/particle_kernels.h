#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

void KerInitEddy(ACCP<double> &eddy_x, ACCP<double> &eddy_r, ACCP<double> &eddy_eps,
                 const ACCP<double> &eddy_rng) {

  eddy_x(0) = x_min + eddy_rng(0) * (x_max - x_min);
  eddy_r(0) = eddy_radius;

  eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
  eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
  eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
}

// oSEM_2d_particles' KerConvectEddies with the (y,z) respawn removed, so nothing
// migrates. Deliberately not oSEM's physics -- see README.md.
void KerConvectEddies(ACCP<double> &eddy_pos, ACCP<double> &eddy_x, ACCP<double> &eddy_r,
                      ACCP<double> &eddy_eps, const ACCP<double> &eddy_rng) {

  (void)eddy_pos;                       // frozen for the life of the run

  eddy_x(0) += increment;

  if (eddy_x(0) > x_max) {
    eddy_x(0) = x_min;
    eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
    eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
    eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
    eddy_r(0) = eddy_radius;
  }
}

#endif /* _PARTICLE_KERNELS_H_ */
