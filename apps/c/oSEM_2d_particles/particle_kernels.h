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

void KerConvectEddies(ACCP<double> &eddy_yz, ACCP<double> &eddy_x, ACCP<double> &eddy_r,
                      ACCP<double> &eddy_eps, const ACCP<double> &eddy_rng) {

  eddy_x(0) += increment;

  if (eddy_x(0) > x_max) {
    eddy_x(0) = x_min;
    eddy_yz(0) = eddy_y_min + (eddy_y_max - eddy_y_min) * eddy_rng(1);
    eddy_yz(1) = eddy_z_min + (eddy_z_max - eddy_z_min) * eddy_rng(2);
    eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
    eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
    eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
    eddy_r(0) = eddy_radius;
  }
}

void KerGatherEddies(const ACCP<double> &eddy_yz, const ACCP<double> &eddy_x,
                    const ACCP<double> &eddy_r, const ACCP<double> &eddy_eps,
                    const ACCP<int> &eddy_id, double *eddy_all) {
  const int s = NCOMP * eddy_id(0);
  eddy_all[s + E_X] += eddy_x(0);
  eddy_all[s + E_Y] += eddy_yz(0);
  eddy_all[s + E_Z] += eddy_yz(1);
  eddy_all[s + E_R] += eddy_r(0);
  eddy_all[s + E_SX] += eddy_eps(0);
  eddy_all[s + E_SY] += eddy_eps(1);
  eddy_all[s + E_SZ] += eddy_eps(2);
}

// check if eddy counts from all ranks are same
void KerCountEddies(const ACCP<int> &eddy_id, int *count) {
  (void)eddy_id;
  *count += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
