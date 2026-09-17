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

// increment x: eddy past x_max is marked for deletion
void KerConvectEddies(ACCP<double> &eddy_x, ACCP<int> &eddy_exit) {

  eddy_x(0) += increment;
  eddy_exit(0) = (eddy_x(0) > x_max) ? 1 : 0;
}

void KerGatherEddies(const ACCP<double> &eddy_pos, const ACCP<double> &eddy_x,
                     const ACCP<double> &eddy_r, const ACCP<double> &eddy_eps,
                     const ACCP<int> &eddy_id, double *eddy_all) {
  const int s = NCOMP * eddy_id(0);
  eddy_all[s + E_X] += eddy_x(0);
  eddy_all[s + E_Y] += eddy_pos(0);
  eddy_all[s + E_Z] += eddy_pos(1);
  eddy_all[s + E_R] += eddy_r(0);
  eddy_all[s + E_SX] += eddy_eps(0);
  eddy_all[s + E_SY] += eddy_eps(1);
  eddy_all[s + E_SZ] += eddy_eps(2);
}

// test check: copies of each gid, plus (last slot) owned eddies outside this rank's box
void KerCheckEddies(const ACCP<double> &eddy_pos, const ACCP<int> &eddy_id,
                    const double *box, int *count) {
  count[eddy_id(0)] += 1;
  if (eddy_pos(0) < box[0] || eddy_pos(0) >= box[2] ||
      eddy_pos(1) < box[1] || eddy_pos(1) >= box[3])
    count[eddies] += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
