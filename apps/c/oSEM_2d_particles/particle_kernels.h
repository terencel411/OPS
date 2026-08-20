/*
 * particle_kernels.h -- the eddies, as ops_particle_par_loop kernels. The
 * particle position is (y, z), the inlet plane, which is also what OPS
 * decomposes; the streamwise x rides as an ordinary particle dat.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

// instantiate_eddies, minus the (y,z) position, which seed_eddies has already
// set on the host. Each uniform is a separate hash of (seed, gid, counter,
// component), not successive states of one stream -- see the README.
void KerInitEddy(ACCP<double> &eddy_x, ACCP<double> &eddy_r, ACCP<double> &eddy_eps,
                 ACCP<double> &eddy_vt, const ACCP<double> &eddy_rng) {

  eddy_x(0) = x_min + eddy_rng(0) * (x_max - x_min);
  eddy_r(0) = eddy_radius;

  // Transverse velocity, drawn once and kept: this replaces oSEM's
  // re-randomisation, so the eddy drifts to a new (y,z) instead of jumping.
  eddy_vt(0) = vt_y * (2.0 * eddy_rng(1) - 1.0);
  eddy_vt(1) = vt_z * (2.0 * eddy_rng(2) - 1.0);

  eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
  eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
  eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
}

// convect_eddies, with continuous re-injection in place of oSEM's jump to a new
// random (y,z), which OPS migration cannot express. The eddy drifts and reflects
// off the box faces; the signs are still re-drawn on recycle. README has the why.
void KerConvectEddies(ACCP<double> &eddy_pos, ACCP<double> &eddy_x, ACCP<double> &eddy_r,
                      ACCP<double> &eddy_eps, ACCP<double> &eddy_vt,
                      const ACCP<double> &eddy_rng) {

  eddy_x(0) += increment;

  // Transverse drift, reflected at the box faces: a small local correction,
  // never a jump.
  eddy_pos(0) += eddy_vt(0);
  eddy_pos(1) += eddy_vt(1);

  if (eddy_pos(0) < eddy_y_min) { eddy_pos(0) = 2.0 * eddy_y_min - eddy_pos(0); eddy_vt(0) = -eddy_vt(0); }
  if (eddy_pos(0) > eddy_y_max) { eddy_pos(0) = 2.0 * eddy_y_max - eddy_pos(0); eddy_vt(0) = -eddy_vt(0); }
  if (eddy_pos(1) < eddy_z_min) { eddy_pos(1) = 2.0 * eddy_z_min - eddy_pos(1); eddy_vt(1) = -eddy_vt(1); }
  if (eddy_pos(1) > eddy_z_max) { eddy_pos(1) = 2.0 * eddy_z_max - eddy_pos(1); eddy_vt(1) = -eddy_vt(1); }

  if (eddy_x(0) > x_max) {
    eddy_x(0) = x_min;
    eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
    eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
    eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
    eddy_r(0) = eddy_radius;
  }
}

// The gather, in place of oSEM's seven ops_dat_fetch_data calls. Must be
// ITERATE_LOCAL: a ghost eddy carries its owner's id and would be added twice.
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

// Population check: eddies recycle but are never created or destroyed.
void KerCountEddies(const ACCP<int> &eddy_id, int *count) {
  (void)eddy_id;
  *count += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
