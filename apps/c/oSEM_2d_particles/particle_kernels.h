/*
 * particle_kernels.h -- the eddies, as ops_particle_par_loop kernels using
 * ACCP<T>&. Counterparts of oSEM's instantiate_eddies and convect_eddies, plus
 * the publish/count pair the gather needs.
 *
 * The particle POSITION is (y, z), the inlet plane, which is also what OPS
 * decomposes. The streamwise x rides as an ordinary particle dat: nothing is
 * decomposed along it and no neighbour search uses it.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

/* instantiate_eddies. Positions (y,z) are seeded on the host, since the
   particle count has to be set there; everything else is drawn here from six
   pre-filled uniforms. Each component is a separate hash of
   (seed, gid, counter, component), not successive states of one stream -- see
   the position/sign correlation bug in the README. */
void KerInitEddy(ACCP<double> &px, ACCP<double> &pr, ACCP<double> &peps,
                 ACCP<double> &pvt, const ACCP<double> &rnd) {

  px(0) = x_min + rnd(0) * (x_max - x_min);
  pr(0) = eddy_radius;

  /* Transverse velocity, drawn once and kept: this is what replaces oSEM's
     re-randomisation, so the eddy drifts to a new (y,z) instead of jumping. */
  pvt(0) = vt_y * (2.0 * rnd(1) - 1.0);
  pvt(1) = vt_z * (2.0 * rnd(2) - 1.0);

  peps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  peps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  peps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
}

/* convect_eddies: one step of streamwise convection, plus the recycle when the
   eddy leaves the downstream face.

   CONTINUOUS RE-INJECTION, a deliberate change from oSEM. It recycles by
   assigning a new random (y,z); that is a jump to an arbitrary point, and OPS
   migration only reaches a neighbouring rank. Here the eddy drifts on a
   transverse velocity and reflects off the box faces instead. The SIGNS are
   still re-drawn on every recycle, which is what refreshes the statistics.
   README has the measurements. */
void KerConvectEddies(ACCP<double> &pos, ACCP<double> &px, ACCP<double> &pr,
                      ACCP<double> &peps, ACCP<double> &pvt,
                      const ACCP<double> &rnd) {

  px(0) += increment;

  /* Continuous transverse drift, with reflection off the box faces. A
     reflection is a small local correction, never a jump. */
  pos(0) += pvt(0);
  pos(1) += pvt(1);

  if (pos(0) < eddy_y_min) { pos(0) = 2.0 * eddy_y_min - pos(0); pvt(0) = -pvt(0); }
  if (pos(0) > eddy_y_max) { pos(0) = 2.0 * eddy_y_max - pos(0); pvt(0) = -pvt(0); }
  if (pos(1) < eddy_z_min) { pos(1) = 2.0 * eddy_z_min - pos(1); pvt(1) = -pvt(1); }
  if (pos(1) > eddy_z_max) { pos(1) = 2.0 * eddy_z_max - pos(1); pvt(1) = -pvt(1); }

  if (px(0) > x_max) {
    px(0) = x_min;
    peps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
    peps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
    peps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
    pr(0) = eddy_radius;
  }
}

/* The gather, in place of oSEM's seven ops_dat_fetch_data calls. OPS_INC into
   the slot the global id names; every other rank contributes 0.0, so the
   MPI_Allreduce is an allgather. Must be ITERATE_LOCAL: a ghost eddy carries
   its owner's id and would be added twice. */
void KerPublishEddy(const ACCP<double> &pos, const ACCP<double> &px,
                    const ACCP<double> &pr, const ACCP<double> &peps,
                    const ACCP<int> &gid, double *all) {
  const int s = NCOMP * gid(0);
  all[s + E_X] += px(0);
  all[s + E_Y] += pos(0);
  all[s + E_Z] += pos(1);
  all[s + E_R] += pr(0);
  all[s + E_SX] += peps(0);
  all[s + E_SY] += peps(1);
  all[s + E_SZ] += peps(2);
}

/* Population check: eddies recycle but are never created or destroyed. */
void KerCountEddies(const ACCP<int> &gid, int *count) {
  (void)gid;
  *count += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
