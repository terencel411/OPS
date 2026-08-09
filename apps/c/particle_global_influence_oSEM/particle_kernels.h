/*
 * particle_kernels.h  --  the eddies
 *
 * In oSEM the eddies are grid dats on a second ops_block. Here they are OPS
 * particles, so these are ops_particle_par_loop kernels using ACCP<T>&.
 *
 *   KerInitEddy       <- oSEM instantiate_eddies
 *   KerConvectEddies  <- oSEM convect_eddies      (the "advance" kernel)
 *   KerPublishEddy    -- scatter into the global gather buffer
 *   KerCountEddies    -- population check, see the note on recycling
 *
 * The particle POSITION is (y, z) -- the inlet plane, which is also what OPS
 * decomposes. The streamwise coordinate x is carried as an ordinary particle
 * dat, because it is not a spatial dimension of this problem: nothing is
 * decomposed along it and no neighbour search uses it.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

/* ------------------------------------------------------------------ *
 * Initialisation  <- instantiate_eddies
 * ------------------------------------------------------------------ *
 * Positions (y,z) are seeded on the host, because the particle count has to be
 * set there. Everything else is drawn here.
 *
 * The randoms arrive PRE-FILLED in a dat, exactly as in oSEM: the driver calls
 * ops_fill_random_uniform_particle() before the loop, the kernel just reads.
 * `rnd` carries six independent uniforms per eddy:
 *
 *     0        x position
 *     1, 2     transverse velocity (y, z)
 *     3, 4, 5  the signs eps_x, eps_y, eps_z
 *
 * Independent is the operative word. Every component is a separate hash of
 * (seed, gid, counter, component), not a successive state of one stream, so
 * the position/sign correlation that skewed the earlier version cannot recur
 * (see ops_particle_random.h).
 */
void KerInitEddy(ACCP<double> &px, ACCP<double> &pr, ACCP<double> &peps,
                 ACCP<double> &pvt, const ACCP<double> &rnd,
                 const double *prm) {

  px(0) = prm[P_XMIN] + rnd(0) * (prm[P_XMAX] - prm[P_XMIN]);
  pr(0) = prm[P_RADIUS];

  /* Transverse velocity, drawn once and kept: this is what replaces oSEM's
     re-randomisation, so the eddy drifts to a new (y,z) instead of jumping. */
  pvt(0) = prm[P_VTY] * (2.0 * rnd(1) - 1.0);
  pvt(1) = prm[P_VTZ] * (2.0 * rnd(2) - 1.0);

  peps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  peps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  peps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
}

/* ------------------------------------------------------------------ *
 * Convection  <- convect_eddies
 * ------------------------------------------------------------------ *
 * One step of streamwise convection, plus the recycle when the eddy leaves the
 * downstream face. Body for body this is oSEM's kernel; what differs is the
 * accessors, the source of the randoms, and the transverse motion.
 *
 * CONTINUOUS RE-INJECTION. oSEM recycles by assigning a new random (y,z). That
 * is a jump to an arbitrary point in the plane, and OPS particle migration only
 * hands a particle to a NEIGHBOURING rank -- measured to fail from three ranks
 * up (README).
 *
 * Only one of the three coordinates was ever the problem. x is not a spatial
 * dimension of this block: it is an ordinary particle dat, nothing is
 * decomposed along it, so resetting x to x_min involves no migration at all.
 * It was purely the (y,z) assignment that jumped.
 *
 * So the eddy carries a transverse velocity and drifts, reflecting off the box
 * faces -- a fraction of a cell per step, always local. The statistical
 * refreshment survives because the part that matters is not spatial: the SIGNS
 * are still re-drawn on every recycle, and they are what randomises u',v',w'.
 *
 * `rnd` slots 3,4,5 hold this step's sign draws, pre-filled by the driver.
 */
void KerConvectEddies(ACCP<double> &pos, ACCP<double> &px, ACCP<double> &pr,
                      ACCP<double> &peps, ACCP<double> &pvt,
                      const ACCP<double> &rnd, const double *prm) {

  px(0) += prm[P_INCREMENT];

  /* Continuous transverse drift, with reflection off the box faces. A
     reflection is a small local correction, never a jump. */
  pos(0) += pvt(0);
  pos(1) += pvt(1);

  if (pos(0) < prm[P_EYMIN]) { pos(0) = 2.0 * prm[P_EYMIN] - pos(0); pvt(0) = -pvt(0); }
  if (pos(0) > prm[P_EYMAX]) { pos(0) = 2.0 * prm[P_EYMAX] - pos(0); pvt(0) = -pvt(0); }
  if (pos(1) < prm[P_EZMIN]) { pos(1) = 2.0 * prm[P_EZMIN] - pos(1); pvt(1) = -pvt(1); }
  if (pos(1) > prm[P_EZMAX]) { pos(1) = 2.0 * prm[P_EZMAX] - pos(1); pvt(1) = -pvt(1); }

  if (px(0) > prm[P_XMAX]) {
    px(0) = prm[P_XMIN];
    peps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
    peps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
    peps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
    pr(0) = prm[P_RADIUS];
  }
}

/* ------------------------------------------------------------------ *
 * The gather  -- this is what replaces ops_dat_fetch_data
 * ------------------------------------------------------------------ *
 * oSEM fetches its seven eddy dats into host arrays and passes them to
 * compute_fluct as ops_arg_gbl. Under MPI that is broken: ops_dat_fetch_data
 * copies only THIS RANK's slice of the decomposed eddy block, and writes it at
 * offset 0 (ops_mpi_rt_support.cpp:2125 computes a displacement and then never
 * uses it). compute_fluct then loops over the global eddy count and reads
 * uninitialised memory past the local part.
 *
 * Here the buffer is an ops_arg_reduce declared OPS_INC: every eddy writes
 * into the slot picked out by its global id, every other rank contributes 0.0
 * there, so the MPI_Allreduce inside ops_reduction_result is a bit-exact
 * allgather. Every rank ends up with the COMPLETE eddy list, in id order --
 * exactly the array shape compute_fluct already expects.
 *
 * Must be OPS_PARTICLE_ITERATE_LOCAL: a ghost eddy carries its owner's id and
 * would be added into the same slot twice.
 */
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

/* Population check. The eddy count is a conserved quantity in oSEM -- eddies
   recycle, they are never created or destroyed -- so anything other than the
   seeded total means the recycle teleport lost one. */
void KerCountEddies(const ACCP<int> &gid, int *count) {
  (void)gid;
  *count += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
