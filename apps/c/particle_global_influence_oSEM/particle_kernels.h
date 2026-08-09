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
 * Positions (y,z) are seeded on the host, because the particle count has to
 * be set there. Everything else is drawn here from the eddy's own LCG, in the
 * same order oSEM draws it: x, then the three signs.
 */
void KerInitEddy(ACCP<double> &px, ACCP<double> &pr, ACCP<double> &peps,
                 ACCP<double> &pvt, ACCP<int> &rng, const double *prm) {

  unsigned int s = (unsigned int)rng(0);

  s = lcg_next(s);
  px(0) = prm[P_XMIN] + lcg_unit(s) * (prm[P_XMAX] - prm[P_XMIN]);

  pr(0) = prm[P_RADIUS];

  /* A transverse velocity, drawn once and kept. This is what replaces the
     re-randomisation: instead of jumping to a new (y,z) on recycle, the eddy
     drifts there continuously. Uniform in [-1,1] times the scale. */
  s = lcg_next(s);
  pvt(0) = prm[P_VTY] * (2.0 * lcg_unit(s) - 1.0);
  s = lcg_next(s);
  pvt(1) = prm[P_VTZ] * (2.0 * lcg_unit(s) - 1.0);

  s = lcg_next(s);
  peps(0) = (double)lcg_sign(s);
  s = lcg_next(s);
  peps(1) = (double)lcg_sign(s);
  s = lcg_next(s);
  peps(2) = (double)lcg_sign(s);

  rng(0) = (int)s;
}

/* ------------------------------------------------------------------ *
 * Convection  <- convect_eddies
 * ------------------------------------------------------------------ *
 * One step of streamwise convection, and a recycle when the eddy leaves the
 * downstream face: it re-enters at x_min with a fresh random (y,z), a fresh
 * set of signs and a reset radius. Body for body, this is oSEM's kernel; only
 * the accessors and the source of the randoms differ.
 *
 * CONTINUOUS RE-INJECTION. oSEM recycles an exiting eddy by assigning it a new
 * random (y,z). That is a jump to an arbitrary point in the plane, and OPS
 * particle migration only hands a particle to a NEIGHBOURING rank -- measured
 * here to segfault from three ranks up (README).
 *
 * The key observation is that only ONE of the three coordinates is a problem.
 * x is not a spatial dimension of this block: it is an ordinary particle dat,
 * nothing is decomposed along it, so resetting x to x_min is free -- no
 * migration is involved at all. It is purely the (y,z) assignment that jumps.
 *
 * So the eddy is given a transverse VELOCITY instead. It drifts across the
 * plane continuously, a fraction of a cell per step, and reflects off the eddy
 * box faces. Every step is small and local, so migration only ever sees a move
 * into an adjacent subdomain -- exactly the regime particle_global_influence
 * lives in, and it works at every rank count.
 *
 * The statistical refreshment survives: the SIGNS are still re-drawn on every
 * recycle, and they are what randomises u',v',w'. The signs are not spatial, so
 * re-drawing them costs nothing. The positions decorrelate by drift instead of
 * by teleport, on a timescale set by P_VTY / P_VTZ (chosen so an eddy crosses
 * the box in roughly one flow-through time).
 */
void KerConvectEddies(ACCP<double> &pos, ACCP<double> &px, ACCP<double> &pr,
                      ACCP<double> &peps, ACCP<double> &pvt, ACCP<int> &rng,
                      const double *prm) {

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

    unsigned int s = (unsigned int)rng(0);
    s = lcg_next(s);
    peps(0) = (double)lcg_sign(s);
    s = lcg_next(s);
    peps(1) = (double)lcg_sign(s);
    s = lcg_next(s);
    peps(2) = (double)lcg_sign(s);

    pr(0) = prm[P_RADIUS];
    rng(0) = (int)s;
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
