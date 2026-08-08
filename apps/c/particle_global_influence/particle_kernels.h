/*
 * particle_kernels.h  --  all-to-all particle influence
 *
 * Kernel A : KerInitState      -- initialise velocity / influence
 * Kernel B : KerAdvance        -- constant acceleration, forward Euler
 * Kernel C1: KerPublishState   -- scatter this rank's particles into the
 *                                 global reduction buffer
 * Kernel C2: KerInfluence      -- read the (now global) buffer and sum the
 *                                 contribution of EVERY particle
 * Check    : KerCheckInfluence -- compare against a host reference
 *
 * C1 and C2 are the whole trick. See influence.cpp section 12.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

#include <cmath>

/* Components gathered per particle in the global buffer: x, y, strength.
   Shared between the kernels and the driver so the layout is stated once. */
#define NCOMP 3

/* ------------------------------------------------------------------ *
 * Kernel A -- initial state
 * ------------------------------------------------------------------ *
 * Positions come from seeding (they have to: the particle count is set
 * there), everything else is initialised here, in a kernel, exactly as
 * you would in a real application.
 */
void KerInitState(ACCP<double> &up, ACCP<double> &phi) {
  up(0) = 0.0;
  up(1) = 0.0;
  phi(0) = 0.0;
}

/* ------------------------------------------------------------------ *
 * Kernel B -- advance
 * ------------------------------------------------------------------ *
 * v <- v + a*dt ;  x <- x + v*dt
 *
 * Purely local: a particle's new state depends only on its own dats, so
 * every rank can do this for the particles it owns with no communication
 * at all. This is the "easy" half of the problem.
 */
void KerAdvance(ACCP<double> &xp, ACCP<double> &up, const double *a,
                const double *dt) {
  up(0) += a[0] * (*dt);
  up(1) += a[1] * (*dt);
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

/* ------------------------------------------------------------------ *
 * Kernel C1 -- publish
 * ------------------------------------------------------------------ *
 * `all` is an ops_arg_reduce buffer of NPART*NCOMP doubles, declared
 * OPS_INC, so OPS has zeroed it before this loop and will MPI_Allreduce
 * it (SUM, over all ranks) when ops_reduction_result() is called.
 *
 * Each particle owns exactly one slot, picked out by its GLOBAL id, and
 * writes into it with +=. Because every other rank contributes 0.0 to
 * that slot, the sum is the value itself -- bit for bit. A sum-reduction
 * over a disjointly-filled array IS an allgather.
 *
 * Two things make this correct and both are easy to get wrong:
 *   - the loop must be OPS_PARTICLE_ITERATE_LOCAL. A ghost (virtual)
 *     copy has the same gid as its owner, so iterating ALL would add the
 *     same particle in twice and every position would come out doubled.
 *   - gid must be globally unique and dense in [0, NPART).
 */
void KerPublishState(const ACCP<double> &xp, const ACCP<double> &mp,
                     const ACCP<int> &gid, double *all) {
  const int s = NCOMP * gid(0);
  all[s + 0] += xp(0);
  all[s + 1] += xp(1);
  all[s + 2] += mp(0);
}

/* ------------------------------------------------------------------ *
 * Kernel C2 -- the all-to-all influence
 * ------------------------------------------------------------------ *
 * By the time this runs, `all` has been through ops_reduction_result()
 * and holds the state of EVERY particle in the domain, on EVERY rank.
 * It arrives here as a plain read-only ops_arg_gbl, so the kernel simply
 * loops over the whole population:
 *
 *     phi_i = sum_{j != i}  m_j / (eps + |r_i - r_j|)
 *
 * Nearest particle contributes most, farthest contributes least, nothing
 * is truncated -- which is exactly the requirement. eps is a softening
 * length that keeps the sum finite if two particles coincide.
 *
 * Cost is O(N) per particle. That is inherent to an all-to-all, not to
 * this implementation.
 */
void KerInfluence(ACCP<double> &phi, const ACCP<double> &xp,
                  const ACCP<int> &gid, const double *all, const int *n,
                  const double *eps) {

  const int me = gid(0);
  const double x = xp(0);
  const double y = xp(1);

  double sum = 0.0;
  for (int j = 0; j < *n; j++) {
    if (j == me) continue; /* a particle does not influence itself */
    const double dx = x - all[NCOMP * j + 0];
    const double dy = y - all[NCOMP * j + 1];
    sum += all[NCOMP * j + 2] / (*eps + sqrt(dx * dx + dy * dy));
  }

  phi(0) = sum;
}

/* ------------------------------------------------------------------ *
 * Check kernel
 * ------------------------------------------------------------------ *
 * `ref` is the host-computed reference influence for all NPART
 * particles, indexed by gid. worst/count are ops_arg_reduce handles, so
 * the verification is itself collective without a line of MPI in the
 * application.
 */
void KerCheckInfluence(const ACCP<double> &phi, const ACCP<int> &gid,
                       const double *ref, double *worst, int *count) {

  const double r = ref[gid(0)];
  double d = fabs(phi(0) - r);
  if (fabs(r) > 1.0) d /= fabs(r); /* relative once the value is O(1) */

  if (d > *worst) *worst = d;
  *count += 1;
}

#endif /* _PARTICLE_KERNELS_H_ */
