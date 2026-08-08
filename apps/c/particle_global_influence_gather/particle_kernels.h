/*
 * particle_kernels.h  --  all-to-all particle influence
 *
 * Kernel A : KerInitState        -- initial velocity (a swirl) and influence
 * Kernel B : KerAdvance          -- constant acceleration, forward Euler
 * Kernel C : KerInfluence        -- read the gathered buffer and sum the
 *                                   contribution of EVERY particle
 * Check    : KerCheckInfluence   -- compare against a host reference
 *
 * There is no publish kernel here. The gather is MPI_Allgatherv, packed on the
 * host from the dats' contiguous storage -- see gather_state_mpi() in
 * influence_gather.cpp.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

#include <cmath>

/* ------------------------------------------------------------------ *
 * Layout of the global gather buffer
 * ------------------------------------------------------------------ *
 * NCOMP doubles per particle, indexed by GLOBAL id:
 *
 *     all[NCOMP*gid + C_X] ... all[NCOMP*gid + C_M]
 *
 * The interaction law only needs position and strength; velocity rides
 * along because the gather is also what feeds HDF5 output (the particle
 * API has no HDF5 path of its own), and quiver arrows want it. Put
 * whatever your interaction needs here -- that is the point.
 */
#define NCOMP 5
#define C_X 0
#define C_Y 1
#define C_VX 2
#define C_VY 3
#define C_M 4

/* ------------------------------------------------------------------ *
 * Kernel A -- initial state
 * ------------------------------------------------------------------ *
 * Positions come from seeding (they have to: the particle count is set
 * there), everything else is initialised here, in a kernel.
 *
 * The initial velocity is a rigid-body swirl about the domain centre,
 *
 *     v = omega * ( -(y - cy),  (x - cx) )
 *
 * with no centripetal force to hold it, so the blob rotates AND spreads
 * outward as it drifts. That matters for more than looks: under a rigid
 * translation every inter-particle distance is constant, so the influence
 * would be frozen for the whole run and there would be nothing to watch.
 * A deforming cloud makes phi genuinely time-dependent.
 */
void KerInitState(ACCP<double> &up, ACCP<double> &phi, const ACCP<double> &xp,
                  const double *omega, const double *centre) {
  up(0) = -(*omega) * (xp(1) - centre[1]);
  up(1) = (*omega) * (xp(0) - centre[0]);
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
    const double dx = x - all[NCOMP * j + C_X];
    const double dy = y - all[NCOMP * j + C_Y];
    sum += all[NCOMP * j + C_M] / (*eps + sqrt(dx * dx + dy * dy));
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
