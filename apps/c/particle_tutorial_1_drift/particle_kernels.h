/*
 * particle_kernels.h  --  OPS Particles tutorial 1
 *
 * Particle kernels use ACCP<T>& instead of ACC<T>&.
 *
 *   ACC<T>&   grid accessor    ->  a(component, i_offset, j_offset)
 *   ACCP<T>&  particle accessor->  a(component)
 *
 * A particle accessor is already positioned on the current particle when your
 * kernel is called, so you only index the component. There is no (i,j) because
 * a particle has no grid index -- it has a position.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

/*
 * Give every particle the same constant velocity.
 * Called once, before the time loop.
 */
void KerSetVelocity(ACCP<double> &up, const double *v) {
  up(0) = v[0];
  up(1) = v[1];
}

/*
 * Forward Euler advection:  x <- x + u*dt
 *
 * Because u is constant here, after n steps a particle has moved exactly
 * n*u*dt from where it started. That makes the result trivially checkable,
 * which is the whole point of this tutorial.
 */
void KerUpdatePosition(ACCP<double> &xp, const ACCP<double> &up,
                       const double *dt) {
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

/*
 * Periodic re-injection in x, done by hand.
 *
 * SERIAL BUILDS ONLY. Under MPI this same assignment teleports a particle the
 * full width of the domain, and the migration cycle only hands particles to a
 * NEIGHBOURING rank: whenever the decomposition puts more than one rank
 * between the two edges the particle is silently lost or the map rebuild
 * segfaults. The MPI builds use a periodic particle halo instead -- see the
 * halo group declarations in drift.cpp.
 *
 * With no ranks there is no migration and none of that applies, so in serial
 * the direct wrap is both correct and the simplest thing that works. It is
 * also the only option: the particle halo path throws
 * "Particle Halo exchange must be set after block boxes are set" in the
 * single-node library (ops/c/src/sequential/ops_particle_host_single_node.cpp:1908).
 *
 * WHEN THIS RUNS MATTERS. It must be called after the position update and
 * BEFORE the map update: the map update is where OPS deletes particles that
 * lie outside the bounding box, so a particle that stepped past the right edge
 * has to be brought back inside while it is still only just outside.
 *
 * There is nothing here for y, and nothing anywhere else in this app either:
 * the drift is along x only, so a particle's y never changes and it can never
 * reach the top or bottom of the domain.
 */
void KerWrapX(ACCP<double> &xp, const double *lo, const double *hi) {

  const double Lx = hi[0] - lo[0];

  /* Half-open interval [lo,hi): OPS bins on the same convention, so a particle
     left sitting exactly on hi would be treated as outside. while() rather
     than if() costs nothing and stays correct for any dt.                   */
  while (xp(0) >= hi[0]) xp(0) -= Lx;
  while (xp(0) <  lo[0]) xp(0) += Lx;
}

#endif /* _PARTICLE_KERNELS_H_ */
