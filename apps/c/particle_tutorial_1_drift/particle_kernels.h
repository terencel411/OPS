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

#endif /* _PARTICLE_KERNELS_H_ */
