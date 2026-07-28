/*
 * particle_kernels.h  --  OPS Particles tutorial 3
 *
 * Particle-only kernels. ACCP<T>& is the particle accessor:
 *     up(component)   -- no spatial offsets, a particle is a point
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

/* Constant-velocity drift, exactly as in tutorial 1. Particles move so that
   the deposit loop is exercised with a changing bin population, migration and
   ghost particles -- not just a static configuration. */
void KerUpdatePosition(ACCP<double> &xp, const ACCP<double> &up,
                       const double *dt) {
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

#endif /* _PARTICLE_KERNELS_H_ */
