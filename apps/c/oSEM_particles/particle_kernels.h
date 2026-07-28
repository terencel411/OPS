/*
 * particle_kernels.h -- oSEM on OPS Particles
 *
 * ACCP<T>& is the particle accessor: up(component). A particle is a point, so
 * there are no spatial offsets.
 *
 * THE DATA LAYOUT, AND WHY IT WORKS
 * ---------------------------------
 * Eddies live in 3-D (x, y, z) but the inlet plane is 2-D (y, z). That is not
 * a problem, because x is not a SEARCH dimension: eddies convect in x and x
 * only ever enters the shape function. So
 *
 *     position dat = (y, z)      2-D, matching the block
 *     x            = an ordinary particle dat, like radius or eps
 *
 * and all the binning, migration and ghost-particle machinery operates in the
 * plane -- exactly where the neighbour search needs to happen.
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

/* ------------------------------------------------------------------ *
 *  Convection and recycling
 * ------------------------------------------------------------------ *
 * Same rule as oSEM's convect_eddies(): march in x, and on leaving the box
 * respawn at x_min with a fresh (y,z) and fresh signs.
 *
 * The random numbers come from an LCG state carried BY THE PARTICLE, so the
 * stream is a property of the eddy rather than of the rank. It migrates with
 * the eddy, which makes results independent of how many ranks are used and of
 * how eddies happen to be distributed. Nothing is fetched to the host and
 * nothing is gathered.
 *
 * NOTE: the respawn TELEPORTS the eddy to an unrelated (y,z). That is far
 * larger than any map skin, so the caller must take the full rebuild branch of
 * the migration cycle on any step where a respawn may have happened;
 * ops_particle_update_map_lists_actual_hybrid() decides this collectively.
 */
void KerConvectEddy(ACCP<double> &px, ACCP<double> &ppos, ACCP<double> &pr,
                    const ACCP<double> &pinc, ACCP<int> &peps,
                    ACCP<int> &pseed, const int *jump,
                    const double *lbox) {
  px(0) += pinc(0);

  if (px(0) > x_max) {
    /* UNSIGNED, deliberately. Signed overflow is undefined behaviour, and an
       LCG overflows on essentially every step: written with `int` and built at
       -O3 the compiler folds the recurrence and every eddy respawns at the
       same point. Unsigned arithmetic has well-defined wraparound. The state
       is masked to 31 bits so it round-trips through the int dat unchanged. */
    unsigned int s = (unsigned int)pseed(0);

    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double ynew = eddy_y_min + (eddy_y_max - eddy_y_min) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double znew = eddy_z_min + (eddy_z_max - eddy_z_min) * ((double)s / 2147483648.0);
    /* jump == 2 : respawn anywhere in the GLOBAL eddy box (textbook SEM).
       jump == 1 : respawn anywhere in THIS RANK's subdomain (lbox).
       jump == 0 : do not move at all (diagnostic).
       See the note on migration range in sem.cpp. */
    if (*jump == 2) { ppos(0) = ynew; ppos(1) = znew; }
    else if (*jump == 1) {
      ppos(0) = lbox[0] + (lbox[1] - lbox[0]) * (ynew - eddy_y_min) / (eddy_y_max - eddy_y_min);
      ppos(1) = lbox[2] + (lbox[3] - lbox[2]) * (znew - eddy_z_min) / (eddy_z_max - eddy_z_min);
    }

    /* bit 8, not bit 0: the low bit of an LCG modulo a power of two has
       period 2, so its parity is not random at all. */
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  peps(0) = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  peps(1) = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu;  peps(2) = ((s >> 8) & 1u) ? 1 : -1;

    pseed(0) = (int)s;
    px(0)    = x_min;
    pr(0)    = 0.2 * delta;
  }
}

#endif /* _PARTICLE_KERNELS_H_ */
