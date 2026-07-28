/*
 * grid_kernels.h  --  OPS Particles tutorial 3
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

/*
 * Fill the coordinate dat: node (i,j) sits at (i*dx, j*dx).
 * Same as tutorial 1 -- the bounding box and the mapping are derived from it.
 */
void KerInitGrid(ACC<double> &xf, const double *dx, const int *idx) {
  xf(0, 0, 0) = (*dx) * static_cast<double>(idx[0]);
  xf(1, 0, 0) = (*dx) * static_cast<double>(idx[1]);
}

/*
 * Zero the accumulator before each deposit sweep.
 *
 * This MUST be a separate loop. In the deposit loop below the kernel body runs
 * once per (grid node, nearby particle) pair rather than once per node, so it
 * can only accumulate -- there is no "first visit" at which to zero.
 */
void KerZeroDensity(ACC<double> &rho) { rho(0, 0) = 0.0; }

/* ------------------------------------------------------------------ *
 *  The scatter kernel -- the whole point of this tutorial
 * ------------------------------------------------------------------ *
 *
 * Body of the grid-outer / particle-inner loop
 * (ops_par_loop overload in ops_grid_part_seq_v2.h:562), whose structure is
 *
 *     for each grid node (i,j):
 *         for each offset s in map_stencil:
 *             for each particle in bin(node -> map + s):
 *                 kernel(grid args at (i,j), particle args at that particle)
 *
 * Grid dats may be WRITTEN here. That is specific to this loop form: it sets
 * halo dirtybits for non-OPS_READ args (ops_grid_part_seq_v2.h:527) and has no
 * read-only check. The other coupling form, ops_par_particle_grid_loop (used
 * in tutorial 2), enforces read-only grid args and would throw.
 *
 * UNCONDITIONAL variant: every particle contributes its weight at every
 * stencil point it is reachable from, with no geometric test at all. That
 * makes the total exactly
 *
 *     sum(rho) == N_particles * stencil_points
 *
 * an integer identity independent of grid spacing, particle positions and rank
 * count -- so it isolates bin traversal, ghost particles and the halo from any
 * question about geometry.
 */
void KerDepositAll(ACC<double> &rho, const ACCP<double> &w) {
  rho(0, 0) += w(0);
}

/*
 * RADIUS-LIMITED variant: the shape a real application has. Only nodes within
 * RADIUS of the particle receive anything. There is no closed-form total, so
 * this one is checked by requiring the field to be identical between serial
 * and MPI runs.
 */
void KerDepositRadius(ACC<double> &rho, const ACC<double> &xf,
                      const ACCP<double> &xp, const ACCP<double> &w,
                      const double *radius) {
  const double dx = xp(0) - xf(0, 0, 0);
  const double dy = xp(1) - xf(1, 0, 0);
  const double r2 = dx * dx + dy * dy;
  if (r2 < (*radius) * (*radius)) rho(0, 0) += w(0);
}

/* Global sum of the accumulator, via an OPS reduction so it works unchanged
   under MPI. The loop range covers owned nodes only, so each node is counted
   exactly once across all ranks. */
void KerSumDensity(const ACC<double> &rho, double *sum) { *sum += rho(0, 0); }

#endif /* _GRID_KERNELS_H_ */
