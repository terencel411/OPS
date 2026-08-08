/*
 * grid_kernels.h  --  particle-mesh influence
 *
 * The mesh side of the pipeline:
 *
 *   KerInitGrid          node coordinates (the mapping and box derive from it)
 *   KerZeroScalar        clear an accumulator
 *   KerDepositCIC        particles -> mesh charge  (grid-outer scatter loop)
 *   KerSumScalar         global sum of a grid dat, via an OPS reduction
 *   KerBoundaryDirect    Dirichlet values from the exact pairwise sum
 *   KerSOR               one red-or-black SOR sweep of the Poisson solve
 *   KerResidual          ||lap(phi) - q||^2, for the convergence report
 *
 * Convention: dim-1 grid dats index as f(i, j); the dim-2 coordinate dat
 * indexes as xf(component, i, j). That is what OPS's ACC<T> overloads give
 * and it matches apps/c/particle_tutorial_3_deposit.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

#include <cmath>

void KerInitGrid(ACC<double> &xf, const double *dx, const int *idx) {
  xf(0, 0, 0) = (*dx) * static_cast<double>(idx[0]);
  xf(1, 0, 0) = (*dx) * static_cast<double>(idx[1]);
}

void KerZeroScalar(ACC<double> &f) { f(0, 0) = 0.0; }

/* ------------------------------------------------------------------ *
 * Deposit -- the only loop form that may WRITE a grid dat
 * ------------------------------------------------------------------ *
 * Body of the grid-outer / particle-inner loop
 * (`ops_par_loop` overload in ops_grid_part_seq_v2.h:562, called here through
 * the ops_par_scatter_loop wrapper so the translator leaves it alone):
 *
 *     for each grid node (i,j):
 *       for each offset s in map_stencil:
 *         for each particle in bin(node -> map + s):
 *           kernel(grid args at (i,j), particle args at that particle)
 *
 * So it runs once per (node, nearby particle) pair and must accumulate --
 * hence the separate KerZeroScalar pass before it.
 *
 * This is cloud-in-cell: a particle at distance (dx,dy) from the node gives it
 * (1-|dx|/h)(1-|dy|/h) of its charge, zero beyond one cell. Summed over the
 * four surrounding nodes the weights are exactly 1, so the deposited charge
 * equals the particle's charge -- checked at startup on the |m| field, which
 * unlike the signed field has a non-trivial total.
 *
 * A ±1 search stencil is not just sufficient, it is REQUIRED: the particle
 * ghost band is only correct to a depth of one cell under MPI
 * (measured in apps/c/particle_tutorial_3_deposit, README §3a). CIC is exactly
 * the depth-1 case, so this pipeline sits in the regime that works.
 */
void KerDepositCIC(ACC<double> &q, ACC<double> &qabs, const ACC<double> &xf,
                   const ACCP<double> &xp, const ACCP<double> &mp,
                   const double *h) {
  const double ax = 1.0 - fabs(xp(0) - xf(0, 0, 0)) / (*h);
  const double ay = 1.0 - fabs(xp(1) - xf(1, 0, 0)) / (*h);
  if (ax > 0.0 && ay > 0.0) {
    const double w = ax * ay;
    q(0, 0) += mp(0) * w;
    qabs(0, 0) += fabs(mp(0)) * w;
  }
}

void KerSumScalar(const ACC<double> &f, double *sum) { *sum += f(0, 0); }

/* ------------------------------------------------------------------ *
 * Boundary values
 * ------------------------------------------------------------------ *
 * The mesh solve needs a boundary condition, and which one you pick is the
 * central accuracy/cost trade in this whole app.
 *
 *   -bc zero    phi = 0 on the box edge. Needs nothing from other ranks, so
 *               the pipeline has NO global collective at all. But phi=0 at a
 *               finite boundary is the potential of the charges PLUS their
 *               mirror images, so it is only a good approximation when the
 *               cloud is small compared with its distance to the wall.
 *
 *   -bc direct  evaluate the exact pairwise sum at the boundary nodes only.
 *               Restores the free-space answer, so the mesh error becomes
 *               purely a discretisation error. Costs O(N * N_boundary) and
 *               needs the gathered particle array -- i.e. it keeps one
 *               collective per step. Still cheap: N_boundary ~ 4*NX << N.
 *
 * `all` is the gathered state buffer, laid out NCOMP doubles per particle.
 */
void KerBoundaryDirect(ACC<double> &phig, const ACC<double> &xf,
                       const double *all, const int *n, const double *eps2,
                       const double *k) {
  const double x = xf(0, 0, 0);
  const double y = xf(1, 0, 0);

  double s = 0.0;
  for (int j = 0; j < *n; j++) {
    const double dx = x - all[NCOMP * j + C_X];
    const double dy = y - all[NCOMP * j + C_Y];
    s += all[NCOMP * j + C_M] * log(dx * dx + dy * dy + (*eps2));
  }
  phig(0, 0) = (*k) * s;
}

/* ------------------------------------------------------------------ *
 * The Poisson solve
 * ------------------------------------------------------------------ *
 * We want phi to be the potential whose Green's function is the interaction
 * law, i.e. lap(phi) = rho. Deposit gives the CHARGE q at each node, which is
 * a density of q/h^2 over that node's cell, and the 5-point Laplacian carries
 * a 1/h^2 -- so h^2 cancels exactly and the update needs no spacing at all:
 *
 *     (phiE + phiW + phiN + phiS - 4 phi) / h^2 = q / h^2
 *     =>  phi = (phiE + phiW + phiN + phiS - q) / 4
 *
 * Red-black ordering: a red point reads only black neighbours, so the sweep is
 * order-independent and gives the same answer at any rank count. `colour`
 * selects which half updates; call it twice for a full sweep, and OPS
 * exchanges the halo between the two calls because phi is read with a 5-point
 * stencil.
 *
 * Over-relaxation with the optimal omega turns O(n^2) Jacobi iterations into
 * O(n). Warm-starting from the previous timestep's phi then leaves only a few
 * dozen sweeps of work per step.
 */
void KerSOR(ACC<double> &phig, const ACC<double> &q, const int *idx,
            const double *omega, const int *colour) {
  if (((idx[0] + idx[1]) & 1) != *colour) return;

  const double gs = 0.25 * (phig(1, 0) + phig(-1, 0) + phig(0, 1) +
                            phig(0, -1) - q(0, 0));
  phig(0, 0) += (*omega) * (gs - phig(0, 0));
}

void KerResidual(const ACC<double> &phig, const ACC<double> &q, double *r2) {
  const double lap = phig(1, 0) + phig(-1, 0) + phig(0, 1) + phig(0, -1) -
                     4.0 * phig(0, 0);
  const double d = lap - q(0, 0);
  *r2 += d * d;
}

#endif /* _GRID_KERNELS_H_ */
