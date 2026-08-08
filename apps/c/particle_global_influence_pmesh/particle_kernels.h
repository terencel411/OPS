/*
 * particle_kernels.h  --  particle-mesh vs. direct all-to-all
 *
 * Motion (shared by both methods)
 *   KerInitState        swirl initial velocity
 *   KerAdvance          constant acceleration, forward Euler
 *
 * Direct O(N^2) method  (the one particle_global_influence uses)
 *   KerPublishState     scatter into the global gather buffer
 *   KerInfluenceDirect  sum over every particle in the domain
 *
 * Particle-mesh method
 *   KerZeroPhiP         clear the accumulator (interpolation accumulates)
 *   KerInterpCIC        read the mesh potential back at the particle
 *   KerFitSelf          one-parameter self-energy calibration
 *   KerSubtractSelf     apply it
 *
 * Comparison
 *   KerCompare          max / RMS difference between the two fields
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

#include <cmath>

/* Global gather buffer layout, indexed by global id: NCOMP doubles each. */
#define NCOMP 5
#define C_X 0
#define C_Y 1
#define C_VX 2
#define C_VY 3
#define C_M 4

/* ------------------------------------------------------------------ *
 * Motion
 * ------------------------------------------------------------------ */

void KerInitState(ACCP<double> &up, ACCP<double> &phid, ACCP<double> &phim,
                  const ACCP<double> &xp, const double *omega,
                  const double *centre) {
  up(0) = -(*omega) * (xp(1) - centre[1]);
  up(1) = (*omega) * (xp(0) - centre[0]);
  phid(0) = 0.0;
  phim(0) = 0.0;
}

void KerAdvance(ACCP<double> &xp, ACCP<double> &up, const double *a,
                const double *dt) {
  up(0) += a[0] * (*dt);
  up(1) += a[1] * (*dt);
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

/* ------------------------------------------------------------------ *
 * Direct method
 * ------------------------------------------------------------------ *
 * `all` is an ops_arg_reduce buffer declared OPS_INC. Each particle writes
 * into the slot picked out by its global id; every other rank contributes 0.0
 * there, so the Allreduce inside ops_reduction_result is a bit-exact
 * allgather. MUST be OPS_PARTICLE_ITERATE_LOCAL -- a ghost carries its owner's
 * gid and would be counted twice.
 */
void KerPublishState(const ACCP<double> &xp, const ACCP<double> &up,
                     const ACCP<double> &mp, const ACCP<int> &gid,
                     double *all) {
  const int s = NCOMP * gid(0);
  all[s + C_X] += xp(0);
  all[s + C_Y] += xp(1);
  all[s + C_VX] += up(0);
  all[s + C_VY] += up(1);
  all[s + C_M] += mp(0);
}

/* The interaction law: the softened 2-D Coulomb potential.
 *
 *     phi_i = sum_{j != i}  m_j * (1/4pi) ln(r_ij^2 + eps^2)
 *
 * The log is not cosmetic. (1/2pi) ln r is the Green's function of the 2-D
 * Laplacian, which is the ONLY reason a mesh solve can reproduce this sum at
 * all. The 1/r law of particle_global_influence has no local PDE behind it in
 * 2-D, so it has no particle-mesh equivalent -- switching kernel is the price
 * of admission, and it is the honest way to compare the two methods.
 *
 * eps is set to the mesh spacing by default, so the pairwise sum is smoothed
 * at the same scale the mesh can represent.
 */
void KerInfluenceDirect(ACCP<double> &phi, const ACCP<double> &xp,
                        const ACCP<int> &gid, const double *all, const int *n,
                        const double *eps2, const double *k) {
  const int me = gid(0);
  const double x = xp(0);
  const double y = xp(1);

  double sum = 0.0;
  for (int j = 0; j < *n; j++) {
    if (j == me) continue;
    const double dx = x - all[NCOMP * j + C_X];
    const double dy = y - all[NCOMP * j + C_Y];
    sum += all[NCOMP * j + C_M] * log(dx * dx + dy * dy + (*eps2));
  }
  phi(0) = (*k) * sum;
}

/* ------------------------------------------------------------------ *
 * Particle-mesh method
 * ------------------------------------------------------------------ */

void KerZeroPhiP(ACCP<double> &phi) { phi(0) = 0.0; }

/* Read the mesh potential back with the same CIC weights used to deposit.
 * Using the same weights both ways is what keeps the scheme momentum
 * conserving; it is also why the kernel accumulates -- the enclosing
 * ops_par_particle_grid_loop runs it once per stencil node.
 */
void KerInterpCIC(ACCP<double> &phi, const ACCP<double> &xp,
                  const ACC<double> &phig, const ACC<double> &xf,
                  const double *h) {
  const double ax = 1.0 - fabs(xp(0) - xf(0, 0, 0)) / (*h);
  const double ay = 1.0 - fabs(xp(1) - xf(1, 0, 0)) / (*h);
  if (ax > 0.0 && ay > 0.0) phi(0) += ax * ay * phig(0, 0);
}

/* ---- the self-energy term -----------------------------------------
 *
 * A particle deposits its own charge onto the mesh and then reads the field
 * back, so it feels itself. The direct sum excludes j == i, so it does not.
 * The difference is proportional to the particle's own charge, and with signed
 * charges it pushes positive and negative particles opposite ways -- it does
 * not average out, and it does not shrink when you refine the mesh.
 *
 * It is not one number, though. Writing w_a for the four CIC weights and
 * Ghat(a,b) for the discrete Green's function between two nodes,
 *
 *     phi_self = m * sum_{a,b} w_a w_b Ghat(a,b)
 *
 * and by symmetry Ghat takes only three distinct values over the four nodes of
 * a cell: same node, edge-adjacent, diagonal. Grouping the weight products
 * accordingly gives three shape factors that depend only on where the particle
 * sits inside its cell,
 *
 *     A = sum_a w_a^2                 (same node)
 *     B = edge-adjacent pairs         C = diagonal pairs,     A + B + C = 1
 *
 * so phi_self = m (a A + b B + c C) for three constants fitted ONCE, at step 0,
 * by least squares against the direct sum. A is largest when the particle sits
 * on a node and smallest at the cell centre, which is exactly the sub-cell
 * dependence a single constant cannot express.
 *
 * With px = (1-fx)^2 + fx^2 and qx = 2 fx (1-fx) (so px + qx = 1),
 *
 *     A = px py     B = qx py + qy px     C = qx qy
 *
 * This is a calibration of a known systematic, not a fit to the answer: the
 * coefficients are frozen at step 0 while the cloud then rotates and spreads
 * for the rest of the run, so the agreement reported at the END is a genuine
 * test. The app prints the difference before and after.
 */
static inline void cic_shape_factors(double x, double y, double h, double *A,
                                     double *B, double *C) {
  const double fx = x / h - floor(x / h);
  const double fy = y / h - floor(y / h);
  const double px = (1.0 - fx) * (1.0 - fx) + fx * fx;
  const double qx = 2.0 * fx * (1.0 - fx);
  const double py = (1.0 - fy) * (1.0 - fy) + fy * fy;
  const double qy = 2.0 * fy * (1.0 - fy);
  *A = px * py;
  *B = qx * py + qy * px;
  *C = qx * qy;
}

/* Accumulate the normal equations for the 3-term fit, plus the 1-term fit, in
 * a single dim-11 reduction:
 *
 *   0..5  the symmetric 3x3 M^T M  (AA AB AC BB BC CC)
 *   6..8  M^T d
 *   9,10  sum m*d and sum m^2, i.e. the one-constant fit
 */
void KerFitSelf(const ACCP<double> &phim, const ACCP<double> &phid,
                const ACCP<double> &mp, const ACCP<double> &xp,
                const double *h, double *acc) {

  double A, B, C;
  cic_shape_factors(xp(0), xp(1), *h, &A, &B, &C);

  const double m = mp(0);
  const double d = phim(0) - phid(0);
  const double a = m * A, b = m * B, c = m * C;

  acc[0] += a * a;  acc[1] += a * b;  acc[2] += a * c;
  acc[3] += b * b;  acc[4] += b * c;  acc[5] += c * c;
  acc[6] += a * d;  acc[7] += b * d;  acc[8] += c * d;
  acc[9] += m * d;  acc[10] += m * m;
}

/* Subtract m (cA A + cB B + cC C). Passing the same value three times gives
 * the one-constant model, because A + B + C = 1. */
void KerSubtractSelf(ACCP<double> &phim, const ACCP<double> &mp,
                     const ACCP<double> &xp, const double *h,
                     const double *coef) {
  double A, B, C;
  cic_shape_factors(xp(0), xp(1), *h, &A, &B, &C);
  phim(0) -= mp(0) * (coef[0] * A + coef[1] * B + coef[2] * C);
}

/* ------------------------------------------------------------------ *
 * Comparison
 * ------------------------------------------------------------------ *
 * Accumulates the pieces of a max-abs and an RMS difference, plus the RMS of
 * the reference itself so the result can be quoted as a relative error.
 */
void KerCompare(const ACCP<double> &phim, const ACCP<double> &phid,
                double *maxdiff, double *sum_d2, double *sum_r2, int *count) {
  const double d = fabs(phim(0) - phid(0));
  if (d > *maxdiff) *maxdiff = d;
  *sum_d2 += d * d;
  *sum_r2 += phid(0) * phid(0);
  *count += 1;
}

/* Gather phi by global id -- used for the optional exact check. */
void KerPublishScalar(const ACCP<double> &f, const ACCP<int> &gid,
                      double *all_f) {
  all_f[gid(0)] += f(0);
}

#endif /* _PARTICLE_KERNELS_H_ */
