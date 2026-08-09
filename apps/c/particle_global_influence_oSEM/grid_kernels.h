/*
 * grid_kernels.h  --  the inlet plane
 *
 *   KerInitGrid     <- oSEM instantiate_grid
 *   KerInitRST      <- oSEM instantiate_RST
 *   KerComputeFluct <- oSEM compute_fluct     (the "influence" kernel)
 *
 * Ordinary OPS grid kernels; nothing particle-specific. The plane spans the
 * EDDY box, not just the physical inlet -- that is what oSEM's instantiate_grid
 * already does (it starts at z_min - r_max and runs to z_max + r_max), and it
 * matters here for a second reason: the bounding box that OPS derives from
 * this coordinate dat is what decides whether an eddy is inside the domain, so
 * it has to cover everywhere an eddy may legally be.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

void KerInitGrid(ACC<double> &crd, const double *org, const double *span,
                 const int *n, const int *idx) {
  crd(0, 0, 0) = org[0] + span[0] * (double)idx[0] / (double)n[0];
  crd(1, 0, 0) = org[1] + span[1] * (double)idx[1] / (double)n[1];
}

/* The Cholesky factor of the Reynolds stress tensor. oSEM also carries a
   boundary-layer profile variant (instantiate_RST_TBL) driven by a tabulated
   dataset; this is the isotropic one, which keeps the app self-contained. */
void KerInitRST(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                ACC<double> &a31, ACC<double> &a32, ACC<double> &a33,
                const double *prm) {
  a11(0, 0) = prm[P_U0TI];
  a21(0, 0) = 0.0;
  a22(0, 0) = prm[P_U0TI];
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = prm[P_U0TI];
}

/* ------------------------------------------------------------------ *
 * compute_fluct -- the all-to-all
 * ------------------------------------------------------------------ *
 * Every inlet node sums a contribution from EVERY eddy in the domain. This is
 * the same shape of problem as the influence kernel in
 * particle_global_influence: an unrestricted sum over a globally gathered
 * array, here with the outer loop over grid points rather than particles.
 *
 * The body is oSEM's, unchanged apart from reading one interleaved buffer
 * instead of seven separate arrays. The Gaussian shape function, the r^2
 * rejection test, the normalisation constant and the a_ij * eps_j assembly are
 * all as they were.
 *
 * `all` arrives from ops_reduction_result and holds every eddy, on every rank.
 */
void KerComputeFluct(ACC<double> &uprime, ACC<double> &vprime,
                     ACC<double> &wprime, const ACC<double> &crd,
                     const ACC<double> &a11, const ACC<double> &a21,
                     const ACC<double> &a22, const ACC<double> &a31,
                     const ACC<double> &a32, const ACC<double> &a33,
                     const double *all, const int *neddy, const double *prm) {

  const double y = crd(0, 0, 0);
  const double z = crd(1, 0, 0);

  double u = 0.0, v = 0.0, w = 0.0;

  for (int i = 0; i < *neddy; i++) {
    const double ex = all[NCOMP * i + E_X];
    const double ey = all[NCOMP * i + E_Y];
    const double ez = all[NCOMP * i + E_Z];
    const double er = all[NCOMP * i + E_R];

    const double dy = ey - y;
    const double dz = ez - z;
    const double rsq = ex * ex + dy * dy + dz * dz;

    if (rsq < er * er) {
      const double dx = ex - prm[P_XPLANE];
      double shape = (fabs(dx) < er)
                         ? exp(-0.5 * dx * dx / (er * er))
                         : 0.0;
      shape *= prm[P_SHAPENORM];
      shape *= exp(-0.5 * dy * dy / (er * er));
      shape *= exp(-0.5 * dz * dz / (er * er));

      const double sx = all[NCOMP * i + E_SX];
      const double sy = all[NCOMP * i + E_SY];
      const double sz = all[NCOMP * i + E_SZ];

      u += a11(0, 0) * sx * shape;
      v += a21(0, 0) * sx * shape + a22(0, 0) * sy * shape;
      w += a31(0, 0) * sx * shape + a32(0, 0) * sy * shape +
           a33(0, 0) * sz * shape;
    }
  }

  uprime(0, 0) = u;
  vprime(0, 0) = v;
  wprime(0, 0) = w;
}

/* Sum of squares over the plane, for the turbulence-intensity check. */
void KerFluctStats(const ACC<double> &uprime, const ACC<double> &vprime,
                   const ACC<double> &wprime, double *acc) {
  acc[0] += uprime(0, 0) * uprime(0, 0);
  acc[1] += vprime(0, 0) * vprime(0, 0);
  acc[2] += wprime(0, 0) * wprime(0, 0);
  acc[3] += 1.0;
}

#endif /* _GRID_KERNELS_H_ */
