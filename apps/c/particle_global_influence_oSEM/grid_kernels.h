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

void KerInitGrid(ACC<double> &crd, const int *idx) {
  crd(0, 0, 0) = eddy_y_min + (eddy_y_max - eddy_y_min) * (double)idx[0] / (double)ny;
  crd(1, 0, 0) = eddy_z_min + (eddy_z_max - eddy_z_min) * (double)idx[1] / (double)nz;
}

/* The Cholesky factor of the Reynolds stress tensor. oSEM also carries a
   boundary-layer profile variant (instantiate_RST_TBL) driven by a tabulated
   dataset; this is the isotropic one, which keeps the app self-contained. */
void KerInitRST(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                ACC<double> &a31, ACC<double> &a32, ACC<double> &a33) {
  a11(0, 0) = u0ti;
  a21(0, 0) = 0.0;
  a22(0, 0) = u0ti;
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = u0ti;
}

/* ------------------------------------------------------------------ *
 * KerInitRST_TBL  <- oSEM instantiate_RST_TBL
 * ------------------------------------------------------------------ *
 * The realistic alternative to the isotropic RST above: Reynolds stresses read
 * from a tabulated boundary-layer profile (TBL_data.h, 260 points),
 * interpolated to each node's wall-normal position, then Cholesky factorised.
 *
 * Two things this buys over the isotropic version:
 *   - the stresses VARY WITH y, so the inlet has a real boundary-layer profile
 *     instead of uniform intensity
 *   - a21 != 0, so u' and v' become correlated -- the shear stress <u'v'>,
 *     which is what drives turbulence production downstream
 *
 * a31 = a32 = 0 encodes <u'w'> = <v'w'> = 0, correct for a 2-D boundary layer
 * with z spanwise. The 0.001 clamp on a11 guards the wall, where R11 -> 0 and
 * the division for a21 would blow up.
 *
 * ONE DEPARTURE FROM THE REFERENCE. oSEM's search is
 *
 *     for (i = 1; i < 260-1; i++) if ((y - ydata[i]) < 0) { idx = i-1; break; }
 *
 * with idx initialised to 0. If y lies beyond the last tabulated point the
 * loop never breaks and idx stays 0 -- so it silently interpolates from the
 * first two rows, applying WALL values out in the freestream. It cannot
 * trigger with the shipped numbers (table reaches y = 0.01925, grid only
 * 0.01187), but it would the moment anyone raised y_max or r_max. Initialising
 * idx to the LAST interval makes it saturate instead, which is the behaviour
 * the silent version was presumably meant to have.
 */
void KerInitRST_TBL(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                    ACC<double> &a31, ACC<double> &a32, ACC<double> &a33,
                    const ACC<double> &crd, const double *ydata,
                    const double *r11data, const double *r21data,
                    const double *r22data, const double *r33data) {

  const double y = crd(0, 0, 0);

  int idx = ntbl - 2;                    /* clamp: saturate at the last row */
  for (int i = 1; i < ntbl - 1; i++) {
    if ((y - ydata[i]) < 0) { idx = i - 1; break; }
  }

  const double w = (y - ydata[idx]) / (ydata[idx + 1] - ydata[idx]);
  const double r11 = r11data[idx] + w * (r11data[idx + 1] - r11data[idx]);
  const double r21 = r21data[idx] + w * (r21data[idx + 1] - r21data[idx]);
  const double r22 = r22data[idx] + w * (r22data[idx + 1] - r22data[idx]);
  const double r33 = r33data[idx] + w * (r33data[idx + 1] - r33data[idx]);

  a11(0, 0) = sqrt(r11);

  double a11c = fabs(a11(0, 0));
  if (a11c < 0.001) a11c = 0.001;

  a21(0, 0) = r21 / a11c;
  a22(0, 0) = sqrt(r22 - a21(0, 0) * a21(0, 0));
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = sqrt(r33 - a31(0, 0) * a31(0, 0) - a32(0, 0) * a32(0, 0));
}

/* Per-row sums of u'^2, v'^2, w'^2, indexed by the wall-normal grid index.
 * With the TBL profile the three rms values are SUPPOSED to differ, so the
 * isotropic invariant (all three equal) no longer applies -- the test becomes
 * whether the computed PROFILE follows the tabulated one. */
void KerFluctProfile(const ACC<double> &uprime, const ACC<double> &vprime,
                     const ACC<double> &wprime, const int *idx, double *acc) {
  const double u = uprime(0, 0), v = vprime(0, 0), w = wprime(0, 0);
  const int r = 6 * idx[0];
  acc[r + 0] += u * u;
  acc[r + 1] += v * v;
  acc[r + 2] += w * w;
  /* The off-diagonal stresses. <u'v'> should reproduce R21 -- that is the
     SHEAR stress, and it is the only thing that tests a21: the rms values
     depend on a21 only through a22 = sqrt(R22 - a21^2), which collapses to
     R22 whatever a21 is. <u'w'> and <v'w'> should vanish, because a31 = a32 =
     0, so they are a free check that nothing is leaking between components. */
  acc[r + 3] += u * v;
  acc[r + 4] += u * w;
  acc[r + 5] += v * w;
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
                     const double *all) {

  const double y = crd(0, 0, 0);
  const double z = crd(1, 0, 0);

  double u = 0.0, v = 0.0, w = 0.0;

  for (int i = 0; i < eddies; i++) {
    const double ex = all[NCOMP * i + E_X];
    const double ey = all[NCOMP * i + E_Y];
    const double ez = all[NCOMP * i + E_Z];
    const double er = all[NCOMP * i + E_R];

    const double dy = ey - y;
    const double dz = ez - z;
    const double rsq = ex * ex + dy * dy + dz * dz;

    if (rsq < er * er) {
      const double dx = ex - x_plane;
      double shape = (fabs(dx) < er)
                         ? exp(-0.5 * dx * dx / (er * er))
                         : 0.0;
      shape *= shape_norm;
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
