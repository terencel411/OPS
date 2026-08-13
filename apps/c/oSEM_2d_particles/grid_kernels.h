/*
 * grid_kernels.h -- the inlet plane. KerInitGrid / KerInitRST / KerComputeFluct
 * correspond to oSEM's instantiate_grid / instantiate_RST / compute_fluct.
 *
 * The plane spans the EDDY box, not just the physical inlet: the bounding box
 * OPS derives from this coordinate dat decides whether an eddy is in the
 * domain, so it must cover everywhere an eddy may legally be.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

void KerInitGrid(ACC<double> &crd, const int *idx) {
  crd(0, 0, 0) = eddy_y_min + (eddy_y_max - eddy_y_min) * (double)idx[0] / (double)ny;
  crd(1, 0, 0) = eddy_z_min + (eddy_z_max - eddy_z_min) * (double)idx[1] / (double)nz;
}

/* Cholesky factor of the Reynolds stress tensor, isotropic variant. The
   tabulated boundary-layer alternative is KerInitRST_TBL below. */
void KerInitRST(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                ACC<double> &a31, ACC<double> &a32, ACC<double> &a33) {
  a11(0, 0) = u0ti;
  a21(0, 0) = 0.0;
  a22(0, 0) = u0ti;
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = u0ti;
}

/* oSEM's instantiate_RST_TBL: Reynolds stresses interpolated from a tabulated
   boundary-layer profile, then Cholesky factorised. Unlike the isotropic
   variant the stresses vary with y and a21 != 0, giving a real shear stress.
   The 0.001 clamp on a11 guards the wall, where R11 -> 0.

   CARRIES A LATENT EXTRAPOLATION HAZARD, kept for parity with oSEM. It cannot
   trigger as shipped, but will if y_max or r_max is raised -- silently. Read
   the README section before changing either. tbl_at() in the driver mirrors
   this search exactly, idx = 0 included, and must. */
void KerInitRST_TBL(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                    ACC<double> &a31, ACC<double> &a32, ACC<double> &a33,
                    const ACC<double> &crd, const double *ydata,
                    const double *r11data, const double *r21data,
                    const double *r22data, const double *r33data) {

  const double y = crd(0, 0, 0);

  int idx = 0;                           /* oSEM's; see the hazard note above */
  for (int i = 1; i < ntbl - 1; i++) {
    if ((y - ydata[i]) < 0) { idx = i - 1; break; }
  }

  /* Written in oSEM's exact form -- slope first, then multiply by (y - y0) --
     rather than the tidier "compute the weight once and share it". The two are
     the same interpolation, but (a/b)*c and (c/b)*a do not round identically,
     so the shared-weight version differed from the reference in the last bit
     (measured: 1.1e-16 relative, worst case, on vv and uv). Harmless, and
     still not worth diverging for. */
  const double r11 = (r11data[idx + 1] - r11data[idx]) /
                     (ydata[idx + 1] - ydata[idx]) * (y - ydata[idx]) + r11data[idx];
  const double r21 = (r21data[idx + 1] - r21data[idx]) /
                     (ydata[idx + 1] - ydata[idx]) * (y - ydata[idx]) + r21data[idx];
  const double r22 = (r22data[idx + 1] - r22data[idx]) /
                     (ydata[idx + 1] - ydata[idx]) * (y - ydata[idx]) + r22data[idx];
  const double r33 = (r33data[idx + 1] - r33data[idx]) /
                     (ydata[idx + 1] - ydata[idx]) * (y - ydata[idx]) + r33data[idx];

  a11(0, 0) = sqrt(r11);

  double a11c = fabs(a11(0, 0));
  if (a11c < 0.001) a11c = 0.001;

  a21(0, 0) = r21 / a11c;
  a22(0, 0) = sqrt(r22 - a21(0, 0) * a21(0, 0));
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = sqrt(r33 - a31(0, 0) * a31(0, 0) - a32(0, 0) * a32(0, 0));
}

/* compute_fluct: every inlet node sums a contribution from every eddy in the
   domain. Body is oSEM's, unchanged apart from reading one interleaved buffer
   instead of seven arrays. `all` comes from ops_reduction_result and holds
   every eddy, on every rank. */
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
