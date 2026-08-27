#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

void instantiate_grid(ACC<double> &grid, const int *idx) {
  grid(0, 0, 0) = eddy_y_min + (eddy_y_max - eddy_y_min) * (double)idx[0] / (double)ny;
  grid(1, 0, 0) = eddy_z_min + (eddy_z_max - eddy_z_min) * (double)idx[1] / (double)nz;
}

void instantiate_RST(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                ACC<double> &a31, ACC<double> &a32, ACC<double> &a33) {
  a11(0, 0) = u0ti;
  a21(0, 0) = 0.0;
  a22(0, 0) = u0ti;
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = u0ti;
}

void instantiate_RST_TBL(ACC<double> &a11, ACC<double> &a21, ACC<double> &a22,
                    ACC<double> &a31, ACC<double> &a32, ACC<double> &a33,
                    const ACC<double> &grid) {

  const double y = grid(0, 0, 0);

  int idx = 0;                           // oSEM's; see the hazard note above
  for (int i = 1; i < ntbl - 1; i++) {
    if ((y - y_inp[i]) < 0) { idx = i - 1; break; }
  }

  const double r11 = (uu_inp[idx + 1] - uu_inp[idx]) /
                     (y_inp[idx + 1] - y_inp[idx]) * (y - y_inp[idx]) + uu_inp[idx];
  const double r21 = (uv_inp[idx + 1] - uv_inp[idx]) /
                     (y_inp[idx + 1] - y_inp[idx]) * (y - y_inp[idx]) + uv_inp[idx];
  const double r22 = (vv_inp[idx + 1] - vv_inp[idx]) /
                     (y_inp[idx + 1] - y_inp[idx]) * (y - y_inp[idx]) + vv_inp[idx];
  const double r33 = (ww_inp[idx + 1] - ww_inp[idx]) /
                     (y_inp[idx + 1] - y_inp[idx]) * (y - y_inp[idx]) + ww_inp[idx];

  a11(0, 0) = sqrt(r11);

  double a11c = fabs(a11(0, 0));
  if (a11c < 0.001) a11c = 0.001;

  a21(0, 0) = r21 / a11c;
  a22(0, 0) = sqrt(r22 - a21(0, 0) * a21(0, 0));
  a31(0, 0) = 0.0;
  a32(0, 0) = 0.0;
  a33(0, 0) = sqrt(r33 - a31(0, 0) * a31(0, 0) - a32(0, 0) * a32(0, 0));
}

void compute_fluct(ACC<double> &uprime, ACC<double> &vprime,
                     ACC<double> &wprime, const ACC<double> &grid,
                     const ACC<double> &a11, const ACC<double> &a21,
                     const ACC<double> &a22, const ACC<double> &a31,
                     const ACC<double> &a32, const ACC<double> &a33,
                     const double *eddy_all) {

  const double y = grid(0, 0, 0);
  const double z = grid(1, 0, 0);

  double u = 0.0, v = 0.0, w = 0.0;

  for (int i = 0; i < eddies; i++) {
    const double ex = eddy_all[NCOMP * i + E_X];
    const double ey = eddy_all[NCOMP * i + E_Y];
    const double ez = eddy_all[NCOMP * i + E_Z];
    const double er = eddy_all[NCOMP * i + E_R];

    const double dy = ey - y;
    const double dz = ez - z;
    const double rsq = ex * ex + dy * dy + dz * dz;

    if (rsq < er * er) {
      const double dx = ex - x_plane;
      double shape = (fabs(dx) < er)
                         ? exp(-0.5 * dx * dx / (er * er))
                         : 0.0;
      shape *= 1/1.5829045;
      shape *= exp(-0.5 * dy * dy / (er * er));
      shape *= exp(-0.5 * dz * dz / (er * er));

      const double sx = eddy_all[NCOMP * i + E_SX];
      const double sy = eddy_all[NCOMP * i + E_SY];
      const double sz = eddy_all[NCOMP * i + E_SZ];

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

#endif /* _GRID_KERNELS_H_ */
