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

  int idx = 0;                     // oSEM's search; stays 0 if y is past y_inp
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

// compute_fluct accumulates one eddy at a time, so the node must start at zero.
void zero_fluct(ACC<double> &uprime, ACC<double> &vprime, ACC<double> &wprime) {
  uprime(0, 0) = 0.0;
  vprime(0, 0) = 0.0;
  wprime(0, 0) = 0.0;
}

// Called once per (node, eddy in a bin of the search stencil) either way round,
// so oSEM_2d_particles' inner eddy loop is gone.
void compute_fluct(ACC<double> &uprime, ACC<double> &vprime, ACC<double> &wprime,
                   const ACC<double> &grid,
                   const ACC<double> &a11, const ACC<double> &a21,
                   const ACC<double> &a22, const ACC<double> &a31,
                   const ACC<double> &a32, const ACC<double> &a33,
                   const ACCP<double> &eddy_pos, const ACCP<double> &eddy_x,
                   const ACCP<double> &eddy_r, const ACCP<double> &eddy_eps) {

  const double y = grid(0, 0, 0);
  const double z = grid(1, 0, 0);

  const double ex = eddy_x(0);
  const double er = eddy_r(0);

  const double dy = eddy_pos(0) - y;
  const double dz = eddy_pos(1) - z;
  const double rsq = ex * ex + dy * dy + dz * dz;

  if (rsq < er * er) {
    const double dx = ex - x_plane;
    double shape = (fabs(dx) < er)
                       ? exp(-0.5 * dx * dx / (er * er))
                       : 0.0;
    shape *= 1/1.5829045;
    shape *= exp(-0.5 * dy * dy / (er * er));
    shape *= exp(-0.5 * dz * dz / (er * er));

    const double sx = eddy_eps(0);
    const double sy = eddy_eps(1);
    const double sz = eddy_eps(2);

    uprime(0, 0) += a11(0, 0) * sx * shape;
    vprime(0, 0) += a21(0, 0) * sx * shape + a22(0, 0) * sy * shape;
    wprime(0, 0) += a31(0, 0) * sx * shape + a32(0, 0) * sy * shape +
                    a33(0, 0) * sz * shape;
  }
}

#endif /* _GRID_KERNELS_H_ */
