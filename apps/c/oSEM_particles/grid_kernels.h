/*
 * grid_kernels.h -- oSEM on OPS Particles
 *
 * Ordinary OPS grid kernels: nothing particle-specific here.
 */

#ifndef _GRID_KERNELS_H_
#define _GRID_KERNELS_H_

/* ------------------------------------------------------------------ *
 *  The inlet plane -- the physics grid
 * ------------------------------------------------------------------ *
 * Identical to oSEM's instantiate_grid(), so the two apps sample the same
 * points. Note it divides by ny/nz rather than (ny-1)/(nz-1), so the last node
 * sits one cell short of the top of the eddy box. That is why the particle
 * coordinate dat below is a separate, slightly larger field.
 */
void KerInitInletGrid(ACC<double> &y, ACC<double> &z, const int *idx) {
  y(0, 0) = y_min + (y_max + r_max - y_min) * (double)(idx[0]) / ny;
  z(0, 0) = z_min - r_max + (z_max + r_max - z_min + r_max) * (double)(idx[1]) / nz;
}

/* ------------------------------------------------------------------ *
 *  The particle coordinate field -- box and bins only, no physics
 * ------------------------------------------------------------------ *
 * A dim-2 coordinate field spanning the eddy box EXACTLY, carrying one node
 * more than the inlet grid in each direction.
 *
 * Why it has to be its own dat:
 *   - BoundingBox is built as exactly [first node .. last node].
 *     _ops_construct_local_box_from_dat (ops_particle_box_host_funcs.h:44)
 *     accepts a `grid_size`/dx argument and never uses it, so there is no
 *     slack to lean on.
 *   - KerInitInletGrid stops one cell short of eddy_y_max / eddy_z_max, so
 *     eddies in that last sliver would fall outside the box and be deleted.
 *
 * Spacing is the same dy, dz as the inlet grid, so bin k stays aligned with
 * inlet node k and the search stencil means the same thing on both.
 */
void KerInitPartCoords(ACC<double> &c, const double *d, const int *idx) {
  c(0, 0, 0) = eddy_y_min + d[0] * (double)(idx[0]);
  c(1, 0, 0) = eddy_z_min + d[1] * (double)(idx[1]);
}


/* ------------------------------------------------------------------ *
 *  Reynolds stress tensor from the tabulated TBL profile
 * ------------------------------------------------------------------ *
 * Copied unchanged from oSEM's instantiate_RST_TBL so both apps build the
 * same a_ij from the same data. Linear interpolation in y through the 260
 * tabulated points, then a Cholesky-style factorisation.
 *
 * The ops_arg_gbl arrays here are TABULATED PROFILE DATA, not eddy data --
 * they are a fixed 260-point table read once at setup. Nothing about the
 * eddies travels this way.
 */
void instantiate_RST_TBL(ACC<double>& a11, ACC<double>& a21, ACC<double>& a22, ACC<double>& a31, ACC<double>& a32, ACC<double>& a33,
                    const ACC<double>& y, const ACC<double>& z, const double* ydata, const double* r11data, const double* r21data, const double* r22data,
                    const double* r33data){
                        
    int idx;
    idx = 0;
    for(int i{1}; i < 260 - 1; i++){
        // assume data passed in is within bounds
        if((y(0, 0) - ydata[i]) < 0){
            idx = i-1;
            break;
        }
    }                   
    double r11temp = (r11data[idx+1] - r11data[idx]) / (ydata[idx+1] - ydata[idx]) * (y(0, 0) - ydata[idx]) + r11data[idx];
    double r21temp = (r21data[idx+1] - r21data[idx]) / (ydata[idx+1] - ydata[idx]) * (y(0, 0) - ydata[idx]) + r21data[idx];
    double r22temp = (r22data[idx+1] - r22data[idx]) / (ydata[idx+1] - ydata[idx]) * (y(0, 0) - ydata[idx]) + r22data[idx];
    double r33temp = (r33data[idx+1] - r33data[idx]) / (ydata[idx+1] - ydata[idx]) * (y(0, 0) - ydata[idx]) + r33data[idx];

    a11(0, 0) = sqrt(r11temp);
    double a11temp = std::abs(a11(0, 0));
    if(a11temp < 0.001){
        a11temp = 0.001;
    }
    a21(0, 0) = r21temp / a11temp;
    a22(0, 0) = sqrt(r22temp - a21(0, 0)*a21(0, 0));
    a31(0, 0) = 0;
    a32(0, 0) = 0;
    a33(0, 0) = sqrt(r33temp - a31(0,0)*a31(0,0) - a32(0,0)*a32(0,0));
}

/* ------------------------------------------------------------------ *
 *  Zero the fluctuation field
 * ------------------------------------------------------------------ *
 * Must be a separate loop: in the scatter loop below the kernel body runs
 * once per (node, nearby eddy) pair, so it can only accumulate.
 */
void KerZeroFluct(ACC<double> &uprime, ACC<double> &vprime, ACC<double> &wprime) {
  uprime(0, 0) = 0.0;
  vprime(0, 0) = 0.0;
  wprime(0, 0) = 0.0;
}

/* ================================================================== *
 *  compute_fluct -- THE POINT OF THE WHOLE EXERCISE
 * ================================================================== *
 *
 * Body of the grid-outer / particle-inner loop (ops_grid_part_seq_v2.h:562,
 * called here through ops_par_scatter_loop -- see scatter_loop.h):
 *
 *     for each inlet node (i,j):
 *         for each offset s in the search stencil:
 *             for each eddy in bin(node + s):
 *                 kernel(grid args at (i,j), eddy args for that eddy)
 *
 * This is oSEM's compute_fluct body with the `for (i < eddies)` loop REMOVED.
 * The library supplies that iteration, and only over eddies binned near this
 * node -- roughly 50 of 1718 rather than all of them.
 *
 * Every eddy quantity arrives as ACCP<T>&, i.e. through
 * ops_arg_dat_particle. Nothing about the eddies is passed as ops_arg_gbl,
 * which is the whole reason for moving to OPS Particles.
 *
 * The substitution from the reference implementation is exactly:
 *     x_gbl[i]     -> px(0)
 *     y_gbl[i]     -> ppos(0)
 *     z_gbl[i]     -> ppos(1)
 *     r_gbl[i]     -> pr(0)
 *     eps_*_gbl[i] -> peps(0..2)
 * so any difference between the two paths is the coupling machinery, never
 * the physics.
 */
void KerComputeFluct(ACC<double> &uprime, ACC<double> &vprime, ACC<double> &wprime,
                     const ACC<double> &y, const ACC<double> &z,
                     const ACC<double> &a11, const ACC<double> &a21,
                     const ACC<double> &a22, const ACC<double> &a31,
                     const ACC<double> &a32, const ACC<double> &a33,
                     const ACCP<double> &ppos, const ACCP<double> &px,
                     const ACCP<double> &pr, const ACCP<int> &peps,
                     double *nvisit, double *nhit) {
  *nvisit += 1.0;

  const double rr  = pr(0);
  const double ddy = ppos(0) - y(0, 0);
  const double ddz = ppos(1) - z(0, 0);
  const double rsq = px(0) * px(0) + ddy * ddy + ddz * ddz;

  if (rsq < rr * rr) {
    double shape = (std::abs(px(0) - x_plane) < rr)
                 ? exp(-0.5 * (px(0) - x_plane) * (px(0) - x_plane) / (rr * rr))
                 : 0.0;
    shape *= 1 / 1.5829045;
    shape *= exp(-0.5 * ddy * ddy / (rr * rr));
    shape *= exp(-0.5 * ddz * ddz / (rr * rr));

    *nhit += 1.0;
    uprime(0, 0) += a11(0, 0) * peps(0) * shape;
    vprime(0, 0) += a21(0, 0) * peps(0) * shape + a22(0, 0) * peps(1) * shape;
    wprime(0, 0) += a31(0, 0) * peps(0) * shape + a32(0, 0) * peps(1) * shape
                  + a33(0, 0) * peps(2) * shape;
  }
}

/* ------------------------------------------------------------------ *
 *  Reference implementation -- validation only
 * ------------------------------------------------------------------ *
 * oSEM's original formulation, verbatim: an ordinary grid loop that walks
 * ALL eddies at every node, with the eddy state arriving as ops_arg_gbl
 * arrays. Used only under -validate, so the production path never builds
 * these arrays.
 *
 * Running both from the same eddy field in one binary makes the comparison
 * exact. The two will not agree bitwise -- the reference sums in global eddy
 * order, the particle path in bin order, and floating point addition is not
 * associative -- so agreement to a small multiple of machine epsilon is the
 * correct expectation.
 */
void KerComputeFluctRef(ACC<double> &uprime, ACC<double> &vprime, ACC<double> &wprime,
                        const ACC<double> &y, const ACC<double> &z,
                        const ACC<double> &a11, const ACC<double> &a21,
                        const ACC<double> &a22, const ACC<double> &a31,
                        const ACC<double> &a32, const ACC<double> &a33,
                        const double *x_gbl, const double *y_gbl,
                        const double *z_gbl, const double *r_gbl,
                        const int *eps_x_gbl, const int *eps_y_gbl,
                        const int *eps_z_gbl, double *nhit) {

  uprime(0, 0) = 0.0;
  vprime(0, 0) = 0.0;
  wprime(0, 0) = 0.0;

  for (int i = 0; i < eddies; i++) {
    const double rr  = r_gbl[i];
    const double ddy = y_gbl[i] - y(0, 0);
    const double ddz = z_gbl[i] - z(0, 0);
    const double rsq = x_gbl[i] * x_gbl[i] + ddy * ddy + ddz * ddz;

    if (rsq < rr * rr) {
      double shape = (std::abs(x_gbl[i] - x_plane) < rr)
                   ? exp(-0.5 * (x_gbl[i] - x_plane) * (x_gbl[i] - x_plane) / (rr * rr))
                   : 0.0;
      shape *= 1 / 1.5829045;
      shape *= exp(-0.5 * ddy * ddy / (rr * rr));
      shape *= exp(-0.5 * ddz * ddz / (rr * rr));

      *nhit += 1.0;
      uprime(0, 0) += a11(0, 0) * eps_x_gbl[i] * shape;
      vprime(0, 0) += a21(0, 0) * eps_x_gbl[i] * shape + a22(0, 0) * eps_y_gbl[i] * shape;
      wprime(0, 0) += a31(0, 0) * eps_x_gbl[i] * shape + a32(0, 0) * eps_y_gbl[i] * shape
                    + a33(0, 0) * eps_z_gbl[i] * shape;
    }
  }
}

/* Node-by-node comparison of the two fields, as OPS reductions so it works
   unchanged under MPI. */
void KerCompareFluct(const ACC<double> &u,  const ACC<double> &v,  const ACC<double> &w,
                     const ACC<double> &ur, const ACC<double> &vr, const ACC<double> &wr,
                     double *maxdiff, double *maxval, double *maxpart) {
  double d;
  d = std::abs(u(0, 0) - ur(0, 0)); if (d > *maxdiff) *maxdiff = d;
  d = std::abs(v(0, 0) - vr(0, 0)); if (d > *maxdiff) *maxdiff = d;
  d = std::abs(w(0, 0) - wr(0, 0)); if (d > *maxdiff) *maxdiff = d;

  d = std::abs(ur(0, 0)); if (d > *maxval) *maxval = d;
  d = std::abs(vr(0, 0)); if (d > *maxval) *maxval = d;
  d = std::abs(wr(0, 0)); if (d > *maxval) *maxval = d;

  d = std::abs(u(0, 0)); if (d > *maxpart) *maxpart = d;
  d = std::abs(v(0, 0)); if (d > *maxpart) *maxpart = d;
  d = std::abs(w(0, 0)); if (d > *maxpart) *maxpart = d;
}

/* Whole-field checksum, as OPS reductions so it works unchanged under MPI.
 * Each owned node is counted exactly once across all ranks, so the result is a
 * property of the FIELD, not of the decomposition -- which makes it a valid
 * comparison between different rank counts. */
void KerFluctChecksum(const ACC<double> &u, const ACC<double> &v,
                      const ACC<double> &w, double *sum2, double *amax) {
  const double q = u(0, 0) * u(0, 0) + v(0, 0) * v(0, 0) + w(0, 0) * w(0, 0);
  *sum2 += q;
  if (std::abs(u(0, 0)) > *amax) *amax = std::abs(u(0, 0));
  if (std::abs(v(0, 0)) > *amax) *amax = std::abs(v(0, 0));
  if (std::abs(w(0, 0)) > *amax) *amax = std::abs(w(0, 0));
}

#endif /* _GRID_KERNELS_H_ */
