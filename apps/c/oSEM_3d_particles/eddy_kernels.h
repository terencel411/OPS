/*
 * eddy_kernels.h -- oSEM_3d's eddies as OPS particles.
 * ACC<T> is the grid accessor c(comp,i,j,k); ACCP<T> is the particle accessor
 * p(comp), already positioned on the current particle. See the README.
 */

#ifndef _EDDY_KERNELS_H_
#define _EDDY_KERNELS_H_

// The three coordinate dats packed into one dim-3 dat, which is the shape
// ops_create_bounding_box and ops_decl_mapping require.
void KerPackCoords(ACC<double> &c, const ACC<double> &x0, const ACC<double> &x1,
                   const ACC<double> &x2) {
  c(0, 0, 0, 0) = x0(0, 0, 0);
  c(1, 0, 0, 0) = x1(0, 0, 0);
  c(2, 0, 0, 0) = x2(0, 0, 0);
}

// ../oSEM_3d's instantiate_eddies, minus the position, which seed_eddies has
// already set. The sign test differs deliberately -- see the README.
void KerInitEddy(ACCP<double> &r, ACCP<double> &eps, const ACCP<double> &rnd) {
  r(0) = radius;

  eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
}

// ../oSEM_3d's convect_eddies, statement for statement: advance the eddy, and
// on leaving the box put it back at the inlet with fresh y, z and signs.
void KerConvectEddies(ACCP<double> &e, ACCP<double> &eps,
                      const ACCP<double> &rnd) {

  e(0) = e(0) + increment;

  if (e(0) > eddy_x_max) {
    e(0) = eddy_x_min;
    e(1) = eddy_y_min + rnd(1) * (eddy_y_max - eddy_y_min);
    e(2) = eddy_z_min + rnd(2) * (eddy_z_max - eddy_z_min);

    eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
    eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
    eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
  }
}

// The gather, in place of the MPI_Allgatherv: each eddy adds itself into the
// slot its global id names. Must be ITERATE_LOCAL or ghosts double-count.
void KerPublishEddy(const ACCP<double> &e, const ACCP<double> &r,
                    const ACCP<double> &eps, const ACCP<int> &id, double *all) {
  const int s = NCOMP * id(0);
  all[s + E_X] += e(0);
  all[s + E_Y] += e(1);
  all[s + E_Z] += e(2);
  all[s + E_R] += r(0);
  all[s + E_SX] += eps(0);
  all[s + E_SY] += eps(1);
  all[s + E_SZ] += eps(2);
}

// Population check. Eddies recycle but are never created or destroyed.
void KerCountEddies(const ACCP<int> &id, int *count) {
  (void)id;
  *count += 1;
}

#endif /* _EDDY_KERNELS_H_ */
