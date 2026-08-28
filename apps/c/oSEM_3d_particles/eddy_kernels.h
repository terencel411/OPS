#ifndef _EDDY_KERNELS_H_
#define _EDDY_KERNELS_H_

// initialise the 3d grid for eddies
void KerInitGrid(ACC<double> &grid, const ACC<double> &x0, const ACC<double> &x1,
                   const ACC<double> &x2) {
  grid(0, 0, 0, 0) = x0(0, 0, 0);
  grid(1, 0, 0, 0) = x1(0, 0, 0);
  grid(2, 0, 0, 0) = x2(0, 0, 0);
}

// instantiate_eddies, minus the position, which seed_eddies has already set. 
void KerInitEddy(ACCP<double> &eddy_r, ACCP<double> &eddy_eps, const ACCP<double> &eddy_rng) {
  eddy_r(0) = radius;

  eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
  eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
  eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
}

// convect_eddies
void KerConvectEddies(ACCP<double> &eddy, ACCP<double> &eddy_eps,
                      const ACCP<double> &eddy_rng) {

  eddy(0) = eddy(0) + increment;

  if (eddy(0) > eddy_x_max) {
    eddy(0) = eddy_x_min;
    eddy(1) = eddy_y_min + eddy_rng(1) * (eddy_y_max - eddy_y_min);
    eddy(2) = eddy_z_min + eddy_rng(2) * (eddy_z_max - eddy_z_min);

    eddy_eps(0) = (eddy_rng(3) < 0.5) ? -1.0 : 1.0;
    eddy_eps(1) = (eddy_rng(4) < 0.5) ? -1.0 : 1.0;
    eddy_eps(2) = (eddy_rng(5) < 0.5) ? -1.0 : 1.0;
  }
}

// each eddy adds itself into the slot its global id names
void KerGatherEddies(const ACCP<double> &eddy, const ACCP<double> &eddy_r,
                    const ACCP<double> &eddy_eps, const ACCP<int> &eddy_id, double *eddy_all) {
  const int s = NCOMP * eddy_id(0);
  eddy_all[s + E_X] += eddy(0);
  eddy_all[s + E_Y] += eddy(1);
  eddy_all[s + E_Z] += eddy(2);
  eddy_all[s + E_R] += eddy_r(0);
  eddy_all[s + E_SX] += eddy_eps(0);
  eddy_all[s + E_SY] += eddy_eps(1);
  eddy_all[s + E_SZ] += eddy_eps(2);
}

// eddy count check
void KerCountEddies(const ACCP<int> &eddy_id, int *count) {
  (void)eddy_id;
  *count += 1;
}

#endif /* _EDDY_KERNELS_H_ */
