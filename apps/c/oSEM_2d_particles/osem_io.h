#ifndef OSEM_IO_H
#define OSEM_IO_H

#include <cstdio>
#include <vector>

#include <ops_hdf5.h>

inline void write_osem_step(ops_block &block, ops_dat &d_grid, ops_dat &uprime,
                            ops_dat &vprime, ops_dat &wprime,
                            const std::vector<double> &all, int step) {

  char file[160];
  snprintf(file, sizeof(file), "osem_output_%06d.h5", step);

  ops_fetch_block_hdf5_file(block, file);
  ops_fetch_dat_hdf5_file(d_grid, file);
  ops_fetch_dat_hdf5_file(uprime, file);
  ops_fetch_dat_hdf5_file(vprime, file);
  ops_fetch_dat_hdf5_file(wprime, file);

  const int N = eddies;

  // Split the interleaved gather buffer so a plot script can read one quantity
  // at a time. A few tens of kB per frame.
  std::vector<double> ex(N), ey(N), ez(N), er(N), sx(N), sy(N), sz(N);
  for (int i = 0; i < N; i++) {
    ex[i] = all[NCOMP * i + E_X];
    ey[i] = all[NCOMP * i + E_Y];
    ez[i] = all[NCOMP * i + E_Z];
    er[i] = all[NCOMP * i + E_R];
    sx[i] = all[NCOMP * i + E_SX];
    sy[i] = all[NCOMP * i + E_SY];
    sz[i] = all[NCOMP * i + E_SZ];
  }

  double time = dt * step;
  const double box[4] = {eddy_y_min, eddy_y_max, eddy_z_min, eddy_z_max};

  ops_write_const_hdf5("neddy", 1, "int", (char *)&eddies, file);
  ops_write_const_hdf5("NY", 1, "int", (char *)&ny, file);
  ops_write_const_hdf5("NZ", 1, "int", (char *)&nz, file);
  ops_write_const_hdf5("NITER", 1, "int", (char *)&niter, file);
  ops_write_const_hdf5("timestep", 1, "int", (char *)&step, file);
  ops_write_const_hdf5("time", 1, "double", (char *)&time, file);
  ops_write_const_hdf5("u0ti", 1, "double", (char *)&u0ti, file);
  ops_write_const_hdf5("x_plane", 1, "double", (char *)&x_plane, file);
  ops_write_const_hdf5("box", 4, "double", (char *)box, file);
  ops_write_const_hdf5("use_tbl", 1, "int", (char *)&use_tbl, file);
  ops_write_const_hdf5("eddy_x", N, "double", (char *)ex.data(), file);
  ops_write_const_hdf5("eddy_y", N, "double", (char *)ey.data(), file);
  ops_write_const_hdf5("eddy_z", N, "double", (char *)ez.data(), file);
  ops_write_const_hdf5("eddy_r", N, "double", (char *)er.data(), file);
  ops_write_const_hdf5("eddy_sx", N, "double", (char *)sx.data(), file);
  ops_write_const_hdf5("eddy_sy", N, "double", (char *)sy.data(), file);
  ops_write_const_hdf5("eddy_sz", N, "double", (char *)sz.data(), file);
}

#endif /* OSEM_IO_H */
