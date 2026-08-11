/*
 * osem_io.h  --  per-timestep HDF5 output
 *
 * Same approach as the other apps in this series: the OPS particle API has no
 * HDF5 path, but the eddies have already been all-gathered onto every rank by
 * the reduction that compute_fluct needs anyway. So the writer costs one extra
 * copy and no communication of its own, and the eddy arrays come out ordered by
 * global id -- row i is eddy i in every frame, at any rank count.
 *
 * The grid dats go through the ordinary collective ops_fetch_dat_hdf5_file.
 */

#ifndef OSEM_IO_H
#define OSEM_IO_H

#include <cerrno>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <ops_hdf5.h>

/* Frames go in a subdirectory rather than the app root -- 50 files per run
   otherwise sit alongside the sources. Both plot scripts read from here. */
#define OSEM_OUTDIR "h5files"

/* The run constants below are read straight off the file-scope globals that
   osem_constants.h defines -- the same thing oSEM's io.h write_constants()
   does, and the same thing grid_kernels.h and particle_kernels.h already do
   with ny, nz, eddies and shape_norm. This header therefore has to be included
   AFTER osem_constants.h, which the driver does.
   Note that being an OPS constant is not what makes this work: ops_decl_const
   only makes a value visible inside KERNELS (device constant memory on the
   accelerated backends), and this is host code. niter, nprint and use_tbl are
   not ops_decl_const'd at all and are read here just the same.
   An osem_io_params struct used to carry these into write_osem_step. It bought
   nothing: its own aggregate initialiser was an unlabelled positional list of
   five ints then three doubles, exactly the transposition hazard a struct is
   supposed to remove, and one of its fields (NPRINT) was never read. */

/* Collective barrier built out of OPS: ops_reduction_result ends in an
   Allreduce, and ops_arg_reduce is what arms the handle. */
inline void ops_sync_barrier(ops_reduction handle) {
  ops_arg armed = ops_arg_reduce(handle, 1, "int", OPS_INC);
  (void)armed;
  int discard = 0;
  ops_reduction_result(handle, &discard);
}

inline void remove_stale_output(const char *prefix, int niter, int nprint,
                                ops_reduction sync) {
  /* Create the output directory before anything tries to write into it.
     Every rank calls mkdir and EEXIST is the expected answer on all but one --
     racing is harmless here and avoids needing a rank query just for this.
     ops_fetch_*_hdf5_file is collective, so the barrier below still has to be
     reached by everyone before the first write. */
  if (mkdir(OSEM_OUTDIR, 0777) != 0 && errno != EEXIST)
    ops_printf("warning: could not create %s/ (errno %d); "
               "HDF5 writes will fail\n", OSEM_OUTDIR, errno);

  char name[160];
  for (int s = nprint; s <= niter; s += nprint) {
    snprintf(name, sizeof(name), "%s/%s_%06d.h5", OSEM_OUTDIR, prefix, s);
    remove(name);
  }
  ops_sync_barrier(sync);
}

/**
 * One frame: the inlet fields, the complete eddy list, and the run constants.
 *
 * @param all  the gathered eddy buffer, NEDDY*NCOMP doubles, id-ordered
 */
inline void write_osem_step(ops_block &block, ops_dat &crd, ops_dat &uprime,
                            ops_dat &vprime, ops_dat &wprime,
                            const std::vector<double> &all,
                            const std::vector<double> &targ, int step) {

  char file[160];
  snprintf(file, sizeof(file), "%s/osem_output_%06d.h5", OSEM_OUTDIR, step);

  ops_fetch_block_hdf5_file(block, file);
  ops_fetch_dat_hdf5_file(crd, file);
  ops_fetch_dat_hdf5_file(uprime, file);
  ops_fetch_dat_hdf5_file(vprime, file);
  ops_fetch_dat_hdf5_file(wprime, file);

  const int N = eddies;

  /* Split the interleaved gather buffer so a plot script can read one
     quantity at a time. A few tens of kB per frame. */
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
  /* The tabulated target the profile should follow. A plane-averaged rms has
     nothing meaningful to compare against under the TBL profile -- the target
     is a PROFILE -- so each frame carries it.
     NEITHER the computed profile NOR the plane rms is written: both are
     derivable from the uprime/vprime/wprime fields above, and the plot script
     forms them there instead. Dropping the profile lost a kernel, an MPI
     reduction and the ny <= 100 restriction; dropping the plane rms lost a
     second KerFluctStats loop and its per-frame Allreduce from the output
     path. Python and the reduction agreed to 4.5e-15 relative over 50 frames.
     KerFluctStats itself stays -- the end-of-run report still uses it, and
     nothing is written at that point for a script to work from. */
  ops_write_const_hdf5("rms_target", 3 * (ny + 1), "double",
                       (char *)targ.data(), file);

  ops_write_const_hdf5("eddy_x", N, "double", (char *)ex.data(), file);
  ops_write_const_hdf5("eddy_y", N, "double", (char *)ey.data(), file);
  ops_write_const_hdf5("eddy_z", N, "double", (char *)ez.data(), file);
  ops_write_const_hdf5("eddy_r", N, "double", (char *)er.data(), file);
  /* All three signs: eps_x drives u', eps_y drives v', eps_z drives w', so a
     plot can overlay each panel with the signs that actually produced it. */
  ops_write_const_hdf5("eddy_sx", N, "double", (char *)sx.data(), file);
  ops_write_const_hdf5("eddy_sy", N, "double", (char *)sy.data(), file);
  ops_write_const_hdf5("eddy_sz", N, "double", (char *)sz.data(), file);
}

#endif /* OSEM_IO_H */
