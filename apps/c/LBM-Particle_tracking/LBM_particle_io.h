/*
 * LBM_particle_io.h
 *
 *  Periodic HDF5 output for the LBM + particle tracking application.  This
 *  follows the pattern used by apps/c/oSEM/OPS_oSEM_io.h: every call produces
 *  one self-contained .h5 file holding the block, the grid ops_dats and the
 *  simulation constants.
 *
 *  The grid data goes through the standard ops_fetch_*_hdf5_file() calls.  The
 *  particle data has no equivalent in the OPS particle API, so it is gathered
 *  onto every rank and written as flat arrays with ops_write_const_hdf5().
 */

#ifndef LBM_PARTICLE_IO_H
#define LBM_PARTICLE_IO_H

#include <stdio.h>
#include <vector>

#include <ops_hdf5.h>

#ifdef OPS_MPI
#include <ops_mpi_core.h>
#endif

/** Scalar simulation parameters copied into every output file. */
struct LBM_io_params {
  int NX;
  int NY;
  int Nsteps;
  int nprint;
  double dx;
  double dt;
  double tau;
  double OMEGA;
  double rho0;
  double cs;
};

/**
 * ops_write_const_hdf5() refuses to overwrite a dataset whose dim differs from
 * the one already in the file, and the particle count changes from step to
 * step.  Dropping any stale file up front keeps re-runs reproducible.
 */
inline void remove_stale_hdf5_file(const char *file_name) {
#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(OPS_MPI_GLOBAL, &my_rank);
  if (my_rank == 0)
    remove(file_name);
  MPI_Barrier(OPS_MPI_GLOBAL);
#else
  remove(file_name);
#endif
}

/**
 * Write the given particle ops_dats to file_name as flat arrays of
 * (global number of particles) * dat->dim elements, laid out particle by
 * particle.  ops_write_const_hdf5() is collective and expects the same data on
 * every rank, so the per-rank slices are all-gathered first.
 */
inline void write_particle_dats(ops_particle particle, ops_dat *dats, int ndats,
                                const char *file_name) {

  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR,
                         "ERROR: ops_dat is not linked to particle data");
    ops_get_data(dats[idat]);
  }

  int nlocal = (int)particle->no_particles;
  int nglobal = nlocal;

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[particle->block->index];
  int nranks;
  MPI_Comm_size(sb->comm, &nranks);

  std::vector<int> counts(nranks), displs(nranks);
  MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, sb->comm);

  nglobal = 0;
  for (int r = 0; r < nranks; r++) {
    displs[r] = nglobal;
    nglobal += counts[r];
  }
#endif

  ops_write_const_hdf5("no_particles", 1, "int", (char *)&nglobal, file_name);

  for (int idat = 0; idat < ndats; idat++) {
    ops_dat dat = dats[idat];

#ifdef OPS_MPI
    // Gather as raw bytes so the same code serves every particle dat type
    std::vector<char> gathered((size_t)nglobal * dat->elem_size);
    std::vector<int> byte_counts(nranks), byte_displs(nranks);
    for (int r = 0; r < nranks; r++) {
      byte_counts[r] = counts[r] * dat->elem_size;
      byte_displs[r] = displs[r] * dat->elem_size;
    }

    MPI_Allgatherv(dat->data, nlocal * dat->elem_size, MPI_BYTE,
                   gathered.data(), byte_counts.data(), byte_displs.data(),
                   MPI_BYTE, sb->comm);

    ops_write_const_hdf5(dat->name, nglobal * dat->dim, dat->type,
                         gathered.data(), file_name);
#else
    ops_write_const_hdf5(dat->name, nglobal * dat->dim, dat->type, dat->data,
                         file_name);
#endif
  }
}

/** Write the simulation constants, so a plot script needs nothing else. */
inline void write_constants(const char *file_name, const LBM_io_params &params,
                            int timestep) {
  ops_write_const_hdf5("NX", 1, "int", (char *)&params.NX, file_name);
  ops_write_const_hdf5("NY", 1, "int", (char *)&params.NY, file_name);
  ops_write_const_hdf5("Nsteps", 1, "int", (char *)&params.Nsteps, file_name);
  ops_write_const_hdf5("nprint", 1, "int", (char *)&params.nprint, file_name);
  ops_write_const_hdf5("timestep", 1, "int", (char *)&timestep, file_name);
  ops_write_const_hdf5("dx", 1, "double", (char *)&params.dx, file_name);
  ops_write_const_hdf5("dt", 1, "double", (char *)&params.dt, file_name);
  ops_write_const_hdf5("tau", 1, "double", (char *)&params.tau, file_name);
  ops_write_const_hdf5("OMEGA", 1, "double", (char *)&params.OMEGA, file_name);
  ops_write_const_hdf5("rho0", 1, "double", (char *)&params.rho0, file_name);
  ops_write_const_hdf5("cs", 1, "double", (char *)&params.cs, file_name);
}

/**
 * Write one HDF5 file per output step: LBM_output_<timestep>.h5
 */
inline void HDF5_IO_Write_lb_block_dynamic(
    ops_block &lb_block,
    int i,
    ops_dat &rho, ops_dat &u_x, ops_dat &u_y, ops_dat &x_grid,
    ops_particle particle, ops_dat *particle_dats, int nparticle_dats,
    const LBM_io_params &params
){
  double cpu_start0, elapsed_start0;
  ops_timers(&cpu_start0, &elapsed_start0);

  char name0[80];
  sprintf(name0, "LBM_output_%06d.h5", i + 1);
  remove_stale_hdf5_file(name0);

  // Writing OPS datasets
  ops_fetch_block_hdf5_file(lb_block, name0);
  ops_fetch_dat_hdf5_file(rho, name0);
  ops_fetch_dat_hdf5_file(u_x, name0);
  ops_fetch_dat_hdf5_file(u_y, name0);
  ops_fetch_dat_hdf5_file(x_grid, name0);

  // Writing particle datasets
  write_particle_dats(particle, particle_dats, nparticle_dats, name0);

  // Writing simulation constants
  write_constants(name0, params, i + 1);

  double cpu_end0, elapsed_end0;
  ops_timers(&cpu_end0, &elapsed_end0);
  ops_printf("-----------------------------------------\n");
  ops_printf("Time to write HDF5 file: %s: %lf\n", name0,
             elapsed_end0 - elapsed_start0);
  ops_printf("-----------------------------------------\n");
}

/**
 * Write the final state to a single file: LBM_output.h5
 */
inline void HDF5_IO_Write_lb_block(
    ops_block &lb_block,
    int i,
    ops_dat &rho, ops_dat &u_x, ops_dat &u_y, ops_dat &x_grid,
    ops_particle particle, ops_dat *particle_dats, int nparticle_dats,
    const LBM_io_params &params
){
  double cpu_start0, elapsed_start0;
  ops_timers(&cpu_start0, &elapsed_start0);

  char name0[80];
  sprintf(name0, "LBM_output.h5");
  remove_stale_hdf5_file(name0);

  // Writing OPS datasets
  ops_fetch_block_hdf5_file(lb_block, name0);
  ops_fetch_dat_hdf5_file(rho, name0);
  ops_fetch_dat_hdf5_file(u_x, name0);
  ops_fetch_dat_hdf5_file(u_y, name0);
  ops_fetch_dat_hdf5_file(x_grid, name0);

  // Writing particle datasets
  write_particle_dats(particle, particle_dats, nparticle_dats, name0);

  // Writing simulation constants
  write_constants(name0, params, i + 1);

  double cpu_end0, elapsed_end0;
  ops_timers(&cpu_end0, &elapsed_end0);
  ops_printf("-----------------------------------------\n");
  ops_printf("Time to write HDF5 file: %s: %lf\n", name0,
             elapsed_end0 - elapsed_start0);
  ops_printf("-----------------------------------------\n");
}

#endif /* LBM_PARTICLE_IO_H */
