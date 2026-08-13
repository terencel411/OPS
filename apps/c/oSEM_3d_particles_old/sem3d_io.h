/*
 * sem3d_io.h -- oSEM_3d eddies as OPS particles
 *
 *  One self-contained .h5 file per output: the eddy-box block, its coordinate
 *  ops_dat, every particle dat, and the run constants.
 *
 *  Grid data goes through the standard ops_fetch_*_hdf5_file() calls, which are
 *  dimension-agnostic. Particle data has no equivalent in the OPS particle API,
 *  so it is gathered onto every rank and written as flat arrays with
 *  ops_write_const_hdf5().
 *
 *  This gather is I/O only. It runs once, at output, and nothing in the solve
 *  path depends on it -- which is the whole point of the exercise. The gather
 *  being removed is the per-timestep MPI_Allgatherv in
 *  ../oSEM_3d/opensbli.cpp:456-531, which exists so Kernel030 can see every
 *  eddy on every rank.
 */

#ifndef _SEM3D_IO_H_
#define _SEM3D_IO_H_

#include <stdio.h>
#include <vector>

#include <ops_hdf5.h>

#ifdef OPS_MPI
#include <ops_mpi_core.h>
#endif

/** Scalar parameters copied into every output file, so a plot script needs
    nothing but the .h5. */
struct sem3d_io_params {
  int    eddies;
  int    nex, ney, nez;   /* eddy-box grid, nodes */
  double delta;
  double radius;
  double dt;
  double u0;
  double span_z;
  double domain[6];       /* xlo xhi ylo yhi zlo zhi -- the eddy box */
};

/**
 * ops_write_const_hdf5() refuses to overwrite a dataset whose dim differs from
 * the one already in the file, and the particle count can change from step to
 * step. Dropping any stale file up front keeps re-runs reproducible.
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
 * Write the given particle ops_dats as flat arrays of
 * (global particle count) * dat->dim elements, laid out particle by particle.
 * ops_write_const_hdf5() is collective and expects the same data on every rank,
 * so the per-rank slices are all-gathered first.
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
    /* Gathered as raw bytes so one code path serves every particle dat type. */
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

inline void write_constants(const char *file_name,
                            const sem3d_io_params &p, int timestep) {
  ops_write_const_hdf5("eddies", 1, "int", (char *)&p.eddies, file_name);
  ops_write_const_hdf5("nex", 1, "int", (char *)&p.nex, file_name);
  ops_write_const_hdf5("ney", 1, "int", (char *)&p.ney, file_name);
  ops_write_const_hdf5("nez", 1, "int", (char *)&p.nez, file_name);
  ops_write_const_hdf5("timestep", 1, "int", (char *)&timestep, file_name);
  ops_write_const_hdf5("delta", 1, "double", (char *)&p.delta, file_name);
  ops_write_const_hdf5("radius", 1, "double", (char *)&p.radius, file_name);
  ops_write_const_hdf5("dt", 1, "double", (char *)&p.dt, file_name);
  ops_write_const_hdf5("u0", 1, "double", (char *)&p.u0, file_name);
  ops_write_const_hdf5("span_z", 1, "double", (char *)&p.span_z, file_name);
  ops_write_const_hdf5("domain", 6, "double", (char *)p.domain, file_name);
}

/** Write one output file: <prefix>_<timestep>.h5 */
inline void HDF5_IO_Write_eddy_box(ops_block &block, const char *prefix,
                                   int timestep, ops_dat &coords,
                                   ops_particle particle, ops_dat *particle_dats,
                                   int nparticle_dats,
                                   const sem3d_io_params &params) {
  double cpu_start, elapsed_start;
  ops_timers(&cpu_start, &elapsed_start);

  char name[96];
  sprintf(name, "%s_%06d.h5", prefix, timestep);
  remove_stale_hdf5_file(name);

  ops_fetch_block_hdf5_file(block, name);
  ops_fetch_dat_hdf5_file(coords, name);

  write_particle_dats(particle, particle_dats, nparticle_dats, name);
  write_constants(name, params, timestep);

  double cpu_end, elapsed_end;
  ops_timers(&cpu_end, &elapsed_end);
  ops_printf("wrote %s in %.4f s\n", name, elapsed_end - elapsed_start);
}

#endif /* _SEM3D_IO_H_ */
