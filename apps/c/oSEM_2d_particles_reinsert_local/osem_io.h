#ifndef OSEM_IO_H
#define OSEM_IO_H

#include <cstdio>
#include <vector>

#include <ops_hdf5.h>

inline void write_osem_step(ops_block &block, ops_dat &d_grid, ops_dat &uprime,
                            ops_dat &vprime, ops_dat &wprime, int step) {

  char file[160];
  snprintf(file, sizeof(file), "osem_output_%06d.h5", step);

  ops_fetch_block_hdf5_file(block, file);
  ops_fetch_dat_hdf5_file(d_grid, file);
  ops_fetch_dat_hdf5_file(uprime, file);
  ops_fetch_dat_hdf5_file(vprime, file);
  ops_fetch_dat_hdf5_file(wprime, file);

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
}

// per-rank box bounds, the local d_grid size, and which ranks sit on the boundary
inline void print_decomp(ops_particle p, ops_dat grid) {

  BoundingBox<double> *box = (BoundingBox<double> *)p->box_block;
  const double lo[2] = {box->getLocalMin().x, box->getLocalMin().y};
  const double hi[2] = {box->getLocalMax().x, box->getLocalMax().y};

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[p->block->index];
  if (!sb->owned) return;
  sub_dat_list sd = OPS_sub_dat_list[grid->index];

  int nr, curr_r;
  MPI_Comm_size(sb->comm, &nr);
  MPI_Comm_rank(sb->comm, &curr_r);

  if (curr_r == 0) {
    printf("\n=== DECOMP: %d ranks on %s, cart %d x %d ===\n",
           nr, p->block->name, sb->pdims[0], sb->pdims[1]);
    printf("rank coord |        y box (owned)        |        z box (owned)        "
           "| block owned   dat owned    alloc     halo y   halo z    disp    | id_m(y,z)  id_p(y,z)\n");
    fflush(NULL);
  }

  for (int r = 0; r < nr; r++) {
    if (r == curr_r) {
      printf("%4d (%d,%d) | [%+.6e,%+.6e) | [%+.6e,%+.6e) |"
             " %4d x%4d  %4d x%4d  %4d x%4d  %+d/%+d  %+d/%+d  (%d,%d) | (%s,%s) (%s,%s)\n",
             curr_r, sb->coords[0], sb->coords[1], lo[0], hi[0], lo[1], hi[1],
             sb->decomp_size[0], sb->decomp_size[1],
             sd->decomp_size[0], sd->decomp_size[1], grid->size[0], grid->size[1],
             grid->d_m[0], grid->d_p[0], grid->d_m[1], grid->d_p[1],
             sd->decomp_disp[0], sd->decomp_disp[1],
             sb->id_m[0] == MPI_PROC_NULL ? "NULL" : "rank",
             sb->id_m[1] == MPI_PROC_NULL ? "NULL" : "rank",
             sb->id_p[0] == MPI_PROC_NULL ? "NULL" : "rank",
             sb->id_p[1] == MPI_PROC_NULL ? "NULL" : "rank");
      fflush(NULL);
    }
    MPI_Barrier(sb->comm);
  }

  int mine[4] = {sb->id_m[0] == MPI_PROC_NULL, sb->id_m[1] == MPI_PROC_NULL,
                 sb->id_p[0] == MPI_PROC_NULL, sb->id_p[1] == MPI_PROC_NULL};
  std::vector<int> all(4 * nr, 0);
  MPI_Gather(mine, 4, MPI_INT, all.data(), 4, MPI_INT, 0, sb->comm);

  if (curr_r == 0) {
    const char *what[4] = {"id_m[0]", "id_m[1]",
                           "id_p[0]", "id_p[1]"};
    for (int k = 0; k < 4; k++) {
      printf("edge ranks, %-17s MPI_PROC_NULL:", what[k]);
      for (int r = 0; r < nr; r++)
        if (all[4 * r + k]) printf(" %d", r);
      printf("\n");
    }
    fflush(NULL);
  }
#else
  printf("\n=== DECOMP: serial ===\n");
  printf("box y [%+.6e,%+.6e)  z [%+.6e,%+.6e)\n", lo[0], hi[0], lo[1], hi[1]);
  printf("d_grid alloc %d x %d, halo y %+d/%+d, z %+d/%+d\n",
         grid->size[0], grid->size[1], grid->d_m[0], grid->d_p[0],
         grid->d_m[1], grid->d_p[1]);
  fflush(NULL);
#endif
}

#endif /* OSEM_IO_H */
