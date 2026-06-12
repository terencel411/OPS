/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file
  * @brief OPS mpi run-time support routines for particle structures
  * @author Valantis Tsinginos
  * @details Implements the runtime support routines for the OPS mpi backend
  *          for particle structures in host
  */

#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <math.h>
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_mpi_particle_core.h>

#include <string>
#include <assert.h>
#include <array>
#include <limits>


//Array for copying data to particles in reverse communications
int ntmp_size_max = 0;
char *temp = NULL;

static inline  uint64_t pack_pair(int tagI, int tagJ) {

  uint32_t a = (tagI < tagJ) ? tagI : tagJ;
  uint32_t b = (tagI < tagJ) ? tagJ : tagI;

  return ((uint64_t)a << 32) | b;
}

static int virtual_within(int bin[], int bin_limits[], double *xpos,
                          BoundingBox *box, int dim) {

  int flag = 0;

  for (int i = 0; i < dim; i++)
    if (bin[i] == bin_limits[2 * i] || bin[i] == bin_limits[2 * i + 1]) {
      flag = 1; break;
    }

  if (flag)
    return (int) box->isCoordinateInBoundingBox(xpos);

  return flag;
}

static int particle_is_within(const double *xpos, const double *region, const int dim)  {
  int flag  =  1;

  for (int i = 0; i < dim; i++) {
    if (xpos[i] < region[2 * i] || xpos[i] >= region[2 * i + 1]) return 0;
  }

  return flag;
}

static void  get_coord_point(double *xp, int *ilocal, int *d_m,
                             ops_point xmin, double *dx,
                             int dim, double skin, int stag) {

  xp[0] = xmin.x + (ilocal[0] - d_m[0]) * dx[0] + 0.5 * static_cast<double>(stag) * dx[0];
  xp[1] = xmin.y + (ilocal[1] - d_m[1]) * dx[1] + 0.5 * static_cast<double>(stag) * dx[1];
  if (dim == 3)
    xp[2] = xmin.z + (ilocal[2] - d_m[2]) * dx[2] + 0.5 * static_cast<double>(stag) * dx[2];
}

static void get_local_point(const int point, const int size[],const int d_m[],
                            const int dim,  int grid[]) {
  int address = point;
  for (int i = dim - 1; i >= 0; i--) {
    int prod = 1;
    for (int j = 0; j < i; j++)
      prod *= size[j];
    grid[i] = address / prod;
    address -= grid[i] * prod;
    grid[i] += d_m[i];
  }


}

static int local_decide_rebuild(const double *xPart, const double *xGrid, const double dx,
                                const int dim) {

  for (int isou = 0; isou < dim; isou++)
    if (fabs(xPart[isou] - xGrid[isou]) > 0.5 * dx + DBL_EPSILON) {
      return 1;
    }


  return 0;
}

static void _remove_particle_from_bins(int address, int i, int *binhead, int *bins) {

  int iPart = binhead[address];

  if (iPart == i)  {
    binhead[address] = bins[i];
  }
  else {
    int iPrev;
    while (iPart != i) {
      iPrev = iPart;
      iPart = bins[iPart];
    }

    bins[iPrev] = bins[i];
  }

  bins[i] = -1;
}

void _ops_construct_local_box_from_dat(ops_dat coords, double *grid_size, int dim,
                                       double *xmin, double *xmax) {

  int imin[OPS_MAX_DIM], imax[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];

  sub_block *sb = OPS_sub_block_list[coords->block->index];
  if (!sb->owned) return;

  for (int i = 0; i < dim; i++) {
    int d_m = coords->d_m[i] + OPS_sub_dat_list[coords->index]->d_im[i];
    int d_p = coords->d_p[i] + OPS_sub_dat_list[coords->index]->d_ip[i];
    imin[i] = -d_m;
    imax[i] = -d_m + (coords->size[i] - d_p + d_m) - 1;
    size[i] = coords->size[i];
  }

  double *data = (double *)coords->data;
  OPS_instance *instance = coords->block->instance;

  if (dim == 2) {
    if (instance->OPS_soa) {//TODO: Debug
      xmin[0] = *(data + imin[0] + imin[1] * size[0]);
      xmin[1] = *(data + imin[0] + imin[1] * size[0] + size[0] * size[1]);

      xmax[0] = *(data + imax[0] + imax[1] * size[0]);
      xmax[1] = *(data + imax[0] + imax[1] * size[0] + size[0] * size[1]);

 //     printf("xmin = [%f %f] xmax =[%f %f]\n", xmin[0], xmin[1], xmax[0], xmax[1]);

    }
    else {
      xmin[0] = *(data + imin[0] * coords->dim + imin[1] * coords->dim * size[0]);
      xmin[1] = *(data + 1 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]);

      /* xmax */
      xmax[0] = *(data + imax[0] * coords->dim + imax[1] * coords->dim * size[0]);
      xmax[1] = *(data + 1 + imax[0] * coords->dim + imax[1] * coords->dim * size[0]);
    }
  }
  else if (dim == 3) {
    if (instance->OPS_soa) {
      xmin[0] = * (data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]);
      xmin[1] = * (  data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]
                   + size[0] * size[1] * size[2]);
      xmin[2] = *(  data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]
                   + 2 * size[0] * size[1] * size[2]);

      //Getting xmax
      xmax[0] = *(data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]);
      xmax[1] = *(  data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]
                   + size[0] * size[1] * size[2]);
      xmax[2] = *(  data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]
                  + 2 * size[0] * size[1] * size[2]);
    }
    else {
      xmin[0] = *(  data + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);
      xmin[1] = *(  data + 1 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);
      xmin[2] = *(  data + 2 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);

      xmax[0] = *( data + imax[0] * coords->dim + imax[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
      xmax[1] = *(  data + 1 + imax[0] * coords->dim + imax[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
      xmax[2] = *(  data + 2 + imax[0] * coords->dim + imin[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
    }
  }
  else {
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: The size of the physical domain"
                                         "must be 2 or 3\n");
  }

  if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
    printf("Rank %d ix = [%d %d]x[%d %d]\n", ops_get_proc(), imin[0], imax[0],
                                             imin[1], imax[1]);

    printf("Rank %d x = [%12.9e %12.9e]x[%12.9e %12.9e]\n", ops_get_proc(), xmin[0],
                                                            xmax[0], xmin[1], xmax[1]);
  }
}

void _ops_particle_allocate_tmp_array(int size_elem) {

  if (size_elem == 0) return;

  ntmp_size_max = size_elem * OPS_MAX_PART;
  temp = (char *) ops_malloc(ntmp_size_max);

}

void _ops_particle_free_tmp_array() {
  ops_free(temp);
  ntmp_size_max = 0;
}


void _ops_particle_setup_tmp_array(OPS_instance *instance) {

  int nhalo_groups = instance->OPS_particle_halo_data_index;
  ops_particle_halo_data *halos = instance->OPS_particle_halo_data_list;

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Setting up tmp array before domain "
                                          "partition");

  if (nhalo_groups == 0) return;

  int nelem_max = 0;
  for (int ihalo = 0; ihalo < nhalo_groups; ihalo++) {

    if (halos[ihalo]->halo_type !=  OPS_EXCHANGE_PARTICLE_DAT) continue;
    sub_block *sb = OPS_sub_block_list[halos[ihalo]->to->block->index];
    if (!sb->owned) continue;

    nelem_max = MAX(nelem_max, halos[ihalo]->to->elem_size);
  }

  if (nelem_max > 0)
    _ops_particle_allocate_tmp_array(nelem_max);


}

void _ops_particle_number_of_particles_in_range(const int *region, const int *binhead,
                                                const int *bins, const int *size,
                                                int *nsend) {
  int nwithin = 0;
  for (int k = region[4]; k < region[5];k++) {
    for (int  j = region[2]; j < region[3]; j++) {
      for (int i = region[0]; i < region[1]; i++) {
        int address = i +  j *size[0] + k * size[0] * size[1];
        int iPart = binhead[address];
        while (iPart != -1) {
          nwithin++;
          iPart = bins[iPart];
        }
      }
    }
  }

  (*nsend) = nwithin;
}

void _ops_particle_number_of_particles_in_range(BoundingBox *box,int dim,double  *xcrds,
                                                int noParticles, int *nsend) {
  int nwithin = 0;
  double xpoint[OPS_MAX_DIM];


  for (int i = 0; i < noParticles; i++) {
    for (int isou = 0; isou < dim; isou++)
      xpoint[isou] = xcrds[dim * i + isou];
    bool isin = box->isCoordinateInBoundingBox(xpoint);
    if (isin) nwithin++;
  }

  (*nsend) = nwithin;
}

void _ops_particle_number_of_particles_in_range(double *region, int dim, double *xcrds,
                                                int ifirst, int ilast, int *nwithin) {

  int n_in = 0;
  for (int i = ifirst; i < ilast; i++)  {
    if (particle_is_within(xcrds + dim * i, region, dim)) n_in++;
  }

  *nwithin = n_in;
}

void _ops_particle_mapped_into_region(const int *region, const int *binhead,
                                      const int *bins, const int *size,
                                      int *sendlist) {

  int nwithin = 0;
  for (int k = region[4]; k < region[5]; k++) {
    for (int j = region[2]; j < region[3]; j++) {
      for (int i = region[0]; i < region[1]; i++) {
        int address = i + j * size[0] + k * size[0] * size[1];
        int iPart = binhead[address];
        while (iPart != - 1) {
          sendlist[nwithin] = iPart;
          nwithin++;
          iPart = bins[iPart];
        }
      }
    }
  }

}

void _ops_particle_mapped_into_region(BoundingBox *box, int dim, double  *xcrds,
                                      int noParticles, int *sendlist) {
  int nwithin = 0;
  double xpoint[OPS_MAX_DIM];

  for (int  iPart  = 0; iPart < noParticles; iPart++) {
    for (int isou = 0; isou < dim; isou++)
      xpoint[isou] = xcrds[dim * iPart + isou];
    bool isin = box->isCoordinateInBoundingBox(xpoint);
    if (isin) {
      sendlist[nwithin] = iPart;
      nwithin++;
    }
  }
}

void _ops_particle_remove_from_region(BoundingBox *box, const double *env,
                                      const double *xcrds, int *mark_deletion,
                                      size_t noParticles,
                                      const int dim, int *sendlist) {

  int nwithin = 0;

  for (size_t ipart = 0; ipart < noParticles; ipart++) {

    if (env != nullptr) {
      if (env[ipart] < 0) continue;
    }

    if (mark_deletion[ipart] != 1) continue;

    bool decide = box->isCoordinateInBoundingBox(xcrds + dim * ipart);
    if (decide) {
      mark_deletion[ipart] = 2;
      sendlist[nwithin] = ipart;
      nwithin++;
    }

  }


}


//TODO: Do we have similar function for sequential (??)
void _ops_particle_mark_for_removal(BoundingBox *box, double *xcrds, int *mark_del,
                                    int dim, int first, int last) {

  for (int ipart = first; ipart < last; ipart++) {
    mark_del[ipart] = 0;
    bool iswithin = box->isCoordinateInBoundingBox(xcrds + dim * ipart);
    if (!iswithin)
      mark_del[ipart] = 1;
  }
}


void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin, ops_dat part_to_bin) {

  int size[dim];
  size_t no_elems{1};
  for (int i = 0; i < dim; i++) {
    size[i] = binhead->size[i];
    no_elems *= size[i];
  }

  /* Initialize elements */
  memset(binhead->data, -1, sizeof(int) * no_elems);

  memset(bin->data, -1, sizeof(int) * Np);
  memset(part_to_bin->data, -1, sizeof(int) * Np);

  int *bin_data = (int *)bin->data;
  int *part_2_bin = (int *)part_to_bin->data;
  int* binhead_data = (int *)binhead->data;


  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positive value Rank %d: Particle %d (%f %f)\n", ops_get_proc(), i,
                 xp_data[dim * i], xp_data[dim * i + 1]);
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
    part_2_bin[i] = ibin;
  }
}

int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {

  int nhistories = particle->nhistories;

  if (enforce) {
    for (int i = 0; i < nhistories; i++) particle->histories[i]->flag_update = true;
    return 1;
  }

  if (map->decide) {
    for (int i = 0; i < nhistories; i++) particle->histories[i]->flag_update = true;
    return 1;
  }

  if (!OPS_sub_block_list[particle->block->index]->owned) return 0;

  int flag = 0;
  double *x = (double *)particle->particle_pos_dat->data;
  double *x_old = (double *)map->pos_old->data;\

  double dx = map->skin;

  int dim = particle->block->dims;
  int nParticles = (int) particle->no_particles;

  for (int i = 0; i < nParticles; i++) {
    int a1 = 0;
    for (int idir = 0; idir < dim; idir++) {
      if (fabs(x[i * dim + idir] - x_old[i * dim + idir]) > 0.5 * dx + DBL_EPSILON) {
        a1 = 1;
        break;
      }
    }
    if (a1 == 1) {
      flag = 1;
      break;
    }
  }

  int global_flag = 0;
  MPI_Allreduce(&flag, &global_flag, 1, MPI_INT, MPI_MAX,
                OPS_sub_block_list[particle->block->index]->comm);

  if (global_flag) { map->decide = true;
    for (int i = 0; i < nhistories; i++) particle->histories[i]->flag_update = true;
  }
  return global_flag;
}


void _ops_particle_update_map_int_halos(ops_particle_mapping map, ops_particle particle,
                                       int ifirst, int ilast) {
  map->nParticles = particle->no_particles + particle->no_virtual;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[OPS_MAX_DIM];

  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }
  xmin.x = xmin.x + static_cast<double>(d_m[0]) * dx[0];
  xmin.y = xmin.y + static_cast<double>(d_m[1]) * dx[0];

  xmax.x = xmax.x + static_cast<double>(d_p[0]) * dx[0];
  xmax.y = xmax.y + static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
    xmin.z = xmin.z + static_cast<double>(d_m[2]) * dx[2];
    xmax.z = xmax.z + static_cast<double>(d_p[2]) * dx[2];
  }



  //TODO: Check if we reset the maps at the beginning
  double *xp = (double *)particle->particle_pos_dat->data;
  double *xp_old = (double *)map->pos_old->data;
  int *binhead_data = (int *)map->binhead->data;
  int *bin_data = (int *) map->bin->data;
  int *part2bin = (int *) map->parts_to_grid;

  int dim = particle->particle_pos_dat->dim;
  int *size = map->binhead->size;
  for (int i = ifirst; i < ilast; i++) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp + dim * i);
    if (ibin < 0) continue;

    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
    part2bin[i] = ibin;

    //Set particle to map old //
    int ilocal[OPS_MAX_DIM];
    get_local_point(part2bin[i], map->binhead->size, d_m, dim, ilocal);
    get_coord_point(xp_old + i * dim, ilocal, d_m, xmin, dx, dim, map->skin, 1);
  }

}


/*-----------------------------------------------------------------------*
 * Function to update maps of actual particles after particle exchange
 *-----------------------------------------------------------------------*/


//TODO: Check its usage: If necessary remove it
void _ops_particle_map_from_exchange(ops_particle_mapping map, ops_particle
                                     particle, int ifirst, int ilast) {

  /*Sanity check for block ownership */
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  //TODO: Check for re-allocations

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int *mark_deletion = (int *)particle->mark_deletion;

  //Get grid structure
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[OPS_MAX_DIM];

  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  /* Upgrade due to virtual bins */
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }
  xmin.x += static_cast<double>(d_m[0]) * dx[0];
  xmax.x += static_cast<double>(d_p[0]) * dx[0];

  xmin.y += static_cast<double>(d_m[1]) * dx[1];
  xmax.y += static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(d_m[2]) * dx[2];
   xmax.z += static_cast<double>(d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = ifirst; i < ilast; i++) {
    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                    xpos + dim * i);


    if (!particle->box_block->isCoordinateInBoundingBox(xpos + i * dim)) {
      bin2grid[i] = -1;
      bins[i] = -1;
      mark_deletion[i] = 1;
      continue;
    }

    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m, dim, ilocal);


    get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);


  }

  for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = true;

}

int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return 0;

  map->decide = false;

  int stag = 1;
  int changed{0};

  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part_to_grid = (int *)map->parts_to_grid->data;
  int *mark_del = particle->mark_deletion;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  int *size = map->binhead->size;

  for (int i = 0; i < dim; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  int local_flag = 0;
  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  double dx[particle->block->dims];

  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  int nmapping =
      (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                             particle->no_particles + particle->no_virtual;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  xmin.x -= dx[0];
  xmin.y -= dx[1];

  xmax.x += dx[0];
  xmax.y += dx[1];

  if (dim == 3) {
    xmin.z += dx[2];
    xmax.z += dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  int rmv_limits[2 * OPS_MAX_DIM];
  int exch_limits[2 * OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    exch_limits[2 *i] = (d_m[i]< 0) ? 0 : -size[i]; //Constant for the model
    exch_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - d_p[i] - d_p[i] : 2 * size[i];
    rmv_limits[2 * i] = 0;
    rmv_limits[2 * i + 1] = size[i] + d_m[i] - d_p[i] - 1;
  }

  int nactual = particle->no_particles;

  for (size_t i = 0; i < particle->no_particles; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold  + i * dim, map->skin,
                                    dim); //TODO:

    if (flag) {
      int address = part_to_grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, d_m, dim, ilocal);

      _remove_particle_from_bins(address, i, binhead, bins); //TODO:

      int del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                       xpos + i * dim, particle->box_block);

      if (del_flag) {
        part_to_grid[i] = -1;
        local_flag = 1;
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
      }

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);
      for (int ih = 0; ih < particle->nhistories; ih++)
        particle->histories[ih]->flag_update
         = (particle->histories[ih]->flag_update) ? true : false;

      if (address < 0) local_flag = 1;

      bins[i] = binhead[address];
      binhead[address] = i;

      part_to_grid[i] = address;

      int ilocal[OPS_MAX_DIM];
      get_local_point(address, map->binhead->size, d_m, dim, ilocal);

      get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

      bool flag_build = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new,exch_limits, dim);
      if (!local_flag) local_flag = (int )flag_build;

    }
  }

  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

  for (int i = ifirst; i < nmapping; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin, dim);

    if (flag) {
      int address = part_to_grid[i];
      _remove_particle_from_bins(address, i, binhead, bins);
      part_to_grid[i] = -1;

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                 xpos + dim * i);

      if (address < 0) {
        local_flag = 1;
        continue;
      }

      get_local_point(address, map->binhead->size, d_m,
                      dim, ilocal);

      int flag_in = virtual_within(ilocal, rmv_limits, xpos + dim * i,
                                   particle->box_block, dim);

      if (flag_in) { //Virtual become actual
        nactual++;
        nvirtual_act--;

        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;

        _ops_particle_swap_data(particle->particle_pos_dat->data, i,
                                nactual - 1, particle->particle_pos_dat->elem_size);

        if (particle->particle_envelope != nullptr) {
           _ops_particle_swap_data(particle->particle_envelope->data, i, nactual - 1,
                                   particle->particle_envelope->elem_size);
         }

        if (particle->ids != nullptr)
          _ops_particle_swap_data(particle->ids->data, i, nactual - 1,
                                   particle->ids->elem_size);

         for (int idat = 0; idat < particle->particle_dat_index; idat++) {
           ops_dat dat = particle->particle_dat[idat];
           _ops_particle_swap_data(dat->data, i, nactual - 1, dat->elem_size);
         }

         particle->mark_deletion[i] = 0;
         _ops_particle_swap_data((char *)particle->mark_deletion, i, nactual - 1,
                                 sizeof(int));

         _ops_particle_swap_mapping_data(map, i, nactual - 1);
         _ops_particle_swap_data(map->pos_old->data, i, nactual -1, \
                                 map->pos_old->elem_size);

         local_flag = 1;
      }

      if (!local_flag) {
        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;

        get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

      }
    }
  }

  if (nactual > particle->no_particles)
    particle->no_particles = nactual;

  int global_flag = 0;
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, sb->comm);

  map->decide = (bool) global_flag;
  if (map->decide) for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = true;

  return global_flag;
}

int _ops_particle_decide_build_only_local_uniform(ops_particle_mapping map,
                                                  ops_particle         particle) {

  map->decide = false;
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return 0;

//  printf("Building zones\n");

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;
  int *mark_del = particle->mark_deletion;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  int *size = map->binhead->size;
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  int nmapping = particle->no_particles;
  int local_flag = 0;

  /* Compute bounding points for mapping structures */
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();
  double dx[particle->block->dims];

  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x += static_cast<double>(d_m[0]) * dx[0];
  xmax.x += static_cast<double>(d_p[0]) * dx[0];

  xmin.y += static_cast<double>(d_m[1]) * dx[1];
  xmax.y += static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(d_m[2]) * dx[2];
   xmax.z += static_cast<double>(d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  int dim = particle->block->dims;
  int nlimits = 2 * dim;
  int border_limits[2 * OPS_MAX_DIM];
  int rmv_limits[2 * OPS_MAX_DIM];

  //TODO: Need adaptation
  for (int i =  0; i < particle->block->dims; i++) {
    border_limits[2 *i] = (d_m[i]< 0) ? 0 : -size[i]; //Constant for the model
    border_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - d_p[i] - d_p[i] : 2 * size[i];
    rmv_limits[ 2 * i ] = 0;
    rmv_limits[2 * i + 1] = size[i] + d_m[i] - d_p[i] - 1;
  }

  for (int i = 0; i < nmapping; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, dx[0],
                                    dim);

    if (flag) {
      int address = part2grid[i];
      int iPart = binhead[address];

      _remove_particle_from_bins(address, i, binhead, bins);



      //Add new address prior deletion
      get_local_point(address, map->binhead->size, d_m, dim, ilocal);


      int del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                      xpos + i * dim, particle->box_block);

/*
      printf("Proc %d: Build list decided for %d [%f %f] xold =[%f %f] and del_flag  = %d\n", ops_get_proc(),
             i, *(xpos + i * dim), *(xpos + i * dim + 1),
             *(xold + i * dim), *(xold + i * dim + 1), del_flag);
      printf("Rank %d rmv_limits: [-INF %d]-[%d INF] x [-INF %d]-[%d INF] local = [%d %d] wiht address =%d\n", ops_get_proc(),
             rmv_limits[0], rmv_limits[1], rmv_limits[2], rmv_limits[3], ilocal[0], ilocal[1], address);
*/

      if (del_flag) {
        part2grid[i] = -1;
        local_flag = 1;

        particle->mark_deletion[i] = 1;
        continue;
      }

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                  xpos + dim * i);

      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;

      //Check for particle moving in or out of exchange zone
      get_local_point(address, map->binhead->size, d_m,
                       dim, ilocal_new);

      get_coord_point(xold + dim * i, ilocal_new, d_m, xmin, dx, dim, map->skin, 1); //TODO: Check both

      bool flag_build = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new,
                                                             border_limits, dim);

      if (!local_flag) local_flag = (int) flag_build;

    }
  }

  int global_flag = 0;

  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, sb->comm);

  map->decide = (bool) global_flag;
  for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = true;

  return global_flag;
}

void _ops_particle_remap_virtual(ops_particle_mapping map, ops_particle particle,
                                 int istart, int ilast) {
  //Sanity checks
  if (istart < particle->no_particles)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function for mapping virtual "
                                             "called for actual particles");

  if (ilast > particle->no_particles + particle->no_virtual)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function called for non-existing "
                                             " particle");

  if (map->mapping_type != OPS_WITH_VIRTUAL) return;

  sub_block *sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  int stag = 1;

  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;
  int *size = map->binhead->size;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];

  for (int i = 0; i < dim; i++) {
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
  }

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];

  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = istart; i < ilast; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin,
                                        dim);

    if (flag) {
      int address = part2grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, d_m, dim, ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                  xpos+ dim * i);


      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;

      get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

    }
  }
}
void  _ops_particle_build_map_to_dir(int idir,ops_particle particle,
                                    ops_int_particle_halos halo,
                                    const ops_dat binhead, const ops_dat bin,
                                    const sub_block_list sb) {

  if (!sb->owned) return;

  int *bin_head = (int *)binhead->data;
  int *part2bin = (int *)bin->data;

  if (sb->id_m[idir] == MPI_PROC_NULL) {
    halo->nforward_neg[0] = 0;
  }
  else {

    int nforward = 0;
    for (int k = halo->region_neg[4]; k < halo->region_neg[5]; k++) {
      for (int j = halo->region_neg[3]; j < halo->region_neg[2]; j++) {
        for (int i = halo->region_neg[0]; i < halo->region_neg[1]; i++) {
          int iPart = bin_head[ k * binhead->size[0] * binhead->size[1]
                              + j * binhead->size[0] + i];
          while (iPart != - 1) {
            nforward++;
            if (nforward > halo->nalloc_max_neg) {
              halo->nalloc_max_neg += 100;
              halo->particle_send_neg =
                  (int *)ops_realloc(halo->particle_send_neg,
                                     sizeof(int) * halo->nalloc_max_neg);
            }
            halo->particle_send_neg[nforward - 1] = iPart;
            iPart = part2bin[iPart];

          }
        }
      }
    }

    halo->nforward_neg[0] = nforward;
  }

  // Shift into positive direction
  if (sb->id_p[idir] == MPI_PROC_NULL) {
    halo->nforward_pos[0] = 0;
  }
  else {
    int nforward = 0;
    for (int k = halo->region_pos[4]; k < halo->region_pos[5]; k++) {
      for (int j = halo->region_pos[2]; halo->region_pos[3]; j++) {
        for (int i = halo->region_pos[0]; halo->region_pos[1]; i++) {
          int iPart = bin_head[k * binhead->size[0] * binhead->size[1]
                              + j * binhead->size[0] + i];

          while (iPart != -1) {
            nforward++;
            if (nforward > halo->nalloc_max_pos) {
              halo->nalloc_max_pos += 100;
              halo->particle_send_pos =
                  (int *)ops_realloc(halo->particle_send_pos,
                                     sizeof(int) * halo->nalloc_max_pos);
            }
            halo->particle_send_pos[nforward-1] = iPart;
            iPart = part2bin[iPart];
          }
        }
      }
    }

    halo->nforward_pos[0] = nforward;
  }

  //Sending and receiving data structures
  MPI_Request request[4];
  MPI_Isend(&halo->nforward_neg, 1, MPI_INT,
            sb->id_m[idir], idir, sb->comm,
            & request[0]);
  MPI_Isend(&halo->nforward_pos, 1, MPI_INT,
            sb->id_p[idir], idir + OPS_MAX_DIM, sb->comm,
            & request[0]);


  MPI_Irecv(&halo->nrecv_pos, 1, MPI_INT,
            sb->id_p[idir], idir, sb->comm,
            &request[2]);

  MPI_Irecv(&halo->nrecv_neg, 1, MPI_INT,
            sb->id_m[idir], idir + OPS_MAX_DIM, sb->comm,
            &request[3]);

  MPI_Status status[4];
  MPI_Waitall(2, &request[2], &status[2]);

  //Update structures: TODO



  halo->irecv_neg[0] = particle->no_particles + particle->no_virtual;
  halo->irecv_pos[0] = halo->irecv_neg[0] + halo->nrecv_neg[0];

  MPI_Waitall(2, &request[0], &status[0]);

  if (particle->no_particles + particle->no_virtual > particle->Nmax) {
    ops_particle_realloc_data(particle, particle->no_particles + particle->no_virtual);
  }


}

void _ops_particle_build_local_uniform(ops_particle_mapping map, ops_particle particle) {
  size_t Np = particle->no_particles + particle->no_virtual;


  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  map-> nParticles = Np;

  //Initialize structures
  int prod = 1;
  for (int i = 0; i < particle->block->dims; i++) prod *= map->binhead->size[i];
  for (int i = 0; i < prod;i++)
    ((int *)map->binhead->data)[i] = -1;

  for (int i = 0; i < Np; i++) {
    ((int *)map->bin->data)[i] = -1;
    ((int *)map->parts_to_grid->data)[i] = -1;
  }

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  double dx[particle->block->dims];

  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x = xmin.x + static_cast<double>(d_m[0]) * dx[0];
  xmin.y = xmin.y + static_cast<double>(d_m[1]) * dx[1];

  xmax.x = xmax.x + static_cast<double>(d_p[0]) * dx[0];
  xmax.y = xmax.y + static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
    xmax.z = xmax.z + static_cast<double>(d_p[2]) * dx[2];
    xmin.z = xmin.z + static_cast<double>(d_m[2]) * dx[2];
  }

  _ops_build_uniform_dats(1, particle->block->dims, map->grid, particle->particle_pos_dat,
                          Np, dx, xmin, xmax, map->binhead, map->bin, map->parts_to_grid);

  int *part_to_bin = (int *)map->parts_to_grid->data;

  double *xold = (double *)map->pos_old->data;
  int dim = particle->block->dims;
  int ilocal[OPS_MAX_DIM];
  for (int i = 0; i < Np; i++) {
    int address = part_to_bin[i];
    get_local_point(address, map->binhead->size, d_m, particle->block->dims, ilocal);
    get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

  }


}

void _ops_particle_map_validation(ops_particle_mapping map) {

  //No need to check for matching decompositions

  ops_block block = map->particle->block;
  sub_block *sb = OPS_sub_block_list[block->index];

  if (!sb->owned) return;

  //TODO: Check if we can remove it
 // if (map->grid == nullptr) return;

  int stride_flag = 0;
  for (int i = 0; i < map->particle->block->dims; i++) {
    int str = (map->binhead->stride[i] > 1) ? 1 : 0;
    stride_flag += str;
  }

  for (int i = 0; i < map->particle->block->dims; i++) {
    int d_m = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    int d_p = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
    int size_loc = map->binhead->size[i] + d_m - d_p;

    if (map->binhead->stride[i] != 1) {
      int size_fine = (sb->id_p[i] != MPI_PROC_NULL) ? sb->decomp_size[i] : sb->decomp_size[i] - 1;
      double stride_meas = (double) size_fine / size_loc;
      if (fabs(stride_meas - ((double) map->binhead->stride[i])) > 1.e-12)
        throw OPSException(OPS_RUNTIME_ERROR,"Error: A non-uniform map grid is generated (Due to non-matching projections)");
    }

    map->dx[i] = (map->particle->box_block->getMaxCoordDir(i) - map->particle->box_block->getMinCoordDir(i))
            / static_cast<double>(size_loc);

  }


}


void  _ops_particle_init_map(ops_particle_mapping map) {

  int size = 1;
  for (int i = 0; i < map->binhead->block->dims; i++)
    size *= map->binhead->size[i];


  int *binhead = (int *)map->binhead->data;
  for (int i = 0; i < size; i++)
    binhead[i] = -1;
}

bool  ops_get_bounding_box_local_to_global(ops_block block,double* xmin,double *xmax,double* xglb_min, double* xglb_max)
{
  sub_block *sb = OPS_sub_block_list[block->index];
  if (!sb->owned)
    return false;

  MPI_Allreduce(xmin, xglb_min, 3, MPI_DOUBLE, MPI_MIN, sb->comm);
  MPI_Allreduce(xmax, xglb_max, 3, MPI_DOUBLE, MPI_MAX, sb->comm);

  return true;
}

void _ops_particle_setup_map(ops_particle particle, ops_particle_mapping map) {


  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  if (particle->no_particles == 0) return;

  map->nParticles = particle->no_particles;

  /* Data initialization */
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *) map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  size_t binsize = 1;
  for (int i = 0; i < particle->block->dims; i++)
    binsize *= map->binhead->size[i];

  memset(binhead, -1, sizeof(int) * binsize);
  memset(bin2grid, -1, sizeof(int) * particle->no_particles);
  memset(bins, -1, sizeof(int) * particle->no_particles);


  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }
  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  //Upate xmin and xmax due to special conditions
  xmin.x += static_cast<double>(d_m[0]) * dx[0];
  xmax.x += static_cast<double>(d_p[0]) * dx[0];

  xmin.y += static_cast<double>(d_m[1]) * dx[1];
  xmax.y += static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(d_m[2]) * dx[2];
   xmax.z += static_cast<double>(d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = 0; i < particle->no_particles; i++) {
    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    bin2grid[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);
    get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

  }

}

void _ops_particle_setup_map_virtual(ops_particle particle, ops_particle_mapping map) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  if (particle->no_virtual == 0) return;

  map->nParticles += particle->no_virtual;

  //Initialize structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }


  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  //Upate xmin and xmax due to special conditions
  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = particle->no_particles;
           i < particle->no_particles + particle->no_virtual; i++) {
    bins[i] = -1;
    bin2grid[i] = -1;

    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    if (address < 0) continue;
    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);

    get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

  }

}

void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map) {

 // printf("Rank %d: nfirst =%d and to_map = %d\n",ops_get_proc(), ifirst, n_to_map);
  map->nParticles = ifirst + n_to_map;

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  //Upate xmin and xmax due to special conditions

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < map->binhead->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  xmin.x += static_cast<double>(d_m[0]) * dx[0];
  xmax.x += static_cast<double>(d_p[0]) * dx[0];

  xmin.y += static_cast<double>(d_m[1]) * dx[1];
  xmax.y += static_cast<double>(d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(d_m[2]) * dx[2];
   xmax.z += static_cast<double>(d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *) particle->particle_pos_dat->data;
  double *xold = (double *) map->pos_old->data;

  for (int i = ifirst; i < ifirst + n_to_map; i++) {

    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);
    if (address < 0) printf("Particle %d address = %d xpos =[%f %f] dx =%12.9e dy = %12.9e\n",i, address, xpos[i* dim],
                            xpos[i * dim + 1], (xpos[i * dim] - xmin.x)/dx[0], (xpos[i * dim + 1] - xmin.y)/dx[1]);

    if (address < 0) continue;
    bin2grid[i] = address;

    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);


    //TODO:
    get_coord_point(xold + dim * i, ilocal, d_m, xmin, dx, dim, map->skin, 1);

  }


}


void _ops_particle_halo_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                   int nhalos, ops_particle_halo_exchange  halo_info,
                                   int *ntot_bites, int flag) {

  int *sendlist = halo_info->sendlist;
  int nsend = halo_info->nsend;

  int nsend_bites = 0;

  for (int ihalo = 0; ihalo < nhalos; ihalo++) {

    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->from;
      int nbites = dat->elem_size;
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        memcpy(buff + nsend_bites, dat->data + ipart * nbites, nbites);
        nsend_bites += dat->elem_size;
      }
    }
    else if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_HISTORY && flag) {
      ops_neighbor_history history = halo_data[ihalo]->history_from;
      ops_dat data = history->data;
      ops_dat npartners = history->n_partnersI;
      ops_dat partnersI = history->partnersI;

      //Part I: Shift npartners
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        int nbites = history->n_partnersI->elem_size;
        memcpy(buff + nsend_bites, history->n_partnersI->data + ipart * nbites, nbites);
        nsend_bites += nbites;
      }

      //Part II: Shift partnersI only
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        int npartners = ((int *)history->n_partnersI->data)[ipart];
        int ishift = ipart * history->partnersI->elem_size;
        int nbites = npartners * history->partnersI->type_size;
        memcpy(buff + nsend_bites, history->partnersI->data + ishift, nbites);
        nsend_bites += nbites;
      }

      //Part III: Pack contacts
      int nbites = history->data->elem_size;
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        for (int j = 0; j < ((int *)history->n_partnersI->data)[ipart]; j++) {
          int index = ((int *) history->indexI->data)[ipart * history->num_neighsI + j];
          memcpy(buff + nsend_bites, history->data->data + index * nbites, nbites);
          nsend_bites += nbites;
        }
      }

    }
  }

  (*ntot_bites) = nsend_bites;

}

//Only mpi: TODO: Check if there is a similar function for sequential
void _ops_particle_halo_dat_to_buf(char *buff, ops_dat dat,
                                   ops_particle_halo_exchange halo_info,
                                   int *ntot_bites) {

  int *sendlist  = halo_info->sendlist;
  int nsend = halo_info->nsend;

  int nsend_bites = 0;
  int nbites = dat->elem_size;
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];
    memcpy(buff + nsend_bites, dat->data + ipart * nbites, nbites);
    nsend_bites += dat->elem_size;
  }

  (*ntot_bites) = nsend_bites;
}

void _ops_particle_intra_dat_to_buff(char *buff,  ops_dat dat,
                                     int *sendlist, int nsend) {

  int nbites = dat->elem_size;

  //Copy to negative direction
  int send_bites = 0;
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];
    memcpy(buff + send_bites, dat->data + ipart * nbites, nbites);
    send_bites += dat->elem_size;
  }
}

int _ops_particle_intra_hist_to_buff(char *buff, ops_neighbor_history history,
                                     int *sendlist, int nsend) {
  int nsend_bites = 0;

  //Part I: Cpy n_partnersI to list

  int nbites = history->n_partnersI->elem_size;
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];
    memcpy(buff + nsend_bites, history->n_partnersI->data + ipart * nbites, nbites);
    nsend_bites += history->n_partnersI->elem_size;
  }

  //Part II: Copy partners to buffer
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];

    nbites = ((int *)history->n_partnersI->data)[ipart] * history->n_partnersI->type_size;
    int shift_bites = history->partnersI->elem_size* ipart;
    memcpy(buff + nsend_bites, history->partnersI->data + shift_bites,
           nbites);
    nsend_bites += nbites;
  }

  //Part III: Copy data to buffer
  nbites = history->data->elem_size;
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];
    for (int j = 0; j < ((int *)history->n_partnersI->data)[ipart]; j++) {
      int index = ((int *)history->indexI->data)[ipart * history->num_neighsI + j];
      int shift_bites = index * nbites;
      memcpy(buff + nsend_bites, history->data->data + shift_bites, nbites);
      nsend_bites += nbites;
    }
  }

  return nsend_bites;

}


void _ops_particle_intra_pack_rev_dat_to_buff(char *buff, ops_dat dat, const int ifirst,
                                              const int nrecv) {
  int nbites  = dat->elem_size;
  memcpy(buff, dat->data + nbites * ifirst, nbites * nrecv);
}

void _ops_particle_halo_reverse_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                           int nhalos, ops_particle_halo_exchange halo_info,
                                           int *ntot_bites) {

  int nfirst =  halo_info->firstrecv;
  int nsend = halo_info->nsend;

  int nsend_bites = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    ops_dat dat = halo_data[ihalo]->to;
    int send_size = dat->elem_size * nsend;
    memcpy(buff + nsend_bites, dat->data + nfirst, send_size);
    nsend_bites += send_size;
  }

  (*ntot_bites) = nsend_bites;
}


void _ops_particle_halo_copy_from_buff(char *buff, ops_particle_halo_data *halo_data,
                                       int nhalos, ops_particle_halo_exchange halo_info,
                                       int dir_to[], int dir_from[], double translate[],
                                       int *ntot_bites, int flag) {

  int nfirst = halo_info->firstrecv;
  int nrecv = halo_info->nrecv;
  int max_size = 0;
  char *temp = nullptr;

  int a1 = 0;
  int nrecv_bites = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    ops_dat dat= halo_data[ihalo]->to;
    if (halo_data[ihalo]->orient == OPS_PART_ORIENT_ON ||
        halo_data[ihalo]->orient == OPS_PART_POSITION) {
      a1 = 1;
      max_size = MAX(max_size, dat->elem_size);
    }
  }

  if (a1 == 1) {
    int nmax = MAX(nrecv, 1);
    temp = (char *) ops_malloc(max_size * nmax);
  }

  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->to;
      ops_part_orient orient = halo_data[ihalo]->orient;
      int nsize = nrecv * dat->elem_size;
      if (orient == OPS_PART_ORIENT_OFF) {
        int nbite_first = nfirst * dat->elem_size;
        memcpy(dat->data + nbite_first, buff + nrecv_bites, nsize);
      }
      else {
        int dim = dat->dim;
        int dims = dat->block->dims;

        if (dim != dims || dat->type_size != sizeof(double))
          throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat with orient"
                                                           " must be a vector (of size dim per particle point) and of "
                                                           "type double");

        memcpy(temp, buff + nrecv_bites, nsize);

        double *tmp = (double *)temp;
        double *data = (double *)dat->data;

        for (int i = 0; i < nrecv; i++) {
          int ipart = nfirst + i;
          for (int isou = 0; isou < dim; isou++)
            data[ipart * dim + dir_to[isou]] =
                tmp[i * dim + dir_from[isou]] + translate[dir_from[isou]];
        }
      }
      nrecv_bites +=nsize;
    }
    else if (flag && halo_data[ihalo]->halo_type == OPS_EXCHANGE_HISTORY) {
      ops_neighbor_history history = halo_data[ihalo]->history_to;
      ops_dat npartnersI = history->n_partnersI;
      ops_dat partnersI = history->partnersI;
      ops_dat indexing = history->indexing;
      ops_dat indexI = history->indexI;

      ops_dat data = history->data;
      ops_particle particle = history->particleI;
      int *tags = (int *)particle->ids->data;
      int nlocal_elems;

      int nshift = nfirst * npartnersI->elem_size;
      int nbites = nrecv * npartnersI->elem_size;
      memcpy(npartnersI->data + nshift, buff + nrecv_bites, nbites);

      nrecv_bites += nbites;

      //Part II: Receive partners and update index
      int new_neighs;
      for (int  i = 0; i < nrecv; i++) {
        int ipart = nfirst + nrecv;
        nshift = ipart * partnersI->elem_size;
        nbites = partnersI->type_size * ((int *) npartnersI->data)[ipart];
        memcpy(partnersI->data + nshift, buff + nrecv_bites, nbites);
        nrecv_bites += nbites;

        for (int j = 0; j <((int *) npartnersI->data)[ipart]; j++) {
          ((int *) indexI->data)[ipart * history->num_neighsI + j]
                               = history->nconts + new_neighs + j;
        }
        new_neighs += ((int *) npartnersI->data)[i];//TODO: Check that for sanity

      }

      //Part III: Reallocate neighbor histories
      if (new_neighs + history->nconts > history->nmax_cont) {
        history->nmax_cont += (new_neighs + OPS_MAX_PART) * MAX(history->num_neighsI,
                                                                history->num_neighsJ);
        int size = history->data->elem_size * history->nmax_cont;
        history->data->data = (char *) ops_realloc(history->data->data, size);

        size = history->indexing->elem_size * history->nmax_cont;
        history->indexing->data = (char *) ops_realloc(history->indexing->data, size);

        size = history->flag->elem_size * history->nmax_cont;
        history->flag->data = (char *) ops_realloc(history->flag->data, size);
      }

      //Part IV: Update history data
      nshift = history->nconts * history->data->elem_size;
      nbites = new_neighs * history->data->elem_size;
      memcpy(history->data->data + nshift, buff + nrecv_bites, nbites);
      nrecv_bites += nbites;

      //Part V: Update keys
      int ineighs = 0;
      for (int i = 0; i < nrecv; i++) {
        int ipart = nfirst + i;
        int tag = tags[ipart];

        for (int j = 0; j < ((int *) history->n_partnersI->data)[ipart]; j++) {
          int tagJ = ((int *)partnersI->data)[ipart * history->num_neighsI + j];
          int pair = ineighs + history->nconts;
          ((uint64_t *) history->indexing->data)[history->indexing->dim * pair] =
              pack_pair(tag, tagJ);
          ((uint64_t *) history->indexing->data)[history->indexing->dim * pair + 1] =
              pack_pair(ipart, 0);
          ineighs++;
        }
      }

      history->nconts += new_neighs;
    }
  }

  (*ntot_bites) = nrecv_bites;

  ops_free(temp);
}

void _ops_particle_dat_copy_from_buff(char *buff, ops_dat dat, ops_part_orient orient,
                                      ops_particle_halo_exchange halo_info,
                                      int dir_to[], int dir_from[], double translate[],
                                      int *ntot_bites) {

  int nfirst = halo_info->firstrecv;
  int nrecv = halo_info->nrecv;
  int max_size = 0;

  int a1 = 0;
  int nrecv_bites = 0;
  int nsize = dat->elem_size;

  if (orient == OPS_PART_ORIENT_ON) {

    if (nsize * nrecv > ntmp_size_max) {
      temp = (char *) ops_realloc(temp, nsize * (nrecv + OPS_MAX_PART));
      ntmp_size_max = nsize * (nrecv + OPS_MAX_PART);
    }

    memcpy(temp, buff + nrecv_bites, nsize);
    double *tmp = (double *)temp;
    double *data = (double *)dat->data;

    for (int i = 0; i < nrecv; i++) {
      for (int isou = 0; isou < dat->dim; isou++)
      data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
    }
  }
  else {
    int nbite_first = nfirst * dat->elem_size;
    memcpy(dat->data + nbite_first, buff + nrecv_bites, nsize);
  }

  (*ntot_bites) = nsize * nrecv;

}

void _ops_particle_intra_buff_to_dat(char  *buff, ops_dat dat,
                                     size_t nexist, int nrecv) {

  int nsize  = dat->elem_size;
  int nfirst = nexist * nsize;

  memcpy(dat->data + nfirst, buff, nrecv * nsize);

}

int _ops_particle_intra_buff_to_hist(char *buff, ops_neighbor_history history,
                                     size_t nexist, int nrecv) {

  int nrecv_bites = 0;

  //Part I: Output n_partners
  int nsize = history->n_partnersI->elem_size;
  int nfirst = nexist * nsize;

  int nbytes = nrecv * history->n_partnersI->elem_size;
  memcpy(history->n_partnersI + nfirst, buff + nrecv_bites, nbytes);
  nrecv_bites += nbytes;

  //Part II: Push back partnersI of particle to the list
  int new_neighs = 0;
  for (int i = 0; i < nrecv; i++) {
    int ipart = nexist + i;
    int nshift = ipart * history->partnersI->elem_size;
    int nbites = ((int *)history->n_partnersI->data)[ipart]
               * history->partnersI->type_size;
    memcpy(history->partnersI + nshift, buff + nrecv_bites, nbytes);
    nrecv_bites += nbites;

    //Set the index for fast finding
    for (int j = 0; j < ((int *) history->n_partnersI->data)[ipart]; j++) {
      ((int *) history->indexI->data)[ipart * history->num_neighsI + j] =
           history->nconts + new_neighs + j;
    }
    new_neighs += ((int *) history->n_partnersI->data)[ipart];
  }

  //Part III: Reallocate neighbor histories
  if (new_neighs + history->nconts > history->nmax_cont) {
    history->nmax_cont += (new_neighs + OPS_MAX_PART)
                        * MAX(history->num_neighsI, history->num_neighsJ);
    int size = history->data->elem_size * history->nmax_cont;
    history->data->data = (char *) ops_realloc(history->data->data, size);

    size = history->indexing->elem_size * history->nmax_cont;
    history->indexing->data = (char *) ops_realloc(history->indexing->data, size);

    size = history->flag->elem_size * history->nmax_cont;
    history->flag->data = (char *) ops_realloc(history->flag->data, size);

  }


  ///Part IV: Update history data
  int nshift = history->nconts * history->data->elem_size;
  int nbites = new_neighs * history->data->elem_size;
  memcpy(history->data->data + nshift, buff + nrecv_bites, nbites);
  nrecv_bites += nbites;

  //Initialize structures-Update keys
  int ineighs = 0;
  int *tags = (int *) history->particleI->ids->data;
  for (int i = 0; i < nrecv; i++) {
    int ipart = nfirst + i;
    int tag = tags[ipart];
    for (int j = 0; j < ((int *)history->n_partnersI->data)[ipart]; j++) {
      int tagJ = ((int *)history->partnersI->data)[ipart * history->num_neighsI + j];
      int pair = ineighs + history->nconts;
      ((uint64_t *) history->indexing->data)[history->indexing->dim * pair] =
          pack_pair(tag, tagJ);
      ((uint64_t *) history->indexing->data)[history->indexing->dim * pair + 1] =
          pack_pair(ipart, 0);
      ineighs++;
    }
  }

  history->nconts += new_neighs;
  return nrecv_bites;
}

void _ops_particle_unpack_reverse_buff_to_dat(char *buff, ops_dat dat,
                                              const int *recv_list,
                                              const int  nrecv,
                                              ops_access access) {


  if (dat->elem_size * nrecv > ntmp_size_max) {
    ntmp_size_max = dat->elem_size * (nrecv + OPS_MAX_PART);
    temp = (char *) ops_realloc(temp, ntmp_size_max);
  }
  memcpy(temp, buff, dat->elem_size * nrecv);

  if (strcmp(dat->type,"float")==0) {
    float *data  = (float *)dat->data;
    float *tmp_fl = (float *)temp;
    for (int i = 0; i < nrecv; i++) {
      int ipart = recv_list[i];
      _ops_inc_element(data + dat->dim * ipart, tmp_fl + i * dat->dim,
                       dat->dim, access);
    }
  }
  else if (strcmp(dat->type, "int") == 0) {
    int *data = (int *)dat->data;
    int *tmp_int = (int *) temp;
    for (int i = 0; i < nrecv; i++) {
      int ipart = recv_list[i];

      _ops_inc_element(data + dat->dim * ipart, tmp_int + i * dat->dim,
                       dat->dim, access);
    }
  }
  else if (strcmp(dat->type,"double") == 0) {
    double *data  = (double *)dat->data;
    double *tmp_fl = (double *)temp;
    for (int i = 0; i < nrecv; i++) {
      int ipart = recv_list[i];
      _ops_inc_element(data + dat->dim * ipart, tmp_fl + i * dat->dim,
                       dat->dim, access);
    }
  }

}

//TODO: Need to set access type & remove the continuous allocation.
void _ops_particle_halo_reverse_copy_from_buff(char *buff, ops_particle_halo_data *halo_data,
                                               int nhalos, ops_particle_halo_exchange info,
                                               int dir_from[], int dir_to[],
                                               int *ntot_bites, ops_access access) {

  int *sendlist = info->sendlist;
  int nsend = info->nsend;

  int recv_bites = 0;

  int nmax = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++)
    nmax = MAX(halo_data[ihalo]->to->elem_size, nmax);

  if (nmax * nsend > ntmp_size_max) {
    ntmp_size_max = (nsend + OPS_MAX_PART) * nmax;
    temp = (char *) ops_realloc(temp, ntmp_size_max);
  }

  int ifirst = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->from;
      int size_dat = dat->elem_size;
      memcpy(temp, buff + ifirst, size_dat);
      ifirst += size_dat;

      if (strcmp(dat->type, "float") == 0) {
        float *data = (float *)dat->data;
        float *tmp_fl = (float *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_fl + i * dat->dim,
                                dat->dim, access);
        }
      }
      else if (strcmp(dat->type, "int") == 0) {
        int *data = (int *)dat->data;
        int *tmp_i = (int *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_i + i * dat->dim,
                                dat->dim, access);
        }
      }
      else if (strcmp(dat->type, "double") == 0) {
        double *data = (double *)dat->data;
        double *tmp_dbl = (double *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_dbl + i * dat->dim,
                                dat->dim, access);
        }
      }
      else {
        throw OPSException(OPS_RUNTIME_ERROR, "ERROR: This variable type is not supported for reverse "
                                            " halo exchanges" );
      }
    }

  }

  (*ntot_bites) = ifirst;

}

//
void _ops_particle_remove_flag_reset_map(ops_particle particle, int flag) {

  if (particle->no_particles == 0)
    return;

  int dim = particle->block->dims;
  int Nlocal = particle->no_particles;

  for (int i = 0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] == flag) {
      if (i == Nlocal - 1) {
        Nlocal--; break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      if (particle->ids != nullptr)
        _ops_particle_swap_data(particle->ids->data, i, Nlocal - 1,
                                particle->ids->elem_size);

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal - 1, dat->elem_size);
      }


      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      for (int imaps = 0; imaps < particle->particle_map_index; imaps++) {
        ops_particle_mapping map = particle->map_list[imaps];
        _ops_particle_copy_mapping_data_to(map, i, Nlocal - 1);
      }

      Nlocal--;
      if (Nlocal == 0) break;
    }
  }

  particle->no_particles = Nlocal;
}

void _ops_particle_find_intra_box(ops_particle particle, ops_int_particle_halos halo,
                                  int iswap, int ifirst, int ilast) {

  double *xpos = (double *)particle->particle_pos_dat->data;
  int dim = particle->block->dims;

  int nsend_neg = 0;
  int nsend_pos = 0;

  for (int ipart = ifirst; ipart < ilast; ipart++) {
    if (iswap < halo->nswap_neg && particle_is_within(xpos + dim * ipart, halo->region_bord_neg, dim)) {
      nsend_neg++;
      continue;
    }

    if (iswap < halo->nswap_pos && particle_is_within(xpos + dim * ipart, halo->region_bord_pos, dim)) {
      nsend_pos++;
      continue;
    }
  }

  halo->nforward_pos[iswap] = nsend_pos;
  halo->nforward_neg[iswap] = nsend_neg;

}

void _ops_particle_find_intra_map(ops_particle particle, ops_int_particle_halos halo,
                                  ops_dat bin_head, ops_dat bin, int *size) {
  int nsend_neg = 0;
  int nsend_pos = 0;

  int *binhead = (int *) bin_head->data;
  int *bins = (int *)bin->data;
  //accessing negative directions

  for (int iz = halo->region_neg[4]; iz < halo->region_neg[5]; iz++)  {
    for (int iy = halo->region_neg[2]; iy < halo->region_neg[3]; iy++)  {
      for (int ix = halo->region_neg[0]; ix < halo->region_neg[1]; ix++) {
        int address = ix + size[0] * iy + iz * size[1] * size[0];
        int ipart = binhead[address];
        while (ipart != -1) {
          nsend_neg++;
          ipart = bins[ipart];
        }
      }
    }
  }

  //Assesing negative direction
  for (int iz = halo->region_pos[4]; iz < halo->region_pos[5]; iz++) {
    for (int iy = halo->region_pos[2]; iy < halo->region_pos[3]; iy++)  {
      for (int ix = halo->region_pos[0]; ix < halo->region_pos[1]; ix++) {
        int address = ix + size[0] * iy + iz * size[1] * size[0];
        int ipart = binhead[address];
        while (ipart != -1) {
          nsend_pos++;
          ipart = bins[ipart];
        }
      }
    }
  }

  halo->nforward_pos[0] = nsend_pos;
  halo->nforward_neg[0] = nsend_neg;
}

void _ops_particle_set_intra_map(ops_particle particle, ops_int_particle_halos halo,
                                 ops_dat binhead, ops_dat bins, int size[]) {

  int *bin_head = (int *) binhead->data;
  int *bin = (int *)bins->data;

  int nsend_neg = 0;
  for (int iz = halo->region_neg[4]; iz < halo->region_neg[5]; iz++)  {
     for (int iy = halo->region_neg[2]; iy < halo->region_neg[3]; iy++)  {
       for (int ix = halo->region_neg[0]; ix < halo->region_neg[1]; ix++) {
         int address = ix + iy * size[0] + iz * size[0] * size[1];
         int ipart = bin_head[address];
         while (ipart != -1) {
           halo->particle_send_neg[nsend_neg] = ipart;
           nsend_neg++;
           ipart = bin[ipart];
         }
       }
     }
  }

  int nsend_pos = 0;
  for (int iz = halo->region_pos[4]; iz < halo->region_pos[5]; iz++)  {
     for (int iy = halo->region_pos[2]; iy < halo->region_pos[3]; iy++)  {
       for (int ix = halo->region_pos[0]; ix < halo->region_pos[1]; ix++) {
         int address = ix + iy * size[0] + iz * size[0] * size[1];
         int ipart = bin_head[address];
         while (ipart != -1) {
           halo->particle_send_pos[nsend_pos] = ipart;
           nsend_pos++;
           ipart = bin[ipart];
         }
       }
     }
  }

}



void _ops_particle_set_intra_border_box(ops_particle particle, ops_int_particle_halos halo,
                                        int iswap, int ifirst, int ilast) {

  int nshift_pos = 0;
  int nshift_neg = 0;
  for (int i = 0; i < iswap; i++) {
    nshift_pos += halo->nforward_pos[iswap];
    nshift_neg += halo->nforward_neg[iswap];
  }

  int dim = particle->block->dims;
  int nsend_neg = 0;
  int nsend_pos = 0;

  double *xpos = (double *)particle->particle_pos_dat->data;

  for (int ipart = ifirst; ipart < ilast; ipart++) {
    if (iswap < halo->nswap_neg && particle_is_within(xpos + dim * ipart, halo->region_bord_neg, dim)) {
      halo->particle_send_neg[nsend_neg] = ipart;
      nsend_neg++;

    }

    if (iswap < halo->nswap_pos && particle_is_within(xpos + dim * ipart, halo->region_bord_pos, dim)) {
      halo->particle_send_pos[nsend_pos] = ipart;
      nsend_pos++;
    }

  }
}

void ops_mpi_particle_host_write_to_file(ops_particle particle, const char *file_name,
                                         char *buff, size_t len) {

  //OPEN MPI_FILE
  MPI_File fh;

  sub_block *sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  int num_ranks, my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  if (my_rank == 0)
    MPI_File_delete(file_name, MPI_INFO_NULL);

  MPI_File_open(sb->comm, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY , MPI_INFO_NULL, &fh);



  //compute offset prior to passage



  //Testing the MPI_Exscan
  MPI_Offset offset = 0;
  MPI_Offset mylen = len;
  MPI_Exscan(&mylen, &offset, 1, MPI_OFFSET, MPI_SUM, sb->comm);
  if (my_rank == 0) offset = 0;

  if (len > 0)
    MPI_File_write_at(fh, offset, buff, len, MPI_BYTE, MPI_STATUS_IGNORE);
  else
    MPI_File_write_at(fh, offset, NULL, 0, MPI_BYTE, MPI_STATUS_IGNORE);


  MPI_File_close(&fh);

}

void ops_particle_print_mpi_dats_to_txt_file_core(ops_particle particle, ops_dat *dats,
                                                  int ndats,const char *file_name) {

  size_t len = 0;
  size_t size = 0;
  char *buff = nullptr;

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int local_rank;
  MPI_Comm_rank(sb->comm, &local_rank);
  int ntotal = 0;
  MPI_Reduce(&particle->no_particles, &ntotal, 1, MPI_INT,
             MPI_SUM, 0, sb->comm);

  if (local_rank == 0) {
    _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ", particle->block->name,
                     particle->box_block->getGlobalMin().x,
                     particle->box_block->getGlobalMax().x,
                     particle->box_block->getGlobalMin().y,
                     particle->box_block->getGlobalMax().y);

    if (particle->block->dims == 3)
      _ops_append_char(buff, len, size, "[%f %f] ", particle->box_block->getGlobalMin().z,
                       particle->box_block->getGlobalMax().z);

    _ops_append_char(buff, len, size, "\nParticle: %s Number of particles: %d\n", particle->name, ntotal);

    //Create title
    for (int idat = 0; idat < ndats; idat++) {
      for (int isou = 0; isou < dats[idat]->dim; isou++)
      _ops_append_char(buff, len, size, "%16s[%d]", dats[idat]->name, isou);
    }

    _ops_append_char(buff, len, size,"\n");

  }

  int nparticles = particle->no_particles;
  for (int ip = 0; ip < nparticles; ip++) {
    _ops_append_char(buff, len, size, "   ");
    for (int idat = 0; idat < ndats; idat++) {
      _ops_particle_append_dat_point(dats[idat], buff, len, size, ip);
    }
    _ops_append_char(buff, len, size, "\n");
  }

  ops_mpi_particle_host_write_to_file(particle, file_name, buff, len);

  ops_free(buff);

}


void ops_particle_print_mpi_data_to_txt_file_core(ops_particle particle,
                                                  const char *file_name) {

  size_t len = 0;
  size_t size = 0;
  char *buff = nullptr;

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  int local_rank;
  MPI_Comm_rank(sb->comm, &local_rank);
  int ntotal = 0;
  MPI_Reduce(&particle->no_particles, &ntotal, 1, MPI_INT,
             MPI_SUM, 0, sb->comm);


  if (local_rank == 0) {
    _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ",
                     particle->block->name, particle->box_block->getGlobalMin().x,
                     particle->box_block->getGlobalMax().x,
                     particle->box_block->getGlobalMin().y,
                     particle->box_block->getGlobalMax().y);

    if (particle->block->dims == 3)
      _ops_append_char(buff, len, size, "[%f %f]\n",  particle->box_block->getGlobalMin().z,
                       particle->box_block->getGlobalMax().z);
    else
      _ops_append_char(buff, len, size, "\n");

    _ops_append_char(buff, len, size, "\nParticle: %s Number of particles: %d\n", particle->name, ntotal);

    //Write titles

    if (particle->ids != nullptr)
      _ops_append_char(buff, len, size, "%16s[%d] ", particle->ids->name);


    for (int i = 0; i < particle->particle_pos_dat->dim; i++)
        _ops_append_char(buff, len, size, "%16s[%d] ", particle->particle_pos_dat->name, i);

    if (particle->particle_envelope != nullptr)
      _ops_append_char(buff, len, size, "%16s ", particle->particle_envelope->name);

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      for (int i = 0; i < particle->particle_dat[idat]->dim; i++)
      _ops_append_char(buff, len, size, "%16s[%d] ", particle->particle_dat[idat]->name, i);
    }

    _ops_append_char(buff, len, size, "\n");
  }


  int nparticles = particle->no_particles;



  for (int ip = 0; ip < nparticles; ip++) {

    if (particle->ids != nullptr)
      _ops_particle_append_dat_point(particle->ids, buff, len, size, ip);

    _ops_particle_append_dat_point(particle->particle_pos_dat, buff, len, size, ip);

    if (particle->particle_envelope != nullptr)
      _ops_particle_append_dat_point(particle->particle_pos_dat, buff, len, size, ip);

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_append_dat_point(dat, buff, len, size, ip);
    }

    _ops_append_char(buff, len, size ,"\n");
  }

  //TODO: PASS TO MPI_COMMS
  ops_mpi_particle_host_write_to_file(particle, file_name, buff, len);

  ops_free(buff);

}

void ops_particle_map_get_dx(ops_particle_mapping map, double dx[]) {

  for (int i = 0; i < map->particle->block->dims; i++) dx[i] = map->dx[i];
}


