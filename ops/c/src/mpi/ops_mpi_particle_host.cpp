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

#include "ops_particle_mapping_functions.h"

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

static int _particle_is_within(char *xpos, char *region, int ipart,
                               int dim, int type_size) {
  switch(type_size) {
  case sizeof(float):
    return particle_is_within((float *) xpos + dim * ipart, (float *) region, dim);
    break;
  case sizeof(double):
    return particle_is_within((double *) xpos + dim * ipart, (double *) region, dim);
    break;
  case sizeof(long double):
    return particle_is_within((long double *) xpos + dim * ipart, (long double *) region, dim);
    break;
  }

  return 0;
}

//TODO: DO I need that??
void _ops_particle_allocate_tmp_array(int size_elem) {

  if (size_elem == 0) return;

  ntmp_size_max = size_elem * OPS_MAX_PART;
  temp = (char *) ops_malloc(ntmp_size_max);

}

void _ops_particle_free_tmp_array() {
  ops_free(temp);
  ntmp_size_max = 0;
}

//TODO: DO I need that
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
  ops_dat xps = particle->particle_pos_dat;
  ops_dat xps_old = map->pos_old;

  int dim = particle->block->dims;
  int nParticles = (int) particle->no_particles;

  for (int i = 0; i < nParticles; i++) {
    int a1;
    if (xps->type_size == sizeof(float))
      flag = _ops_check_particle_movement((float *)xps->data + dim *i,
                                         (float *) xps_old->data + dim * i,
                                         (float *) map->dx, dim);
    else if (xps->type_size == sizeof(double))
      flag = _ops_check_particle_movement((double *)xps->data + dim *i,
                                         (double *) xps_old->data + dim * i,
                                         (double *) map->dx, dim);
    else if (xps->type_size == sizeof(long double))
      flag = _ops_check_particle_movement((long double *)xps->data + dim *i,
                                         (long double *) xps_old->data + dim * i,
                                         (long double *) map->dx, dim);

    if (flag == 1) break;

  }

  int global_flag = 0;
  MPI_Allreduce(&flag, &global_flag, 1, MPI_INT, MPI_MAX,
                OPS_sub_block_list[particle->block->index]->comm);

  if (global_flag) { map->decide = true;
    for (int i = 0; i < nhistories; i++) particle->histories[i]->flag_update = true;
  }
  else map->decide = false;
  return global_flag;
}


void _ops_particle_update_map_int_halos(ops_particle_mapping map, ops_particle particle,
                                       int ifirst, int ilast) {
  map->nParticles = particle->no_particles + particle->no_virtual;

  char xmin[320], xmax[320];

  //Compute d_m
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  char *dx = map->dx;

  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
  }

  //TODO: Check if we reset the maps at the beginning
  int dim = particle->particle_pos_dat->dim;
  int *size = map->binhead->size;
  char* xps = particle->particle_pos_dat->data;
  char *xps_old = map->pos_old->data;
  for (int i = ifirst; i < ilast; i++) {
    int ibin;
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      ibin = _ops_coord_to_bin(dim, (float *)xmin, (float *)xmax,
                               (float *) dx, size, (float *)xps + dim * i);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      ibin = _ops_coord_to_bin(dim, (double *)xmin, (double *)xmax,
                               (double *) dx, size, (double *)xps + dim * i);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      ibin = _ops_coord_to_bin(dim, (long double *)xmin, (long double *)xmax,
                               (long double *) dx, size, (long double *)xps + dim * i);
    }

    if (ibin < 0) continue;

    ((int *)map->bin->data)[i] = ((int *) map->binhead->data)[ibin];
    ((int *)map->binhead->data)[ibin] = i;
    ((int *)map->parts_to_grid->data)[i] = ibin;

    //Set particle to map old //
    int ilocal[OPS_MAX_DIM];
    get_local_point(((int *)map->parts_to_grid->data)[i], map->binhead->size, d_m, dim, ilocal);

    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      get_coord_point((float *) xps_old + i * dim, ilocal, d_m, (float *) xmin,
                      (float *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      get_coord_point((double *) xps_old + i * dim, ilocal, d_m, (double *) xmin,
                      (double *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      get_coord_point((long double *) xps_old + i * dim, ilocal, d_m, (long double *) xmin,
                      (long double *) dx, dim, 1);
    }
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

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  char xmin[320], xmax[320];
  char *dx = map->dx;

  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
  }

  int dim = particle->block->dims;
  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (int i = ifirst; i < ilast; i++) {
    int address;

    particle->mark_deletion[i] = 0;
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      address = _ops_coord_to_bin(dim, (float *)xmin, (float *)xmax,
                                 (float *) dx, map->binhead->size,
                                 (float *)xpos + dim * i);

      if (!((BoundingBox<float> *) particle->box_block)->isCoordinateInBoundingBox((float *) xpos + i * dim)) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        ((int *)map->bin->data)[i] = -1;
        particle->mark_deletion[i] = 1;
        continue;
      }
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      address = _ops_coord_to_bin(dim, (double *)xmin, (double *)xmax,
                                 (double *) dx, map->binhead->size,
                                 (double *)xpos + dim * i);

      if (!((BoundingBox<double> *)particle->box_block)->isCoordinateInBoundingBox((double *) xpos + i * dim)) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        ((int *)map->bin->data)[i] = -1;
        particle->mark_deletion[i] = 1;
        continue;
      }
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      address = _ops_coord_to_bin(dim, (long double *)xmin, (long double *)xmax,
                                 (long double *) dx, map->binhead->size,
                                 (long double *)xpos + dim * i);

      if (!((BoundingBox<long double> *)particle->box_block)->isCoordinateInBoundingBox((long double *) xpos + i * dim)) {
        ((int *)map->bin->data)[i] = -1;
        ((int *)map->parts_to_grid->data)[i] = -1;
        particle->mark_deletion[i] = 1;
        continue;
      }
    }



    ((int *) map->bin->data)[i] = ((int *)map->binhead->data)[address];
    ((int *) map->binhead->data)[address] = i;
    ((int *) map->parts_to_grid->data)[i] = address;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m, dim, ilocal);
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      get_coord_point((float *) xold + i * dim, ilocal, d_m, (float *)xmin, (float *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      get_coord_point((double *) xold + i * dim, ilocal, d_m, (double *)xmin, (double *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double))
      get_coord_point((long double *) xold + i * dim, ilocal, d_m, (long double *)xmin,
                      (long double *) dx, dim, 1);

  }

  for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = true;

}

int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return 0;

  map->decide = false;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  int *size = map->binhead->size;

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  int local_flag = 0;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int nmapping =
      (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                             particle->no_particles + particle->no_virtual;

  //TODO: Shift to other structures
  char *dx = map->dx;
  char xmin[320], xmax[320];
  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
  }



  int dim = particle->block->dims;
  int rmv_limits[2 * OPS_MAX_DIM];
  int exch_limits[2 * OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    exch_limits[2 *i] = (d_m[i]< 0) ? 0 : -size[i]; //Constant for the model
    exch_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - d_p[i] - d_p[i] : 2 * size[i];
    rmv_limits[2 * i] = 0;
    rmv_limits[2 * i + 1] = size[i] + d_m[i] - d_p[i] - 1;
  }

  size_t nactual = particle->no_particles;
  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (size_t i = 0; i < particle->no_particles; i++) {
    int flag{0};
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      flag = local_decide_rebuild((float *)xpos + i * dim, (float *)xold + i * dim,
                                  ((float *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      flag = local_decide_rebuild((double *)xpos + i * dim, (double *)xold + i * dim,
                                  ((double *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      flag = local_decide_rebuild((long double *)xpos + i * dim, (long double *)xold + i * dim,
                                  ((long double *)dx)[0], dim);
    }

    if (flag) {
      int address = ((int *)map->parts_to_grid->data)[i];

      get_local_point(address, map->binhead->size, d_m, dim, ilocal);


      _remove_particle_from_bins(address, i, (int *)map->binhead->data, (int *)map->bin->data);


      int del_flag;
      if (particle->particle_pos_dat->type_size == sizeof(float))
         del_flag=  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                     (float *)xpos + i * dim,
                                                     (BoundingBox<float> *) particle->box_block);
      else if (particle->particle_pos_dat->type_size == sizeof(double))
        del_flag=  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (double *)xpos + i * dim,
                                                    (BoundingBox<double> *)particle->box_block);
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        del_flag=  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (long double *)xpos + i * dim,
                                                    (BoundingBox<long double> *)particle->box_block);

      if (del_flag) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        local_flag = 1;
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
      }

      if (particle->particle_pos_dat->type_size == sizeof(float)) {
        address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                    map->binhead->size, (float *) xpos + dim * i);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(double)) {
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax,
                                    (long double *) dx, map->binhead->size,
                                    (long double *) xpos + dim * i);
      }

      if (address < 0) local_flag = 1;

      ((int *)map->bin->data)[i] = ((int *) map->binhead->data)[address];
      ((int *)map->binhead->data)[address] = i;
      ((int *)map->parts_to_grid->data)[i] = address;
      int ilocal[OPS_MAX_DIM];

      get_local_point(address, map->binhead->size, d_m, dim, ilocal);
      if (particle->particle_pos_dat->type_size == sizeof(float)) {
        get_coord_point((float *) xold + i * dim, ilocal, d_m, (float *) xmin, (float *) dx,
                        dim, 1);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(double)) {
        get_coord_point((double *) xold + i * dim, ilocal, d_m, (double *) xmin, (double *) dx,
                        dim, 1);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        get_coord_point((long double *) xold + i * dim, ilocal, d_m, (long double *) xmin, (long double *) dx,
                        dim, 1);

      bool flag_build = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new,exch_limits, dim);
      if (!local_flag) local_flag = (int )flag_build;

    }
  }


  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

  for (int i = ifirst; i < nmapping; i++) {
    int flag;
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      flag = local_decide_rebuild((float *) xpos + i * dim, (float *) xold + i * dim,
                                  ((float *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      flag = local_decide_rebuild((double *) xpos + i * dim, (double *) xold + i * dim,
                                  ((double *)dx)[0],dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      flag = local_decide_rebuild((long double *) xpos + i * dim, (long double *) xold + i * dim,
                                  ((long double *)dx)[0], dim);
    }

    if (flag) {
      int address = ((int *)map->parts_to_grid->data)[i];
      _remove_particle_from_bins(address, i, (int *) map->binhead->data,
                                 (int *) map->bin->data);
      ((int *)map->parts_to_grid->data)[i] = -1;

      if (particle->particle_pos_dat->type_size == sizeof(float)) {
        address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                    map->binhead->size, (float *) xpos + dim * i);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(double)) {
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax,
                                    (long double *) dx, map->binhead->size,
                                    (long double *) xpos + dim * i);
      }

      if (address < 0) {
        local_flag = 1;
        continue;
      }

      get_local_point(address, map->binhead->size, d_m,
                      dim, ilocal);

      int flag_in;

      if (particle->particle_pos_dat->type_size == sizeof(float)) {
        flag_in= virtual_within(ilocal, rmv_limits, (float *)xpos + dim * i,
                                (BoundingBox<float> *)particle->box_block, dim);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(double)) {
        flag_in= virtual_within(ilocal, rmv_limits, (double *)xpos + dim * i,
                                (BoundingBox<double> *)particle->box_block, dim);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
        flag_in= virtual_within(ilocal, rmv_limits, (long double *)xpos + dim * i,
                                (BoundingBox<long double> *)particle->box_block, dim);
      }


      if (flag_in) { //Virtual become actual
        nactual++;
        nvirtual_act--;

        ((int *)map->parts_to_grid->data)[i] = address;
        ((int *)map->bin->data)[i] = ((int *)map->binhead->data)[address];
        ((int *)map->binhead->data)[address] = i;

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
        ((int *)map->parts_to_grid->data)[i] = address;
        ((int *)map->bin->data)[i] = ((int *) map->binhead->data)[address];
        ((int *)map->binhead->data)[address] = i;

        if (particle->particle_pos_dat->type_size == sizeof(float))
          get_coord_point((float *) xold + i * dim, ilocal, d_m, (float *)xmin, (float *) dx,
                                  dim, 1);
        else if (particle->particle_pos_dat->type_size == sizeof(double))
          get_coord_point((double *) xold + i * dim, ilocal, d_m, (double *) xmin, (double *) dx,
                                  dim, 1);
        else if (particle->particle_pos_dat->type_size == sizeof(long double))
          get_coord_point((long double *) xold + i * dim, ilocal, d_m, (long double *) xmin, (long double *) dx,
                           dim, 1);
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
  map->flag_history = 0;
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return 0;

  int *size = map->binhead->size;

  int nmapping = particle->no_particles;
  int local_flag = 0;
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  char *xmin[320], *xmax[320];
  char *dx = map->dx;
  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  int dim = particle->block->dims;
  int border_limits[2 * OPS_MAX_DIM];
  int rmv_limits[2 * OPS_MAX_DIM];
  int ilocal[OPS_MAX_DIM];
  //TODO: Need adaptation
  for (int i =  0; i < particle->block->dims; i++) {
    border_limits[2 *i] = (d_m[i]< 0) ? 0 : -size[i]; //Constant for the model
    border_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - d_p[i] - d_p[i] : 2 * size[i];
    rmv_limits[ 2 * i ] = 0;
    rmv_limits[2 * i + 1] = size[i] + d_m[i] - d_p[i] - 1;
  }

  for (int i = 0; i < nmapping; i++) {


    int flag;
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      flag = local_decide_rebuild((float *) xpos + i * dim, (float *) xold + i * dim,
                                ((float *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      flag = local_decide_rebuild((double *) xpos + i * dim, (double *) xold + i * dim,
                                ((double *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      flag = local_decide_rebuild((long double *) xpos + i * dim, (long double *) xold + i * dim,
                                  ((long double *)dx)[0], dim);
    }

    if (flag) {
      int address = ((int *) map->parts_to_grid->data)[i];
      _remove_particle_from_bins(address, i, (int *)map->binhead->data,
                                 (int *) map->bin->data);


      map->flag_history= 1;

      //Add new address prior deletion
      get_local_point(address, map->binhead->size, d_m, dim, ilocal);


      int del_flag;
      if (particle->particle_pos_dat->type_size == sizeof(float))
       del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                         (float *) xpos + i * dim,
                                         (BoundingBox<float> *)particle->box_block);
      else if (particle->particle_pos_dat->type_size == sizeof(double))
        del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                   (double *) xpos + i * dim,
                                                   (BoundingBox<double> *)particle->box_block);
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        del_flag= _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                   (long double *) xpos + i * dim,
                                                   (BoundingBox<long double> *)particle->box_block);

/*
      printf("Proc %d: Build list decided for %d [%f %f] xold =[%f %f] and del_flag  = %d\n", ops_get_proc(),
             i, *(xpos + i * dim), *(xpos + i * dim + 1),
             *(xold + i * dim), *(xold + i * dim + 1), del_flag);
      printf("Rank %d rmv_limits: [-INF %d]-[%d INF] x [-INF %d]-[%d INF] local = [%d %d] wiht address =%d\n", ops_get_proc(),
             rmv_limits[0], rmv_limits[1], rmv_limits[2], rmv_limits[3], ilocal[0], ilocal[1], address);
*/

      if (del_flag) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        local_flag = 1;

        particle->mark_deletion[i] = 1;
        continue;
      }

      switch (particle->type_box) {
      case sizeof(float):
         address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                     map->binhead->size, (float *) xpos + dim * i);
        break;
      case sizeof(double):
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
        break;
      case sizeof(long double):
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *) dx,
                                    map->binhead->size, (long double *) xpos + dim * i);
        break;
      }

      ((int *) map->bin->data)[i] = ((int *) map->binhead->data)[address];
      ((int *) map->binhead->data)[address] = i;
      ((int *) map->parts_to_grid->data)[i] = address;

      //Check for particle moving in or out of exchange zone
      int ilocal_new[OPS_MAX_DIM];
      get_local_point(address, map->binhead->size, d_m,
                       dim, ilocal_new);
      if (particle->particle_pos_dat->type_size == sizeof(float))
        get_coord_point((float *) xold + i * dim, ilocal_new, d_m, (float *)xmin,
                        (float *) dx, dim, 1);
      else if (particle->particle_pos_dat->type_size == sizeof(double))
        get_coord_point((double *) xold + i * dim, ilocal_new, d_m, (double *)xmin,
                        (double *) dx, dim, 1);
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        get_coord_point((long double *) xold + i * dim, ilocal_new, d_m, (long double *) xmin,
                        (long double *) dx,  dim, 1);

      bool flag_build = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new,
                                                             border_limits, dim);

      if (!local_flag) local_flag = (int) flag_build;

    }
  }

  int global_flag = 0;

  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, sb->comm);

  map->decide = (bool) global_flag;

  if (map->flag_history)
   for (int i = 0; i < particle->nhistories; i++)
     if (!particle->histories[i]->flag_update)
     particle->histories[i]->flag_update = true;

  return global_flag;
}

//TODO:
void _ops_particle_remap_virtual(ops_particle_mapping map, ops_particle particle,
                                 int istart, int ilast) {
  //Sanity checks
  if ((size_t) istart < particle->no_particles)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function for mapping virtual "
                                             "called for actual particles");

  if ((size_t) ilast > particle->no_particles + particle->no_virtual)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function called for non-existing "
                                             " particle");

  if (map->mapping_type != OPS_WITH_VIRTUAL) return;

  sub_block *sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  int dim = particle->block->dims;
  int *size = map->binhead->size;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
  }

  char *dx = map->dx;
  char xmin[320], xmax[320];
  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
  }

  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (int i = istart; i < ilast; i++) {

    //Part I: Decide if rebuild for particle
    int flag;
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      flag = local_decide_rebuild((float *) xpos + i * dim, (float *) xold + i * dim,
                                  ((float *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      flag = local_decide_rebuild((double *) xpos + i * dim, (double *) xold + i * dim,
                                  ((double *)dx)[0], dim);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      flag = local_decide_rebuild((long double *) xpos + i * dim, (long double *) xold + i * dim,
                                  ((long double *)dx)[0], dim);
    }

    if (flag) {
      int address = ((int *) map->parts_to_grid->data)[i];
      get_local_point(address, map->binhead->size, d_m, dim, ilocal);

      _remove_particle_from_bins(address, i, (int *) map->binhead->data,
                                 (int *) map->bin->data);

      if (particle->particle_pos_dat->type_size == sizeof(float))
        address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                    size, (float *) xpos + i * dim);
      else if (particle->particle_pos_dat->type_size == sizeof(double))
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *)dx,
                                    size, (double *) xpos + i * dim);
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *)dx,
                                    size, (long double *) xpos + i * dim);


      ((int *)map->bin->data)[i] = ((int *)map->binhead->data)[address];
      ((int *)map->binhead->data)[address] = i;
      ((int *)map->parts_to_grid->data)[i] = address;


      if (particle->particle_pos_dat->type_size == sizeof(float))
        get_coord_point((float *) xold + i * dim, ilocal_new, d_m, (float *)xmin,
                        (float *) dx, dim, 1);
      else if (particle->particle_pos_dat->type_size == sizeof(double))
        get_coord_point((double *) xold + i * dim, ilocal_new, d_m, (double *) xmin,
                        (double *) dx, dim, 1);
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        get_coord_point((long double *) xold + i * dim, ilocal_new, d_m,
                        (long double *)xmin, (long double *) dx, dim, 1);
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

    size_t nforward = 0;
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
    size_t nforward = 0;
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

  for (size_t i = 0; i < Np; i++) {
    ((int *)map->bin->data)[i] = -1;
    ((int *)map->parts_to_grid->data)[i] = -1;
  }

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }

  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, dim);
    ops_point<float> x_min;
    ops_point<float> x_max;
    x_min.x = ((float *)xmin)[0]; x_max.x = ((float *)xmax)[0];
    x_min.y = ((float *)xmin)[1]; x_max.y = ((float *)xmax)[1];
    x_min.z = (dim == 3) ? ((float *)xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((float *)xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (float *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);

    ops_point<double> x_min;
    ops_point<double> x_max;
    x_min.x = ((double *)xmin)[0]; x_max.x = ((double *)xmax)[0];
    x_min.y = ((double *)xmin)[1]; x_max.y = ((double *)xmax)[1];
    x_min.z = (dim == 3) ? ((double *)xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((double *) xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (double *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
    ops_point<long double> x_min;
    ops_point<long double> x_max;
    x_min.x = ((long double *) xmin)[0]; x_max.x = ((long double *) xmax)[0];
    x_min.y = ((long double *) xmin)[1]; x_max.y = ((long double *) xmax)[1];
    x_min.z = (dim == 3) ? ((long double *) xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((long double *) xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (long double *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
  }


  int *part_to_bin = (int *)map->parts_to_grid->data;

  char *xold = map->pos_old->data;
  int ilocal[OPS_MAX_DIM];
  for (size_t i = 0; i < Np; i++) {
    int address = part_to_bin[i];
    get_local_point(address, map->binhead->size, d_m, particle->block->dims, ilocal);

    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      ops_point<float> x_min;
      x_min.x = ((float *)xmin)[0];
      x_min.y = ((float *)xmin)[1];
      x_min.z = (dim == 3) ? ((float *)xmin)[2] : 0.0;
      get_coord_point((float *) xold + i * dim, ilocal, d_m, x_min, (float *) dx,
                              dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      ops_point<double> x_min;
      x_min.x = ((double *)xmin)[0];
      x_min.y = ((double *)xmin)[1];
      x_min.z = (dim == 3) ? ((double *)xmin)[2] : 0.0;
      get_coord_point((double *) xold + i * dim, ilocal, d_m, x_min, (double *) dx,
                              dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      ops_point<long double> x_min;
      x_min.x = ((long double *)xmin)[0];
      x_min.y = ((long double *)xmin)[1];
      x_min.z = (dim == 3) ? ((long double *)xmin)[2] : 0.0;
      get_coord_point((long double *) xold + i * dim, ilocal, d_m, x_min,
                      (long double *) dx, dim, 1);
    }
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


    switch (map->particle->particle_pos_dat->type_size) {
    case sizeof(float):
      _ops_compute_map_grid_size_dir<float>(i, map->dx, map->particle->box_block,
                                              size_loc);
      break;
    case sizeof(double):
       _ops_compute_map_grid_size_dir<double>(i, map->dx, map->particle->box_block,
                                              size_loc);
      break;
    case sizeof(long double):
       _ops_compute_map_grid_size_dir<long double>(i, map->dx, map->particle->box_block,
                                                   size_loc);
      break;
    }
  }

}

void _ops_partition_flat_wall(ops_particle particle) {

  if (particle->particle_pos_dat == nullptr || particle->normal_vector == nullptr)
     throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_dat is defined for one of the following: "
                                           "wall position or normal vector\n");

  int dim = particle->block->dims;

  if (!ops_partitioned()) //TODO: Find the right function
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Domain is not partitioned");

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  //identify(range not within)
  switch(particle->type_box) {
  case sizeof(float): {
    float vector_point[OPS_MAX_DIM];
    int idir_nz;
    for (int i = 0; i < dim; i++) {
      if (ops_abs(((float *) particle->normal_vector->data)[i]) ==1) {
        idir_nz = i; break;
      }
    }

    for (int i = 0; i <dim; i++)
      vector_point[i] = 0.5 * (((BoundingBox<float> *) particle->box_block)->getMinCoordDir(i)
                               + ((BoundingBox<float> *) particle->box_block)->getMaxCoordDir(i));

    vector_point[idir_nz] = ((float *) particle->xcm)[idir_nz];

    bool isin
        = ((BoundingBox<float> *) particle->box_block)->isCoordinateInBoundingBox(vector_point);

    particle->no_particles = 0;
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((float *) particle->particle_pos_dat->data)[i] = vector_point[i];
        ((float *) particle->normal_vector->data)[i] = ((float *) particle->nx)[i]; //TODO:
      }
    }

    } break;
  case sizeof(double): {
    double vector_point[OPS_MAX_DIM];
    int idir_nz;
    for (int i = 0; i < dim; i++) {
      if (ops_abs(((double *) particle->normal_vector->data)[i]) ==1) {
        idir_nz = i; break;
      }
    }

    for (int i = 0; i <dim; i++)
      vector_point[i] = 0.5 * (((BoundingBox<double> *) particle->box_block)->getMinCoordDir(i)
                               + ((BoundingBox<double> *) particle->box_block)->getMaxCoordDir(i));

    vector_point[idir_nz] = ((double *) particle->xcm)[idir_nz];

    bool isin
        = ((BoundingBox<double> *) particle->box_block)->isCoordinateInBoundingBox(vector_point);

    particle->no_particles = 0;
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((double *) particle->particle_pos_dat->data)[i] = vector_point[i];
        ((double *) particle->normal_vector->data)[i] = ((float *) particle->nx)[i]; //TODO:
      }
    }

    } break;
  case sizeof(long double): {
    long double vector_point[OPS_MAX_DIM];
    int idir_nz;
    for (int i = 0; i < dim; i++) {
      if (ops_abs(((long double *) particle->normal_vector->data)[i]) ==1) {
        idir_nz = i; break;
      }
    }

    for (int i = 0; i <dim; i++)
      vector_point[i] = 0.5 * (((BoundingBox<long double> *) particle->box_block)->getMinCoordDir(i)
                               + ((BoundingBox<long double> *) particle->box_block)->getMaxCoordDir(i));

    vector_point[idir_nz] = ((long double *) particle->xcm)[idir_nz];

    bool isin
        = ((BoundingBox<long double> *) particle->box_block)->isCoordinateInBoundingBox(vector_point);

    particle->no_particles = 0;
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((long double *) particle->particle_pos_dat->data)[i] = vector_point[i];
        ((long double *) particle->normal_vector->data)[i] = ((float *) particle->nx)[i]; //TODO:
      }
    }
    } break;
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


//TODO:
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


  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             d_m, d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              d_m, d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             d_m, d_p, dim);
    break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;
  for (size_t i = 0; i < particle->no_particles; i++) {

    int address;
    switch(particle->particle_pos_dat->type_size) {
    case sizeof(float):
       address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                   (float *) xpos + dim * i);
       break;
    case sizeof(double):
       address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                   (double *) xpos + dim * i);
       break;
    case sizeof(long double):
       address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                   map->binhead->size, (long double *) xpos + dim * i);
       break;
    }

    bin2grid[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);
    switch (particle->particle_pos_dat->type_size) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
        get_coord_point((double *) xold + dim * i, ilocal, d_m, (double *) xmin,
                        (double *)dx, dim, 1);
        break;
    case sizeof(long double):
        get_coord_point((long double *) xold + dim * i, ilocal, d_m, (long double *) xmin,
                       (long double *)dx, dim, 1);
        break;
    }

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

  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             d_m, d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              d_m, d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             d_m, d_p, dim);
    break;
  }


  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (size_t i = particle->no_particles;
              i < particle->no_particles + particle->no_virtual; i++) {
    bins[i] = -1;
    bin2grid[i] = -1;

    int address;
    switch(particle->particle_pos_dat->type_size) {
    case sizeof(float):
       address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                   (float *) xpos + dim * i);
       break;
    case sizeof(double):
       address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                   (double *) xpos + dim * i);
       break;
    case sizeof(long double):
       address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                   map->binhead->size, (long double *) xpos + dim * i);
       break;
    }

    if (address < 0) continue;
    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);

    switch (particle->particle_pos_dat->type_size) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
        get_coord_point((double *) xold + dim * i, ilocal, d_m, (double *) xmin,
                        (double *)dx, dim, 1);
        break;
    case sizeof(long double):
        get_coord_point((long double *) xold + dim * i, ilocal, d_m, (long double *) xmin,
                       (long double *)dx, dim, 1);
        break;
    }

  }

}

//TODO:
void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map) {

  map->nParticles = ifirst + n_to_map;

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = map->binhead->d_m[i] + OPS_sub_dat_list[map->binhead->index]->d_im[i];
    d_p[i] = map->binhead->d_p[i] + OPS_sub_dat_list[map->binhead->index]->d_ip[i];
  }


  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             d_m, d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              d_m, d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             d_m, d_p, dim);
    break;
  }


  char *xpos =  particle->particle_pos_dat->data;
  char *xold =  map->pos_old->data;

  /* Redundant once _ops_coord_to_bin() bounds its index, but this is the site
     that corrupted the heap: a virtual particle arriving from a halo can be
     outside this rank's binning box, and binhead[address] is a write. */
  int nbins = 1;
  for (int d = 0; d < dim; d++) nbins *= map->binhead->size[d];

  for (int i = ifirst; i < ifirst + n_to_map; i++) {

    int address;
    switch(particle->particle_pos_dat->type_size) {
    case sizeof(float):
       address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                   (float *) xpos + dim * i);
       break;
    case sizeof(double):
       address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                   (double *) xpos + dim * i);
       break;
    case sizeof(long double):
       address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                   map->binhead->size, (long double *) xpos + dim * i);
       break;
    }

    if (address < 0 || address >= nbins) continue;
    bin2grid[i] = address;

    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, d_m,
                     dim, ilocal);
    switch (particle->particle_pos_dat->type_size) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
        get_coord_point((double *) xold + dim * i, ilocal, d_m, (double *) xmin,
                        (double *)dx, dim, 1);
        break;
    case sizeof(long double):
        get_coord_point((long double *) xold + dim * i, ilocal, d_m, (long double *) xmin,
                       (long double *)dx, dim, 1);
        break;
    }

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
                                       int dir_to[], int dir_from[], char* translate,
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

        if (dim != dims || !(dat->type_size != sizeof(double) ||
            dat->type_size != sizeof(float) ||
            dat->type_size != sizeof(long double)))
          throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat with orient"
                                                           " must be a vector (of size dim per particle point) and of "
                                                           "type double");

        memcpy(temp, buff + nrecv_bites, nsize);

        char *data = dat->data;

        for (int i = 0; i < nrecv; i++) {
          int ipart = nfirst + i;
          for (int isou = 0; isou < dim; isou++)
            switch(dat->type_size) {
            case sizeof(float):
              ((float *) data)[ipart * dim + dir_to[isou]] =
                ((float *) temp)[i * dim + dir_from[isou]]
                + ((float  *) translate)[dir_from[isou]];
            break;
            case sizeof(double):
              ((double *) data)[ipart * dim + dir_to[isou]] =
              ((double *) temp)[i * dim + dir_from[isou]]
              + ((double  *) translate)[dir_from[isou]];
            break;
            case sizeof(long double):
              ((long double *) data)[ipart * dim + dir_to[isou]] =
              ((long double *) temp)[i * dim + dir_from[isou]]
              + ((long double  *) translate)[dir_from[isou]];
            break;
            }
        }
      }
      nrecv_bites +=nsize;
    }
    else if (flag && halo_data[ihalo]->halo_type == OPS_EXCHANGE_HISTORY) {
      ops_neighbor_history history = halo_data[ihalo]->history_to;
      ops_dat npartnersI = history->n_partnersI;
      ops_dat partnersI = history->partnersI;
      ops_dat indexI = history->indexI;

      ops_particle particle = history->particleI;
      int *tags = (int *)particle->ids->data;

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
                                      int dir_to[], int dir_from[], char *translate,
                                      int *ntot_bites) {

  int nfirst = halo_info->firstrecv;
  int nrecv = halo_info->nrecv;

  int nrecv_bites = 0;
  int nsize = dat->elem_size;

  if (orient == OPS_PART_ORIENT_ON) {

    if (nsize * nrecv > ntmp_size_max) {
      temp = (char *) ops_realloc(temp, nsize * (nrecv + OPS_MAX_PART));
      ntmp_size_max = nsize * (nrecv + OPS_MAX_PART);
    }



    memcpy(temp, buff + nrecv_bites, nsize);

    switch(dat->type_size) {
    case sizeof(float): {
      float *tmp = (float *)temp;
      float *data = (float *) dat->data;

      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }

      } break;
    case sizeof(double): {
      double *tmp = (double *) temp;
      double *data = (double *) dat->data;
      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }
      } break;
    case sizeof(long double): {
      long double *tmp = (long double *)temp;
      long double *data = (long double *) dat->data;

      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }
      } break;
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
  memcpy(history->n_partnersI->data + nfirst, buff + nrecv_bites, nbytes);
  nrecv_bites += nbytes;

  //Part II: Push back partnersI of particle to the list
  int new_neighs = 0;
  for (int i = 0; i < nrecv; i++) {
    int ipart = nexist + i;
    int nshift = ipart * history->partnersI->elem_size;
    int nbites = ((int *)history->n_partnersI->data)[ipart]
               * history->partnersI->type_size;
    memcpy(history->partnersI->data + nshift, buff + nrecv_bites, nbytes);
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

  char *xpos = particle->particle_pos_dat->data;
  int dim = particle->block->dims;

  int nsend_neg = 0;
  int nsend_pos = 0;

  for (int ipart = ifirst; ipart < ilast; ipart++) {
    if (iswap < halo->nswap_neg && _particle_is_within(xpos, halo->region_bord_neg, ipart, dim ,
                                                       particle->particle_pos_dat->type_size)) {
      nsend_neg++;
      continue;
    }

    if (iswap < halo->nswap_pos && _particle_is_within(xpos, halo->region_bord_pos, ipart, dim,
                                                       particle->particle_pos_dat->type_size)) {
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

  char *xpos = particle->particle_pos_dat->data;

  for (int ipart = ifirst; ipart < ilast; ipart++) {
    if (iswap < halo->nswap_neg && _particle_is_within(xpos, halo->region_bord_neg, ipart,
                                                       dim, particle->type_box)) {

      halo->particle_send_neg[nsend_neg] = ipart;
      nsend_neg++;

    }

    if (iswap < halo->nswap_pos && _particle_is_within(xpos, halo->region_bord_pos, ipart,
                                                       dim, particle->type_box)) {
      halo->particle_send_pos[nsend_pos] = ipart;
      nsend_pos++;
    }

  }
}


//TODO:
void ops_mpi_particle_host_write_to_file(ops_particle particle, const char *file_name,
                                         char *buff, size_t len) {

  //OPEN MPI_FILE
  MPI_File fh;

  sub_block *sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  int my_rank;
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

    switch (particle->type_box) {
    case sizeof(float): {
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ", particle->block->name,
                       ((BoundingBox<float> *)particle->box_block)->getGlobalMin().x,
                       ((BoundingBox<float> *)particle->box_block)->getGlobalMax().x,
                       ((BoundingBox<float> *)particle->box_block)->getGlobalMin().y,
                       ((BoundingBox<float> *)particle->box_block)->getGlobalMax().y);

      if (particle->block->dims == 3)
        _ops_append_char(buff, len, size, "[%f %f] ",
                         ((BoundingBox<float> *)particle->box_block)->getGlobalMin().z,
                         ((BoundingBox<float> *) particle->box_block)->getGlobalMax().z);
    } break;
    case sizeof(double): {
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ", particle->block->name,
                             ((BoundingBox<double> *)particle->box_block)->getGlobalMin().x,
                             ((BoundingBox<double> *)particle->box_block)->getGlobalMax().x,
                             ((BoundingBox<double> *)particle->box_block)->getGlobalMin().y,
                             ((BoundingBox<double> *)particle->box_block)->getGlobalMax().y);

      if (particle->block->dims == 3)
        _ops_append_char(buff, len, size, "[%f %f] ",
                        ((BoundingBox<double> *)particle->box_block)->getGlobalMin().z,
                        ((BoundingBox<double> *) particle->box_block)->getGlobalMax().z);
    } break;
    case sizeof(long double): {
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ", particle->block->name,
                             ((BoundingBox<long double> *)particle->box_block)->getGlobalMin().x,
                             ((BoundingBox<long double> *)particle->box_block)->getGlobalMax().x,
                             ((BoundingBox<long double> *)particle->box_block)->getGlobalMin().y,
                             ((BoundingBox<long double> *)particle->box_block)->getGlobalMax().y);

            if (particle->block->dims == 3)
              _ops_append_char(buff, len, size, "[%f %f] ",
                               ((BoundingBox<long double> *)particle->box_block)->getGlobalMin().z,
                               ((BoundingBox<long double> *) particle->box_block)->getGlobalMax().z);
    } break;
    }

    _ops_append_char(buff, len, size, "\nParticle: %s \nNumber of particles: %d\n", particle->name, ntotal);

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
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      BoundingBox<float>* box = (BoundingBox<float> *)particle->box_block;
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ",
                       particle->block->name, box->getGlobalMin().x,
                       box->getGlobalMax().x,
                       box->getGlobalMin().y,
                       box->getGlobalMax().y);
      if (particle->block->dims == 3)
        _ops_append_char(buff, len, size, "[%f %f]\n",  box->getGlobalMin().z,
                         box->getGlobalMax().z);
      else
        _ops_append_char(buff, len, size, "\n");

    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      BoundingBox<double>* box = (BoundingBox<double> *)particle->box_block;
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ",
                       particle->block->name, box->getGlobalMin().x,
                       box->getGlobalMax().x,
                       box->getGlobalMin().y,
                       box->getGlobalMax().y);
      if (particle->block->dims == 3)
        _ops_append_char(buff, len, size, "[%f %f]\n",  box->getGlobalMin().z,
                         box->getGlobalMax().z);
      else
        _ops_append_char(buff, len, size, "\n");

    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
      BoundingBox<long double>* box = (BoundingBox<long double> *)particle->box_block;
      _ops_append_char(buff, len, size, "Block %s: [%f %f] x [%f %f] ",
                       particle->block->name, box->getGlobalMin().x,
                       box->getGlobalMax().x,
                       box->getGlobalMin().y,
                       box->getGlobalMax().y);
      if (particle->block->dims == 3)
        _ops_append_char(buff, len, size, "[%f %f]\n",  box->getGlobalMin().z,
                         box->getGlobalMax().z);
      else
        _ops_append_char(buff, len, size, "\n");

    }

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

