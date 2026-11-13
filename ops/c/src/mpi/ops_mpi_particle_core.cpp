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
  * @brief OPS MPI functions for particle functionalities
  * @author  C. Tsinginos
  * @details Implements runtime support functions applicable to MPI backend for
  *          particle functionalities
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

#include "ops_bounding_box.h"
#include "ops_particle_box_mpi_funcs.h"
#include "ops_particle_mapping_functions.h"

static void  _compute_local_region(char *send_local, const char *box_block, const int dim,
                                   const int type_size) {

  switch(type_size) {
  case sizeof(float): {
    float *send = (float *)send_local;
    BoundingBox<float> *box = (BoundingBox<float> *)box_block;
    for (int i = 0; i < dim; i++)  {
      send[2 * i] = box->getMinCoordDir(i);
      send[2 * i + 1] = box->getMaxCoordDir(i);
    }
  } break;
  case sizeof(double): {
    double *send = (double *)send_local;
    BoundingBox<double> *box = (BoundingBox<double> *)box_block;
    for (int i = 0; i < dim; i++) {
      send[2 * i] = box->getMinCoordDir(i);
      send[2 * i + 1] = box->getMaxCoordDir(i);
    }
  } break;
  case sizeof(long double): {
    long double *send = (long double *)send_local;
    BoundingBox<long double> *box = (BoundingBox<long double> *)box_block;
    for (int i = 0; i < dim; i++) {
      send[2 * i] = box->getMinCoordDir(i);
      send[2 * i + 1] = box->getMaxCoordDir(i);
    }
  } break;
  }
}

static void _get_halo_recv_region(char *xrecv_min, char *xrecv_max, char *recv_box_regions,
                                  char *translate, const int i, const int dir_to[],
                                  const int dir_from[], const int dim, const int type_size) {

  switch(type_size) {
  case sizeof(float): {
   for (int isou = 0; isou < dim; isou++) {
     int irecv_dir = dir_to[isou];
     int isend_dir = dir_from[isou];
     ((float *)xrecv_min)[isend_dir] = ((float *) recv_box_regions)[2 * dim * i + 2 * irecv_dir]
                                      - ((float *)translate)[isend_dir];
     ((float *)xrecv_max)[isend_dir] = ((float *) recv_box_regions)[2 * dim * i + 2 * irecv_dir + 1]
                                      - ((float *) translate)[isend_dir];
   }
  } break;
  case sizeof(double): {
    for (int isou = 0; isou < dim; isou++) {
      int irecv_dir = dir_to[isou];
      int isend_dir = dir_from[isou];
      ((double *)xrecv_min)[isend_dir] = ((double *) recv_box_regions)[2 * dim * i + 2 * irecv_dir]
                                     - ((double *)translate)[isend_dir];
      ((double *)xrecv_max)[isend_dir] = ((double *) recv_box_regions)[2 * dim * i + 2 * irecv_dir + 1]
                                        - ((double *) translate)[isend_dir];
    }
  } break;
  case sizeof(long double): {
    for (int isou = 0; isou < dim; isou++) {
      int irecv_dir = dir_to[isou];
      int isend_dir = dir_from[isou];
      ((long double *)xrecv_min)[isend_dir] = ((long double *) recv_box_regions)[2 * dim * i + 2 * irecv_dir]
                                          - ((long double *)translate)[isend_dir];
      ((long double *)xrecv_max)[isend_dir] = ((long double *) recv_box_regions)[2 * dim * i + 2 * irecv_dir + 1]
                                        - ((long double *) translate)[isend_dir];
    }

  } break;
  }
}

static void _get_halo_send_region(char *xsend_min, char *xsend_max, char *send_box_regions,
                                  char *translate, const int i, const int dir_to[],
                                  const int dir_from[], const int dim, const int type_size) {

  switch(type_size) {
  case sizeof(float): {
    for (int isou = 0; isou < dim; isou++) {
      int isend_dir = dir_to[isou];
      int irecv_dir = dir_from[isou];
      ((float *)xsend_min)[irecv_dir] = ((float *)send_box_regions)[2 * dim * i + 2 * isend_dir]
                                      + ((float *)translate)[irecv_dir];
      ((float *)xsend_max)[irecv_dir] = ((float *)send_box_regions)[2 * dim * i + 2 * isend_dir + 1]
                                      + ((float *)translate)[irecv_dir];

    }

  } break;
  case sizeof(double): {
    for (int isou = 0; isou < dim; isou++) {
      int isend_dir = dir_to[isou];
      int irecv_dir = dir_from[isou];
      ((double *)xsend_min)[irecv_dir] = ((double *)send_box_regions)[2 * dim * i + 2 * isend_dir]
                                      + ((double *)translate)[irecv_dir];
      ((double *)xsend_max)[irecv_dir] = ((double *)send_box_regions)[2 * dim * i + 2 * isend_dir + 1]
                                      + ((double *)translate)[irecv_dir];

    }
  } break;
  case sizeof(long double): {
    for (int isou = 0; isou < dim; isou++) {
      int isend_dir = dir_to[isou];
      int irecv_dir = dir_from[isou];
      ((long double *)xsend_min)[irecv_dir] = ((long double *)send_box_regions)[2 * dim * i + 2 * isend_dir]
                                      + ((long  double *)translate)[irecv_dir];
      ((long double *)xsend_max)[irecv_dir] = ((long double *)send_box_regions)[2 * dim * i + 2 * isend_dir + 1]
                                      + ((long double *)translate)[irecv_dir];

    }
  } break;
  }
}




static void * _declaire_exchange_sending_box(const char *xmin, const char *xmax,const int dim,
                                             const char *xrecv_min, const char *xrecv_max,
                                             int idir, int iswap, int type_size) {


  if (type_size == sizeof(float)) {

    float xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((float *)xmin)[j];
      xsend_max[j] = ((float *)xmax)[j];
    }

    xsend_min[idir] = (iswap == 0) ? ((float *)xmin)[idir] - 0.5 * BIG : ((float *)xmax)[idir];
    xsend_max[idir] = (iswap == 0) ? ((float *)xmin)[idir] : ((float *)xmax)[idir] + 0.5 * BIG;

    int a1 = ops_check_box_intersections2(dim, xsend_min, xsend_max,
                                          (float *)xrecv_min, (float *)xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (j != idir) {
          xsend_min[j]
               = (ops_abs(xsend_min[j] - ((float *)xrecv_min)[j]) < std::numeric_limits<float>::epsilon()) ? -BIG :
                   MAX(xsend_min[j], ((float *)xrecv_min)[j]);
          xsend_max[j]
               = ops_abs(xsend_max[j] - ((float *) xrecv_max)[j]) < std::numeric_limits<float>::epsilon() ? BIG :
                   MIN(xsend_max[j], ((float *)xrecv_max)[j]);
        }
      }

      BoundingBox<float> *box = new BoundingBox<float>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);

      return (void *) box;
    }
  }
  else if (type_size == sizeof(double)) {
    double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((double *)xmin)[j];
      xsend_max[j] = ((double *)xmax)[j];
    }

    xsend_min[idir] = (iswap == 0) ? ((double *)xmin)[idir] - 0.5 * BIG : ((double *)xmax)[idir];
    xsend_max[idir] = (iswap == 0) ? ((double *)xmin)[idir] : ((double *)xmax)[idir] + 0.5 * BIG;


    int a1 = ops_check_box_intersections2(dim, xsend_min, xsend_max,
                                          (double *)xrecv_min, (double *)xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (j != idir) {
          xsend_min[j]
               = (ops_abs(xsend_min[j] - ((double *)xrecv_min)[j]) < std::numeric_limits<double>::epsilon()) ? -BIG :
                   MAX(xsend_min[j], ((double *)xrecv_min)[j]);
          xsend_max[j]
               = ops_abs(xsend_max[j] - ((double *) xrecv_max)[j]) < std::numeric_limits<double>::epsilon() ? BIG :
                   MIN(xsend_max[j], ((double *)xrecv_max)[j]);
        }
      }

      BoundingBox<double> *box = new BoundingBox<double>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);

      return (void *) box;
    }
  }
  else if (type_size == sizeof(long double)) {
    long double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((long double *) xmin)[j];
      xsend_max[j] = ((long double *) xmax)[j];
    }

    xsend_min[idir] = (iswap == 0) ? ((long double *)xmin)[idir] - 0.5 * BIG : ((long double *)xmax)[idir];
    xsend_max[idir] = (iswap == 0) ? ((long double *)xmin)[idir] : ((long double *)xmax)[idir] + 0.5 * BIG;

    int a1 = ops_check_box_intersections2(dim, xsend_min, xsend_max,
                                          (long double *)xrecv_min, (long double *)xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (j != idir) {
          xsend_min[j]
               = (ops_abs(xsend_min[j] - ((long double *)xrecv_min)[j]) < std::numeric_limits<long double>::epsilon()) ? -BIG :
                   MAX(xsend_min[j], ((long double *)xrecv_min)[j]);
          xsend_max[j]
               = ops_abs(xsend_max[j] - ((long double *) xrecv_max)[j]) < std::numeric_limits<long double>::epsilon() ? BIG :
                   MIN(xsend_max[j], ((long double *)xrecv_max)[j]);
        }
      }

      BoundingBox<long double> *box = new BoundingBox<long double>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);

      return (void *) box;
    }

  }

  return nullptr;
}

static void * _declaire_border_sending_box(const char *xrecv_min, const char *xrecv_max,
                                           const char * xmin, const char *xmax, const char *dx,
                                           const int dim, const int idir, const int iswap,
                                           const int size_type) {

  if (size_type == sizeof(float)) {
    float xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((float *)xmin)[j];
      xsend_max[j] = ((float *)xmax)[j];
    }

    float dx[OPS_MAX_DIM];
    xsend_min[idir] = (iswap == 0) ? ((float *)xmin)[idir] - 0.5 * BIG :
                                     ((float *)xmax)[idir] - ((float *)dx)[idir];
    xsend_max[idir] = (iswap == 0) ? ((float *)xmin)[idir] + ((float *)dx)[idir] :
                                     ((float *)xmax)[idir] + 0.5 * BIG;

    int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                         (float *) xrecv_min, (float *) xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (idir != j) {
          xsend_min[j] = (ops_abs(xsend_min[j] - ((float *)xrecv_min)[j])
              < std::numeric_limits<float>::epsilon()) ?
                          -BIG : MAX(xsend_min[j], ((float *)xrecv_min)[j]);
          xsend_max[j] = (ops_abs(xsend_max[j] - ((float *)xrecv_max)[j])
              < std::numeric_limits<float>::epsilon()) ?
                           BIG : MIN(xsend_max[j], ((float *)xrecv_max)[j]);
        }
      }

      BoundingBox<float> * box = new BoundingBox<float>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);
      return (void *) box;

    }
  }
  else if (size_type == sizeof(double)) {
    double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((double *)xmin)[j];
      xsend_max[j] = ((double *)xmax)[j];
    }

    xsend_min[idir] = (iswap == 0) ? ((double *)xmin)[idir] - 0.5 * BIG :
                                     ((double *)xmax)[idir] - ((double *)dx)[idir];
    xsend_max[idir] = (iswap == 0) ? ((double *)xmin)[idir] + ((double *)dx)[idir] :
                                     ((double *)xmax)[idir] + 0.5 * BIG;

    int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                         (double *) xrecv_min, (double *) xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (idir != j) {
          xsend_min[j] = (ops_abs(xsend_min[j] - ((double *)xrecv_min)[j])
              < std::numeric_limits<double>::epsilon()) ?
                          -BIG : MAX(xsend_min[j], ((double *)xrecv_min)[j]);
          xsend_max[j] = (ops_abs(xsend_max[j] - ((double *)xrecv_max)[j])
              < std::numeric_limits<double>::epsilon()) ?
                           BIG : MIN(xsend_max[j], ((double *)xrecv_max)[j]);
        }
      }

      BoundingBox<double> *box = new BoundingBox<double>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);
      return (void *) box;
    }
  }
  else if (size_type == sizeof(long double)) {
    long double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
    for (int j = 0; j < dim; j++) {
      xsend_min[j] = ((long double *)xmin)[j];
      xsend_max[j] = ((long double *)xmax)[j];
    }

    xsend_min[idir] = (iswap == 0) ? ((long double *)xmin)[idir] - 0.5 * BIG :
                                     ((long double *)xmax)[idir] - ((long double *)dx)[idir];
    xsend_max[idir] = (iswap == 0) ? ((long double *)xmin)[idir] + ((long double *)dx)[idir] :
                                     ((long double *)xmax)[idir] + 0.5 * BIG;

    int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                         (long double *) xrecv_min, (long double *) xrecv_max);

    if (a1 == 1) {
      for (int j = 0; j < dim; j++) {
        if (idir != j) {
          xsend_min[j] = (ops_abs(xsend_min[j] - ((long double *)xrecv_min)[j])
              < std::numeric_limits<long double>::epsilon()) ?
                          -BIG : MAX(xsend_min[j], ((long double *)xrecv_min)[j]);
          xsend_max[j] = (ops_abs(xsend_max[j] - ((long double *)xrecv_max)[j])
              < std::numeric_limits<long double>::epsilon()) ?
                           BIG : MIN(xsend_max[j], ((long double *)xrecv_max)[j]);
        }
      }

      BoundingBox<long double> * box = new BoundingBox<long double>(dim);
      box->setBoundingBoxLocalBound(xsend_min, xsend_max);
      return (void *) box;
    }
  }

  return nullptr;
}

static bool _ops_box_get_ownership(char *box, int size_type) {

  switch (size_type) {
  case sizeof(float):
    return ((BoundingBox<float> *) box)->getOwnership();
    break;
  case sizeof(double):
    return ((BoundingBox<double> *) box)->getOwnership();
    break;
  case sizeof(long double):
    return ((BoundingBox<long double> *) box)->getOwnership();
    break;
  }

  return false;
}

static void   _get_local_box(char *xmin, char *xmax, char *box_block,
                             const int dim, const int size_type) {

  switch(size_type) {
  case sizeof(float): {
    ((BoundingBox<float > * ) box_block)->getLocalMaxMin((float *) xmin, (float *) xmax);
  } break;
  case sizeof(double): {
    ((BoundingBox<double> * ) box_block)->getLocalMaxMin((double *) xmin, (double *) xmax);
  } break;
  case sizeof(long double): {
    ((BoundingBox<long double> *) box_block)->getLocalMaxMin((long double *) xmin,
                                                             (long double *) xmax);
  } break;
  }
}

void   _ops_particle_init_map_grid(ops_particle_mapping map, char *dx) {

  if (map->map_type == 0 || map->map_type == 1) return;

  sub_block *sb = OPS_sub_block_list[map->particle->block->index];
  if (!sb->owned) return;

  int flag_m[OPS_MAX_DIM] = {};
  int flag_p[OPS_MAX_DIM] = {};

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  int dim = map->particle->block->dims;
  int dgrid_m[OPS_MAX_DIM], dgrid_p[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    d_m[i] = map->grid_head->d_m[i] + OPS_sub_dat_list[map->grid_head->index]->d_im[i];
    d_p[i] = map->grid_head->d_p[i] + OPS_sub_dat_list[map->grid_head->index]->d_ip[i];
    dgrid_m[i] = map->grid->d_m[i] + OPS_sub_dat_list[map->grid->index]->d_im[i];
    dgrid_p[i] = map->grid->d_p[i] + OPS_sub_dat_list[map->grid->index]->d_ip[i];

  }
  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0;
    d_p[i] = 0;
    dgrid_m[i] = 0;
    dgrid_p[i] = 0;
  }


  //Get points in range
  int nPoints = 0;
  int type_size = map->grid->type_size;
  switch (type_size) {
  case sizeof(float):
    _ops_particle_points_in_map((float *) map->xmin, (float *) map->xmax, (float *) dx,
                                (float *) map->grid->data, map->particle->block->dims,
                                map->grid->size, dgrid_m, dgrid_p, flag_m, flag_p,
                                d_m, d_p,
                                OPS_instance::getOPSInstance()->OPS_soa, &nPoints,
                                map->region_access, (float *) map->region_grid);
    break;
  case sizeof(double):
    _ops_particle_points_in_map((double *) map->xmin, (double *) map->xmax,(double *) dx,
                                (double *) map->grid->data, map->particle->block->dims,
                                map->grid->size, dgrid_m, dgrid_p,
                                flag_m, flag_p, d_m, d_p,
                                OPS_instance::getOPSInstance()->OPS_soa, &nPoints,
                                map->region_access, (double *) map->region_grid);
    break;
  case sizeof(long double):
    _ops_particle_points_in_map((long double *) map->xmin, (long double *) map->xmax,
                                (long double *) dx, (long double *) map->grid->data,
                                map->particle->block->dims,
                                map->grid->size, d_m, d_p,
                                flag_m, flag_p, map->grid_head->d_m, map->grid_head->d_p,
                                OPS_instance::getOPSInstance()->OPS_soa, &nPoints,
                                map->region_access, (long double *) map->region_grid);
    break;
  }

  if (nPoints > 0) {
    map->grid_bin->data = (char *) ops_realloc(map->grid_bin->data, nPoints * sizeof(int));
    map->grid_bin->size[0] = nPoints;
    map->id_grid->data = (map->map_type > 1) ? (char *) ops_realloc(map->id_grid->data, nPoints * sizeof(int)) : nullptr;
    if (map->map_type > 1) {
      map->id_grid->size[0] = nPoints;
    }
  }
  else {
    map->grid_bin->size[0] = 0;
    map->id_grid->size[0] = 0;
  }

  //Update grid_map
  int *grid_head = (int *) map->grid_head->data;
  int *grid_bin = (int *) map->grid_bin->data;
  int *id_grid = (int *) map->id_grid->data;
  int *grid_to_map = (int *) map->grid_to_map->data;

  switch(type_size) {
  case sizeof(float):
    _ops_particle_generate_maps_non_conf( map->map_type, (float *) map->xmin, (float *) map->xmax,
                                         (float *) dx, (float *) map->grid->data,
                                         map->particle->block->dims, grid_head, grid_bin,
                                         id_grid, grid_to_map, map->grid->size, dgrid_m, dgrid_p,
                                         flag_m, flag_p,  map->grid_head->size, d_m,
                                         d_p, OPS_instance::getOPSInstance()->OPS_soa,
                                         nPoints, map->region_access, (float *)map->region_grid);
    break;
  case sizeof(double):
    _ops_particle_generate_maps_non_conf( map->map_type, (double *) map->xmin, (double *) map->xmax,
                                         (double *) dx, (double *) map->grid->data,
                                         map->particle->block->dims, grid_head, grid_bin,
                                         id_grid, grid_to_map, map->grid->size, dgrid_m, dgrid_p,
                                         flag_m, flag_p,  map->grid_head->size, d_m,
                                         d_p, OPS_instance::getOPSInstance()->OPS_soa,
                                         nPoints, map->region_access, (double *)map->region_grid);
    break;
  case sizeof(long double):
    _ops_particle_generate_maps_non_conf( map->map_type, (long double *) map->xmin, (long double *) map->xmax,
                                         (long double *) dx, (long double *) map->grid->data,
                                         map->particle->block->dims, grid_head, grid_bin,
                                         id_grid, grid_to_map, map->grid->size, dgrid_m, dgrid_p,
                                         flag_m, flag_p,  map->grid_head->size, d_m,
                                         d_p, OPS_instance::getOPSInstance()->OPS_soa,
                                         nPoints, map->region_access, (long double *)map->region_grid);
  }


}


void _ops_mapping_def_core(ops_particle particle, ops_dat grid, ops_stencil stencil,
                           ops_with_virtual &include_virtual, int size[],
                           int base[], int d_m[], int d_p[]) {

  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Empty particle structure");

  if (grid == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Empty grid structure. Mapping structures "
                                             "require a user-defined grid (ops_dat) structure");

  ops_block block_particle = particle->block;
  ops_block block_grid = grid->block;
  if (strcmp(block_particle->name, block_grid->name) != 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Inconsistent block assignment: Particle and grid "
                                             "(ops_dat) structure are assigned to different ops_block "
                                             "structures");

  include_virtual = OPS_WITH_VIRTUAL; //Becomes default for modelling particle structures

  int dim = block_particle->dims;
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i] + grid->d_m[i] - grid->d_p[i] -1;

    base[i] = grid->base[i];
    d_m[i] = MIN(grid->d_m[i], -1);
    d_p[i] = MAX(grid->d_p[i], 1);
    for (int p = 0; p < stencil->points; p++) {
      d_m[i] = MIN(d_m[i], stencil->stencil[particle->block->dims * p + i]);
      d_p[i] = MAX(d_p[i], stencil->stencil[particle->block->dims * p + i]);
    }
  }

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    size[i] = 1;
    base[i] = d_m[i] = d_p[i] = 0;
  }

}

void _ops_mapping_set_structure(ops_particle particle, ops_dat mapping_grid,
                                ops_stencil stencil, int d_mm[],
                                int d_mp[], int size[], int base[],
                                const int d_m[], const int d_p[], int stride[],
                                ops_grid_type grid_type) {

  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT,"Error: Empty particle structure\n");

  if (mapping_grid == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty grid ops_dat structure\n");

  //Check if boxes conform
  bool conform{true};
  switch(particle->type_box) {
  case sizeof(float):
    conform = ((BoundingBox<float> *)particle->box_block)->isBoxConformingToBlock();
    break;
  case sizeof(double):
    conform = ((BoundingBox<double> *) particle->box_block)->isBoxConformingToBlock();
    break;
  case sizeof(long double):
    conform = ((BoundingBox<long double> *) particle->box_block)->isBoxConformingToBlock();
  }

  //Get grid type
  int grid_gen = (grid_type == OPS_UNIFORM_STAG || grid_type == OPS_UNIFORM_COLL) ? 0 : 1;

  if (conform == 0 && grid_gen == 0) {
    for (int i = 0; i < particle->block->dims; i++) {
       size[i] = mapping_grid->size[i] + mapping_grid->d_m[i] - mapping_grid->d_p[i];
       base[i] = mapping_grid->base[i];
       stride[i] = mapping_grid->stride[i];
       d_mm[i] = MIN(d_m[i], -1);
       d_mp[i] = MAX(d_p[i], 1);
       for (int p = 0; p < stencil->points; p++) {
         d_mm[i] = MIN(d_mm[i], stencil->stencil[particle->block->dims * p + i]);
         d_mp[i] = MAX(d_mp[i], stencil->stencil[particle->block->dims * p + i]);
       }
    }
  }
  else { //Case II: Non conforming & non-uniform grid
    for (int i = 0; i < particle->block->dims; i++) {
      size[i] = 2;
      base[i] = 0; //no-need for base
      d_mm[i] = MIN(d_m[i], -1);
      d_mp[i] = MAX(d_p[i], 1);
      stride[i] = (grid_gen == 0) ? mapping_grid->stride[i] : 1;
      for (int p = 0; p < stencil->points; p++) {
        d_mm[i] = MIN(d_mm[i], stencil->stencil[particle->block->dims * p + i]);
        d_mp[i] = MAX(d_mp[i], stencil->stencil[particle->block->dims * p + i]);

      }
    }
  }

}


//TODO: Need to pass them to template structures use size to handle them
void _ops_mapping_set_structures(ops_particle particle, char *skin, int  d_m[],
                                 int d_p[], int d_mb[], int d_pb[], int size[],
                                 char *dx_map,  ops_with_virtual &include_virtual) {


  switch(particle->particle_pos_dat->type_size) {
  case sizeof(float):
    if (((BoundingBox<float> *) particle->box_block)->getBlockVolume() < std::numeric_limits<float>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");
    break;
  case sizeof(double):
    if (((BoundingBox<double> *) particle->box_block)->getBlockVolume() < std::numeric_limits<double>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");
    break;
  case sizeof(long double):
    if (((BoundingBox<long double> *) particle->box_block)->getBlockVolume() <
           std::numeric_limits<long double>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");
    break;
  }



  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    float length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<float> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<float> *) particle->box_block)->getGlobalMin(i);


    _ops_map_compute_dx((float *)skin, length, particle->block->dims, (float *) dx_map, size);//TODO
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    double length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<double> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<double> *) particle->box_block)->getGlobalMin(i);

    _ops_map_compute_dx((double *)skin, length, particle->block->dims, (double *) dx_map, size);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    long double length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<long double> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<long double> *) particle->box_block)->getGlobalMin(i);
    _ops_map_compute_dx((long double *)skin, length, particle->block->dims, (long double *)dx_map,
                        size);

  } //TODO: We may want
  else
    throw OPSException(OPS_RUNTIME_ERROR, "Error: This type of float is not supported\n");

  for (int i = 0; i < particle->block->dims; i++) {
    d_mb[i] = (d_m != nullptr) ? MIN(d_m[i], -1) : -1;
    d_pb[i] = (d_p != nullptr) ? MAX(d_p[i], 1) : 1;
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) {
    size[i] = 1;

    switch(particle->type_box) {
    case sizeof(float):
      ((float *) dx_map)[i] = 0.0;
      break;
    case sizeof(double):
      ((double *) dx_map)[i] = 0.0;
      break;
    case sizeof(long double):
      ((long double *) dx_map)[i] = .0;
      break;
    }

    d_mb[i] = d_pb[i] = 0;

  }

  include_virtual = OPS_WITH_VIRTUAL; //Default for MPI
}

void _ops_particle_init_map_structures(ops_particle_mapping map) {

  ops_particle particle = map->particle;
  ops_block block = particle->block;

  char *dx_map = nullptr;
  char *dx_map1 = nullptr;

  //Part I: Get xmin xmax
  if (map->map_type <=2) {
    switch (particle->type_box) {
    case sizeof(float):
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                               (float *) map->xmin, (float *) map->xmax);
      break;
    case sizeof(double):
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                               (double *) map->xmin, (double *) map->xmax);
      break;
    case sizeof(long double):
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                               (long double *) map->xmin, (long double *) map->xmax);
      break;
    }
  }
  else if (map->map_type == 3 &&(map->grid_type == OPS_UNIFORM_STAG ||
                                 map->grid_type == OPS_UNIFORM_COLL)) {
    char *box = _ops_get_block_box(block);
    dx_map1 = (char *) ops_malloc(particle->type_box * block->dims);
    _ops_particle_update_bounds_non_conf_unif_grid(particle, box, dx_map1,
                                                   map->xmin, map->xmax, map->dx,
                                                   map->mapping_grid); //TODO
  }
  else {//Non-conforming stretched box
    switch(particle->type_box) {
    case (sizeof(float)) :
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                             (float *) map->xmin, (float *)map->xmax);
      break;
    case sizeof(double):
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                             (double *) map->xmin, (double *) map->xmax);
      break;
    case sizeof(long double):
      _ops_particle_set_map_bounds_conform_box(particle->box_block, block->dims,
                                             (long double *) map->xmin, (long double *) map->xmax);
      break;
    }
  }



  dx_map = (map->grid_type == OPS_UNIFORM_STAG || map->grid_type == OPS_UNIFORM_COLL) ? dx_map1 : map->dx;

  if (map->map_type >=2 ) {
    dx_map = (map->grid_type == OPS_UNIFORM_STAG || map->grid_type == OPS_UNIFORM_COLL) ? dx_map1 : map->dx;

    int base[OPS_MAX_DIM] ={};
    int size[OPS_MAX_DIM] = {};



    //Compute data structure
    char *box = _ops_get_block_box(block);
    switch(particle->type_box) {
    case sizeof(float): //TODO
      _ops_particle_compute_size_base(box,(float *) map->xmin, (float *) map->xmax,
                                      (float *) dx_map, block->dims, size, base);
      break;
    case sizeof(double):
      _ops_particle_compute_size_base(box, (double *) map->xmin, (double *) map->xmax,
                                      (double *) dx_map, block->dims, size, base);
      break;
    case sizeof(long double):
      _ops_particle_compute_size_base(box, (long double *) map->xmin, (long double *) map->xmax,
                                      (long double *) dx_map, block->dims, size, base);
      break;
    }

    if (map->grid_type == OPS_NON_UNI_STAG || map->grid_type == OPS_NON_UNI_COLL) {
      switch(particle->type_box) {
      case sizeof(float):
        _ops_update_halo_points((float *) map->dx, (float *) map->xmin, (float *) map->xmax,
                                size, block, map->binhead, map->grid_head);
        break;
      case sizeof(double):
        _ops_update_halo_points((double *) map->dx, (double *) map->xmin, (double *) map->xmax,
                                size, block, map->binhead, map->grid_head);
        break;
      case sizeof(long double):
        _ops_update_halo_points((long double *) map->dx, (long double *) map->xmin, (long double *) map->xmax,
                                size, block, map->binhead, map->grid_head);
        break;
      }
    }

    //TODO: Tommorow

    for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) size[i] = 1;

    ops_malloc_non_defined_dat(map->binhead, size, base); //TODO

    ops_malloc_non_defined_dat(map->grid_head, size, base);


  }

  //Part III: Allocate data & initialize grid structures
  _ops_particle_init_map_grid(map, dx_map);


/*
  printf("Maps constructed. Tommorow additional checks\n");
  if (map->map_type > 1) {
  size_t size_grid = map->grid_to_map->size[1];
  int ip = 0;
  for (size_t j = 0; j < map->grid->size[1]; j++) {
    for (size_t i = 0; i < map->grid->size[0]; i++) {
    int address = i + map->grid->size[0] * j;
    int within = ((int *) map->grid_to_map->data)[address];
    if (within > -1 && ops_get_proc() == 0) { ip++;
    printf("R %d: %d of %d (%d) : (%d) [%12.9e %12.9e] in region [%12.9e %12.9e]x[%12.9e %12.9e]\n", ops_get_proc(),
           ip, map->grid_bin->size[0], address, within, ((double *) map->grid->data)[2 * address], ((double *) map->grid->data)[2 * address + 1],
           ((double *) map->xmin)[0], ((double *) map->xmax)[0], ((double *) map->xmin)[1],
           ((double *) map->xmax)[1]);
    }
    }
  }


  if (ops_get_proc() == 0) {
    printf("Map grid_head = [%d %d]\n", map->grid_head->size[0], map->grid_head->size[1]);
    for (size_t j = 0; j < map->grid_head->size[1]; j++) {
      for (size_t i = 0; i < map->grid_head->size[0]; i++ ) {

        int address = i + j * map->grid_head->size[0];
        int grid_bin = ((int *) map->grid_head->data)[address];

        printf("R1 %d [%d %d] (%d) has grid_points (head = %d): ", ops_get_proc(), i, j,
                                                                   address, grid_bin);

       while (grid_bin > - 1) {
         printf("[%d %d], ", grid_bin, ((int *) map->id_grid->data)[grid_bin]);
         grid_bin = ((int *) map->grid_bin->data)[grid_bin];
       }

        printf("\n");
    }
  }
  }
  } */


  ops_free(dx_map1);

}


void ops_build_bounding_box(ops_particle particle) {

  //TODO: Predifine communication map if necessary-
  //For the moment is the first entry map


  if (particle->type_box == sizeof(float)) {
    BoundingBox<float> *boxT = (BoundingBox<float> *)particle->box_block;
    boxT->partitionBoundingBox(particle->block,
                               particle->map_list[0]->binhead,
                               (float *)particle->map_list[0]->dx);
  }
  else if (particle->type_box == sizeof(double)) {


    BoundingBox<double> *boxT = (BoundingBox<double> *)particle->box_block;


    boxT->partitionBoundingBox(particle->block,
                               particle->map_list[0]->binhead,
                               (double *) particle->map_list[0]->dx);

  }
  else if (particle->type_box == sizeof(long double)) {
    BoundingBox<long double> *boxT = (BoundingBox<long double> *)particle->box_block;
    boxT->partitionBoundingBox(particle->block,
                               particle->map_list[0]->binhead,
                               (long double *) particle->map_list[0]->dx);
  }

}

void ops_build_main_bounding_box(ops_block block) {

  char *box = block->instance->OPS_block_list[block->index].box;
  int block_type = block->instance->OPS_block_list[block->index].type_size;

  switch(block_type) {
  case sizeof(float):
    ((BoundingBox<float> *) box)->partitionBoundingBox(block);
    break;
  case sizeof(double):
    ((BoundingBox<double> *) box)->partitionBoundingBox(block);
    break;
  case sizeof(long double):
    ((BoundingBox<long double> *) box)->partitionBoundingBox(block);
     break;
  }

}

void   ops_particle_update_intra_halo_maps(ops_particle particle, int ifirst,
                                           int ilast) {

  sub_block *sb =  OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    _ops_particle_update_map_int_halos(map, particle, ifirst, ilast);
  }
}

//TODO: Check if we need it
bool ops_particle_global_rebuild(bool flag) {
  return flag;
}


/*=====================================================================================*
 *              HALO DEFINITIONS FOR PARALLEL COMMS                                                                       *
 *=====================================================================================*/

void ops_get_send_recv_box(char *recv_box_regions, char *send_box_regions,
                           const ops_mpi_particle_halo *mpi_halo, int dim) {

  ops_particle_halo halo
  = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];
  sub_block *sb_from = OPS_sub_block_list[halo->particle_from->block->index];
  sub_block *sb_to = OPS_sub_block_list[halo->particle_to->block->index];

  char *send_local = (char *) ops_malloc(2 * OPS_MAX_DIM * halo->particle_from->type_box);
  char *recv_local = (char *) ops_malloc(2 * OPS_MAX_DIM * halo->particle_to->type_box);


  MPI_Request request_send[mpi_halo->nproc_from_max + mpi_halo->nproc_to_max];
  MPI_Request request_recv[mpi_halo->nproc_from_max + mpi_halo->nproc_to_max];

  if (sb_from->owned) {
    if (!_ops_box_get_ownership(halo->particle_from->box_block, halo->particle_from->type_box))
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Local Boxes need to be defined prior to "
                         " halo setup (sending block)");

    _compute_local_region(send_local, halo->particle_from->box_block, dim,
                          halo->particle_from->type_box);

    //Send data
    int box_type = halo->particle_from->type_box;
    for (int i = 0; i < mpi_halo->nproc_to_max; i++) {
      int idp = mpi_halo->proclist_complete[i];
      MPI_Isend(send_local, 2 * dim * box_type, MPI_CHAR, idp, 200,
                OPS_MPI_GLOBAL, &request_send[i]);
    }

  }

  if (sb_to->owned) {
    if (!_ops_box_get_ownership(halo->particle_to->box_block, halo->particle_to->type_box))
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications"
                                            " require the partition of bounding box");

    _compute_local_region(recv_local, halo->particle_to->box_block, dim,
                          halo->particle_to->type_box);

    int box_type = halo->particle_to->type_box;
    for (int i = 0; i < mpi_halo->nproc_from_max; i++) {
      int idp = mpi_halo->proclist_complete[mpi_halo->nproc_to_max + i];

      MPI_Isend(recv_local, 2 * dim * box_type, MPI_CHAR, idp, 100, OPS_MPI_GLOBAL,
                &request_send[i +  mpi_halo->nproc_to_max]);
    }
  }

  //Received boxes

  //Perform receive and unpack
  int buffsize = MAX(halo->particle_from->type_box, halo->particle_to->type_box);
  char *buff = (char *) ops_malloc(2 * OPS_MAX_DIM * buffsize);
  if (sb_from->owned) {
    for (int i = mpi_halo->nproc_to_max - 1; i >= 0; i--) {
      int idp = mpi_halo->proclist_complete[i];
      int type_size = halo->particle_to->type_box;
      MPI_Irecv(buff, 2 * dim * type_size, MPI_CHAR, idp, 100, OPS_MPI_GLOBAL,
                &request_recv[i]);
      MPI_Status status;
      MPI_Wait(&request_recv[i], &status);

      memcpy(recv_box_regions + 2 * i * dim * type_size , buff, 2 * dim * type_size);

    }

  }

  if (sb_to->owned) {
    for (int  i = mpi_halo->nproc_from_max - 1; i >= 0; i--) {
      int idp = mpi_halo->proclist_complete[mpi_halo->nproc_to_max + i];
      int type_size = halo->particle_to->type_box;
      MPI_Irecv(buff, 2 * dim * type_size, MPI_CHAR, idp, 200, OPS_MPI_GLOBAL,
                &request_recv[i + mpi_halo->nproc_to]);

      MPI_Status status;
      MPI_Wait(&request_recv[i + mpi_halo->nproc_to], &status);


      memcpy(send_box_regions + 2 * i * dim * type_size, buff, 2 * dim * type_size);
     // _copy_regions(send_box_regions + 2 * i * dim * type_size , buff, dim, type_size); //TODO: memcpy

    }

  }

  MPI_Status status_recv[mpi_halo->nproc_to_max + mpi_halo->nproc_from_max];
  if (sb_from->owned) {
    MPI_Waitall(mpi_halo->nproc_to_max, &request_send[0], &status_recv[0]);
  }
  if (sb_to->owned) {
    MPI_Waitall(mpi_halo->nproc_from_max, &request_send[mpi_halo->nproc_to_max],
                &status_recv[mpi_halo->nproc_to_max]);
  }

  ops_free(send_local);
  ops_free(recv_local);
  ops_free(buff);

}

//TODO: TOUGH FUNCTION
void _ops_particle_halo_set_exchange_send(ops_mpi_particle_halo *mpi_halo,
                                          char *recv_box_regions,
                                          int dim) {

  ops_particle_halo halo =
      OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block->index];
  if (!sb->owned) {mpi_halo->nproc_to = 0; return; }

  int nsend_max = mpi_halo->nproc_to_max;
  int nsend = 0;

  if (!_ops_box_get_ownership(halo->particle_from->box_block, halo->particle_from->type_box))
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications "
                                          " requires the partition of BoundingBoxes");


  int halo_size = halo->particle_from->type_box;
  char *xmin = (char *)ops_malloc(dim * halo_size);
  char *xmax = (char *)ops_malloc(dim * halo_size);
  char *xrecv_min = (char *)ops_malloc(dim * halo_size);
  char *xrecv_max = (char *) ops_malloc(dim * halo_size);


  //TODO: ADD addition
  _get_local_box(xmin, xmax, halo->particle_from->box_block, dim,
                 halo->particle_from->type_box);


  int nmax = mpi_halo->nproc_from_max + mpi_halo->nproc_to_max;

  int *proclist = (int *) ops_malloc(nmax * sizeof(int));
  void **boxes = (void **) ops_malloc(mpi_halo->nproc_to_max * sizeof(void *));


  for (int i = 0; i < nsend_max; i++) {
    //Set box to identify region
    _get_halo_recv_region(xrecv_min, xrecv_max, recv_box_regions, halo->translate, i,
                          halo->dir_to, halo->dir_from, dim, halo->particle_to->type_box);

    int intersect;
    switch(halo->particle_to->type_box) {
    case sizeof(float):
      intersect = ops_check_box_intersections2(dim, (float *) xmin, (float *) xmax,
                                              (float *) xrecv_min, (float *) xrecv_max);
      break;
    case sizeof(double):
      intersect = ops_check_box_intersections2(dim, (double *) xmin, (double *) xmax,
                                              (double *) xrecv_min, (double *) xrecv_max);
      break;
    case sizeof(long double):
      intersect = ops_check_box_intersections2(dim, (long double *) xmin, (long double *) xmax,
                                              (long double *) xrecv_min, (long double *) xrecv_max);
      break;
    }

    if (intersect == 1) {
      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {
          void *pointer_box = _declaire_exchange_sending_box(xmin, xmax, dim, xrecv_min, xrecv_max,
                                                         is, iswap, halo->particle_to->type_box);

           if (pointer_box != nullptr) {


             proclist[nsend] = mpi_halo->proclist_complete[i];
             boxes[nsend] = pointer_box;
             nsend++;
             goto endline;
           }
        }
      }
    }

    endline:
    int a1 = 0;
  }

  mpi_halo->nproc_to = nsend;
  //Allocation of arrays

  if (mpi_halo->nproc_to > 0) {

    mpi_halo->proclist = (int *) ops_malloc(mpi_halo->nproc_to * sizeof(int));
    mpi_halo->sendBox = (void **) ops_malloc(mpi_halo->nproc_to * sizeof(void *));
    mpi_halo->send_region = (int *) ops_malloc(mpi_halo->nproc_to * 6  * sizeof(int));

     for (int i = 0; i < mpi_halo->nproc_to; i++) {
      mpi_halo->proclist[i] = proclist[i];
      mpi_halo->sendBox[i] = boxes[i];
    }
  }

/*  for (int i = 0l i < mpi_halo->nproc_to; i++) {

  }*/

  ops_free(boxes);
  ops_free(proclist);
  ops_free(xmin);
  ops_free(xmax);
  ops_free(xrecv_max);
  ops_free(xrecv_min);


  if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
    printf("Halo %d: Number of processes %d to send from rank %d\n",
           mpi_halo->index, mpi_halo->nproc_to, ops_get_proc());
    if (mpi_halo->nproc_to > 0) {
      printf("Rank %d: Processes to send ", ops_get_proc());
      if (halo->particle_from->type_box == sizeof(float)) {
        for (int i = 0; i < mpi_halo->nproc_to; i++) {
          printf("%d Region [%f %f]x[%f %f]", mpi_halo->proclist[i],
               ((BoundingBox<float> *)mpi_halo->sendBox[i])->getMinCoordDir(0),
               ((BoundingBox<float> *)mpi_halo->sendBox[i])->getMaxCoordDir(0),
               ((BoundingBox<float> *)mpi_halo->sendBox[i])->getMinCoordDir(1),
               ((BoundingBox<float> *)mpi_halo->sendBox[i])->getMaxCoordDir(1));
          if (dim == 3)
            printf("x[%f %f]\n",
                   ((BoundingBox<float> *) mpi_halo->sendBox[i])->getMinCoordDir(2),
                   ((BoundingBox<float> *) mpi_halo->sendBox[i])->getMaxCoordDir(2));
        }
      }
      else if (halo->particle_from->type_box == sizeof(double)) {
        for (int i = 0; i < mpi_halo->nproc_to; i++) {
          printf("%d Region [%f %f]x[%f %f]", mpi_halo->proclist[i],
               ((BoundingBox<double> *)mpi_halo->sendBox[i])->getMinCoordDir(0),
               ((BoundingBox<double> *)mpi_halo->sendBox[i])->getMaxCoordDir(0),
               ((BoundingBox<double> *)mpi_halo->sendBox[i])->getMinCoordDir(1),
               ((BoundingBox<double> *)mpi_halo->sendBox[i])->getMaxCoordDir(1));
          if (dim == 3)
            printf("x[%f %f]\n",
                   ((BoundingBox<double> *) mpi_halo->sendBox[i])->getMinCoordDir(2),
                   ((BoundingBox<double> *) mpi_halo->sendBox[i])->getMaxCoordDir(2));
        }
      }
      else if (halo->particle_from->type_box == sizeof(long double)) {
        for (int i = 0; i < mpi_halo->nproc_to; i++) {
          printf("%d Region [%Lf %Lf]x[%Lf %Lf]", mpi_halo->proclist[i],
               ((BoundingBox<long double> *)mpi_halo->sendBox[i])->getMinCoordDir(0),
               ((BoundingBox<long double> *)mpi_halo->sendBox[i])->getMaxCoordDir(0),
               ((BoundingBox<long double> *)mpi_halo->sendBox[i])->getMinCoordDir(1),
               ((BoundingBox<long double> *)mpi_halo->sendBox[i])->getMaxCoordDir(1));
          if (dim == 3)
            printf("x[%Lf %Lf]\n",
                   ((BoundingBox<long double> *) mpi_halo->sendBox[i])->getMinCoordDir(2),
                   ((BoundingBox<long double> *) mpi_halo->sendBox[i])->getMaxCoordDir(2));
        }
      }

    }
  }

}

void ops_particle_halo_set_border_send(ops_mpi_particle_halo *mpi_halo,
                                       char *recv_box_regions, int dim) {

  ops_particle_halo halo = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block->index];
  if (!sb->owned) {mpi_halo->nproc_to = 0; return; }

  int nsend_max = mpi_halo->nproc_to_max;
  int nsend = 0;

  int nmax = mpi_halo->nproc_to_max;
  int *proclist = (int *) ops_malloc(nmax * sizeof(int));
  void **boxes =
      (void **)ops_malloc(mpi_halo->nproc_to_max *sizeof(void *));
  int *sending_reg = (int *) ops_malloc(mpi_halo->nproc_to_max * 6 * sizeof(int));

  if (!_ops_box_get_ownership(halo->particle_from->box_block, halo->particle_from->type_box))
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications "
                                          " requires the partition of BoundingBoxes");

  int max_size = MAX(halo->particle_from->type_box, halo->particle_to->type_box);
  char *xmin = (char *)ops_malloc(dim * max_size);
  char *xmax = (char *)ops_malloc(dim * max_size);

  char *xrecv_min = (char *) ops_malloc(dim * max_size);
  char *xrecv_max = (char *) ops_malloc(dim * max_size);

  _get_local_box(xmin, xmax, halo->particle_from->box_block, dim,
                 halo->particle_from->type_box);


  for (int i = 0; i < nsend_max; i++) {
    _get_halo_recv_region(xrecv_min, xrecv_max, recv_box_regions, halo->translate, i,
                          halo->dir_to, halo->dir_from, dim, halo->particle_to->type_box);

    int intersect;
    switch(halo->particle_to->type_box) {
    case sizeof(float):
      intersect = ops_check_box_intersections2(dim, (float *) xmin, (float *) xmax,
                                              (float *) xrecv_min, (float *) xrecv_max);
      break;
    case sizeof(double):
      intersect = ops_check_box_intersections2(dim, (double *) xmin, (double *) xmax,
                                              (double *) xrecv_min, (double *) xrecv_max);
      break;
    case sizeof(long double):
      intersect = ops_check_box_intersections2(dim, (long double *) xmin, (long double *) xmax,
                                              (long double *) xrecv_min, (long double *) xrecv_max);
      break;
    }


    if (intersect) {
      //TODO: Need a function
      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {
         void *pointer_box = _declaire_border_sending_box(xrecv_min, xrecv_max, xmin, xmax, halo->dx,
                                                          dim, is, iswap, halo->particle_to->type_box); //TODO: Shift into sth else

         if (pointer_box != nullptr) {
           proclist[nsend] = mpi_halo->proclist_complete[i];
           boxes[nsend] = pointer_box;
           switch (halo->particle_to->type_box  ) {
           case sizeof(float):
             _compute_mapping_region(((BoundingBox<float> **) boxes)[nsend], sending_reg + 6 * nsend, dim,
                                     halo->particle_from, halo->particle_from->map_list[0]);
           break;
           case sizeof(double):
             _compute_mapping_region(((BoundingBox<double> **) boxes)[nsend], sending_reg + 6 * nsend, dim,
                                      halo->particle_from, halo->particle_from->map_list[0]);
           break;
           case sizeof(long double):
             _compute_mapping_region(((BoundingBox<long double> **)boxes)[nsend], sending_reg + 6 * nsend, dim,
                                     halo->particle_from, halo->particle_from->map_list[0]);
           }
           nsend++;

           goto endline;
         }

        }
      }
    }

    endline:
    int a1 = 0;

  }

  mpi_halo->nproc_to = nsend;

  if (mpi_halo->nproc_to > 0) {
    mpi_halo->proclist = (int *) ops_malloc(mpi_halo->nproc_to * sizeof(int));
    mpi_halo->sendBox = (void **) ops_malloc(mpi_halo->nproc_to * sizeof(void *));
    mpi_halo->send_region = (int *)ops_malloc(mpi_halo->nproc_to * 6 * sizeof(int));

    for (int  i = 0; i < mpi_halo->nproc_to; i++) {
      mpi_halo->proclist[i] = proclist[i];
      mpi_halo->sendBox[i] = boxes[i];
      for (int j = 0; j < 6; j++)
        mpi_halo->send_region[6 * i + j] = sending_reg[6 * i + j];
    }

  }



  ops_free(proclist);
  ops_free(boxes);
  ops_free(sending_reg);
  ops_free(xmin);
  ops_free(xmax);
  ops_free(xrecv_min);
  ops_free(xrecv_max);
}

void  ops_particle_halo_set_exchange_recv(ops_mpi_particle_halo *mpi_halo,
                                          char *send_box_regions, int dim) {

  ops_particle_halo halo
    = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];
  sub_block *sb = OPS_sub_block_list[halo->particle_to->block->index];
  if (!sb->owned) {mpi_halo->nproc_from = 0; return;}

  int nrecv_max = mpi_halo->nproc_from_max;
  int nrecv = 0;

  if (!_ops_box_get_ownership(halo->particle_to->box_block, halo->particle_to->type_box))
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications "
                                            " requires the partition of BoundingBoxes");

  if (!_ops_box_get_ownership(halo->particle_to->box_block, halo->particle_to->type_box))
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications "
                                            " requires the partition of BoundingBoxes");

  int max_size = MAX(halo->particle_to->type_box, halo->particle_from->type_box);
  char *xmin = (char *) ops_malloc(dim * max_size);
  char *xmax = (char *) ops_malloc(dim * max_size);
  char *xsend_max = (char *) ops_malloc(dim * max_size);
  char *xsend_min = (char *) ops_malloc(dim * max_size);
  int *proc_recv = (int *)ops_malloc(nrecv_max * sizeof(int));

  _get_local_box(xmin, xmax, halo->particle_to->box_block, dim,
                 halo->particle_to->type_box);

  for (int i = 0; i < nrecv_max; i++) {

    _get_halo_send_region(xsend_min, xsend_max, send_box_regions,
                          halo->translate, i, halo->dir_from,
                          halo->dir_to, dim, halo->particle_from->type_box);

    int intersect = 0;
    switch (halo->particle_to->type_box) {
    case sizeof(float):
      intersect = ops_check_box_intersections2(dim, (float *) xmin, (float *) xmax,
                                               (float *)xsend_min, (float *) xsend_max);
      break;
    case sizeof(double):
      intersect = ops_check_box_intersections2(dim, (double *) xmin, (double *) xmax,
                                              (double *)xsend_min, (double *) xsend_max);
      break;
    case sizeof(long double):
      intersect = ops_check_box_intersections2(dim, (long double *) xmin, (long double *) xmax,
                                              (long double *)xsend_min, (long double *) xsend_max);
      break;
    }

    if (intersect) {
      proc_recv[nrecv] = mpi_halo->proclist_complete[i + mpi_halo->nproc_to_max];
      nrecv++;
    }
  }

  mpi_halo->nproc_from = nrecv;
  int ntot = mpi_halo->nproc_from + mpi_halo->nproc_to;
  if (mpi_halo->nproc_from + mpi_halo->nproc_to > mpi_halo->nproc_to)
    mpi_halo->proclist = (int *) ops_realloc((char *)mpi_halo->proclist,
                                             sizeof(int) * ntot);

  for (int i = 0; i < mpi_halo->nproc_from; i++) {
    mpi_halo->proclist[i + mpi_halo->nproc_to] = proc_recv[i];
  }

  ops_free(proc_recv);
  ops_free(xsend_min);
  ops_free(xsend_max);
  ops_free(xmin);
  ops_free(xmax);
}




void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {

  char *recv_box_regions = nullptr;
  char *send_box_regions = nullptr;
  int size_recv_max = 0;
  int size_send_max = 0;

  //Find processes for data exchange
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (!OPS_sub_block_list[halo->particle_to->block->index]->owned &&
        !OPS_sub_block_list[halo->particle_from->block->index]->owned) continue;

    ops_mpi_particle_halo  *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];


    int dim = halo->particle_to->block->dims;

    int type_size = MAX(halo->particle_to->type_box, halo->particle_from->type_box);
    int size_recv = 2 * type_size * dim * mpi_halo->nproc_to_max;

    if (size_recv_max == 0 && size_recv > 0) {
      recv_box_regions = (char *) ops_malloc( size_recv);
      size_recv_max = size_recv;
    }
    else if (size_recv> size_recv_max) {
      recv_box_regions = (char *)ops_realloc(recv_box_regions, size_recv);
      size_recv_max = size_recv;
    }

    int size_send = 2 * type_size * dim
                  * mpi_halo->nproc_from_max;


    if (size_send_max == 0 && size_send > 0) {
      send_box_regions = (char *) ops_malloc( size_send);
      size_send_max = size_send;
    }
    else if (size_send > size_send_max) {
      send_box_regions = (char *)ops_realloc(send_box_regions, size_send);
      size_send_max = size_send;
    }


    //TODO: Add elements for checking
    ops_get_send_recv_box(recv_box_regions, send_box_regions, mpi_halo, dim);


    _ops_particle_halo_set_exchange_send(mpi_halo,
                                       recv_box_regions, dim);


    ops_particle_halo_set_exchange_recv(mpi_halo,
                                        send_box_regions, dim);

    ops_free(mpi_halo->proclist_complete);
  }

  //Create the halo group
  ops_mpi_particle_halo_group *mpi_group = &OPS_mpi_particle_halo_group_list[halo_grp->index];
  int owned = 0;
  for (int j = 0; j < halo_grp->nhalos; j++) {
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_from > 0  ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_to > 0)
      owned++;
  }

  mpi_group->group = halo_grp;
  mpi_group->nhalos = owned;
  mpi_group->index = halo_grp->index;

  if (owned > 0)
    mpi_group->mpi_halos
      = (ops_mpi_particle_halo **) ops_malloc(owned * sizeof(ops_mpi_particle_halo *));

  for (int i = 0; i < mpi_group->nhalos; i++)
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_from > 0 ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_to > 0) {
      mpi_group->mpi_halos[i] = &OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index];
    }


  mpi_group->num_neighbors_send = 0;
  mpi_group->num_neighbors_recv = 0;
  mpi_group->nhalo_info = 0;


  int *neighbor_send = (int *)ops_malloc(ops_comm_global_size * sizeof(int));
  int *neighbor_recv = (int *)ops_malloc(ops_comm_global_size * sizeof(int));

  for (int i = 0; i < ops_comm_global_size; i++) {
    neighbor_send[i] = 0;
    neighbor_recv[i] = 0;
  }

  for (int i = 0; i < mpi_group->nhalos; i++) {
    for (int k = 0; k < mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_send[iproc]++;
    }

    for (int k = mpi_group->mpi_halos[i]->nproc_to;
             k < mpi_group->mpi_halos[i]->nproc_from
               + mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_recv[iproc]++;
    }
  }

  for (int i = 0; i < ops_comm_global_size; i++) {
    mpi_group->num_neighbors_send += (neighbor_send[i] > 0) ? 1 : 0;
    mpi_group->num_neighbors_recv += (neighbor_recv[i] > 0) ? 1 : 0;
  }

  int ntot = (mpi_group->num_neighbors_send + mpi_group->num_neighbors_recv)
           * mpi_group->nhalos;

  if (ntot > 0)
    mpi_group->halo_info = (ops_particle_halo_exchange *)ops_malloc(sizeof(ops_particle_halo_exchange) * ntot);

  mpi_group->nhalo_info = ntot;


  for (int igroup = 0; igroup < ntot; igroup++)
    mpi_group->halo_info[igroup] =
        (ops_particle_halo_exchange) ops_malloc(sizeof(OPS_particle_halo_exchange_info_core));

  if (mpi_group->num_neighbors_send > 0) {
    mpi_group->neighbors_send =
       (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_shift
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_sizes
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_bites
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  }
  if (mpi_group->num_neighbors_recv > 0) {
    mpi_group->neighbors_recv =
       (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_bites
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
    mpi_group->recv_shift
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
    mpi_group->recv_sizes
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  }

  if (mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send > 0) {
    mpi_group->requests
       = (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                    sizeof(MPI_Request));
    mpi_group->statuses
       = (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                   sizeof(MPI_Status));
  }

  int k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j) {
    if (neighbor_send[j] > 0) {
      mpi_group->neighbors_send[k] = j;
      k++;
    }
  }

  k = 0;
  for (int j = 0; j < ops_comm_global_size; j++)
    if (neighbor_recv[j] > 0) {
      mpi_group->neighbors_recv[k] = j;
      k++;
    }

  /* Allocationn of arrays for sending & receiving data */
  for (int ihalo = 0; ihalo < mpi_group->nhalo_info; ihalo++) {
    mpi_group->halo_info[ihalo]->nmax = 10;
    mpi_group->halo_info[ihalo]->sendlist = (int *)ops_malloc(sizeof(int) * 10);
  }

  ops_free(recv_box_regions);
  ops_free(send_box_regions);
  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}

void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  char *recv_box_regions = NULL;
  char *send_box_regions = NULL;
  int size_recv_max = 0;
  int size_send_max = 0;

  /*Part I: Find processes to which this process will send */
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo  = halo_grp->halo_list[ihalo];

    if (!OPS_sub_block_list[halo->particle_to->block->index]->owned &&
        !OPS_sub_block_list[halo->particle_to->block->index]->owned) continue;


    int dim = halo->particle_to->block->dims;
    ops_mpi_particle_halo  *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];
    int size_block = MAX(halo->particle_to->type_box, halo->particle_from->type_box);
    int size_recv = 2 * size_block * dim * mpi_halo->nproc_to_max;

    if (size_recv_max ==  0 && size_recv > 0) {
      recv_box_regions = (char *) ops_malloc(size_recv);
      size_recv_max = size_recv;
    }
    else if (size_recv > size_recv_max) {
      recv_box_regions = (char *) ops_realloc(recv_box_regions, size_recv);
      size_recv_max = size_recv;
    }

    int size_send = 2 * size_block * dim * mpi_halo->nproc_from_max;
    if (size_send_max == 0 && size_send  > 0) {
      send_box_regions = (char *) ops_malloc(size_send);
      size_send_max = size_send;
    }
    else if (size_send > size_send_max) {
      send_box_regions = (char *) ops_realloc(send_box_regions, size_send);
      size_send_max = size_send;
    }

    //Find send and receive regions
    ops_get_send_recv_box(recv_box_regions, send_box_regions, mpi_halo, dim);

    ops_particle_halo_set_border_send(mpi_halo, recv_box_regions, dim); //TODO

    ops_particle_halo_set_exchange_recv(mpi_halo, send_box_regions, dim);

    ops_free(mpi_halo->proclist_complete);

  }

  /* Part II: Create mpi halo groups for particle structures */
  ops_mpi_particle_halo_group *mpi_group = &OPS_mpi_particle_halo_group_list[halo_grp->index];
  int owned = 0;
  for (int j = 0; j < halo_grp->nhalos; j++) {
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_from > 0  ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_to > 0)
      owned++;
  }

  mpi_group->group = halo_grp;
  mpi_group->nhalos = owned;
  mpi_group->index = halo_grp->index;

  mpi_group->mpi_halos
    = (ops_mpi_particle_halo **) ops_malloc(owned * sizeof(ops_mpi_particle_halo *));

  for (int i = 0; i < mpi_group->nhalos; i++)
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_from > 0 ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_to > 0) {
      mpi_group->mpi_halos[i] = &OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index];
    }


  mpi_group->num_neighbors_send = 0;
  mpi_group->num_neighbors_recv = 0;
  mpi_group->nhalo_info = 0;

  /*Part III: Start populating particle halo group structures */
  int *neighbor_send = (int *)ops_malloc(ops_comm_global_size * sizeof(int));
  int *neighbor_recv = (int *)ops_malloc(ops_comm_global_size * sizeof(int));

  for (int i = 0; i < ops_comm_global_size; i++) {
    neighbor_send[i] = 0;
    neighbor_recv[i] = 0;
  }

  for (int i = 0; i < mpi_group->nhalos; i++) {
    for (int k = 0; k < mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_send[iproc]++;
    }

    for (int k = mpi_group->mpi_halos[i]->nproc_to;
             k < mpi_group->mpi_halos[i]->nproc_from
               + mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_recv[iproc]++;
    }
  }

  for (int i = 0; i < ops_comm_global_size; i++) {
    mpi_group->num_neighbors_send += (neighbor_send[i] > 0) ? 1 : 0;
    mpi_group->num_neighbors_recv += (neighbor_recv[i] > 0) ? 1 : 0;
  }

  int ntot = (mpi_group->num_neighbors_send + mpi_group->num_neighbors_recv)
           * mpi_group->nhalos;

  if (ntot > 0)
    mpi_group->halo_info
       = (ops_particle_halo_exchange *)ops_malloc(sizeof(ops_particle_halo_exchange) * ntot);

  mpi_group->nhalo_info = ntot;


  for (int igroup = 0; igroup < ntot; igroup++)
    mpi_group->halo_info[igroup]
       = (ops_particle_halo_exchange) ops_malloc(sizeof(OPS_particle_halo_exchange_info_core));


    //TODO: We do not need the process, we will need to simply copy them
  if (mpi_group->num_neighbors_send > 0) {
    mpi_group->neighbors_send =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_shift =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_sizes =
      (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_pos_bites =
        (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->shift_send_pos =
        (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  }

  if (mpi_group->num_neighbors_recv > 0) {
    mpi_group->neighbors_recv =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_shift =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_sizes =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_pos_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->shift_recv_pos =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
  }


  if (mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send) {
    mpi_group->requests =
        (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                   sizeof(MPI_Request));
  mpi_group->statuses =
      (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                sizeof(MPI_Status));
  }

  int k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j) {
    if (neighbor_send[j] > 0) {
      mpi_group->neighbors_send[k] = j;
      k++;
    }
  }

  k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j)
    if (neighbor_recv[j] > 0) {
      mpi_group->neighbors_recv[k] = j;
      k++;
    }

  /* Allocation of arrays for sending & receiving data */
  for (int ihalo = 0; ihalo < mpi_group->nhalo_info; ihalo++) {
    mpi_group->halo_info[ihalo]->nmax = 10;
    mpi_group->halo_info[ihalo]->sendlist = (int *)ops_malloc(sizeof(int) * 10); //TODO: Need to add only recv
  }

  ops_free(recv_box_regions);
  ops_free(send_box_regions);
  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}

void _ops_particle_setup_for_rev_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp) {

  ops_particle_halo_group halo_main = halo_grp->halo_master;

  if (halo_main == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR: Main halo for forward/backward halo types is"
                       "not defined");
  if (halo_main->halo_type == OPS_HALO_GRP_BORDER)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR: The type of the main halo must be of border type");

  ops_mpi_particle_halo_group *mpi_main_grp =
      &OPS_mpi_particle_halo_group_list[halo_main->index];

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];


  if (halo_main->nhalos != halo_grp->nhalos)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR:  Inconsistent halo configuration: Reverse and main particle "
                       "halos have different counts");


  //Allocate halo structures and initialize
  for (int i = 0; i < mpi_group->nhalos; i++) {
    ops_mpi_particle_halo *mpi_halo = mpi_group->mpi_halos[i];
    ops_mpi_particle_halo *mpi_halo_main = mpi_main_grp->mpi_halos[i];

    mpi_halo->nproc_from = mpi_halo_main->nproc_from;
    mpi_halo->nproc_to = mpi_halo_main->nproc_to;

    //Allocate structures
    mpi_halo->particle_halo = instance->OPS_particle_halo_list[mpi_halo->index];

    if (mpi_halo->nproc_from + mpi_halo->nproc_to > 0) {
      mpi_halo->proclist = (int *) ops_malloc((mpi_halo->nproc_to + mpi_halo->nproc_from)
                                               * sizeof(int));
    }

    for (int i = 0; i < mpi_halo->nproc_to + mpi_halo->nproc_from; i++) {
      mpi_halo->proclist[i] = mpi_halo_main->proclist[i];
    }

    ops_free(mpi_halo->proclist_complete);
  }

  //Allocate mpi_group structures
  mpi_group->num_neighbors_send = mpi_main_grp->num_neighbors_send;
  mpi_group->num_neighbors_recv = mpi_main_grp->num_neighbors_recv;

  //Allocate structures

  mpi_group->group = halo_grp;
  mpi_group->nhalos = mpi_main_grp->nhalos;
  mpi_group->nhalo_info = mpi_main_grp->nhalo_info;

  mpi_group->mpi_halos
   = (ops_mpi_particle_halo **) ops_malloc(mpi_group->nhalos
                                           * sizeof(ops_mpi_particle_halo *));

  mpi_group->neighbors_send
   = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->neighbors_recv
   = (int *) ops_malloc(mpi_group->num_neighbors_recv
                        * sizeof(int));

  mpi_group->halo_info =
      (ops_particle_halo_exchange *) ops_malloc(sizeof(ops_particle_halo_exchange)
                                                * mpi_group->nhalo_info);

  for (int i = 0; i < mpi_group->nhalo_info; i++)
    mpi_group->halo_info[i] = mpi_main_grp->halo_info[i]; //TODO: Check

  mpi_group->requests = (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                   sizeof(MPI_Request));
  mpi_group->statuses = (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                  sizeof(MPI_Status));

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    mpi_group->neighbors_send[i] = mpi_main_grp->neighbors_send[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    mpi_group->neighbors_recv[i] = mpi_main_grp->neighbors_recv[i];

  mpi_group->send_shift = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->send_sizes = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->send_bites = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->recv_bites = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  mpi_group->recv_shift = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  mpi_group->recv_sizes = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));

}

void ops_particle_setup_intrablock_comms(ops_particle particle) {

  sub_block_list  sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  //TODO:
  if  (!_ops_box_get_ownership(particle->box_block, particle->type_box))
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Box block is not partitioned\n");


  if (particle->particle_map_index == 0)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: At least one ops_particle_mapping structure "
                       "must be defined\n");

  sub_particle sp = sb->sb_particle_list[particle->index];

  int dim = particle->block->dims;

  //Get bounding box
  char *box = particle->box_block;
  char *box_block = particle->block->instance->OPS_block_list[particle->block->index].box;
  int block_type = particle->type_box;

  for (int idim = 0; idim < particle->block->dims; idim++) {

    //aLlocate exchange regions
    sp->particle_halos[idim]->region_exch_neg = (char *) ops_malloc(2 * block_type);
    sp->particle_halos[idim]->region_exch_pos = (char *) ops_malloc(2 * block_type);
    sp->particle_halos[idim]->region_bord_pos = (char *) ops_malloc(2 * dim * block_type);
    sp->particle_halos[idim]->region_bord_neg = (char *) ops_malloc(2 * dim * block_type);

    //Setting up exchange regions
    switch(particle->type_box) {
    case sizeof(float):
      _ops_particle_set_intra_exch_regs((float *) sp->particle_halos[idim]->region_exch_neg,
                                        (float *) sp->particle_halos[idim]->region_exch_pos,
                                        (BoundingBox<float> *) box, (BoundingBox<float> *) box_block,
                                         idim, dim);
      break;
    case sizeof(double):
      _ops_particle_set_intra_exch_regs((double *) sp->particle_halos[idim]->region_exch_neg,
                                        (double *) sp->particle_halos[idim]->region_exch_pos,
                                        (BoundingBox<double> *) box, (BoundingBox<double> *) box_block,
                                        idim, dim);
      break;
    case sizeof(long double):
      _ops_particle_set_intra_exch_regs((long double *) sp->particle_halos[idim]->region_exch_neg,
                                        (long double *) sp->particle_halos[idim]->region_exch_pos,
                                        (BoundingBox<long double> *) box,
                                        (BoundingBox<long double> *) box_block, idim, dim);
      break;
    }

    ops_dat binhead = particle->map_list[0]->binhead; //TODO: Change that to a proper list
    int d_m = binhead->d_m[idim] + OPS_sub_dat_list[binhead->index]->d_im[idim];
    int d_p = binhead->d_p[idim] + OPS_sub_dat_list[binhead->index]->d_ip[idim];

    switch(particle->type_box) {
    case sizeof(float):
      _ops_particle_set_intra_bord_regs((float *) sp->particle_halos[idim]->region_bord_neg,
                                        (float *) sp->particle_halos[idim]->region_bord_pos,
                                        (BoundingBox<float> *) box, (BoundingBox<float> *) box_block,
                                        d_m, d_p, binhead->size,
                                        sb->id_m[idim], sb->id_p[idim],idim, dim);
      break;
    case sizeof(double):
      _ops_particle_set_intra_bord_regs((double *) sp->particle_halos[idim]->region_bord_neg,
                                        (double *) sp->particle_halos[idim]->region_bord_pos,
                                        (BoundingBox<double> *) box, (BoundingBox<double> *) box_block,
                                        d_m, d_p, binhead->size,
                                        sb->id_m[idim], sb->id_p[idim], idim, dim);
      break;
    case sizeof(long double):
      _ops_particle_set_intra_bord_regs((long double *) sp->particle_halos[idim]->region_bord_neg,
                                        (long double *) sp->particle_halos[idim]->region_bord_pos,
                                        (BoundingBox<long double> *) box,
                                        (BoundingBox<long double> *) box_block,
                                        d_m, d_p, binhead->size,
                                        sb->id_m[idim], sb->id_p[idim], idim, dim);
      break;
    }

    if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
      switch (particle->type_box) {
      case sizeof(float): {
        printf("Rank %d Region_pos[%d] =[%f %f]x[%f %f]x[%f %f] "
               "Region_neg[%d] = [%f %f]x[%f %f] x[%f %f]\n",
              ops_get_proc(), idim, ((float *) sp->particle_halos[idim]->region_bord_pos)[0],
              ((float *) sp->particle_halos[idim]->region_bord_pos)[1],
              ((float *) sp->particle_halos[idim]->region_bord_pos)[2],
              ((float *) sp->particle_halos[idim]->region_bord_pos)[3],
              (dim == 3) ?  ((float *) sp->particle_halos[idim]->region_bord_pos)[4] : 0.0,
              (dim == 3) ?  ((float *) sp->particle_halos[idim]->region_bord_pos)[5] : 0.0,
              idim,
              ((float *) sp->particle_halos[idim]->region_bord_neg)[0],
              ((float *) sp->particle_halos[idim]->region_bord_neg)[1],
              ((float *) sp->particle_halos[idim]->region_bord_neg)[2],
              ((float *) sp->particle_halos[idim]->region_bord_neg)[3],
              (dim == 3) ? ((float *) sp->particle_halos[idim]->region_bord_neg)[4] : 0.0,
              (dim == 3) ? ((float *) sp->particle_halos[idim]->region_bord_neg)[5] : 0.0);
      } break;
      case sizeof(double): {
        printf("Rank %d Region_pos[%d] =[%f %f]x[%f %f]x[%f %f] "
               "Region_neg[%d] = [%f %f]x[%f %f] x[%f %f]\n",
              ops_get_proc(), idim, ((double *) sp->particle_halos[idim]->region_bord_pos)[0],
              ((double *) sp->particle_halos[idim]->region_bord_pos)[1],
              ((double *) sp->particle_halos[idim]->region_bord_pos)[2],
              ((double *) sp->particle_halos[idim]->region_bord_pos)[3],
              (dim == 3) ?  ((double *) sp->particle_halos[idim]->region_bord_pos)[4] : 0.0,
              (dim == 3) ?  ((double *) sp->particle_halos[idim]->region_bord_pos)[5] : 0.0,
               idim,
              ((double *) sp->particle_halos[idim]->region_bord_neg)[0],
              ((double *) sp->particle_halos[idim]->region_bord_neg)[1],
              ((double *) sp->particle_halos[idim]->region_bord_neg)[2],
              ((double *) sp->particle_halos[idim]->region_bord_neg)[3],
              (dim == 3) ? ((double *) sp->particle_halos[idim]->region_bord_neg)[4] : 0.0,
              (dim == 3) ? ((double *) sp->particle_halos[idim]->region_bord_neg)[5] : 0.0);
      } break;
      case sizeof(long double): {
        printf("Rank %d Region_pos[%d] =[%Lf %Lf]x[%Lf %Lf]x[%Lf %Lf] "
               "Region_neg[%d] = [%Lf %Lf]x[%Lf %Lf] x[%Lf %Lf]\n",
              ops_get_proc(), idim, ((long double *) sp->particle_halos[idim]->region_bord_pos)[0],
              ((long double *) sp->particle_halos[idim]->region_bord_pos)[1],
              ((long double *) sp->particle_halos[idim]->region_bord_pos)[2],
              ((long double *) sp->particle_halos[idim]->region_bord_pos)[3],
              (dim == 3) ?  ((long double *) sp->particle_halos[idim]->region_bord_pos)[4] : 0.0,
              (dim == 3) ?  ((long double *) sp->particle_halos[idim]->region_bord_pos)[5] : 0.0,
              idim,
              ((long double *) sp->particle_halos[idim]->region_bord_neg)[0],
              ((long double *) sp->particle_halos[idim]->region_bord_neg)[1],
              ((long double *) sp->particle_halos[idim]->region_bord_neg)[2],
              ((long double *) sp->particle_halos[idim]->region_bord_neg)[3],
              (dim == 3) ? ((long double *) sp->particle_halos[idim]->region_bord_neg)[4] : 0.0,
              (dim == 3) ? ((long double *) sp->particle_halos[idim]->region_bord_neg)[5] : 0.0);
      } break;
      }

    }
    //Set up region in bin cells
    if (dim < 3) {
      sp->particle_halos[idim]->region_neg[4] = 0;
      sp->particle_halos[idim]->region_neg[5] = 1;

      sp->particle_halos[idim]->region_pos[4] = 0;
      sp->particle_halos[idim]->region_pos[5] = 1;
    }

    for (int isou = 0; isou < dim; isou++) {
      if (isou != idim) {
        sp->particle_halos[idim]->region_neg[2 * isou] = 0;
        sp->particle_halos[idim]->region_neg[2 * isou + 1] = binhead->size[isou];
        sp->particle_halos[idim]->region_pos[2 * isou] = 0;
        sp->particle_halos[idim]->region_pos[2 * isou + 1] = binhead->size[isou];
      }
    }

    //Set-up


    int conforming_neg, conforming_pos;
    switch (particle->type_box) {
    case sizeof(float):
      conforming_neg = check_box_conformation_in_dir(idim, 0, (BoundingBox<float> *) box,
                                                    (BoundingBox<float> *) box_block);
     conforming_pos = check_box_conformation_in_dir(idim, 1, (BoundingBox<float> *) box,
                                                    (BoundingBox<float> *) box_block);
        break;
    case sizeof(double):
      conforming_neg = check_box_conformation_in_dir(idim, 0, (BoundingBox<double> *) box,
                                                     (BoundingBox<double> *) box_block);
      conforming_pos = check_box_conformation_in_dir(idim, 1, (BoundingBox<double> *) box,
                                                     (BoundingBox<double> *) box_block);
      break;
    case sizeof(long double):
      conforming_neg = check_box_conformation_in_dir(idim, 0, (BoundingBox<long double> *) box,
                                                    (BoundingBox<long double> *) box_block);
      conforming_pos = check_box_conformation_in_dir(idim, 1, (BoundingBox<long double> *) box,
                                                     (BoundingBox<long double> *) box_block);
      break;
    }

    //Setting halos in negative direction
    sp->particle_halos[idim]->region_neg[2 * idim] = 0;
    if (conforming_neg)
      sp->particle_halos[idim]->region_neg[2 * idim + 1] = (sb->id_m[idim] != MPI_PROC_NULL) ?
       -2 * OPS_sub_dat_list[binhead->index]->d_im[idim] : 0;
    else sp->particle_halos[idim]->region_neg[2 * idim + 1] = 0;

    //Setting halos in positive direction
    sp->particle_halos[idim]->region_pos[2 * idim + 1] =  binhead->size[idim];
    if (conforming_pos)
      sp->particle_halos[idim]->region_pos[2 * idim] = (sb->id_p[idim] != MPI_PROC_NULL) ?
         binhead->size[idim] -2 * OPS_sub_dat_list[binhead->index]->d_ip[idim] : binhead->size[idim];
    else
      sp->particle_halos[idim]->region_pos[2 * idim] = binhead->size[idim];


    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Rank %d Forward regions in %d: Neg (%d) [%d %d]x[%d %d]x[%d %d] "
             "and pos (%d): [%d %d]x[%d %d]x[%d %d]\n",
             ops_get_proc(), idim, sb->id_m[idim], sp->particle_halos[idim]->region_neg[0],
             sp->particle_halos[idim]->region_neg[1], sp->particle_halos[idim]->region_neg[2],
             sp->particle_halos[idim]->region_neg[3], sp->particle_halos[idim]->region_neg[4],
             sp->particle_halos[idim]->region_neg[5],
             sb->id_p[idim], sp->particle_halos[idim]->region_pos[0],
             sp->particle_halos[idim]->region_pos[1], sp->particle_halos[idim]->region_pos[2],
             sp->particle_halos[idim]->region_pos[3], sp->particle_halos[idim]->region_pos[4],
             sp->particle_halos[idim]->region_pos[5]);
  }
}

ops_dat ops_decl_particle_dat_char(ops_particle particle, int dim, int *dataset_size,
                                   int *base, int *d_m, int *d_p, int *stride,
                                   char *data, int type_size, char const *type,
                                   char const *name, bool assign, bool exchanging) {

  if (ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_decl_particle_dat_char "
                                                         "called after ops_partition");

  ops_dat dat = ops_decl_dat_temp_core(particle->block, dim, dataset_size, base,
                                       d_m, d_p, stride, data, type_size, type, name, 1);

  dat->user_managed = 0;
  dat->is_hdf5 = 0;

  //Setting up the sd one-Need for
  // create list to hold sub-grid decomposition geometries for each mpi process
    OPS_sub_dat_list = (sub_dat_list *)ops_realloc(
        OPS_sub_dat_list, OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(sub_dat_list));

    sub_dat_list sd = (sub_dat_list)ops_calloc(1, sizeof(sub_dat));
    sd->dat = dat;
    sd->dirty_dir_send =
        (int *)ops_malloc(sizeof(int) * 2 * particle->block->dims * MAX_DEPTH);
    for (int i = 0; i < 2 * particle->block->dims * MAX_DEPTH; i++)
      sd->dirty_dir_send[i] = 1;
    sd->dirty_dir_recv =
        (int *)ops_malloc(sizeof(int) * 2 * particle->block->dims * MAX_DEPTH);
    for (int i = 0; i < 2 * particle->block->dims * MAX_DEPTH; i++)
      sd->dirty_dir_recv[i] = 1;
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      sd->d_ip[i] = 0;
      sd->d_im[i] = 0;
    }
    OPS_sub_dat_list[dat->index] = sd;

    //Set now the required variables
    dat->is_particle = true;
    dat->is_exchangable = exchanging;

    if (assign) {
      particle->particle_dat_index++;
      ops_particle_realloc_list(particle);
      particle->particle_dat[particle->particle_dat_index - 1] = dat;
    }

    return dat;
}




void ops_particle_print_data_to_txtfile(ops_particle particle, const char *file_name) {

  if (!OPS_sub_block_list[particle->block->index]->owned) return;

  ops_get_data(particle->particle_pos_dat);
  if (particle->particle_envelope != nullptr)
    ops_get_data(particle->particle_envelope);

  for (int idat = 0; idat < particle->particle_dat_index; idat++) {
    ops_get_data(particle->particle_dat[idat]);
  }

  ops_particle_print_mpi_data_to_txt_file_core(particle, file_name);
}

void ops_particle_print_dats_to_txtfile(ops_particle particle, ops_dat *dats, int ndats,
                                        const char *file_name) {

  if (!OPS_sub_block_list[particle->block->index]->owned) return;

  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ops_dat is not linked to particle"
                                            " data\n");
    ops_get_data(dats[idat]);
  }

  ops_particle_print_mpi_dats_to_txt_file_core(particle, dats, ndats, file_name);

}
