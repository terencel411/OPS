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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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

/** @brief  Box block functions for MPI-backend
  * @author Valantis Tsinginos
  * @details Box block functions for MPI-backend */


#ifndef __OPS_PARTICLE_BOX_MPI_FUNCS_H_
#define __OPS_PARTICLE_BOX_MPI_FUNCS_H_

#include "ops_mpi_core.h"
#include "ops_bounding_box.h"
#include "ops_util.h"

#include <limits>

#define BIG_1 1e10


template<typename T>
void _ops_construct_local_box_from_dat(ops_dat coords, const int dim,
                                       T *xmin, T *xmax) {

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

  if (coords->type_size != sizeof(T))
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Incompatible sizes between template type "
                                         "and ops_dat structure\n");
  T *data = (T *)coords->data;
  OPS_instance *instance = coords->block->instance;

  if (dim == 2) {
    if (instance->OPS_soa) {
      xmin[0] = *(data + imin[0] + imin[1] * size[0]);
      xmin[1] = *(data + imin[0] + imin[1] * size[0] + size[0] * size[1]);

      xmax[0] = *(data + imax[0] + imax[1] * size[0]);
      xmax[1] = *(data + imax[0] + imax[1] * size[0] + size[0] * size[1]);
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
      xmax[2] = *(  data + 2 + imax[0] * coords->dim + imax[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
    }
  }
  else
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: The size of the physical domain"
                                         "must be 2 or 3\n");

}

template<typename T>
void BoundingBox<T>::partitionBoundingBox(ops_block block, ops_dat map_bin, T *dx) {

  if (owned) return;

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: BoundingBox decomposition "
                                          "failed. Box cannot be decomposed before "
                                          "the domain partition");

  if (this->volume < std::numeric_limits<T>::epsilon()) {
    sub_block *sb = OPS_sub_block_list[block->index];

    T xmin[OPS_MAX_DIM] = {};
    T xmax[OPS_MAX_DIM] = {};
    T xmin_glob[OPS_MAX_DIM];
    T xmax_glob[OPS_MAX_DIM];
    T xmax_loc[OPS_MAX_DIM];

    _ops_construct_local_box_from_dat(coords, block->dims, xmin, xmax); //TODO

    MPI_Status status;
    for (int isou = 0; isou < dim; isou++) {
      xmax_loc[isou] = xmax[isou];
      MPI_Sendrecv((char *)&xmin[isou], sizeof(T), MPI_CHAR, sb->id_m[isou], 10 + isou,
                    (char *)&xmax_loc[isou], sizeof(T), MPI_CHAR, sb->id_p[isou],
                         10+isou, sb->comm, &status);
    }

    this->setBoundingBoxLocalBound(xmin, xmax_loc);

    //Expand base on size
    if (sizeT == sizeof(float)) {
      MPI_Allreduce(xmin, xmin_glob, block->dims, MPI_FLOAT, MPI_MIN,
                    sb->comm);
      MPI_Allreduce(xmax, xmax_glob, block->dims, MPI_FLOAT, MPI_MAX,
                    sb->comm);
    }
    else if (sizeT == sizeof(double)) {
      MPI_Allreduce(xmin, xmin_glob, block->dims, MPI_DOUBLE, MPI_MIN,
                    sb->comm);
      MPI_Allreduce(xmax, xmax_glob, block->dims, MPI_DOUBLE, MPI_MAX,
                    sb->comm);
    }
    else if (sizeT == sizeof(long double)) {
      MPI_Allreduce(xmin, xmin_glob, block->dims, MPI_LONG_DOUBLE, MPI_MIN,
                    sb->comm);
      MPI_Allreduce(xmax, xmax_glob, block->dims, MPI_LONG_DOUBLE, MPI_MAX,
                    sb->comm);
    }
    else
      throw OPSException(OPS_RUNTIME_ERROR, "Error: This type of T is not supported for the MPI version");

    this->setBoundingBoxGlobalBound(xmin_glob, xmax_glob);
    this->owned = true;

  }
  else {
    if (map_bin == nullptr)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: BoundingBox cannot be defined unless"
                                                  "one of the following is not specified:\n "
                                                  "i. Grid structure\n"
                                                  "ii. Mapping structure\n");

    int dim = block->dims;
    T xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
    sub_dat *sdat = OPS_sub_dat_list[map_bin->index];

    //Uniform grid
    for (int i = 0; i < dim; i++) {
      int ifirst = sdat->decomp_disp[i] - map_bin->base[i] -map_bin->d_m[i]; //TODO Check

      int isize = sdat->decomp_size[i]  + map_bin->base[i] + map_bin->d_m[i] - map_bin->d_p[i];

      int ilast = ifirst + isize;

      xmin[i] = this->getGlobalMin(i) + static_cast<double>(ifirst) * dx[i];
      xmax[i] = this->getGlobalMin(i) + static_cast<double>(ilast) * dx[i];
    }


    this->setBoundingBoxLocalBound(xmin, xmax);

    this->owned = true;
  }

}

/*=================================================================================================
 * Functions for accessing BoundingBoxes
 *=================================================================================================*/

template<typename T>
bool ops_get_bounding_box_local_to_global(ops_block block, T *x_min, T *xmax,
                                          T *xmin, T* xglb_min, T* xglb_max) {
  sub_block *sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return false;

  if (sizeof(T) == sizeof(float)) {
    MPI_Allreduce(xmin, xglb_min, 3, MPI_FLOAT, MPI_MIN, sb->comm);
    MPI_Allreduce(xmax, xglb_max, 3, MPI_FLOAT, MPI_MAX, sb->comm);
  }
  else if (sizeof(T) == sizeof(double)) {
    MPI_Allreduce(xmin, xglb_min, 3, MPI_DOUBLE, MPI_MIN, sb->comm);
    MPI_Allreduce(xmax, xglb_max, 3, MPI_DOUBLE, MPI_MAX, sb->comm);
  }
  else if (sizeof(T) == sizeof(long double)) {
    MPI_Allreduce(xmin, xglb_min, 3, MPI_LONG_DOUBLE, MPI_MIN, sb->comm);
    MPI_Allreduce(xmax, xglb_max, 3, MPI_LONG_DOUBLE, MPI_MAX, sb->comm);
  }

  return true;
}


template<typename T>
inline void _compute_mapping_region(BoundingBox<T> *box, int *sending_reg, int dim,
                                    ops_particle particle, ops_particle_mapping map)
{
  int size[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    size[i] = (map != nullptr) ? map->binhead->size[i] : 0;
  }

  ops_point<T> xmin, xmax, xmin_send, xmax_send;
  T dx[OPS_MAX_DIM];

  for (int i = 0; i < box->getDim(); i++) dx[i] = ((T*) map->dx)[i];

  xmin = ((BoundingBox<T> *) particle->box_block)->getLocalMin();
  xmax = ((BoundingBox<T> *) particle->box_block)->getLocalMax();

  xmin_send = box->getLocalMin();
  xmax_send = box->getLocalMax();

  ops_dat binhead = map->binhead;

  int d_p[OPS_MAX_DIM], d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) {
    d_m[isou] = binhead->d_m[isou] + OPS_sub_dat_list[binhead->index]->d_im[isou];
    d_p[isou] = binhead->d_p[isou] + OPS_sub_dat_list[binhead->index]->d_ip[isou];
  }

  xmin.x += static_cast<T>(d_m[0]) * dx[0];
  xmax.x += static_cast<T>(d_p[0]) * dx[0];

  xmin.y += static_cast<T>(d_m[1]) * dx[1];
  xmax.y += static_cast<T>(d_p[1]) * dx[1];

  if (dim == 3) {
    xmin.z += static_cast<T>(d_m[2]) * dx[2];
    xmax.z += static_cast<T>(d_p[2]) * dx[2];
  }

  sending_reg[0] = (xmin_send.x < xmin.x) ? 0 : (int) ops_floor((xmin_send.x - xmin.x) / dx[0]);
  sending_reg[1] = (xmax_send.x > xmax.x) ? size[0] : (int) ops_ceil((xmax_send.x - xmin.x) / dx[0]);

  sending_reg[2] = (xmin_send.y < xmin.y) ? 0 : (int ) ops_floor((xmin_send.y - xmin.y) / dx[1]);
  sending_reg[3] =  (xmax_send.y > xmax.y) ? size[1] : (int ) ops_ceil((xmax_send.y - xmin.y) / dx[1]);

  sending_reg[4] = 0;
  sending_reg[5] = 1;
  if (dim == 3) {
    sending_reg[4] = (xmin_send.z < xmin.z) ? 0 : (int ) ops_floor((xmin_send.z - xmin.z) / dx[2]);
    sending_reg[5] = (xmax_send.z > xmax.z) ? size[2] : (int ) ops_ceil((xmax_send.z - xmin.z) / dx[2]);

  }

}

template<typename T>
void _ops_particle_set_intra_exch_regs(T* reg_neg, T* reg_pos,
                                       const BoundingBox<T> * box, const int idir,
                                       const int dim) {
  reg_neg[0] = -BIG_1;
  reg_neg[1] = box->getMinCoordDir(idir);

  reg_pos[0] = box->getMaxCoordDir(idir);
  reg_pos[1] = BIG_1;
}

template<typename T>
void _ops_particle_set_intra_bord_regs(T *bord_neg, T *bord_pos, const BoundingBox<T> *box,
                                       const int d_m, const int d_p, const int size[],
                                       const int id_m, const int id_p,
                                       const int idir, const int dim) {
  for (int isou = 0; isou < dim; isou++) {
    bord_neg[2 * isou] = - BIG_1;
    bord_neg[2 * isou + 1] = BIG_1;
    bord_pos[2 * isou] = - BIG_1;
    bord_pos[2 * isou + 1] = BIG_1;
  }

  T epsilon = std::numeric_limits<T>::epsilon();

  T xmin = box->getMinCoordDir(idir);
  T xmax = box->getMaxCoordDir(idir);

  T dx = (xmax - xmin) / static_cast<T>(size[idir] - d_p + d_m);


  //Setting actual borders
  bord_neg[2 * idir] = -BIG_1;
  bord_neg[2 * idir + 1] = (id_m != MPI_PROC_NULL) ?
      xmin - dx * d_m + epsilon : - BIG_1;

  bord_pos[2 * idir + 1] = BIG_1;
  bord_pos[2 * idir] = (id_p != MPI_PROC_NULL) ? xmax - dx * d_p - epsilon : BIG_1;

}

#endif /* OPS_C_INCLUDE_OPS_PARTICLE_BOX_MPI_FUNCS_H_ */
