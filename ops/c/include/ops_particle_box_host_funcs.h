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

/** @brief  Box block functions for a single CPU host-back end
  * @author Valantis Tsinginos
  * @details Box block functions for a single CPU host back end */

#ifndef __OPS_PARTICLE_BOX_HOST_FUNCS_H_
#define __OPS_PARTICLE_BOX_HOST_FUNCS_H_

#include "ops_bounding_box.h"

#define BIG_1 1.e9

template<typename T>
void _ops_construct_local_box_from_dat(ops_dat coords, const T* grid_size, const int dim,
                                       T* xmin, T* xmax) {
  int imin[OPS_MAX_DIM], imax[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];

  for (int i = 0; i < dim; i++) {
    int d_m = coords->d_m[i];
    int d_p = coords->d_p[i];
    imin[i] = -d_m;
    imax[i] = -d_m + (coords->size[i] - d_p + d_m) - 1;
    size[i] = coords->size[i];
  }

  if (coords->type_size != sizeof(T)) {
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Incompatible sizes between template type "
                                         "and ops_dat structure\n");
  }

  T *data = (T *) coords->data;
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
    }
  }
  else
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: The size of the physical domain"
                                         "must be 2 or 3\n");
}

template<typename T>
void BoundingBox<T>::partitionBoundingBox(ops_block block, ops_dat map_bin, T *dx) {

  if (owned) return;

  if (coords != nullptr) {
    int dim_dat = coords->block->dims;
    if (dim_dat != dim && coords->dim != dim)
      throw OPSException(OPS_RUNTIME_ERROR, "Dimensions of data structure not consisted "
                         "with dimensions");

    T xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];

    _ops_construct_local_box_from_dat(coords, dx, dim, xmin, xmax);

    printf("xmin = [%f %f] xmax = [%f %f]\n", xmin[0], xmin[1],
                                              xmax[0], xmax[1]);
    //Now set local and mininum
    setBoundingBoxLocalBound(xmin, xmax);
    setBoundingBoxGlobalBound(xmin, xmax);

    owned = true;

    printf("Box has bounding box =[%f %f]x[%f %f]\n", boundingBox[0].x,
           boundingBox[1].x, boundingBox[0].y, boundingBox[1].y);
  }
  else {
    ops_point<T> xmin, xmax;
    xmax = getGlobalMax();
    xmin = getGlobalMin();
    setBoundingBoxLocalBound(xmin, xmax);
    owned = true;
  }

}

template<typename T>
bool ops_get_bounding_box_local_to_global(ops_block block,T* xmin, T *xmax, T* xglb_min,
                                          T* xglb_max) {

  for (int i = 0; i < 3; i++) {
     xglb_min[i] = xmin[i];
     xglb_max[i] = xmax[i];
  }

  return true;
}


template<typename T>
BoundingBox<T> *  _ops_setup_exchange_region(BoundingBox<T> *box_from,
                                             BoundingBox<T> *box_to, const T *translate,
                                             const int *dir_to, const int *dir_from,
                                             const int dim) {
  T xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM], xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
  T x_recv_min[OPS_MAX_DIM], x_recv_max[OPS_MAX_DIM];
  T xrecv_act_min[OPS_MAX_DIM], xrecv_act_max[OPS_MAX_DIM];

  for (int i = 0; i < dim; i++) {
    int irecv_dir = dir_to[i];
    int isend_dir = dir_from[i];
    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - translate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - translate[isend_dir];
  }

  int intersect = ops_check_box_intersection(dim, xmin, xmax, xrecv_act_min,
                                              xrecv_act_max);

  if (intersect == 1) {
    for (int i = 0; i < dim; i++) {
      for (int iswap = 0; iswap < 2; iswap++) {

        for (int j = 0; j < dim; j++) {
          xsend_min[j] = xmin[j];
          xsend_max[j] = xmax[j];
        }

        xsend_min[i] = (iswap == 0) ? xmin[i] : xmax[i] - 0.5 * BIG_1;
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * BIG_1;

        xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG_1 : xmax[i];
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * BIG_1;

        int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                            xrecv_act_min, xrecv_act_max);

        if (a1 == 1) {
          for (int j = 0; j < dim; j++) {
            if (i != j) {
              xsend_min[j] = (ops_abs(xsend_min[j] - xrecv_act_min[j]) < std::numeric_limits<T>::epsilon()) ?
                  -BIG : MAX(xsend_min[j], xrecv_act_min[j]);
              xsend_max[j] = (ops_abs(xsend_max[j] - xrecv_act_max[j])
                               < std::numeric_limits<T>::epsilon()) ?
                  BIG : MIN(xsend_max[j], xrecv_act_max[j]);
            }

            BoundingBox<T> * box = new BoundingBox<T>(dim);
            box->setBoundingBoxLocalBound(xsend_min, xsend_max);
            return box;
          }
        }
      }
    }
  }

  return nullptr;
}

template<typename T>
BoundingBox<T> *  _ops_setup_border_region(BoundingBox<T> *box_from, BoundingBox<T> *box_to,
                                           const T * translate, const int *dir_from,
                                           const int *dir_to, const T* dx, const int dim) {

  T xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM], xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];
  T x_recv_min[OPS_MAX_DIM], x_recv_max[OPS_MAX_DIM];

  box_to->getLocalMaxMin(x_recv_min, x_recv_max);
  box_from->getLocalMaxMin(xmin, xmax);

  T xrecv_act_min[OPS_MAX_DIM], xrecv_act_max[OPS_MAX_DIM];
  for (int i = 0; i < dim ; i++) {
    int isend_dir = dir_from[i];
    int irecv_dir = dir_to[i];

    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - translate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - translate[isend_dir];
  }

  for (int i = 0; i < dim; i++) {
    for (int iswap = 0; iswap < 2; iswap++) {
      for (int j = 0; j < dim ; j++) {
        xsend_min[j] = xmin[j];
        xsend_max[j] = xmax[j];
      }

      xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG_1 : xmax[i]- dx[i]; //Keep it this way for the mome
      xsend_max[i] = (iswap == 0) ? xmin[i] + dx[i] : xmax[i] + 0.5 * BIG;

      int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                           xrecv_act_min, xrecv_act_max);

      if (a1 == 1) {
        for (int j = 0; j < dim; j++) {
        //  xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
        //  xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
          if (i != j) {
            xsend_min[j] = (fabs(xsend_min[j] - xrecv_act_min[j]) < 1.e-9) ?
                            -BIG : MAX(xsend_min[j], xrecv_act_min[j]);
            xsend_max[j] = (fabs(xsend_max[j] - xrecv_act_max[j]) < 1.e-9) ?
                             BIG : MIN(xsend_max[j], xrecv_act_max[j]);
           }
        }

        BoundingBox<T> *box = new BoundingBox<T>(dim);
        box->setBoundingBoxLocalBound(xsend_min, xsend_max);
        return box;
      }
    }
  }

  return nullptr;
}
#endif /* OPS_C_INCLUDE_OPS_PARTICLE_BOX_HOST_FUNCS_H_ */
