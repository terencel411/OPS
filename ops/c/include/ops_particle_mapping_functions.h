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

/** @brief  Support functions for maps within OPS-Particle
  * @author Valantis Tsinginos
  * @details Definition of support functions for particle mapping */

#ifndef __OPS_PARTICLE_MAPPING_FUNCTIONS_H_
#define __OPS_PARTICLE_MAPPING_FUNCTIONS_H_

#include "ops_exceptions.h"
#include "ops_util.h"
#include "ops_bounding_box.h"
#include <limits>

template<typename T>
inline int virtual_within(int bin[], int bin_limits[], T *pos,
                          BoundingBox<T> *box, int dim) {
  int flag = 0;
  for (int i = 0; i < dim; i++)
    if (bin[i] == bin_limits[2 * i] || bin[i] == bin_limits[2 * i + 1]) {
      flag = 1; break;
    }

  if (flag)
    return (int) box->isCoordinateInBoundingBox(pos);

  return flag;
}

template<typename T>
inline int particle_is_within(T *xpos, T *region,  int dim)  {
  int flag  =  1;

  for (int i = 0; i < dim; i++) {
    if (xpos[i] < region[2 * i] || xpos[i] >= region[2 * i + 1]) return 0;
  }

  return flag;
}

template<typename T>
inline int _virtual_within(const int bin[], const int bin_limits[], T *pos,
                           BoundingBox<T> *box, int dim) {
  int flag  = 0;

  for (int i = 0; i < dim; i++)
    if (bin[i] == bin_limits[2 * i] || bin[i] == bin_limits[2 * i + 1]) {
      flag = 1; break;
    }

  if (flag)
    return (int) box->isCoordinateInBoundingBox(pos);

  return flag;
}


template<typename T>
inline void  get_coord_point(T *xp, int *ilocal, int *d_m,
                             ops_point<T> xmin, T *dx,
                             int dim, int stag) {

  xp[0] = xmin.x + (ilocal[0] - d_m[0]) * dx[0] + 0.5 * static_cast<T>(stag) * dx[0];
  xp[1] = xmin.y + (ilocal[1] - d_m[1]) * dx[1] + 0.5 * static_cast<T>(stag) * dx[1];
  if (dim == 3)
    xp[2] = xmin.z + (ilocal[2] - d_m[2]) * dx[2] + 0.5 * static_cast<T>(stag) * dx[2];
}

template<typename T>
inline void get_coord_point(T *xp, int *ilocal, int *d_m,
                            const T *xmin, const T *dx,
                            int dim, int stag) {
  xp[0] = xmin[0] + (ilocal[0] - d_m[0]) * dx[0] + 0.5 * static_cast<T>(stag) * dx[0];
  xp[1] = xmin[1] + (ilocal[1] - d_m[1]) * dx[1] + 0.5 * static_cast<T>(stag) * dx[1];
  if (dim == 3)
    xp[2] = xmin[2]+ (ilocal[2] - d_m[2]) * dx[2] + 0.5 * static_cast<T>(stag) * dx[2];
}

template<typename T>
inline int local_decide_rebuild(const T* xP, const T *xGrid, const T dx,
                            const int dim) {

  T epsilon = std::numeric_limits<T>::epsilon();
  for (int isou = 0; isou < dim; isou++)
    if (fabs(xP[isou] - xGrid[isou]) > 0.5 * dx + epsilon) {
      return 1;
    }

  return 0;
}

template <typename T>
inline void _ops_points_map_min_max(T *xmin, T *xmax,
                                    BoundingBox<T> *box, T* dx,
                                    int d_m[],  int d_p[], int dim) {

  for (int i = 0; i < dim; i++) {
    xmin[i] = box->getMinCoordDir(i) + static_cast<T>(d_m[i]) * dx[i];
    xmax[i] = box->getMaxCoordDir(i) + static_cast<T>(d_p[i]) * dx[i];
  }


}

template<typename T>
int _ops_coord_to_bin(const int dim, const ops_point<T> xmin, const ops_point<T> xmax,
                      const T *dx, const int *Ngrid, const T* xp) {

  int ix{-1}, iy{-1}, iz{-1};
  T epsilon = std::numeric_limits<T>::epsilon();

  int within = 0;

  if (xp[0] >= xmin.x - epsilon && xp[0] <= xmax.x + epsilon) {
    ix = (int) ops_floor((xp[0] - xmin.x) / dx[0]);
    within = 1;
  }

  if ((ix < 0) && within) {
    ix = (int) ops_floor((xp[0] - xmin.x) / dx[0] + epsilon);
  }


  within = 0;
  if (xp[1] >= xmin.y - epsilon && xp[1] <= xmax.y + epsilon) {
    iy = (int) ops_floor((xp[1] - xmin.y) / dx[1]);
    within = 1;
  }

  if (iy < 0 && within) {
    iy = (int) ops_floor((xp[1] - xmin.y) / dx[1] + epsilon);
  }

  iz = 0;
  if (dim == 3) {
    within = 0;
    if (xp[2] >= xmin.z - epsilon && xp[2] <= xmax.z + epsilon) {
      iz = (int) ops_floor((xp[2] - xmin.z) /dx[2]);
      within = 1;
    }

    if (iz < 0 && within)
      iz = (int) ops_floor((xp[2] - xmin.z) / dx[2] + epsilon);

  }
  else iz = 0; //Case Dim = 2

  if (ix < 0 || iy < 0 || iz < 0) return -1;

  /* One past the last bin in any direction is not a bin: a point sitting
     exactly on xmax floors to Ngrid[]. iz is only meaningful in 3D; in 2D it
     is forced to 0 above. */
  if (ix >= Ngrid[0] || iy >= Ngrid[1] ||
      (dim == 3 && iz >= Ngrid[2])) return -1;

  return ix + iy * Ngrid[0] + iz * Ngrid[0] * Ngrid[1];
}

template<typename T>
int _ops_coord_to_bin(const int dim, const T *xmin, const T* xmax, const T* dx,
                      int *Ngrid, const T* xp) {

  T epsilon = std::numeric_limits<T>::epsilon();


  int ix[3] = {-1, -1, -1};
  int address = 0;
  int prd = 1;
  for (int i = 0; i < dim; i++) {
    /* A point outside the binning box has no bin here. Return -1, which every
       caller already handles. Previously this case fell through with ix[i]
       never assigned, and the uninitialised value was then read and used to
       build the address -- an out-of-range index the callers cannot detect. */
    if (xp[i] < xmin[i] - epsilon || xp[i] > xmax[i] + epsilon) return -1;

    ix[i] = (int) ops_floor((xp[i] - xmin[i]) / dx[i]);

    if (ix[i] < 0) {
      ix[i] = (int) ops_floor((xp[i] - xmin[i]) / dx[i] + epsilon);
    }

    /* xp[i] == xmax[i] floors to Ngrid[i], one past the last bin. */
    if (ix[i] < 0 || ix[i] >= Ngrid[i]) return -1;

    address += ix[i] * prd;
    prd *= Ngrid[i];
  }

  return address;
}



template<typename T>
void _ops_map_compute_dx(const T *skin,const T* length, const int dim, T* dx_map,
                         int *size) {

  for (int i = 0; i < dim; i++) {

    if (skin[i] < std::numeric_limits<T>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Cell size too small");

    int icells = floor(length[i] / skin[i]);
    size[i] = (icells > 0) ? icells : 1;
    dx_map[i] = length[i] /  static_cast<double>(size[i]);
  }


}

template<typename T>
inline int _ops_check_particle_movement(T *xp, T *xp_old, const T* dx,
                                        const int dim) {
  T epsilon = std::numeric_limits<T>::epsilon();
  for (int i = 0; i < dim; i++)
    if (ops_abs(xp[i] - xp_old[i]) < dx[i] + epsilon)
      return 1;

  return 0;
}


template<typename T>
void _ops_particle_number_of_particles_in_range(const T *region, const int dim,
                                                const T *crds, const int ifirst,
                                                const int ilast, int *nwithin) {
  int n_in = 0;
  for (int i = ifirst ; i < ilast; i++) {
    if (particle_is_within(crds + dim * i, region, dim)) n_in++;
  }

  (*nwithin) = n_in;
}

template<typename T>
void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const T *dx,
                             const ops_point<T> xmin, const ops_point<T> xmax,
                             ops_dat binhead, ops_dat bin, ops_dat part_to_bin) {

  //Sanity checks
  if (grid != nullptr)
    if (xp->type_size != grid->type_size)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: Incompatible types between xp and "
                                         "grid ops_dat structures");

  if (xp->type_size != sizeof(T))
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Incompatbile types");

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

  const T* xp_data = (T *)xp->data;

  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    if (ibin < 0) continue;

    ((int *) bin->data)[i] = ((int *)binhead->data)[ibin];
    ((int *)binhead->data)[ibin] = i;
    ((int *)part_to_bin->data)[i] = ibin;
  }
}

template<typename T>
void _ops_particle_mapped_into_region_by_block(const BoundingBox<T> *box, const int dim,
                                               const T *xcrds,const int noParticles,
                                               int *sendlist) {
  int nwithin = 0;
  for (int iPart = 0; iPart < noParticles; iPart++) {
    bool isin = box->isCoordinateInBoundingBox(xcrds + iPart * dim);
    if (isin) {
      sendlist[nwithin] = iPart;
      nwithin++;
    }
  }
}

template<typename T>
void ops_particle_map_get_dx(ops_particle_mapping map, T dx[]) {

  for (int i = 0; i < map->particle->block->dims; i++) dx[i] = ((T *) map->dx)[i];
}

template<typename T>
void _ops_particle_halo_border_cells(int *send, const BoundingBox<T> *box, const T * xmin,
                                     const T *xmax, const int *size, const T *dx,
                                     const int dim) {

  T xmin_send[OPS_MAX_DIM], xmax_send[OPS_MAX_DIM];
  box->getLocalMaxMin(xmin_send, xmax_send);

  send[0] = (xmin_send[0] < xmin[0]) ? 0 : (int) ops_floor((xmin_send[0] - xmin[0]) / dx[0]);
  send[1] = (xmax_send[0] > xmax[0]) ? size[0] : (int) ops_ceil((xmax_send[0] - xmax[0]) / dx[0]);

  send[2] = (xmin_send[1] < xmin[1]) ? 0 : (int) ops_floor((xmin_send[1] - xmin[1]) / dx[1]);
  send[3] = (xmax_send[1] > xmax[1]) ? size[1] : (int) ops_ceil( (xmax_send[1] - xmax[1]) / dx[1]);

  send[4] = 0;
  send[5] = 1;
  if (dim == 3) {
    send[4] = (xmin_send[2] < xmin[2]) ? 0 : (int) ops_floor( (xmin_send[2] - xmin[2]) / dx[2]);
    send[5] = (xmax_send[2] > xmin[2]) ? size[2] : (int) ops_ceil((xmax_send[2] - xmax[2]) / dx[2]);
  }
}

template<typename T>
void _ops_particle_no_particles_in_range_by_box(const BoundingBox<T> * box, const int dim,
                                                const T *xcrds, const int noParticles,
                                                int *nsend) {
  int nwithin = 0;
  for (int i = 0; i < noParticles; i++) {
    bool isin = box->isCoordinateInBoundingBox(xcrds + dim * i);
    if (isin) nwithin++;
  }

  (*nsend) = nwithin;
}


#endif /* OPS_C_INCLUDE_OPS_PARTICLE_MAPPING_FUNCTIONS_H_ */
