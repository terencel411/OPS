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
  * @brief OPS single-process specific functions for particles
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements runtime support functions applicable to non-MPI backend
  */


#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <string>
#include <assert.h>
#include <array>
#include <limits>
#include <vector>

void BoundingBox::partitionBoundingBox(ops_block block) {

  if (owned) return;

  if (coords != nullptr) {
    int dim_dat = coords->block->dims;
    if (dim_dat != dim && coords->dim != dim)
      throw OPSException(OPS_RUNTIME_ERROR, "Dimensions of data structure not consisted "
                         "with dimensions");

    double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];

    _ops_construct_local_box_from_dat(coords, dx, dim, xmin, xmax);

    //Now set local and mininum
    setBoundingBoxLocalBound(xmin, xmax);
    setBoundingBoxGlobalBound(xmin, xmax);

    owned = true;
  }
  else {
    ops_point xmin, xmax;
    xmax = getGlobalMax();
    xmin = getGlobalMin();
    setBoundingBoxLocalBound(xmin, xmax);
    owned = true;
  }
}



void _ops_get_max_min(double &minv, double &maxv, const double* dat, const size_t size) {
  
  minv = -1. * std::numeric_limits<double>::max();
  maxv = std::numeric_limits<double>::max();
  for (size_t i = 0; i < size; i++) {
     maxv = MAX(maxv, dat[i]);
     minv = MIN(minv, dat[i]);
  }
}

/*---------------------------------------------------------------------------------------*/
/* Computes bin sizes and grid sizes per direction                                       */
/*---------------------------------------------------------------------------------------*/

void  _ops_compute_bin_size(const ops_point xmin, const ops_point xmax, const int dim,
                            const double dx, int &Nx, double &dx_x, int &Ny,
                            double &dx_y, int &Nz, double &dx_z) {
  
  double box_size_x{xmax.x - xmin.x};
  double box_size_y{xmax.y - xmin.y};
  double box_size_z{0.0};

  if (dim == 3)
    box_size_z = {xmax.z - xmin.z};
  
  Nx = static_cast<int>(box_size_x / (2. * dx));


  Ny = static_cast<int>(box_size_y / (2. * dx));
  if (dim == 3) Nz = static_cast<int>(box_size_z / (2. * dx));

  if (Nx == 0) Nx = 1;
  if (Ny == 0) Ny = 1;
  if (Nz == 0) Nz = 1;
  
  dx_x = box_size_x / Nx;
  dx_y = box_size_y / Ny;
  dx_z = box_size_z / Nz; 
}

/*-------------------------------------------------------------------------------------*/
/*  Computes the grid cell on which particle is projected                              *
/--------------------------------------------------------------------------------------*/

int _ops_coord_to_bin(const int dim, const ops_point xmin,const  ops_point xmax, 
                      const double *dx, const int *Ngrid, const double *xp) {
  int ix{-1}, iy{-1}, iz{-1};

  if ((xp[0] >= xmin.x) && (xp[0] <= xmax.x))
    ix = floor((xp[0] - xmin.x) / dx[0]);
  
  if ((xp[1] >= xmin.y) && (xp[1] <= xmax.y))
    iy = floor((xp[1] - xmin.y) / dx[1]);
  
  if (dim == 3) {
    if ((xp[2] >= xmin.z) && (xp[2] <= xmax.z))
      iz = floor((xp[2] - xmin.z) / dx[2]);
  }
  else
    iz = 0;
  
  return (dim == 2) ? ix + iy * Ngrid[0] : ix + iy * Ngrid[0] + iz * Ngrid[0] * Ngrid[1];                    
}


int _ops_particle_moved_outside(ops_particle particle) {


  size_t nParticles = particle->no_particles;
  int dim = particle->block->dims;
  BoundingBox *box = particle->box_block;
  double *xlocal = (double *) particle->particle_pos_dat->data;
  for (size_t i = 0; i < nParticles; i++) {
    ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                    (dim == 3) ? xlocal[dim * i + 2] : 0.0};
     bool decide = box->isCoordinateInBoundingBox(point);
     if (!decide) return 1;
  }

  return 0;
}


/*-------------------------------------------------------------------------------*/
/* Compute the i or j or k of a given point in a computational grid              */
/*-------------------------------------------------------------------------------*/

int _ops_coord_to_bin_dir(const int dir, const int dim, const ops_point xmin,
                          const double *dx, const int *Ngrid, const double *xp) {
  if (dir > dim) {
    ops_printf("Error: Direction %d larger than dimension %d\n", dir,dim);
    exit(-1);
  }
  
  if (dir == 0)
    return static_cast<int>(floor( xp[0] -xmin.x ) / dx[0]);
  else if (dir == 1)
    return static_cast<int>(floor( xp[1] -xmin.y ) / dx[1]);
  else if (dir == 2)
    return static_cast<int>(floor( xp[2] - xmin.z) / dx[2]);

  return -1;
}

/*-------------------------------------------------------------------------------------------*/
/* Construct a bin structure for a uniform grid                                              */
/*-------------------------------------------------------------------------------------------*/

void _ops_uniform_build_map(const int init, const int dim, const size_t Np, const int *Ngrid, 
                            const double* xp, const ops_point xmin, const ops_point xmax, 
                            const double *dx, int *binhead, int *bin) {
  
  /* Initialize structures*/
  size_t grid_size = (dim == 3) ? Ngrid[0] * Ngrid[1] * Ngrid[2] : Ngrid[0] * Ngrid[1];
  
  for (size_t i = 0; i < grid_size; i++)
    binhead[i] = -1;
  if (init) {
    for (size_t i = 0; i < Np; i++)
      bin[i] = -1;
  }
  for (size_t i = Np-1; i >= 0; i++) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, Ngrid, (xp + 3 * i)); //TODO: Need to get elements directly from address
    if (ibin < 0) continue;

    bin[i] = binhead[ibin];
    binhead[ibin] = i;
  }  

}

void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin) {

  int size[dim];
  size_t no_elems{1};
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i];
    no_elems *= size[i];
  }

  /* Initialize elements */
  int* binhead_data = (int *)binhead->data;
  for (size_t i = 0; i < no_elems; i++)
    binhead_data[i] = -1;

  int* bin_data = (int *)bin->data;
  if (init) {
    for (size_t i = 0; i < Np; i++)
      bin_data[i] = -1;
  }

  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + 3 * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
  }
}

/*--------------------------------------------------------------------------------------*/
/* Computes the grid size per direction for a uniform grid                              */
/*--------------------------------------------------------------------------------------*/

void  _ops_compute_uniform_dx(ops_dat grid, const int dims,double *dx) {

  OPS_instance *instance = grid->block->instance;

  /* Get data points structures */
  int size[dims], imin[dims];
  for (int i = 0; i < dims; i++) {
    imin[i] = -grid->d_m[i];
    size[i] = grid->size[i];
  }

  double *grid_points = (double *)grid->data;

  if (dims == 2) {
    if (instance->OPS_soa) {
      dx[0] = *(grid_points + imin[0] + imin[1] * size[0] + 1)
              - *(grid_points + imin[1] * size[0]);
      dx[1] = *(grid_points + imin[0] + (imin[1] +1) * size[0] + size[0] * size[1])
              - *(grid_points + imin[1] * size[0] + size[0] * size[1]);
    }
    else {
      dx[0] = *(grid_points + dims * (imin[0] + 1) + dims * imin[1] * size[0])
            - *(grid_points + dims * imin[0] + dims * imin[1] * size[0]);
      dx[1] = *(grid_points + 1 + dims * imin[0] + dims * (imin[1] + 1) * size[0])
            - *(grid_points + 1 + dims * imin[0] + dims * imin[1] * size[0]);
    }
  }
  else if (dims == 3) {
    if (instance->OPS_soa) {
      dx[0] = *(   grid_points + (imin[0] + 1) + imin[1] * size[0]
                 + imin[2] * size[0] * size[1])
            - *(    grid_points + imin[0] * size[0] + imin[1] * size[0]
                 + imin[2] * size[0] * size[1]);
      dx[1] = *(   grid_points + imin[0] + (imin[1] + 1) * size[0]
                 + imin[2] * size[0] * size[1] + size[0] * size[1] * size[2])
            - *(   grid_points + imin[0] + imin[1] * size[0]
                 + imin[2] * size[0] * size[1] + size[0] * size[1] * size[2]);
      dx[2] = *(   grid_points + imin[0] + imin[1] * size[0]
                 + (imin[2] + 1) * size[0] * size[1] + 2 * size[0] * size[1] * size[2])
            - *(   grid_points + imin[0] + imin[1] * size[0]
                 + imin[2] * size[0] * size[1] + 2 * size[0] * size[1] * size[2]);
    }
    else {
      dx[0] = *(  grid_points + dims * (imin[0] + 1) + dims * size[0] * imin[1]
                 + dims * size[0] * size[1] * imin[2])
            - *(  grid_points + dims * imin[0] + dims * size[0] * imin[1]
                 + dims * size[0] * size[1] * imin[2]);

      dx[1] = *(  grid_points + 1 + dims * imin[0] + (imin[1] + 1) * dims * size[0]
                 + dims * size[0] * size[1] * imin[2])
            - *(  grid_points + 1 + dims * imin[0] + imin[1] * size[0] * dims
                 + dims * size[0] * size[1] * imin[2]);
      dx[2] = *(  grid_points  + 2 + dims * imin[0] + imin[1] * dims * size[0]
                 + dims * size[0] * size[1] * (imin[2] + 1))
            - *(  grid_points + 2 + dims * imin[0] + dims * imin[1] * size[0]
                 + dims * size[0] * size[1] * imin[2]);
    }
  }
}

/*--------------------------------------------------------------------------------------*/
/* Compute grid size for each node of a non-uniform structured grid                     */
/*--------------------------------------------------------------------------------------*/

void _ops_get_grid_size_per_node(const int dims, const ops_dat grid, const ops_point xmin,
                                 const ops_point xmax, double *grid_dx, double *grid_shape)
{
  size_t size[dims], imin[dims], imax[dims];

  for (int i = 0; i < dims; i++)  {
    imin[i] = (size_t) -grid->d_m[i];
    size[i] = (size_t) grid->size[i];
    imax[i] = (size_t) (grid->size[i] - grid->d_p[i]);
  }

  OPS_instance *instance = grid->block->instance;
  double *grid_data = (double *)grid->data;
  if (dims == 2) {
    for (size_t ix = imin[0]; ix < imax[0]; ix++) {
      for (size_t iy = imin[1]; iy < imax[1]; iy++) {
        double dx[2], xm[2], xb[2];
        if (ix == imin[0]) {
          xb[0] = (instance->OPS_soa) ? 2. * xmin.x - *(grid_data + ix + iy * size[0])
                                      : 2. * xmin.x - *(grid_data + 2 * ix + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? *(grid_data + (ix + 1) + iy * size[0])
                                      : *(grid_data + 2*(ix + 1) + 2 * iy * size[0]);
        }
        else if (ix == imax[0] - 1) {
          xb[0] = (instance->OPS_soa) ? *(grid_data + ix - 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix - 1) + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? 2. * xmax.x - *(grid_data + ix+ iy * size[0])
                                      : 2. * xmax.x - *(grid_data + 2 * ix + 2 * iy * size[0]);
        }
        else {
          xb[0] = (instance->OPS_soa) ? *(grid_data + ix - 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix - 1) + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? *(grid_data + ix + 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix + 1) + 2 * iy * size[0]);
        }

        dx[0] = 0.5 * (xm[0] - xb[0]);

        /* y-direction */
        if (iy == 0) {
          xb[1] = (instance->OPS_soa) ? 2. * xmin.y - *(  grid_data + size[0] * size[1]
                                                        + ix + iy * size[0])
                                      : 2. * xmin.y - *(  grid_data + 1 + 2 * ix
                                                        + 2 * iy * size[0]);
          xm[1] = (instance->OPS_soa) ? *(   grid_data + size[0] * size[1]
                                           + ix + (iy + 1) * size[0])
                                      : *(grid_data + 1 + 2 * ix + 2 * (iy + 1) * size[0]);
        }
        else if (iy == imax[1] - 1) {
          xb[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy - 1) * size[0])
                                      : *(grid_data + 1 + 2 * ix + 2 * (iy -1) * size[0]);
          xm[1] = (instance->OPS_soa) ? 2. * xmax.y - *(  grid_data + size[0] * size[1]
                                                        + ix + iy * size[0])
                                      : 2. * xmax.y - *(  grid_data + 1 + 2 * ix
                                                        + 2 * iy * size[0]);
        }
        else {
          xb[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy - 1) * size[0])
                                      : *(  grid_data + 1 + 2 * ix + 2 * (iy - 1) * size[0]);
          xm[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy + 1) * size[0])
                                      : *(  grid_data + 1 + 2 * ix + 2 * (iy + 1) * size[0]);
        }

        dx[1] = 0.5 * (xm[1] - xb[1]);

        size_t loc = ix + iy * size[0];
        grid_dx[loc] = MAX(dx[0], dx[1]);
        grid_shape[loc * dims] = dx[0];
        grid_shape[loc * dims + 1] = dx[1];

      }
    }
  }
  else if (dims == 3) {
    for (size_t  ix = imin[0]; ix < imax[0]; ix++) {
      for (size_t iy = imin[1]; iy < imax[1]; iy++) {
        for (size_t iz = imin[2]; iz < imax[2]; iz++) {
          double xb[3], xm[3], dx[3];
          if (ix == 0) {
            xb[0] = (instance->OPS_soa) ? 2. * xmin.x - *(  grid_data + ix + size[0] * iy
                                                          + iz * size[0] * size[1])
                                        : 2. * xmin.x - *(  grid_data + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
            xm[0] = (instance->OPS_soa) ? *(  grid_data + (ix + 1) + iy * size[0]
                                            + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix + 1) + 3 * iy * size[0]
                                            + 3 * iz * size[0] * size[1]);
          }
          else if (ix == imax[0] - 1) {
            xb[0] = (instance->OPS_soa) ? *(  grid_data + (ix - 1) + iy * size[0]
                                             + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix - 1) + 3 * iy * size[0]
                                             + 3 * iz * size[0] * size[1]);
            xm[0] = (instance->OPS_soa) ? 2. * xmax.x - *(  grid_data + ix + size[0] * iy
                                                          + iz * size[0] * size[1])
                                        : 2. * xmax.x - *(  grid_data + 3 * ix + 3 * iy *size[0]
                                                          + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[0] = (instance->OPS_soa) ? *(  grid_data + (ix - 1) + iy * size[0]
                                            + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix - 1) + 3 * iy * size[0]
                                            + 3 * iz * size[0] * size[1]);
            xm[0] =  (instance->OPS_soa) ? *(  grid_data + (ix + 1) + iy * size[0]
                                             + iz * size[0] * size[1])
                                         : *(  grid_data + 3 * (ix + 1) + 3 * iy * size[0]
                                              + 3 * iz * size[0] * size[1]);
          }

          dx[0] = 0.5 * (xm[0] - xb[0]);

          /* y-direction */
          if (iy == 0) {
            xb[1] = (instance->OPS_soa) ? 2. * xmin.y - *(grid_data + size[0] * size[1] * size[2]
                                                           + ix + iy * size[0] +
                                                           + iz * size[0] * size[1])
                                        : 2. * xmin.y - *(grid_data + 1 + 3 * ix + 3 * iy * size[0]
                                                            + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy + 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix + 3 * (iy + 1) * size[0]
                                             + 3 * iz * size[0] * size[1]);
          }
          else if (iy == imax[1] - 1) {
            xb[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy - 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix  + 3 * (iy - 1) * size[0]
                                              + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? 2. * xmax.y - *( grid_data + size[0] * size[1] * size[2]
                                                           + ix + iy * size[0]
                                                           + iz * size[0] * size[1])
                                        : 2. * xmax.y - *( grid_data + 1 + 3 * ix
                                                           + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy - 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix + 3 * (iy-1) * size[0]
                                             + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? *(grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy + 1) * size[0]  + iz * size[0] * size[1])
                                        : *(grid_data + 1 + 3 * ix + 3 * (iy + 1) * size[0]
                                            + 3 * iz * size[0] * size[1]);
          }

          dx[1] = 0.5 * (xm[1] - xb[1]);

          /* z- direction */
          if (iz == imin[2]) {
            xb[2] = (instance->OPS_soa) ? 2. * xmin.z - *( grid_data + 2 * size[0] * size[1] * size[2]
                                                           + ix + iy * size[0]
                                                           + iz * size[0] * size[1])
                                        : 2. * xmin.z - *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? *(  grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz + 1) * size[0] * size[1])
                                        : *( grid_data + 2 +3 * ix + 3 * iy * size[0]
                                             + 3 * (iz + 1) * size[0] * size[1]);
          }
          else if (iz == imax[2] - 1) {
            xb[2] = (instance->OPS_soa) ? *( grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz - 1) * size[0] * size[1])
                                        : *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz - 1) * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? 2. * xmax.z - *( grid_data + 2 * size[0] * size[1] * size[2]
                                                          + ix + iy * size[0]
                                                          + iz * size[0] * size[1])
                                        : 2. * xmax.z - *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[2] = (instance->OPS_soa) ? *(grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz - 1) * size[0] * size[1])
                                        : *(grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz - 1) * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? *(grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz + 1) * size[0] * size[1])
                                        : *(grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz + 1) * size[0] * size[1]);
          }

          dx[2] = 0.5 * (xm[2] - xb[2]);
          size_t loc = ix + iy * size[0] + iz * size[0] * size[1];
          grid_dx[loc] = MAX3(dx[0], dx[1], dx[2]);
          for (int idir = 0; idir < dims; idir++)
            grid_shape[dims * loc + idir] = dx[idir];
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*/
/* Assert the intersection of a grid cell with a particle center                                 */
/*-----------------------------------------------------------------------------------------------*/

int _ops_check_particle_nunif_grid_inters(const int dim,const double *xGrid, const double *dxGrid,
                                          const double *xp) {
   
  if (xp[0] < xGrid[0] - 0.5 * dxGrid[0] || xp[0] > xGrid[0] + 0.5 * dxGrid[0])
    return 0;

  /* y-direction */
  if (xp[1] > xGrid[1] - 0.5 * dxGrid[1] || xp[1] > xGrid[1] + 0.5 * dxGrid[1])
    return 0;

  if (dim == 3) {
    if (xp[2] > xGrid[2] - 0.5 * dxGrid[2] || xp[2] > xGrid[2] + 0.5 * dxGrid[2])
      return 1;
  }

  return 1;
}

void _ops_particle_to_non_uniform_grid_intersection(const int init, const int dim, 
                                                    const ops_point xmin,
                                                    const int *binhead_grids, const int *Npoints, 
                                                    const int *bin_grid, const double* xGrid, 
                                                    const double *grid_dx, const int Ngrid,
                                                    const int *binhead_particles, const int *Ng_parts,
                                                    const int *bin_parts, const double *xp, const int Np,
                                                    const double *dx_grid, const double *dx_p_grid, 
                                                    int *binhead, const int *size, int *bin)
{
  /* Initialize structure if necessary */
  if (init == 1) {
    //TODO: Vrf that size[i > dim] = 1
    for (int i = 0; i < size[0] * size[1] * size[2]; i++) {
      binhead[i] = -1;
    }

    for (int i = 0; i < Np; i++)
      bin[i] = -1;
  }

  /* Part II: Identify stencil */
  int Nm[dim]; //TODO: We need to map to smaller particle
  for (int i = 0; i < dim; i++) {
    Nm[i] =static_cast<int>(dx_grid[i] / dx_p_grid[i]);
  }

  /* Loop over coarse*/
  double xelem[dim], dxGrid[dim];
  int ngrid = Npoints[0] * Npoints[1] * Npoints[2];
  for (int i = 0; i < ngrid; i++) {
    /* Get local element */
    int igrid = binhead_grids[i];
    while (igrid > - 1) {
      xelem[0] = xGrid[dim * igrid];
      xelem[1] = xGrid[dim * igrid + 1];
      if (dim == 3)
        xelem[2] = xGrid[dim * igrid + 2];
      
      dxGrid[0] = grid_dx[3 * igrid];
      dxGrid[1] = grid_dx[3 * igrid + 1];
      if (dim == 3)
        dxGrid[2] = grid_dx[3 * igrid + 2];

      /* Get stencil*/
      int ix = _ops_coord_to_bin_dir(0, dim, xmin, dx_p_grid, Ng_parts, xelem);
      int imin = (ix - Nm[0] >= 0 ) ? ix - Nm[0] : 0;
      int imax = (ix + Nm[0] < Ng_parts[0]) ? ix + Nm[0] : Ng_parts[0] - 1;

      int iy = _ops_coord_to_bin_dir(1, dim, xmin, dx_p_grid, Ng_parts, xelem);
      int jmin = (iy - Nm[1] >= 0) ? iy - Nm[1] : 0;
      int jmax = (iy + Nm[1] < Ng_parts[1]) ? iy + Nm[1] : Ng_parts[1] - 1;

      int iz, kmin{0}, kmax{0};
      if (dim == 3) {
        iz = _ops_coord_to_bin_dir(2, dim, xmin, dx_p_grid, Ng_parts, xelem);
        kmin = (iz - Nm[2] >= 0) ? iz - Nm[2] : 0;
        kmax = (iz + Nm[2] < Ng_parts[2]) ? iz + Nm[2] : Ng_parts[2] - 1; 
      }

      /* Search for interaction between grid points and particles */
      for (int ibin = imin; ibin <= imax; ibin++) {
        for (int jbin = jmin; jbin <= jmax; jbin++) {
          for (int kbin = kmin; kbin <= kmax; kbin++) {
            int ipart = binhead_particles[   ibin + jbin *  Ng_parts[0]
                                              + kbin * Ng_parts[0] * Ng_parts[1]];
            while(ipart > - 1) {
              //TODO: Need to verify also for OPS_soa or not
              int inter = _ops_check_particle_nunif_grid_inters(dim, xelem, dxGrid, xp + dim * ipart);
              //onst int dim,const double *xGrid,double *dxGrid, double *xp

              if (inter ) {
                bin[ipart] = binhead[igrid];
                binhead[igrid] = ipart;
              }
              ipart = bin_parts[i];
            }
          }
        }
      }

      igrid = bin_grid[igrid];
    }
  }
}


/*-----------------------------------------------------------------------------------------------------------*/
/* Finds the box of each process based on an ops_dat structure for non-MPI backend code-assume a uniform grid
 * TODO: Expand to non-uniform
 */
void _ops_construct_local_box_from_dat(ops_dat coords, double *grid_size, int dim, double *xmin, double *xmax) {
  /* Get minimum lowerbound */
  int imin[OPS_MAX_DIM], imax[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];

  /* Get lower and upper bound */
  for (int i = 0; i < dim; i++) {
    imin[i] = -coords->d_m[i];
    imax[i] = -coords->d_m[i] + (coords->size[i] - coords->d_p[i] + coords->d_m[i]) - 1; //TODO: Check
    size[i] = coords->size[i];
  }

  ops_printf("imin = [ ");
  for (int i = 0; i < dim; i++)
    ops_printf("%d ", imin[i]);
  ops_printf("]\n");

  ops_printf("imax = [ ");
  for (int i = 0; i < dim; i++)
    ops_printf("%d ", imax[i]);
  ops_printf("]\n");

  double *data = (double *)coords->data; //NEED A Sanity check
  OPS_instance *instance = coords->block->instance;

  if (dim == 2) {
    if (instance->OPS_soa) {
      xmin[0] = *(data + imin[0] + imin[1] * size[0]);
      xmin[1] = *(data + imin[0] + imin[1] * size[0] + size[0] * size[1]);

      /* Getting xmax */
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
      xmin[0] = *(data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]);
      xmin[1] = *(  data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]
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
    ops_printf("Error: The size of the spatial domain must be two or zero.\n");
    exit(-1);
  }

  ops_printf("xmin = [");
  for (int i = 0; i < dim; i++)
    ops_printf("%12.9e ", xmin[i]);
  ops_printf("]\n");

  ops_printf("xmax = [");
  for (int i = 0; i < dim; i++)
    ops_printf("%12.9e ", xmax[i]);
  ops_printf("]\n");

  /* Uniform assumption for the moment */
  for (int i = 0; i < dim ; i++) {
    xmin[i] -= 0.5 * grid_size[i];
    xmax[i] += 0.5 *grid_size[i];
  }
}

bool ops_get_bounding_box_local_to_global(ops_block block,double* xmin,double *xmax,double* xglb_min, double* xglb_max) {

  for (int i = 0; i < 3; i++) {
     xglb_min[i] = xmin[i];
     xglb_max[i] = xmax[i];
  }

  return true;
}

/* Get the local bounding box for sequential code */
bool ops_bounding_box_global_to_local(const ops_block block, int dim,
                                      std::array<ops_point, 2>& globalBoundingBox,
                                      std::array<ops_point, 2>& boundingBox) {

  boundingBox[0].x = globalBoundingBox[0].x;
  boundingBox[0].y = globalBoundingBox[0].y;

  boundingBox[1].x = globalBoundingBox[1].x;
  boundingBox[1].y = globalBoundingBox[1].y;

  if (dim == 3) {
    boundingBox[1].z = globalBoundingBox[1].z;
    boundingBox[0].z = globalBoundingBox[0].z;
  }
  return true;
}

/*-------------------------------------------------------------------------------------*/
/* Mapping particle functions
 *-------------------------------------------------------------------------------------*/

int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {

  /* User enforce new map build */
  if (enforce)
    return 1;

  if (map->decide)
    return 1;

  /*1.  Particle structure changed */
  return _ops_particle_moved_outside(particle);

  int flag = 0;
  //TODO: Expand to three different structures

  /* 2. Particles moved substantially */
  double *x = (double *)particle->particle_pos_dat->data;
  double *x_old =(double *)map->pos_old->data;

  double *rad = (double *)map->Rp->data;
  double *rad_old
  = (map->particle_changes== OPS_EVOLV_SHAPE) ?
      (double *) map->Rp_old->data : nullptr;

  int dim = particle->block->dims;
  double dx[dim];
  for (size_t i = 0; i < map->nParticles; i++) {
    double dx_sq{0.0}, dr{0.0};
    for (int isou = 0; isou < dim; isou++) {
      dx[isou] = x[isou + dim * i] - x_old[isou + dim * i];
      dx_sq += dx[isou] * dx[isou];

      if (map->particle_changes== OPS_EVOLV_SHAPE)
        dr = rad[i] - rad_old[i];
    }

    //Case I: Particle of fixed envelope
    if (map->particle_changes== OPS_EVOLV_SHAPE) {
      if (dx_sq > map->skin * map->skin) {
        flag = 1;
        break; //TODO: Check if it works
      }
    }
    else {
      double dr_sq = dr * dr;
      if (dr_sq > map->skin * map->skin || dx_sq > map->skin * map->skin) {
        flag = 1;
        break;
      }
    }
    //TODO: Add radius

  }

  if (flag) map->decide = true;
  else
    map->decide = false;

  if (flag)
    return 1;

  //TODO: Need to verify that also the grid hash't changed as well

  return 0;
}

void _ops_build_particle_to_grid(const size_t nParticles, const ops_dat bin,
                                 const ops_dat binhead, ops_dat part_to_grid) {
  if (nParticles < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Non-positive number of particles\n");

  if (!bin->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat bin is not a particle ops_dat"
                                          "structure");

  if (binhead->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat binhead is defined for particle data");

  if (!part_to_grid->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat part_to_grid must be set"
                       "for grid data");

  int *binheads = (int *)binhead->data;
  int *bins = (int *)bin->data;
  int *par_grid = (int *)part_to_grid->data;

  int dims = bin->block->dims;
  int sizeL{1};
  for (int i = 0; i < dims; i++)
    sizeL*= binhead->size[i];

  for (int  iGrid = 0; iGrid < sizeL; iGrid++) {
    int ip = binheads[iGrid];
    if (ip == -1) continue;

    par_grid[ip] = iGrid;
    while (ip != -1) {
      ip = bins[ip];
      if (ip != -1) par_grid[ip] = iGrid;
    }
  }
}

void  _ops_particle_build_local_uniform(ops_particle_mapping map, ops_particle particle) {
  //TODO: Realloc all lists as larger size is expected
  size_t Np = particle->no_particles;

  //TODO: Allocate local structures
  map->nParticles = Np;
  /* Reallocate particle lists */
  if (Np > map->Nmax) {
    map->Nmax += 100;
    map->bin->data = (char *)ops_realloc(map->bin->data,
                      map->bin->elem_size * map->Nmax);
  }


  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];

  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);

  /* Build bins based on an ops_dat structure */
  _ops_build_uniform_dats(1, particle->block->dims, map->grid, particle->particle_pos_dat,
                          Np, dx, xmin, xmax, map->binhead, map->bin);

  _ops_build_particle_to_grid(map->nParticles, map->bin, map->binhead,
                              map->parts_to_grid);

}

void _ops_particle_exchange(ops_particle particle) {

}

/*---------------------------------------------------------------------------------------*/
/*! Build a bin structure for non uniform structured grids                               */
/*---------------------------------------------------------------------------------------*/

void _ops_particle_build_local_non_uniform(ops_particle_mapping map, ops_particle particle) {
  
  //Reallocate all lists if needed 
  size_t Np = particle->no_particles;
  map->nParticles = Np;

  if (Np > map->Nmax)  {
    map->Nmax+=100;
    map->bin->data = (char *) ops_realloc(map->bin->data, map->bin->elem_size * particle->Nmax);
    map->parts_to_grid->data = (char *) ops_realloc(map->parts_to_grid->data,
                                                    map->parts_to_grid->elem_size *
                                                         particle->Nmax);
  }

  /* Get particle positions */
  double *xp = (double *)particle->particle_pos_dat->data;
  const ops_point xmin = particle->box_block->getLocalMin();
  const ops_point xmax = particle->box_block->getLocalMax();



  /* Get grid details */
  int size[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    size[i] = map->grid->size[i];
  }

  /* Allocate lists */
  //TODO: Add a single function to get results


  size_t grid_points = 1;
  for (int i = 0; i < particle->block->dims; i++)
    grid_points *= size[i];
  
  double *grid = (double *)map->grid->data;
  double *grid_size = (double *) ops_calloc(grid_points, sizeof(double));
  double *grid_shape = (double *) ops_calloc(grid_points * particle->block->dims, 
                                             sizeof(double));

  _ops_get_grid_size_per_node(particle->block->dims, map->grid, xmin, xmax,
                              grid_size, grid_shape);

  /* Create multiple grids */
  double dx_min, dx_max;  
  _ops_get_max_min(dx_min, dx_max, grid_size, size[0] * size[1] * size[2]);

  int Nx_fine[3], Nx_coarse[3];
  double dx_fine[3], dx_coarse[3];

  _ops_compute_bin_size(xmin, xmax, particle->block->dims, dx_min,
                        Nx_fine[0], dx_fine[0], Nx_fine[1], dx_fine[1],
                        Nx_fine[2], dx_fine[2]);

  
  _ops_compute_bin_size(xmin, xmax, particle->block->dims, dx_max,
                        Nx_coarse[0], dx_coarse[1], Nx_coarse[1], dx_coarse[1],
                        Nx_coarse[2], dx_coarse[2]);

  //TODO: Check if we can move backward and retain some of the bin-structures
  int Nf = map->Ngrids;
  int dim = particle->block->dims;
  double multi_dx[3 * Nf];
  int Npoints[3 * Nf];

  /* Generate structures */
  multi_dx[0] = dx_fine[0];
  multi_dx[1] = dx_fine[1];

  multi_dx[3 * (Nf - 1)] = dx_coarse[0];
  multi_dx[3 * (Nf - 1) + 1] = dx_coarse[1];

  if (dim == 3) {
    multi_dx[2] = dx_fine[2];
    multi_dx[3 * (Nf - 1) + 2] = dx_coarse[2];
  }

  double dx_m = (dx_coarse[0] - dx_fine[0]) / static_cast<double>(Nf - 1);

  for (int i = 1; i < Nf - 1; i++) {
    double dx_g = multi_dx[dim * (i-1)] + dx_m;

    _ops_compute_bin_size(xmin, xmax, dim, dx_g, Npoints[3 * i],
                           multi_dx[3 * i], Npoints[3 * i + 1], multi_dx[3 * i + 1], 
                           Npoints[3 * i + 2], multi_dx[3 * i + 2]);
  }


  //TODO: Construct the rest
  std::vector<int *> binhead_grids;

  /* allocate multiple vectors */
  for (int i = 0; i < Nf; i++) {
    binhead_grids.push_back(nullptr);
    size_t grid_points = Npoints[3 * i] * Npoints[3 * i + 1] * Npoints[3 * i + 2]; //TODO: Checl
    binhead_grids[i] = (int *)ops_calloc(grid_points, sizeof(int));
  }

  /* Allocate temporary particle bins and grid point bins*/
  size_t Ntmp = Npoints[0] * Npoints[1] * Npoints[2];
  int *binhead_particles_tmp = (int *)ops_calloc(Ntmp, sizeof(int));
  int *bin_particles_tmp = (int *) ops_calloc( Np, sizeof(int));

  /* Allocate temporary grid bin*/
  int size_points = size[0] * size[1] * size[2]; 
  int *bin_grid_pnts = (int *) ops_calloc(size_points, sizeof(int));

  _ops_uniform_build_map(1, dim, Np, Npoints, xp, xmin, xmax, multi_dx, binhead_particles_tmp, 
                        bin_particles_tmp);

  /* Construct bins for multigrid*/

  for (int i = 0; i < Nf; i++) {
    int init = (i == 0) ? 1 : 0;
    Ntmp = Npoints[3 * i] * Npoints[3 * i + 1] * Npoints[3 * i + 2];
    size_t Npart = size[0] * size[1] * size[2];
    
    _ops_uniform_build_map(init, dim, Npart, (Npoints + 3 * i), grid, xmin, xmax, 
                          (multi_dx + 3 * i), binhead_grids[i], bin_grid_pnts); 
  }

  int *binhead = (int *)map->binhead->data;
  int *bin = (int *)map->bin->data;

  /* Vrf cell to particle interaction */
  for (int i = 0; i < Nf; i++) {
    /* Ready to loop over all grids */
    int init = (i == 0) ? 1 : 0;
    _ops_particle_to_non_uniform_grid_intersection(init, dim, xmin, 
                                                   binhead_grids[i], Npoints + 3 * i,
                                                   bin_grid_pnts, grid, grid_shape, 
                                                   size[0] * size[1] * size[2], 
                                                   binhead_particles_tmp, Npoints, bin_particles_tmp,
                                                   xp, Np, multi_dx + 3 * i, multi_dx, binhead, size, bin);

  }

  _ops_build_particle_to_grid(map->nParticles, map->bin, map->binhead, map->parts_to_grid);
  ops_free(binhead_particles_tmp);
  ops_free(bin_particles_tmp);
  ops_free(grid_size);
  ops_free(grid_shape);
  ops_free(bin_grid_pnts);

  for (int i = 0; i < Nf; i++) {
    ops_free(binhead_grids[i]);
    binhead_grids[i] = nullptr;
  }

}

/*---------------------------------------------------------------------------------------*/
/*  Particle Halos functions                                                             */
/*---------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------*/
/*              Define exchange zone for a border type     halo                          *
 *---------------------------------------------------------------------------------------*/

void _ops_particle_set_exchange_border_zone(OPS_instance *instance,
                                            ops_particle_halo halo) {
  /* Vrf that bounding boxes are defined  */
  BoundingBox *sendingBox = halo->particle_from->box_block;
  BoundingBox *recvBox = halo->particle_from->box_block;
  if (!sendingBox->getOwnership() || !recvBox->getOwnership()) {
    throw OPSException(OPS_RUNTIME_ERROR, "Particle halo must be set after the bounding "
                       "sets are defined");
  }

  /* Get local boxes */
  int dim = sendingBox->getDim();
  double xmin[dim], xmax[dim], xsend_min[dim], xsend_max[dim];
  double x_recv_min[dim], x_recv_max[dim];
  recvBox->getLocalMaxMin(x_recv_min, x_recv_max);
  sendingBox->getLocalMaxMin(xmin, xmax);


  double *xtranslate = halo->translate;


  double xrecv_act_min[dim], xrecv_act_max[dim];
  for (int i = 0; i < dim ; i++) {
    int isend_dir =  halo->dir_from[i];
    int irecv_dir = halo->dir_to[i];

    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - xtranslate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - xtranslate[isend_dir];
  }

  /* Search for intersection box */
  for (int i = 0; i < dim; i++) {
    for (int iswap = 0; iswap < 2; iswap++) {
      for (int j = 0; j < dim ; j++) {
        xsend_min[j] = xmin[j];
        xsend_max[j] = xmax[j];
      }

//      xsend_min[i] = (iswap == 0) ? -0.5 * std::numeric_limits<double>::max() : xmax[i] - halo->dx[i];
//      xsend_max[i] = (iswap == 0) ? 0.5 * std::numeric_limits<double>::max() : xmin[i] + halo->dx[i];

      //TODO: We modiftied the table
      xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG : xmax[i] -  halo->dx[i]; //Keep it this way for the mome
      //TODO: Removing the 0.5 consider whole elements
      xsend_max[i] = (iswap == 0) ? xmin[i] + halo->dx[i] : xmax[i] + 0.5 * BIG;


//      halo->sendBox = ops_find_send_box_projection(i, dim, xsend_min, xsend_max, xrecv_act_min,
//                                                   xrecv_act_max);

      int a1 = ops_check_box_intersection(dim, xsend_min, xsend_max,
                                          xrecv_act_min, xrecv_act_max);

      if (a1 == 1) {
        for (int j = 0; j < dim; j++) {
          xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
          xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
        }

        halo->sendBox = new BoundingBox(dim, xsend_min, xsend_max);
        goto endline;
      }

    }
  }

  endline:
  return;
}

void  _ops_particle_set_exchange_zone(OPS_instance *instance,
                                      ops_particle_halo halo) {

  BoundingBox *sendingBox = halo->particle_from->box_block;
  BoundingBox *recvBox = halo->particle_to->box_block;

  if (!sendingBox->getOwnership() || !recvBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "Particle Halo exchange must be set after"
                                          " block boxes are set.");

  int dim = sendingBox->getDim();
  double xmin[dim], xmax[dim], xsend_min[dim], xsend_max[dim];
  double x_recv_min[dim], x_recv_max[dim];
  recvBox->getLocalMaxMin(x_recv_min, x_recv_max);
  sendingBox->getLocalMaxMin(xmin, xmax);

  /* Shift bounding box of receiving block to the local coordinate system of
   * sending block */
  double xrecv_act_min[dim], xrecv_act_max[dim];
  for (int i = 0; i < dim ; i++) {
    int irecv_dir =  halo->dir_to[i];
    int isend_dir =  halo->dir_from[i];
    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - halo->translate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - halo->translate[isend_dir];
  }

 /* Checking for intersection region */
  int intersect = ops_check_box_intersection(dim, xmin, xmax,
                                            xrecv_act_min, xrecv_act_max);

  if (intersect == 1) {
    for (int i = 0; i < dim; i++) {
      for (int iswap = 0; iswap < 2; iswap++) {
        for (int j = 0; j < dim ; j++) {
          xsend_min[j] = xmin[j];
          xsend_max[j] = xmax[j];
        }

        xsend_min[i] = (iswap == 0) ? xmin[i] : xmax[i] - 0.5 * std::numeric_limits<double>::max();
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * std::numeric_limits<double>::max();


        xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG : xmax[i];
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * BIG;


        int a1 = ops_check_box_intersection(dim, xsend_min, xsend_max,
                                            xrecv_act_min, xrecv_act_max);

        if (a1 == 1) {

          //Modify region to ensure that all particles will be exchanged within this region
          for (int j = 0; j < dim; j++) {
            xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
            xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
          }
          halo->sendBox = new BoundingBox(dim, xsend_min, xsend_max);
          goto endline;
        }
      }
    }
  }

  endline:
  return;

  //TODO:Add bites for the particle_halo
}



void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    _ops_particle_set_exchange_zone(instance, halo);
  }
}


void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  /* Loop over all halos to define exchange zone */
  for (int ihalos = 0; ihalos < halo_grp->nhalos; ihalos++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalos];
    _ops_particle_set_exchange_border_zone(instance, halo);

    if (halo->sendBox != nullptr) {
      printf("Sending Box = [%e %e]x[%e %e] x[%e %e]\n", halo->sendBox->getLocalMin().x,
             halo->sendBox->getLocalMax().x, halo->sendBox->getLocalMin().y,
             halo->sendBox->getLocalMax().y, halo->sendBox->getLocalMin().z,
             halo->sendBox->getLocalMax().z);
    }
  }
}

void _ops_particle_setup_for_rev_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp)
{

  ops_particle_halo_group halo_master = halo_grp->halo_master;

  if (halo_master == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo for forward/backward halo types"
                                                        "has not defined yet");

  if (halo_master->halo_type != OPS_HALO_GRP_BORDER)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo of a forward/backward "
                                                        "halo is not a border type halo");

  if (halo_master->halo_info == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo must be set before forward/backward"
                                                        "halo");

  //TODO: Check if we need to cpy list info by allocation
}


void _ops_particle_setup_default_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp)
{

}

//TODO: Add the mark deletion to particle list:
//TODO: Need activation somewhere

void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp) {

  char *buff = instance->ops_halo_buffer;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;

    BoundingBox *box = halo->sendBox;

    if (box == nullptr) continue;
    info->nsend = 0;
    int no_particles = particle_from->no_particles;
    for (int i = 0; i < no_particles; i++) {

      if (particle_from->mark_deletion[i] == 1) {


        ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                        (dim == 3) ? xlocal[dim * i + 2] : 0.0};
        bool decide = box->isCoordinateInBoundingBox(point);

        if (decide) {
          //mark particle for deletion now or shift it later
          info->nsend++;
          if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
            instance->ops_halo_buffer_size += 10 * halo->nbites;
            buff = (char *)ops_realloc(buff, instance->ops_halo_buffer_size
                                           * sizeof(char));
          }

          particle_from->mark_deletion[i] = 2; //particle is exchanged
         _ops_particle_pack_halo_data(buff + halo->nbites * (info->nsend - 1),
                                     halo->dat,halo->nhalos, i);
        }
      }
    }

    /* Unpack data */
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    int ifirst = particle_to->no_particles;

    particle_to->no_particles += nrecv;
    if (particle_to->no_particles >= particle_to->Nmax)
      ops_particle_realloc_data(particle_to);

    double *particle_crds = (double *)particle_to->particle_pos_dat->data;

    for (int i = 0; i < nrecv; i++)

    for (int i = 0; i < nrecv; i++) {
      int ielem = ifirst + i;

      _ops_particle_unpack_halo_data(buff + halo->nbites * i, halo->dat,
                                     halo->nhalos, ielem);

      //If particle within mark as actual
      ops_point xpoint{particle_crds[dim * ielem], particle_crds[dim * ielem + 1], 0.0};
      if (dim == 3) xpoint.z = particle_crds[dim * ielem + 2];
      bool a1 =  particle_to->box_block->isCoordinateInBoundingBox(xpoint);
      particle_to->mark_deletion[ielem] = (a1) ? 0 : 1;
    }

  }
}


/*-------------------------------------------------------------------------*/
/* Sequential border halo transfer                                         */
/*-------------------------------------------------------------------------*/

/* Exchange type halo  implementation*/
void _ops_particle_halo_border_transfer(OPS_instance  *instance,
                                        ops_particle_halo_group halo_grp) {

  char *buff = instance->ops_halo_buffer;


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo  halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;
    double *xlocal = (double *)particle_from->particle_pos_dat->data;

    int imin =  (halo_grp->loop_type != OPS_PART_LOOP_VIRTUAL) ?
        0 : particle_from->no_particles + particle_from->no_virtual;
    int imax = (halo_grp->loop_type != OPS_PART_LOOP_LOCAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;

    BoundingBox *box = halo->sendBox;
    info->nsend = 0;
    /* Pack all elements */
    for (int i = imin; i < imax; i++) {
      ops_point point{xlocal[dim * i], xlocal[dim * i + 1], (dim == 3) ?
                      xlocal[dim * i + 2] : 0.0};
      bool decide = box->isCoordinateInBoundingBox(point);
      if (decide) {
        info->nsend ++;
        if (info->nsend > info->nmax) {
          info->nmax += 20;
          info->sendlist = (int *) ops_realloc(info->sendlist,
                                               info->nmax* sizeof(int));

        }
        info->sendlist[info->nsend] = i;

        /* Ensure that buffer is large enough to receive data */
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += 10 * halo->nbites;
          buff = (char *)ops_realloc(buff, instance->ops_halo_buffer_size
                                           * sizeof(char));
        }


        //TODO: Add increase of decrease of elements
        _ops_particle_pack_halo_data(buff + (info->nsend - 1) * halo->nbites,
                                     halo->dat, halo->nhalos, i); //TODO:
      }
    }

    /* Receive data */
    ops_particle particle_to = halo->particle_to;
    info->nrecv = info->nsend;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;
    for (int irecv = 0; irecv < info->nrecv; irecv++) {
      particle_to->no_virtual++;

      /* Realocation of particle data */
      if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax)
        ops_particle_realloc_data(particle_to);

      int iloc = particle_to->no_particles + particle_to->no_virtual - 1;
      _ops_particle_unpack_halo_data(buff + irecv * halo->nbites, halo->dat,
                                     halo->nhalos, iloc);

    }

  }
}

void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp)
{
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    /* Packing data */
    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    int *sendlist = info->sendlist;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff=(char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }

    for (int i = 0; i < nsend; i++) {
      int ipart = sendlist[i];
      _ops_particle_pack_halo_data(buff + i * halo->nbites, halo->dat,
                                   halo->nhalos, ipart);
    }

    /* Unpacking data */
    int istart = info->firstrecv;
    int nrecv = info->nrecv;

    for (int i = 0; i < nrecv; i++) {
      int ipart = istart + i;
      _ops_particle_unpack_halo_data(buff + i * halo->nbites, halo->dat,
                                     halo->nhalos, ipart);
    }
  }
}

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp) {
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  for (int ihalo = 0; ihalo< halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    int nsend = info->nrecv;
    int nrecv = info->nsend;

    int *recvlist = info->sendlist;
    int sendfirst = info->firstrecv;

    int nbites_tot = nsend * halo->nbites;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff=(char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }

    /* Packing received data */
    for (int i = 0; i < nsend; i++) {
      int ipart = sendfirst + i;
      _ops_particle_pack_halo_data(buff + i * halo->nbites, halo->dat,
                                   halo->nhalos, ipart);
    }

    /* Unpack send (received) data */
    for (int i = 0; i < nrecv; i++) {
      int ipart = recvlist[i];
      _ops_particle_unpack_halo_data(buff + i * halo->nbites, halo->dat,
                                     halo->nhalos, ipart); //TODO: Shift + DATA OR add another array.
    }
  }
}

void ops_particle_setup_partition() {

}
