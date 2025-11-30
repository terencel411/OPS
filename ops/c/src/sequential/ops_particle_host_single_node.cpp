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



void BoundingBox::partitionBoundingBox(ops_block block, ops_dat map_bin) {


  if (owned) {
 //   printf("Bounding box is already set\n");
    return;
  }

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

  ops_printf("BoundingBox: [%f %f %f]x[%f %f %f]\n",boundingBox[0].x,
             boundingBox[0].y, boundingBox[0].z,boundingBox[1].x, boundingBox[1].y,
             boundingBox[1].z);
}


static int local_decide_rebuild(const double *xPart, const double *xGrid, const double dx,
                                const int dim) {

  for (int isou = 0; isou < dim; isou++)
    if (fabs(xPart[isou] - xGrid[isou]) > 0.5 * dx + DBL_EPSILON) {
      return 1;
    }


  return 0;
}

static void get_local_point(const int point,const int size[],
                            const int d_m[],const int dim, int grid[]) {
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

static int get_address(const int point[], const int size[], const int d_m[],
                       const int dim) {
  int address = 0;

  for (int isou = 0; isou < dim; isou++) {
    size_t points = 1;
    for (int jsou = 0; jsou < dim; jsou++)
      points *= size[jsou];
    address += (point[isou] - d_m[isou] * points);
  }

  return address;
}

static void get_coord_point(const double *coords, const int size[],const int d_m[],
                            const int ilocal[], const int dim, double *xlocal,
                            double skin, int stag) {

#ifdef OPS_SOA
  if (dim == 2) {
    xlocal[0] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0])
              + 0.5 * static_cast<double>(stag) * skin;
    xlocal[1] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                   + size[0] * size[1]) + 0.5* static_cast<double>(stag) * skin;
  }
  else {
    xlocal[0] = *(  coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                   + (ilocal[2] - d_m[2]) * size[0] * size[1]);
    xlocal[1] = *(  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                   + (ilocal[2] - d_m[2]) * size[0] * size[1] +
                  size[1] * size[2] * size[0]) + 0.5 * skin;
    xlocal[2] = (  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                  + (ilocal[2] - d_m[2]) * size[0] * size[1]
                  + 2 * size[1] * size[2] * size[0]) + 0.5 * static_cast<double>(stag) * skin;
  }
#else
  if (dim == 2) {
    xlocal[0] = *(coords + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0])
              + 0.5 * static_cast<double>(stag) * skin;


    xlocal[1] = *(coords +  dim * (ilocal[0] - d_m[0])
               + dim * (ilocal[1] - d_m[1]) * size[0] + 1)
               + 0.5 * static_cast<double>(stag) * skin;

  }
  else {
    xlocal[0] = *( coords + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
             + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
             + 0.5 * static_cast<double>(stag) * skin;
    xlocal[1] =  *( coords + 1 + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
               + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
               + 0.5 * static_cast<double>(stag) * skin;
    xlocal[2] =  *( coords + 2 + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
              + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
              + 0.5 * static_cast<double>(stag) * skin;
  }
#endif
}


static void get_coord_point_virtual_point(double *coords, int size[], int d_m[],
                                          int ilocal[], int dim, double *xold,
                                          double skin, int stag) {

double xmin[OPS_MAX_DIM];
double dx[OPS_MAX_DIM];

#ifdef OPS_SOA
  if (dim == 2) {
    xmin[0] = *(coords - d_[0] - d_m[1] * size[0]);
    xmin[1] = *( cooords - d_m[0] - d_m[1] * size[0]
                + size[0] * size[1]);

    dx[0] = *(coords + ( -d_[0] + 1) - d_m[1] * size[0])
            - xmin[0];

    dx[1] = *(coords - d[0] + (1 - d_m[1]) * size[0]
               + size[0] * size[1]) - xmin[1];
  }
  else if (dim == 3) {
    xmin[0] = *(    coords - d_m[0] - d_m[1] * size[0]
                  - d_m[2] * size[0] * size[1]);
    xmin[1] = *(    coords - d_m[0] - d_m[1] * size[1]
                  - d_m[2] * size[0] * size[1]
                  + size[0] * size[1] * size[2]);
    xmin[2] = *(    coords - d_m[0] - d_m[1] * size[2]
                  - d_m[2] * size[0] * size[1]
                  + 2 * size[0] * size[1] * size[2]);

    dx[0] = *(    coords + (1 - d_m[0]) - d_m[1] * size[0]
                - d_m[2] * size[0] * size[1]) - xmin[0];
    dx[1] = *(    coords - d_m[0] + (1 - d_m[1]) * size[0]
                - d_m[2] * size[1] * size[2]
                + size[0] * size[1] * size[2]) - xmin[1];
    dx[2] = *(    coords - d_m[0] - d_m[1] * size[1]
                + (1 - d_m[2]) * size[0] * size[1]
                + 2 * size[0] * size[1] * size[2]) - xmin[2];
  }
#else
  if (dim == 2) {
    xmin[0] = *(   coords - dim * d_m[0]  - dim * d_m[1] * size[0]);
    xmin[1] = *(   coords + 1 - dim * d_m[0] - dim * d_m[1] * size[0]);

    dx[0] = *(coords + dim * (1 - d_m[0]) - dim * d_m[1] * size[0]) - xmin[0];
    dx[1] = *(coords + 1 - d_m[0] * dim + (1 - d_m[1] * size[0])) - xmin[1];
  }
  else if (dim == 3) {
    xmin[0] = *(   coords - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);
    xmin[1] = *(   coords + 1 - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);
    xmin[2] = *(   coords + 2 - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);

    dx[0] = *(    coords + (1 - d_m[0]) * dim - d_m[1] * size[0]
                - d_m[2] * dim * size[0] * size[1]) - xmin[0];
    dx[1] = *(    coords + 1 - d_m[0] * dim + (1 - d_m[1]) * size[0]
                - d_m[2] * dim * size[0] * size[1]) - xmin[1];
    dx[2] = *(    coords + 2  - d_m[0] * dim + ( 1 - d_m[2]) * size[1]
                + (1 - d_m[2]) * dim * size[0] * size[1]) - xmin[2];
  }

  xold[0] = xmin[0] + static_cast<double>(ilocal[0]) * dx[0] + 0.5 * static_cast<double>(stag) * skin;
  xold[1] = xmin[1] + static_cast<double>(ilocal[1]) * dx[1] + 0.5 * static_cast<double>(stag) * skin;
  if (dim == 3)
    xold[2] = xmin[2]  + static_cast<double>(ilocal[2]) * dx[2] + 0.5 * static_cast<double>(stag) * skin;

#endif
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


int _ops_particle_mark_for_deletion(int ilocal, int  *mark_del, double *xpos, BoundingBox *box,
                                    int dim) {


  ops_point xpoint;
  xpoint.x = xpos[0];
  xpoint.y = xpos[1];
  xpoint.z = (dim == 3) ? xpos[2] : 0.0;

  bool flag = box->isCoordinateInGlobalBoundingBox(xpoint);

  mark_del[ilocal] = (int) (!flag);

  return (int)(!flag);
}



int _virtual_within(int bin[], int bin_limits[], double *xpos, BoundingBox *box, int dim) {

  int flag = 0;

  for (int i = 0; i < dim; i++)
    if (bin[i] == bin_limits[2 * i] || bin[i] == bin_limits[2 * i + 1]) {
      flag = 1; break;
    }

  if (flag)
    return (int) box->isCoordinateInBoundingBox(xpos);

  return flag;
}

void _ops_particle_realloc_map_data(ops_particle_mapping map, size_t Np) {

  if (Np < map->Nmax) return;

  map->parts_to_grid->data = (char *) ops_realloc(map->parts_to_grid->data,
                                                  map->parts_to_grid->elem_size *
                                                  (Np + OPS_MAX_PART));
  map->bin->data = (char *)ops_realloc(map->bin->data,
                                       map->bin->elem_size * (Np + OPS_MAX_PART));
  if (map->particle_changes == OPS_EVOLV_SHAPE)
    map->Rp_old->data = (char *) ops_realloc(map->Rp_old->data, map->Rp_old->elem_size *
                                             (Np + OPS_MAX_PART));

  map->pos_old->data = (char *) ops_realloc(map->pos_old->data, map->pos_old->elem_size *
                                            (Np + OPS_MAX_PART));

  map->Nmax = Np + OPS_MAX_PART;

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

int _ops_particle_moved_outside(ops_particle particle) {


  size_t nParticles = particle->no_particles;
  int dim = particle->block->dims;
  BoundingBox *box = particle->box_block;
  double *xlocal = (double *) particle->particle_pos_dat->data;
  for (size_t i = 0; i < nParticles; i++) {
    ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                    (dim == 3) ? xlocal[dim * i + 2] : 0.0};

     bool decide = box->isCoordinateInBoundingBox(point);
     if (!decide)
       return 1;

  }

  return 0;
}


void  _ops_update_old_positions_radii(ops_particle_mapping map, ops_dat coord,
                                      ops_dat shape,size_t Np) {
  int dim = coord->block->dims;

  double *xold = (double *) map->pos_old->data;
  int *part_to_bin =(int *)map->parts_to_grid->data;
  int local_point[OPS_MAX_DIM];

  double *xgrid = (double *)map->grid->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared (xold, part_to_bin)
//#endif
  for (int i = 0; i <(int) Np; i++) {
    int address = part_to_bin[i];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                    dim, local_point);
    get_coord_point(xgrid, map->grid->size,map->grid->d_m,
                    local_point, dim, xold + dim * i, map->skin, 1);
  }
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
    size[i] = binhead->size[i];
    no_elems *= size[i];
  }

  /* Initialize elements */
  int* binhead_data = (int *)binhead->data;
  for (size_t i = 0; i < no_elems; i++)
    binhead_data[i] = -1;



  int* bin_data = (int *)bin->data;
  if (init) {
//#ifdef _OPENMP
//#  pragma omp parallel for shared(bin_data)
//#endif
    for (size_t i = 0; i < Np; i++)
      bin_data[i] = -1;
  }

  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared(xp_data)
//#endif
  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
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
  int* binhead_data = (int *)binhead->data;
//  for (size_t i = 0; i < no_elems; i++)
//    binhead_data[i] = -1;


  memset(bin->data, -1, sizeof(int) * Np);
  memset(part_to_bin->data, -1, sizeof(int) * Np);
  int *bin_data = (int *)bin->data;
  int *part_2_bin = (int *)part_to_bin->data;

  //if (init) {
//#ifdef _OPENMP
//#  pragma omp parallel for shared(bin_data)
//#endif

// for (size_t i = 0; i < Np; i++) {
//      bin_data[i] = -1;
//      part_2_bin[i] = -1;
//    }
//  }

  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared(xp_data)
//#endif
  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
    part_2_bin[i] = ibin;
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
    imin[i] = -grid->d_m[i]; //TODO: SHIFT TO SEQUENTIAL AS THE d_m + the sublock is need
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

  printf("SOA: %d\n", coords->block->instance->OPS_soa);

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

  printf("Bounding Box = ");
  for (int i = 0; i < dim; i++)
    printf("[%f %f] ", xmin[i], xmax[i]);
  printf("\n");

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
  /*Enforce new build */
  if (enforce)
    return 1;

  if (map->decide)
    return 1;

  //1. Particle outside structure
 // if (_ops_particle_moved_outside(particle) == 1)
 //   return 1;


  int flag = 0;
  double *x = (double *)particle->particle_pos_dat->data;
  double *x_old = (double *)map->pos_old->data;

  double dx = map->skin;

  /* 2. Particle moved outside previous block */
  int dim = particle->block->dims;
  int nParticles =(int) particle->no_particles;

  for (int i = 0; i < nParticles; i++) {
    int a1 = 0;
    for (int idir = 0; idir < dim; idir++) {
//      printf("Dx = %f skin = %f\n", fabs(x[i * dim + idir] - x_old[i * dim + idir]), map->skin);
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

  if (flag) map->decide = true;
  else
    map->decide = false;

  return flag;

}

/*int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {

  // User enforce new map build
  if (enforce)
    return 1;

  if (map->decide)
    return 1;

  //*1.  Particle structure changed
  return _ops_particle_moved_outside(particle);

  int flag = 0;
  //TODO: Expand to three different structures

  //* 2. Particles moved substantially
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
    if (map->particle_changes != OPS_EVOLV_SHAPE) {
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
}*/

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
  size_t Np = (map->mapping_type == OPS_WITH_VIRTUAL) ?
      particle->no_particles + particle->no_virtual : particle->no_particles;

  //TODO: Allocate local structures
  map->nParticles = Np;
  /* Reallocate particle lists */
  if (Np > map->Nmax) {
    _ops_particle_realloc_map_data(map, Np);
  }

 // printf("Passed building local\n");

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];

  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);

  xmin.x = xmin.x + static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmin.y = xmin.y + static_cast<double>(map->binhead->d_m[1]) * dx[1];

  xmax.x = xmax.x + static_cast<double>(map->binhead->d_p[0]) * dx[0];
  xmax.y = xmax.y + static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
    xmax.z = xmax.z + static_cast<double>(map->binhead->d_p[2]) * dx[2];
    xmin.z = xmin.z + static_cast<double>(map->binhead->d_m[2]) * dx[2];
  }

//  printf("Mapping list [%f %f %f]x[%f %f %f]\n",xmin.x, xmin.y, xmin.z
 //        ,xmax.x, xmax.y, xmax.z);
 // printf("Dx = [%f %f %f]", dx[0], dx[1], dx[2]);

  /* Build bins based on an ops_dat structure */
  _ops_build_uniform_dats(1, particle->block->dims, map->grid, particle->particle_pos_dat,
                          Np, dx, xmin, xmax, map->binhead, map->bin, map->parts_to_grid);

//  _ops_build_particle_to_grid(map->nParticles, map->bin, map->binhead,
//s                              map->parts_to_grid);


  _ops_update_old_positions_radii(map, particle->particle_pos_dat,
                                  particle->particle_envelope, Np);

 // printf("Mapped is build\n");

}

void _ops_particle_create_local_uniform(ops_particle_mapping map, ops_particle particle) {

  if (!map->decide) return;

  //printf("Entered herein\n");


  size_t Np = (map->mapping_type == OPS_WITH_VIRTUAL) ?
      particle->no_particles + particle->no_virtual : particle->no_particles;

  size_t Nexist = map->nParticles;
  map->nParticles = Np;
  //At this all elements removed and new elements are added
  if (map->Nmax < Np) {
    _ops_particle_realloc_map_data(map, Np);
  }


  if (Np - Nexist > 0) {
    int bites_init = Nexist * sizeof(int);
    int bites_set = (Np - Nexist) * sizeof(int);
    memset(map->bin->data + bites_init, -1, bites_set);
  }



  //Obtain grid data
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();
  double dx[OPS_MAX_DIM];
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);

  xmin.x = xmin.x + static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmin.y = xmin.y + static_cast<double>(map->binhead->d_m[1]) * dx[1];

  xmax.x = xmax.x + static_cast<double>(map->binhead->d_p[0]) * dx[0];
  xmax.y = xmax.y + static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3)  {
    xmin.z = xmin.z + static_cast<double>(map->binhead->d_m[2]) * dx[2];
    xmax.z = xmax.z + static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  //Loop over particles
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;
  int dim = particle->block->dims;
  int ilocal[OPS_MAX_DIM];

 // printf("Nexist = %d\n", Nexist);

 /// for (int i =0; i < Nexist; i++)
 //   printf("Bin2Grid[%d] [%f %f ]= %d\n", i, part2grid[i], xpos[2 * i], xpos[2 * i + 1]);


  for (size_t i = Nexist; i < Np; i++) {
    int  address =
      _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

    if (address < 0) continue;
  /* Generate local bins for new particles */
    int ipart = binhead[address];
    binhead[address] = i;
    bins[i] = ipart;
    part2grid[i] = address;

    /* Generate xpos for new particles */
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);

    get_coord_point((double *)map->grid->data, map->grid->size,map->grid->d_m,
                     ilocal, dim, xold + dim * i, map->skin, 1);
  }

  map->decide = false;

}

/**
 * Updates the mapping structure for virtual particles if neccessary.
 */

void _ops_particle_update_local_uniform(ops_particle_mapping map, ops_particle particle) {

  if (map->decide) return; //No new particles inserted in the list
  if (map->mapping_type != OPS_WITH_VIRTUAL) return;

  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;


  size_t Np = (map->mapping_type == OPS_WITH_VIRTUAL) ?
      particle->no_particles + particle->no_virtual : particle->no_particles;

  size_t Nfirst = particle->no_particles;


  if (Np > map->Nmax) {
    _ops_particle_realloc_map_data(map, Np);
  }

  if (Np > map->nParticles)
    map->nParticles = Np;

  int *part_to_grid = (int *) map->parts_to_grid->data;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
  //Upate xmin and xmax due to special conditions


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


  for (size_t i = Nfirst; i < Np; i++) {
    int flag =  local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                     dim);
    if (flag) {
      int address = part_to_grid[i];


      _remove_particle_from_bins(address, i, binhead, bins);

      //Assign particle to new bin
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

      if (address < 0) continue;

      bins[i] = binhead[address];
      binhead[address] = i;

      part_to_grid[i] = address;

      //Assign local to different bin
      bins[i] = binhead[address];
      binhead[address] = i;
      part_to_grid[i] = address;

      //update old positions
      int ilocal[OPS_MAX_DIM];
      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal);
      get_coord_point((double *)map->grid->data, map->grid->size,map->grid->d_m,
                       ilocal, dim, xold + dim * i, map->skin, 1);


    }
  }
}


void _ops_particle_update_maps_due_to_halos(ops_particle particle, ops_particle_mapping map) {

  if (!map->decide) return;

  int nparticles = particle->no_particles;
  int no_virtual = particle->no_virtual;

  if (nparticles + no_virtual > map->Nmax)
    _ops_particle_realloc_map_data(map, nparticles + no_virtual);

  map->nParticles = nparticles + no_virtual;

  //Init data
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2bin = (int *)map->parts_to_grid->data;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
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
  double *xpos = (double *) particle->particle_pos_dat->data;
  double *xold = (double *) map->pos_old->data;
  for (int i = nparticles; i < nparticles + no_virtual; i++) {
    int address =  _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    if (address < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Halo particle not mapped\n");

    part2bin[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);
    get_coord_point_virtual_point((double *)map->grid->data, map->grid->size,
                                  map->grid->d_m,
                                   ilocal, dim, xold + dim * i, map->skin, 1);

  }

 // printf("Exit from update map\n");

}

void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map) {

  if (n_to_map + ifirst > map->Nmax) {
    _ops_particle_realloc_map_data(map, n_to_map + ifirst);
  }

  map->nParticles = ifirst + n_to_map;

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
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
  double *xpos = (double *) particle->particle_pos_dat->data;
  double *xold = (double *) map->pos_old->data;

  for (int i = ifirst; i < ifirst + n_to_map; i++) {



    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

    if (address < 0) printf("Rank %d: Particle %d address = %dv xpos =[%f %f] dx = %12.9e dy = %12.9e\n",ops_get_proc(), i, address, xpos[i* dim], xpos[i * dim + 1],
                            (xpos[i * dim] - xmin.x)/dx[0], (xpos[i * dim + 1] - xmin.y) / dx[1]);

    if (address < 0) continue;
    bin2grid[i] = address;

    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);


    //TODO:
    get_coord_point_virtual_point((double *) map->grid->data, map->grid->size,
                                  map->grid->d_m, ilocal, dim, xold + dim * i,
                                  map->skin, 1);

  }


}


void _ops_particle_setup_map(ops_particle particle, ops_particle_mapping map) {

  if (particle->no_particles == 0) return;

  map->nParticles = particle->no_particles;

  if (map->nParticles > map->Nmax)
    _ops_particle_realloc_map_data(map, map->nParticles);

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

  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
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

  for (int i = 0; i < particle->no_particles; i++) {
    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    bin2grid[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);
    get_coord_point((double *)map->grid->data, map->grid->size,map->grid->d_m,
                     ilocal, dim, xold + dim * i, map->skin, 1);
  }

}

void _ops_particle_map_from_exchange(ops_particle_mapping map, ops_particle particle,
                                     int ifirst, int ilast) {

  if (ifirst > particle->no_particles || ilast > particle->no_particles )
    throw OPSException(OPS_RUNTIME_ERROR, "Mapping particles are not actual particles");

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *) map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int *mark_deletion = (int *)particle->mark_deletion;

  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[OPS_MAX_DIM];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
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

  for (int i = ifirst; i < ilast; i++) {

    int address = _ops_coord_to_bin(dim , xmin, xmax, dx, map->binhead->size,
                                    xpos + dim * i);

    if (address < 0) {
      bin2grid[i] = -1;
      bins[i] = -1;
      mark_deletion[i] = 1;
      continue;
    }

    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

  }

}



//TODO: Need to generate a version that splits the generation creation and adding halo for
//      the exchange.
void _ops_particle_setup_map_virtual(ops_particle particle, ops_particle_mapping map) {

  if (particle->no_virtual == 0) return;

  map->nParticles += particle->no_virtual;

  if (map->Nmax < map->nParticles)
    _ops_particle_realloc_map_data(map, map->nParticles);

  //Initialize structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
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
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);

    get_coord_point_virtual_point((double *)map->grid->data, map->grid->size,
                                  map->grid->d_m,
                                   ilocal, dim, xold + dim * i, map->skin, 1);//TODO-Replace function

  }

}


/**
 * Function erases and build particle lists directly
 */

//TODO: We need to add herein what it builds (local or all)
int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {

  //TODO: Do we need this to be false by default???
  map->decide = false;



  int stag = 1; //Staggering is by def 1 for now

 // printf("Entering to check: %d\n");

  /* Checking if mapping declaired particles < total */
  int changed{0};
  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part_to_grid = (int *)map->parts_to_grid->data;
  int *mark_del = particle->mark_deletion;


  /*Accessing local particles only */
  /* 1. Check for deletion */
  /* 2. Check if need to move */

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];


 /* Set particles for mapping */
  int nmapping
    = (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                          particle->no_particles + particle->no_virtual;


  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
  //Upate xmin and xmax due to special conditions

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data; //Assumed here in that
  //Nparticles in list is build properly

  int nlimits = 2 * dim;
  int exch_limits[2 * OPS_MAX_DIM];
  int rmv_limits[2 * OPS_MAX_DIM];

  for (int i =  0; i < dim; i++) {
    exch_limits[2 *i] = (d_m[i] < 0) ? -d_m[i] : -size[i];
    exch_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - 2 * d_p[i]: 2 * size[i];
    rmv_limits[2 *i ] = 0;
    rmv_limits[2 * i + 1] = size[i] - d_p[i] + d_m[i] - 1;
  }

//  printf("rmv_limits = [%d %d]x[%d %d]\n", rmv_limits[0], rmv_limits[1], rmv_limits[2], rmv_limits[3]);

/*
  for (int i = 0; i < particle->no_particles + particle->no_virtual; i++) {
    if (i < particle->no_particles)
      printf("Actual particle  %d: x = [ %f %f ] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
             bins[i], part_to_grid[i]);
    else printf("Virtual %d: x = [ %f %f ] xold =[%f %f] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
                 xold[2 * i], xold[2 * i + 1],bins[i], part_to_grid[i]);
  }
*/

  int nactual = particle->no_particles;

  //TODO: For GPU: Split into two parts
  for (int i = 0; i < particle->no_particles; i++) {
    int flag =  local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                     dim);

    if (flag) {

      int address = part_to_grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      //Actual particle perform two checks

        //TODO-1: Check for possible removal
      int del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                       xpos + i * dim, particle->box_block);

      if (del_flag) {
        part_to_grid[i] = -1;
        map->decide = true;//TODO: Think if it is needed
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
      }

      //Start-mapping
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);
      if (address < 0) map->decide = true;

      bins[i] = binhead[address];
      binhead[address] = i;

      part_to_grid[i] = address;

      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);

      get_coord_point((double *)map->grid->data, map->grid->size, map->grid->d_m,
                       ilocal_new, dim, xold + dim * i, map->skin, 1);

      //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

 //     if (flag_build) printf("Particle %d to exchange zone\n");

      if (!map->decide) map->decide = flag_build;

    }
  }

  nactual = particle->no_particles;

  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

//  printf("Exit actual mapping: ");

  //printf("Building virtual maps\n");


   //TODO: We need a new version which keeps th

  for (int i = ifirst; i < nmapping; i++) {

    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin, dim);

    if (flag) { //To-Rebuild for virtual particle

      int address = part_to_grid[i];
      _remove_particle_from_bins(address, i, binhead, bins);
      part_to_grid[i] = -1;


      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

      if (address < 0) {
  //      printf("Particle %d non-positive address (%d)\n", i, address);
        map->decide = true; continue;}

      //Virtual still projected'


      get_local_point(address, map->binhead->size, map->binhead->d_m,
                      dim, ilocal);

      int flag_in = _virtual_within(ilocal, rmv_limits, xpos + dim * i, particle->box_block, dim);

      //Need to remove



      if (flag_in) { //Virtual become actual

        nactual++; //Increase local.
        nvirtual_act--;
        //swap->particle data

        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


        _ops_particle_swap_data(particle->particle_pos_dat->data, i, nactual - 1,
                                particle->particle_pos_dat->elem_size);

        //compute new positions and swap: TODO

        if (particle->particle_envelope != nullptr) {
          _ops_particle_swap_data(particle->particle_envelope->data, i, nactual - 1,
                                  particle->particle_envelope->elem_size);
        }

        for (int idat = 0; idat < particle->particle_dat_index; idat++) {
          ops_dat dat = particle->particle_dat[idat];
          _ops_particle_swap_data(dat->data, i, nactual - 1, dat->elem_size);
        }

        particle->mark_deletion[i] = 0;
        _ops_particle_swap_data((char *)particle->mark_deletion, i, nactual - 1,
                                sizeof(int));

        //swap->map with first virtual

        //set flag map to true

        _ops_particle_swap_mapping_data(map, i, nactual - 1);
        _ops_particle_swap_data(map->pos_old->data, i, nactual -1, \
                                map->pos_old->elem_size);
        //TODO: Need to swap positions as well then rebuild
        map->decide = true;

      }

      if (!map->decide)  {
        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


        get_coord_point_virtual_point((double *) map->grid->data, map->grid->size,
                                      map->grid->d_m, ilocal, dim, xold + dim * i,
                                      map->skin, 1);
      }

    }


  }
  //Perform action for virtual particles



  //TODO: Add this info only for the mapping list
  if (nactual > particle->no_particles)
    particle->no_particles = nactual;

  particle->no_virtual = nvirtual_act;

  return (int) map->decide;
}

int _ops_particle_decide_build_only_local_uniform(ops_particle_mapping map,
                                                  ops_particle         particle) {
  map->decide = false;

  int changed{0};
  int dim = particle->block->dims;

  //Get mapping structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;

  int *mark_del = particle->mark_deletion; //TODO: Need a global factor

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  int nmapping = particle->no_particles;


  /* Compute bounding points for mapping structures */
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();
  double dx[particle->block->dims];
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);

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

  int nlimits = 2 * dim;
  int rmv_limits[2 * OPS_MAX_DIM];
  int exch_limits[2 * OPS_MAX_DIM];
  //TODO: Need adaptation
  for (int i =  0; i < dim; i++) {
    exch_limits[2 * i] = d_m[i] < 0 ?  -d_m[i] : -size[i];
    exch_limits[2 * i + 1] = d_p[i] > 0? size[i] + d_m[i] -  2 * d_p[i] : 2 * size[i];
    rmv_limits[2 * i] = 0;
    rmv_limits[2 * i + 1] = size[i] - d_p[i] + d_m[i] - 1;
  }

  for (int i = 0; i < nmapping; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                    dim);

    if (flag) {
      int address = part2grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);


      //Remove particle from bin-box
      _remove_particle_from_bins(address, i, binhead, bins);

      //Check for deletion
      int del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits, xpos + i * dim,
                                                      particle->box_block); //TODO: Check

      if (del_flag) {

        printf("Particle %d marked for deletion: [%f %f]", i, xpos[2 * i], xpos[2 * i + 1]);

       part2grid[i] = -1;
       map->decide = true; //Enforce mapping;
       //nactual--;
       particle->mark_deletion[i] = 1;
       continue;
      }

      //Update map of actual particle
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;

      //Checking for particle moving in or out of exchange zone
      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);

      get_coord_point((double *)map->grid->data, map->grid->size, map->grid->d_m,
                       ilocal_new, dim, xold + dim * i, map->skin, 1);

      //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

      if (!map->decide) map->decide = flag_build;

    }
  }


  return (int) map->decide;
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

  int stag = 1;

  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);

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

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                  xpos + dim * i);

      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;
    }
  }
}


void _ops_particle_exchange(ops_particle particle) {

}

void _ops_particle_build_border(ops_particle particle) {

}

void _ops_particle_build_border_maps(ops_particle particle) { }

void _ops_particle_forward_intra_maps(ops_particle particle) { }
void _ops_particle_exchange_map_update(ops_particle particle) { }

/*---------------------------------------------------------------------------------------*/
/*! Build a bin structure for non uniform structured grids                               */
/*---------------------------------------------------------------------------------------*/

void _ops_particle_build_local_non_uniform(ops_particle_mapping map, ops_particle particle) {
  
  //Reallocate all lists if needed 
  size_t Np = particle->no_particles;
  map->nParticles = Np;

  if (Np > map->Nmax)  {
    map->Nmax= Np + OPS_MAX_PART;
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

        halo->sendBox = new BoundingBox(dim);
        halo->sendBox->setBoundingBoxLocalBound(xsend_min, xsend_max);
        goto endline;
      }

    }
  }


  endline:
  if (halo->sendBox == nullptr) {
    for (int i = 0; i < 2 * dim; i++)
      halo->isend[i] = 0;

    return;
  }
  else { //Finding grid points
    int size[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++)
      size[i] = (halo->particle_from->map_list[0] != nullptr) ?
          halo->particle_from->map_list[0]->binhead->size[i] : 0;

    //Get xmin.x of sending border
    ops_point xmin, xmax, xmin_send, xmax_send;

    double dx[OPS_MAX_DIM];
    _ops_compute_uniform_dx(halo->particle_to->map_list[0]->grid,
                            halo->particle_from->block->dims, dx);

    xmin = halo->particle_from->box_block->getLocalMin();
    xmax = halo->particle_from->box_block->getLocalMax();
    xmin_send = halo->sendBox->getLocalMin();
    xmax_send = halo->sendBox->getLocalMax();

    //TODO: Replacd dx with map dx
    if (halo->particle_from->map_list[0]->mapping_type == OPS_WITH_VIRTUAL) {
      xmin.x += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[0]) * dx[0];
      xmax.x += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_p[0]) * dx[0];
      xmin.y += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[1]) * dx[1];
      xmax.y += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_p[1]) * dx[1];

      if (halo->particle_from->block->dims == 3)  {
        xmin.z += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[2]) * dx[2];
        xmax.z += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[2]) * dx[2];
      }
    }

    halo->isend[0] = (xmin_send.x < xmin.x) ? 0 : (int) floor((xmin_send.x - xmin.x) / dx[0]);
    halo->isend[1] = (xmax_send.x > xmax.x) ? size[0] : (int) ceil((xmax_send.x - xmin.x) / dx[0]); //Sanity check

    halo->isend[2] = (xmin_send.y < xmin.y) ? 0 : (int ) floor((xmin_send.y - xmin.y) / dx[1]);
    halo->isend[3] = (xmax_send.y > xmax.y) ? size[1] : (int ) ceil((xmax_send.y - xmin.y) / dx[1]);

    halo->isend[4] = 0;
    halo->isend[5] = 1;
    if (dim == 3) {
      halo->isend[4] = (xmin_send.z < xmin.z) ? 0 : (int ) floor((xmin_send.z - xmin.z) / dx[2]);
      halo->isend[5] = (xmax_send.z > xmax.z) ? size[2] : (int ) ceil((xmax_send.z - xmin.z) / dx[2]);
    }
  }




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

        int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                            xrecv_act_min, xrecv_act_max);


        if (a1 == 1) {

        //Handling multi-block case
          for (int j = 0; j < dim; j++) {
            if (i != j) {
              xsend_min[j] = (fabs(xsend_min[j] - xrecv_act_min[j]) < 1.e-9) ?
                  -BIG : MAX(xsend_min[j], xrecv_act_min[j]);
              xsend_max[j] = (fabs(xsend_max[j] - xrecv_act_max[j]) < 1.e-9) ?
                  BIG : MIN(xsend_max[j], xrecv_act_max[j]);
            }
            //xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
            //xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
          }

          halo->sendBox = new BoundingBox(dim);
          halo->sendBox->setBoundingBoxLocalBound(xsend_min, xsend_max);
          goto endline;
        }
      }
    }
  }

  endline:
  return;

  //TODO:Add bites for the particle_halo
}

void _ops_particle_verify_exchange_zone(OPS_instance *instance,
                                        ops_particle_halo halo) {

  //Set local Box
  halo->sendBox->setBoundingBoxLocalBound(halo->sendBox->getGlobalMin(), halo->sendBox->getGlobalMax());

  //Ensure that sendBox intersects local box.

}


void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {

//    printf("Halo %d\n", ihalo);
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (halo->sendBox == nullptr)
     _ops_particle_set_exchange_zone(instance, halo);
    else
      _ops_particle_verify_exchange_zone(instance, halo);

    if (halo->sendBox != nullptr) {
      printf("Particle halo %d: SendingBox = [%e %e]x[%e %e]x[%e %e]\n", ihalo,
             halo->sendBox->getLocalMin().x,
             halo->sendBox->getLocalMax().x, halo->sendBox->getLocalMin().y,
             halo->sendBox->getLocalMax().y, halo->sendBox->getLocalMin().z,
             halo->sendBox->getLocalMax().z);
    }
  }
}


void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  /* Loop over all halos to define exchange zone */
  for (int ihalos = 0; ihalos < halo_grp->nhalos; ihalos++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalos];
    _ops_particle_set_exchange_border_zone(instance, halo);

    if (halo->sendBox != nullptr) {
      printf("Sending Box = [%e %e]x[%e %e] x[%e %e] SendingPoints = [%d %d]x[%d %d]x[%d %d]\n", halo->sendBox->getLocalMin().x,
             halo->sendBox->getLocalMax().x, halo->sendBox->getLocalMin().y,
             halo->sendBox->getLocalMax().y, halo->sendBox->getLocalMin().z,
             halo->sendBox->getLocalMax().z, halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3],
             halo->isend[4], halo->isend[5]);

    }
  }
}




  //TODO: Add sanity checks by allocation practically we are using particle_halos to shift only
  //data

  //Sanity 1: Same number of halo_particles
  //Sanity 2: Check for particle_to agrees to particle_from
  //Sanity 3: Check for particle_from agrees with particle_from
  //Sanity 4: Orient from is the same or who cares
  //Sanity 5: Oreint to is the same


void _ops_particle_setup_default_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp)
{

}

//TODO: Add the mark deletion to particle list:
//TODO: Need activation somewhere


void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp) {


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

        if (particle_from->particle_envelope != nullptr)
          if (((double *)particle_from->particle_envelope->data)[i] < 0) continue;

        ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                        (dim == 3) ? xlocal[dim * i + 2] : 0.0};
        bool decide = box->isCoordinateInBoundingBox(point);


        if (decide) {
          //mark particle for deletion now or shift it later
          info->nsend++;
          if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
            instance->ops_halo_buffer_size += (10  + info->nsend) * halo->nbites;
            instance->ops_halo_buffer = (char *)ops_realloc(instance->ops_halo_buffer , instance->ops_halo_buffer_size
                                           * sizeof(char));
          }

          particle_from->mark_deletion[i] = 2; //particle is exchanged
         _ops_particle_pack_halo_data(instance->ops_halo_buffer  + halo->nbites * (info->nsend - 1),
                                      halo->dat, halo->nhalos, i);
        }
      }
    }

    /* Unpack data */
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    int ifirst = particle_to->no_particles;

    particle_to->no_particles += nrecv;
    if (particle_to->no_particles > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles); //TODO: Add default number


    double *particle_crds = (double *)particle_to->particle_pos_dat->data;

    for (int i = 0; i < nrecv; i++) {
      int ielem = ifirst + i;

      _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + halo->nbites * i, halo->dat,
                                     halo->nhalos, ielem, halo->dir_to, halo->dir_from, halo->translate);

      //If particle within mark as actual
      ops_point xpoint{particle_crds[dim * ielem], particle_crds[dim * ielem + 1], (dim == 3) ? particle_crds[dim * ielem + 2] : 0.0};
      bool a1 =  particle_to->box_block->isCoordinateInBoundingBox(xpoint);
      particle_to->mark_deletion[ielem] = (a1) ? 0 : 1;

     //printf("Particle for deletion %d: %d\n", ielem, particle_to->mark_deletion[i]);
    }

    //Update the list initialization for inserted points

    for (int imap = 0; imap < particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      if (map->Nmax < particle_to->no_particles) {
        _ops_particle_realloc_map_data(map, particle_to->no_particles);
      }

      //Set memories for bin and ibin2 from previous owned to currently owned


 //     int ifirst = sizeof(int) * particle_to->no_particles - nrecv;
 //     int size = nrecv * sizeof(int);

    //    memset(map->bin->data + ifirst, -1, size);
    //    memset(map->parts_to_grid->data + ifirst, -1, size);
      int *bins = (int *)map->bin->data;
      int *part2grid = (int *)map->parts_to_grid->data;
      for (int i = ifirst; i < particle_to->no_particles; i++) {
        bins[i] = -1;
        part2grid[i] = -1;
      }



//      for (int iPart = 0; iPart < particle_to->no_particles; iPart++)/
 //printf("")

    }

 //   int *part2grid = (int *)particle_to->map_list[0]->parts_to_grid->data;
//  for (int i = 0; i < particle_to->no_particles; i++)
//      printf("Part2Grid[%d] = %d\n", i, part2grid[i]);

//    printf("nparticles after exchange %d\n", particle_to->no_particles);

  }

}

void _ops_particle_halo_exchange_transfer_map(OPS_instance *instance,
                                              ops_particle_halo_group halo_grp) {

  for (int ihalo = 0; ihalo < halo_grp-> nhalos; ihalo++) {
    ops_particle_halo  halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    BoundingBox *box = halo->sendBox;

    //Check for empty box
    if (box == nullptr) continue;

    info->nsend = 0;

    for (int i = 0; i < particle_from->no_particles; i++) {

      if (particle_from->particle_envelope != nullptr)
        if (((double *)particle_from->particle_envelope->data)[i] < 0) continue;

      if (particle_from->mark_deletion[i] != 1) continue;

      ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                      (dim == 3) ? xlocal[dim * i + 2] : 0.0};

      bool decide = box->isCoordinateInBoundingBox(point);

      if (decide) {
        info->nsend++;
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += (10  + info->nsend) * halo->nbites;
          instance->ops_halo_buffer = (char *)ops_realloc(instance->ops_halo_buffer , instance->ops_halo_buffer_size
                                         * sizeof(char));
        }

        particle_from->mark_deletion[i] = 2; // Exchanged occur
        _ops_particle_pack_halo_data(instance->ops_halo_buffer + halo->nbites * (info->nsend- 1),
                                     halo->dat, halo->nhalos, i);
      }
    }

    /* Unpack data */
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    int ifirst = particle_to->no_particles;

    particle_to->no_particles += nrecv;
    if (particle_to->no_particles > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles); //TODO: Add default number
    }

    double *particle_crds = (double *)particle_to->particle_pos_dat->data;

    for (int i = 0; i < nrecv; i++) {

      int ielem = ifirst + i;
      _ops_particle_unpack_halo_data(instance->ops_halo_buffer + halo->nbites * i,
                                     halo->dat, halo->nhalos, ielem, halo->dir_to,
                                     halo->dir_from, halo->translate);

      particle_to->mark_deletion[ielem] = 0;
    }

    //Mapping particles to grids & mark for removal
    for (int imap = 0; imap < particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      map->nParticles = particle_to->no_particles;
      if (particle_to->no_particles > map->Nmax) {
        _ops_particle_realloc_map_data(map, map->nParticles);
      }

      _ops_particle_map_from_exchange(map, particle_to, ifirst, map->nParticles);
    }


  }

}



/*-------------------------------------------------------------------------*/
/* Sequential border halo transfer                                         */
/*-------------------------------------------------------------------------*/

void _ops_particle_halo_border_transfer_vs(OPS_instance *instance,
                                           ops_particle_halo_group halo_grp) {

 // printf("Halo Grp has %d halos\n", halo_grp->nhalos);

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle from = halo->particle_from;
    size_t nlocal = from->no_particles;

    int *binhead = (int *)from->map_list[0]->binhead->data;
    int *bins = (int *)from->map_list[0]->bin->data;
    ops_dat binheads = from->map_list[0]->binhead;
    double *xpos = (double *)from->particle_pos_dat->data;

    info->nsend = 0;

/*
    printf("Virtual particles for halo_group %d: %d\n", ihalo,from->no_virtual);
    printf("Halo limits [%d %d]x[%d %d]\n", halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3]);
    for (int k = halo->isend[4]; k < halo->isend[5]; k++)
      for (int j = halo->isend[2]; j < halo->isend[3]; j++) {
        for (int i = halo->isend[0]; i < halo->isend[1]; i++) {
          int address = i + j * from->map_list[0]->binhead->size[0];
          printf("[%d %d %d] address  = %d binhead = %d\n",i, j, k, address, binhead[address]);
        }
      }
      */
/*      printf("Actual particles: %d and virtual = %d\n", from->no_particles, from->no_virtual);
      printf("Halo %d: Sending bins [%d %d]x[%d %d]x[%d %d] ", ihalo, halo->isend[0], halo->isend[1],
             halo->isend[2], halo->isend[3], halo->isend[4], halo->isend[5]);
      printf("\n");
      printf("Sending list: ");
*/
    for (int k = halo->isend[4]; k <  halo->isend[5]; k++) {
      for (int j = halo->isend[2]; j < halo->isend[3]; j++) {
        for (int i = halo->isend[0]; i < halo->isend[1]; i++) {

          int address =  i + binheads->size[0] * j
                      + k * binheads->size[0] * binheads->size[1];

          int iPart = binhead[address];

          while (iPart != -1) {
            if (halo_grp->with_virtual == OPS_WITH_VIRTUAL ||
                iPart  < nlocal) {
              info->nsend++;

              if (info->nsend > info->nmax) {
                info->nmax = info->nsend + OPS_MAX_PART;
                info->sendlist = (int *) ops_realloc(info->sendlist,
                                                     info->nmax * sizeof(int));
              }

              info->sendlist[info->nsend-1] = iPart;
            }

            iPart = bins[iPart];
          }
        }
      }
    }




    //Re-allocate buffer if neccessary
    if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size += (OPS_MAX_PART + info->nsend) * halo->nbites;
      instance->ops_halo_buffer = (char *)ops_realloc(instance->ops_halo_buffer,
                                                      instance->ops_halo_buffer_size *
                                                      sizeof(char));

    }

    /*
    printf("Halo: %d nsend = %d: ", ihalo, info->nsend);
    for (int i = 0; i < info->nsend; i++)
      printf("%d ",info->sendlist[i]);
    printf("\n");
   */
  }


  //Sending and receiving information
  for (int ihalo = 0; ihalo < halo_grp-> nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    for (int isend = 0; isend < info->nsend; isend++) {
      int i = info->sendlist[isend];
//      printf("")
      _ops_particle_pack_halo_data(instance->ops_halo_buffer + isend * halo->nbites,
                                   halo->dat, halo->nhalos, i);
    }

    //Part II: Unpack
    ops_particle to = halo->particle_to;

    info->firstrecv = to->no_particles + to->no_virtual;
    info->nrecv = info->nsend;

    if (info->firstrecv + info->nsend > to->Nmax) {
      ops_particle_realloc_data(to, info->firstrecv + info->nrecv);
    }

 //   printf("Halo %d: nrecv = %d\n", ihalo, info->nrecv);

    int ifirst = info->firstrecv;
    for (int irecv = 0; irecv < info->nrecv; irecv++) {
      int iloc = ifirst + irecv;
      _ops_particle_unpack_halo_data(instance->ops_halo_buffer + irecv * halo->nbites,
                                    halo->dat, halo->nhalos, iloc, halo->dir_to,
                                    halo->dir_from, halo->translate);
    }

    to->no_virtual += info->nrecv;

    //Part III: Generate maps
    for (int imap = 0; imap < to->particle_map_index; imap++) {
      ops_particle_mapping map = to->map_list[imap];
      if (map->mapping_type == OPS_WITH_VIRTUAL) {
        //realloc particles
        if (map->Nmax < ifirst + info->nrecv)
            _ops_particle_realloc_map_data(map,  ifirst + info->nrecv);
        _ops_particle_mapping_virtual_from_halo(map, to, ifirst, info->nrecv);

      }
    }
  }
}

void _ops_particle_halo_border_transfer(OPS_instance *instance,
                                        ops_particle_halo_group halo_grp) {

  //Construct sending
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo  halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    int imin = 0;
    int imax = (halo_grp->with_virtual != OPS_NO_VIRTUAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;

    BoundingBox *box = halo->sendBox;
    info->nsend = 0;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    for (int i = imin; i < imax; i++) {
      ops_point point{xlocal[dim * i], xlocal[dim * i + 1], (dim == 3) ?
                      xlocal[dim * i + 2] : 0.0};


      bool decide = box->isCoordinateInBoundingBox(point);
      if (decide) {
        info->nsend++;
        if (info->nsend > info->nmax) {
          info->nmax = info->nsend + OPS_MAX_PART;
          info->sendlist = (int *) ops_realloc(info->sendlist,
                                               info->nmax * sizeof(int));
        }

        info->sendlist[info->nsend - 1] = i;

        /* Reallocatr buffer as well */
        /* Ensure that buffer is large enough to receive data */
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += (OPS_MAX_PART + info->nsend) * halo->nbites;
          instance->ops_halo_buffer  = (char *)ops_realloc(instance->ops_halo_buffer,
                                                           instance->ops_halo_buffer_size
                                                           * sizeof(char));
        }

      }

    }

  }

  //Upgrading recv information
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];


    ops_particle particle_to = halo->particle_to;
    info->nrecv = info->nsend;

    //Compute location of first;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;
    particle_to->no_virtual += info->nrecv;
    //Realloc buffer lists -Only for MPI
  }

  //Perform pack exchange
  for (int ihalo = 0; ihalo < halo_grp-> nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    for (int isend = 0; isend < info->nsend; isend++) {
      int i = info->sendlist[isend];
      _ops_particle_pack_halo_data(instance->ops_halo_buffer  + isend * halo->nbites,
                                   halo->dat, halo->nhalos, i);
    }

    ops_particle particle_to = halo->particle_to;
    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles + particle_to->no_virtual);
    }


    //npack data
    for (int irecv = 0; irecv < info->nrecv; irecv++) {
        int ifirst = info->firstrecv;
        /* Realocation of particle data */
        int iloc = ifirst + irecv;
        _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + irecv * halo->nbites,
                                       halo->dat, halo->nhalos, iloc, halo->dir_to,
                                       halo->dir_from, halo->translate); //TODO: check

    }


  }

}

void _ops_particle_halo_border_pos_transfer(OPS_instance *instance,
                                            ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo  halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    int imin = 0;
    int imax = (halo_grp->with_virtual != OPS_NO_VIRTUAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;


    ops_dat position_from = halo->dat[0]->from;
    if (strcmp(position_from->name, particle_from->particle_pos_dat->name) != 0)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: Automatic position type requires "
                                           "position ops_dat structures to be the "
                                           "first element on the ops_particle_halo_data"
                                           " list");

    BoundingBox *box = halo->sendBox;
    info->nsend = 0;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    for (int i = imin; i < imax; i++) {
      ops_point point{xlocal[dim * i], xlocal[dim * i + 1], (dim == 3) ?
                      xlocal[dim * i + 2] : 0.0};

      bool decide = box->isCoordinateInBoundingBox(point);
      if (decide) {
        info->nsend++;
        if (info->nsend > info->nmax) {
          info->nmax = info->nsend + OPS_MAX_PART;
          info->sendlist = (int *) ops_realloc(info->sendlist,
                                               info->nmax * sizeof(int));
        }

        info->sendlist[info->nsend - 1] = i;

        /* Reallocatr buffer as well */
        /* Ensure that buffer is large enough to receive data */
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += (OPS_MAX_PART + info->nsend) * halo->nbites;
          instance->ops_halo_buffer  = (char *)ops_realloc(instance->ops_halo_buffer,
                                                           instance->ops_halo_buffer_size
                                                           * sizeof(char));
        }

      }
    }

  }

  //Upgrading recv information
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_to = halo->particle_to;
    info->nrecv = info->nsend;

    //Compute location of first;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;
    particle_to->no_virtual += info->nrecv;
    //Realloc buffer lists -Only for MPI
  }

  //Perform pack exchange
  for (int ihalo = 0; ihalo < halo_grp-> nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    int nsize = halo->dat[0]->to->elem_size;

    for (int isend = 0; isend < info->nsend; isend++) {
      int i = info->sendlist[isend];
      _ops_particle_pack_halo_data(instance->ops_halo_buffer  + isend * nsize,
                                   halo->dat, 1, i);
    }

    ops_particle particle_to = halo->particle_to;
    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles + particle_to->no_virtual);
    }


    //npack data
    for (int irecv = 0; irecv < info->nrecv; irecv++) {
        int ifirst = info->firstrecv;
        /* Realocation of particle data */
        int iloc = ifirst + irecv;
        _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + irecv * nsize,
                                       halo->dat, 1, iloc, halo->dir_to,
                                       halo->dir_from, halo->translate); //TODO: check

    }


  }



}


/* Exchange type halo  implementation*/
void _ops_particle_halo_border_transfer2(OPS_instance  *instance,
                                        ops_particle_halo_group halo_grp) {

//  printf("Entering into border transfer\n");

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo  halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;
    double *xlocal = (double *)particle_from->particle_pos_dat->data;

    int imin =  0;
    int imax = (halo_grp->with_virtual != OPS_NO_VIRTUAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;


    BoundingBox *box = halo->sendBox;
    info->nsend = 0;
    /* Pack all elements */
 //   printf("Sending Box [%f %f %f]x[%f %f %f]\n", box->getLocalMin().x, box->getLocalMin().y,
 //          box->getLocalMin().z, box->getLocalMax().x, box->getLocalMax().y, box->getLocalMax().z);
    for (int i = imin; i < imax; i++) {
      ops_point point{xlocal[dim * i], xlocal[dim * i + 1], (dim == 3) ?
                      xlocal[dim * i + 2] : 0.0};

 //      printf("Particle %d [%f %f %f]\n", i, point.x, point.y, point.z);

      bool decide = box->isCoordinateInBoundingBox(point);
      if (decide) {
 //       printf("Sending particle %d\n", i);
        info->nsend ++;
        if (info->nsend >= info->nmax) {
          info->nmax = info->nsend + OPS_MAX_PART;
          info->sendlist = (int *) ops_realloc(info->sendlist,
                                               info->nmax * sizeof(int));

        }
        info->sendlist[info->nsend - 1] = i;

 //       printf("Send[%d] = %d\n", info->nsend - 1, info->sendlist[info->nsend - 1]);
//        printf("Number of bites per particle: %d\n", halo->nbites);

        /* Ensure that buffer is large enough to receive data */
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += (OPS_MAX_PART + info->nsend) * halo->nbites;
          instance->ops_halo_buffer  = (char *)ops_realloc(instance->ops_halo_buffer,
                                                           instance->ops_halo_buffer_size
                                                           * sizeof(char));
        }


        //TODO: Add increase of decrease of elements

        //         _ops_particle_pack_halo_data(buff + halo->nbites * (info->nsend - 1),
        //halo->dat, halo->nhalos, i);


        _ops_particle_pack_halo_data(instance->ops_halo_buffer  + (info->nsend - 1) * halo->nbites,
                                     halo->dat, halo->nhalos, i); //TODO:
      }
    }

//    printf("Particles packed\n");
//    printf("Number of particles to be send in this set: %d\n",info->nsend);
//    printf("Sending: ");
//    for (int i = 0; i < info->nsend;i++)
//      printf("%d ", info->sendlist[i]);
//    printf("\n");

    /* Receive data */
    ops_particle particle_to = halo->particle_to;
    info->nrecv = info->nsend;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;

    int ifirst = particle_to->no_particles + particle_to->no_virtual;
    particle_to->no_virtual += info->nsend;

    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles + particle_to->no_virtual);

    for (int irecv = 0; irecv < info->nrecv; irecv++) {

      /* Realocation of particle data */
      int iloc = ifirst + irecv;
      _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + irecv * halo->nbites,
                                     halo->dat, halo->nhalos, iloc, halo->dir_to,
                                     halo->dir_from, halo->translate); //TODO: check

    }

  }

}

void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp)
{
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  ops_particle_halo_group main_halo_grp =
      (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

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
                                     halo->nhalos, ipart, halo_main->dir_to,
                                     halo_main->dir_from, halo_main->translate); //TODO: Check
    }
  }

}

void _ops_particle_halo_forward_map(OPS_instance   *instance,
                                    ops_particle_halo_group halo_grp) {
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  ops_particle_halo_group main_halo_grp =
      (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp : halo_grp->halo_master;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    int *sendlist = info->sendlist;

    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff = (char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }

    for (int i = 0; i < nsend; i++) {
      int ipart = sendlist[i];
      _ops_particle_pack_halo_data(buff + i * halo->nbites, halo->dat,
                                   halo->nhalos, ipart);
    }

    /* Unpacking data  */
    int istart = info->firstrecv;
    int nrecv = info->nrecv;
    int ilast = istart + nrecv;

    for (int i = 0; i < nrecv; i++) {
      int ipart = istart + i;
      _ops_particle_unpack_halo_data(buff + i * halo->nbites, halo->dat,
                                     halo->nhalos, ipart, halo_main->dir_to,
                                     halo_main->dir_from, halo_main->translate);
    }

    //Remapping particles to grid
    ops_particle to = halo_main->particle_to;
    for (int imap = 0; imap < to->particle_map_index; imap++) {
      ops_particle_mapping map = to->map_list[imap];
      _ops_particle_remap_virtual(map, to, istart, ilast);
    }

  }
}


void _ops_particle_halo_forward_pos_transfer(OPS_instance *instance,
                                             ops_particle_halo_group halo_grp) {

  char *buff = instance->ops_halo_buffer;

  for (int ihalo = 0; ihalo <  halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    /* Packing data */
    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    int *sendlist = info->sendlist;

    //Perform sanity check that first data structure reflects position
    ops_dat pos_from =  halo->dat[0]->from;
    ops_particle from = halo->particle_from;

    if (strcmp(pos_from->name, from->particle_pos_dat->name) != 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Particle position requires the ops_dat "
                         "of the particle position to be the first element of the ops_particle_halo_data "
                         "structure");

    int nsize = pos_from->elem_size;
    for (int i = 0; i < nsend; i++) {
      int ipart = sendlist[i];
      _ops_particle_pack_halo_data(buff + i * nsize, halo->dat,
                                   1, ipart);
    }


    /* Unpacking data */
    int istart = info->firstrecv;
    int nrecv = info->nrecv;
    for (int i = 0; i < nrecv; i++) {
      int ipart = istart + i;
      _ops_particle_unpack_halo_data(buff + i * nsize, halo->dat,
                                     1, ipart, halo->dir_to,
                                     halo->dir_from, halo->translate);
    }

  }
}



void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp) {

  char *buff = instance->ops_halo_buffer;

  ops_particle_halo_group main_halo_grp
    = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];



    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    int nsend = info->nrecv;
    int nbites_tot = nsend * halo->nbites;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff=(char *)ops_realloc(buff, instance->ops_halo_buffer_size);

    }

    int sendfirst = info->firstrecv;
    for (int  i = 0; i < nsend; i++) {
      int ipart = sendfirst + i;
      _ops_particle_pack_halo_data(buff + i * halo->nbites, halo->dat,
                                   halo->nhalos, ipart);
    }


    int nrecv = info->nsend;
    int *recvlist = info->sendlist;
    for (int i = 0; i < nrecv; i++) {
      int ipart = recvlist[i];
      _ops_particle_unpack_reverse_halo_data(buff + i * halo->nbites, halo->dat, halo->nhalos,
                                             ipart, halo_main->dir_from,
                                             halo_main->dir_to);
    }


  }
}

//TODO: Pass to
void  ops_particle_init_map(ops_particle_mapping map) {

  int size = 1;
  for (int i = 0; i < map->binhead->block->dims; i++)
    size *= map->binhead->size[i];

  memset(map->binhead->data, -1, sizeof(int) * size);

}

void ops_build_bounding_box(ops_particle particle ) {

  particle->box_block->partitionBoundingBox(particle->block);
}

void ops_particle_setup_intrablock_comms(ops_particle particle) {

}


/*-------------------------------------------------------------------------------------*
 *  Functions to output particle data structures to txt files
 *-------------------------------------------------------------------------------------*/

void ops_particle_print_data_to_txtfile(ops_particle particle, const char *file_name) {

  ops_get_data(particle->particle_pos_dat);

  if (particle->particle_envelope != nullptr) ops_get_data(particle->particle_envelope);

  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat  dat = particle->particle_dat[i];
    ops_get_data(dat);
  }

  ops_particle_print_data_to_txtfile_core(particle, file_name);
}


//TODO: Add a particle pointer within the ops_dat
void ops_particle_print_dat_to_txtfile(ops_dat dat, ops_particle particle,
                                       const char *file_name) {
  ops_get_data(dat);

  ops_particle_print_dat_to_txtfile_core(dat, particle, file_name);

}

bool ops_particle_global_rebuild(bool flag) {
  return flag;
}

void ops_mapping_def_core(ops_particle particle, ops_dat grid, ops_stencil stencil,
                          ops_with_virtual &include_virtual, int size[],
                          int base[], int d_m[], int d_p[]) {

  //Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structure");

  if (grid == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty grid ops_dat structure");

  ops_block block_particle = particle->block;
  ops_block block_grid = grid->block;
  if (strcmp(block_particle->name, block_grid->name) != 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-consistent blocks for grid and particle "
                                             "data structures.");

  int dim = block_particle->dims;
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i] + grid->d_m[i] - grid->d_p[i];
    base[i] = grid->base[i];
    d_m[i] = grid->d_m[i];
    d_p[i] = grid->d_p[i];
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

void ops_particle_halo_exchanges(ops_arg *args, int nargs, double *range_in) { }

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
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo must be setup before forward/backward"
                                                        "halo");

  if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD) {

    for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
      ops_particle_halo particle_halo = halo_grp->halo_list[ihalo];
      for (int i = 0; i < particle_halo->nhalos; i++) {
        ops_particle_halo_data halo_data = particle_halo->dat[i];
        if (halo_data->to->type_size != sizeof(double) ||
            halo_data->to->type_size != sizeof(float))
          throw OPSException(OPS_RUNTIME_ERROR," Reverse communications only for float or real"
                             "data structures");

        if (halo_data->orient == OPS_PART_POSITION) {
          throw OPSException(OPS_RUNTIME_ERROR, " Reverse communications only for orientation");

        }
      }
    }
  }
}

