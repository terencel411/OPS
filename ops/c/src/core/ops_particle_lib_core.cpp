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
  * @brief Functions for handling particles
  * @author Gihan Mudalige
  * @details Implements dummy functions from the MPI backend for the sequential
  * cpu backend (OpenMP and Sequential)
  */

#include <float.h>
#include <stdlib.h>
#include <iostream>

#include <ops_lib_core.h>
#include <ops_exceptions.h>
#include "ops_util.h"

static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)ops_calloc(len, sizeof(char));
  snprintf(dest, len, "%s", src);
  return dest;
}

/*----------------------------------------------------------------------*/
/*  This constructor creates a bounding box prior to partition. The call
 *  of the constructor after partition will lead to a decomposition of
 *  the domain based on the number of processes per direction. This option
 *  works for Lagrangian simulations and not Lagrangian-Eulerian
 *  simulations
 */
BoundingBox::BoundingBox(const ops_block block, int dim,
                         ops_point minCrd, ops_point maxCrd) :
    dim(dim), coords(nullptr){

  if (dim < 2 && dim > 3)
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                        "for 2D or 3D geometries\n");


  /* Sanity check for box */
  if (minCrd.x >= maxCrd.x)
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-positive size in x-direction");


  globalBoundingBox[0].x = minCrd.x;
  globalBoundingBox[1].x = maxCrd.x;

  if (minCrd.y >= maxCrd.y)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The size of the bounding box "
                       "in y-dir is non-positive");


  globalBoundingBox[0].y = minCrd.y;
  globalBoundingBox[1].y = maxCrd.y;

  if (dim == 3) {
    if (minCrd.z >= maxCrd.z)
      throw OPSException(OPS_INVALID_ARGUMENT, "Error: The size "
                         "of the bounding box in the z-dir is non"
                         " positive.");




    globalBoundingBox[0].z = minCrd.z;
    globalBoundingBox[1].z = maxCrd.z;
  }


  for (int i = 0 ; i < dim; i++)
    dx[i] = 0.0;
 // /* Copy local to global box */
 // ops_bounding_box_global_to_local(block, dim, globalBoundingBox, boundingBox);
 owned = false;
}

BoundingBox::BoundingBox(int dim) : dim(dim) , coords(nullptr) {
  if (dim < 2 && dim > 3)
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                        "for 2D or 3D geometries\n");

  owned = false;
}

BoundingBox::BoundingBox(const ops_dat coords, const double *Deltax,
                         int dim) : dim(dim) , coords(coords) {

  if (dim < 2 && dim > 3) {
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                       "for 2D or 3D geometries\n");
  }

  owned = false;

  for (int i = 0; i < dim; i++)
    dx[i] = Deltax[i];
  /* For generating an object with the given function requires domain
   * partition */

}

BoundingBox::BoundingBox(int dim, double *xmin, double *xmax) : dim{dim} , coords(nullptr) {

  if (dim < 2 && dim > 3)
    throw OPSException(OPS_RUNTIME_ERROR,"A bounding box is defined only for "
                       "2D and 3D spaces\n");

  /* Setting local bounding boxes */
  globalBoundingBox[0].x = xmin[0];
  globalBoundingBox[0].y = xmin[1];
  globalBoundingBox[0].z = (dim == 3) ? xmin[2] : 0.0;

  globalBoundingBox[1].x = xmax[0];
  globalBoundingBox[1].y = xmax[1];
  globalBoundingBox[1].z = (dim == 3) ? xmax[2] : 0.0;

  owned = false;

}

BoundingBox::~BoundingBox() { };

const ops_point BoundingBox::getLocalMin() const {
  return this->boundingBox[0];
}

const ops_point BoundingBox::getLocalMax() const {
  return this->boundingBox[1];
}

const ops_point BoundingBox::getGlobalMax() const {
  return this->globalBoundingBox[0];
}

const ops_point BoundingBox::getGlobalMin() const {
  return this->globalBoundingBox[1];
}

void BoundingBox::getLocalMaxMin(double *xmin, double* xmax) {
  if (!this->owned) return;

  xmin[0] = this->boundingBox[0].x;
  xmax[0] = this->boundingBox[1].x;

  xmin[1] = this->boundingBox[0].y;
  xmax[1] = this->boundingBox[1].y;

  if (dim == 3) {
    xmin[2] = this->boundingBox[0].z;
    xmax[2] = this->boundingBox[1].z;
  }
}


void BoundingBox::setBoundingBoxLocalBound(const ops_point &xl, const ops_point &xm) {

  if (xm.x <= xl.x || xm.y <= xl.y)
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: Defined bounding box of non-positive volume.");

  boundingBox[0].x = xl.x;
  boundingBox[0].y = xl.y;

  boundingBox[1].x = xm.x;
  boundingBox[1].y = xm.y;

  if (dim == 3) {
    if (xm.z <= xl.z)
      throw OPSException(OPS_INVALID_ARGUMENT,
                         "Error: Defined bounding box of non-positive volume.");


    boundingBox[0].z = xl.z;
    boundingBox[1].z = xm.z;
  }

  owned = true;
}
void BoundingBox::setBoundingBoxLocalBound(const double* xlow, const double* xmax) {

  if (xmax[0] <= xlow[0] || xmax[1] <= xlow[1])
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: Defined bounding box of non-positive volume.");

  boundingBox[0].x = xlow[0];
  boundingBox[0].y = xlow[1];


  boundingBox[1].x = xmax[0];
  boundingBox[1].y = xmax[1];

  if (dim == 3) {
    if (xmax[2] <= xlow[2])
      throw OPSException(OPS_INVALID_ARGUMENT,
                         "Error: Defined bounding box of non-positive volume.");

    boundingBox[0].z = xlow[2];
    boundingBox[1].z = xmax[2];
  }

  owned = true;
}

void BoundingBox::setBoundingBoxLocalBound(const double *boxregion) {

  if (boxregion[0] > boxregion[1] || boxregion[2] > boxregion[3])
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: Defined bounding box of non-positive volume.");


  boundingBox[0].x = boxregion[0];
  boundingBox[0].y = boxregion[2];

  boundingBox[1].x = boxregion[1];
  boundingBox[1].y = boxregion[3];

  if (dim == 3) {
    if (boxregion[4] > boxregion[5])
      throw OPSException(OPS_INVALID_ARGUMENT,
                         "Error: Defined bounding box of non-positive volume.");


    boundingBox[0].z = boxregion[4];
    boundingBox[1].z = boxregion[5];
  }

  owned =true;
}


void BoundingBox::setBoundingBoxGlobalBound(const ops_point &xlow, const ops_point &xmax) {
  if (xmax.x <= xlow.x || xmax.y <= xlow.y) {
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: Defined global bounding box of non-positive volume.");
  }
  globalBoundingBox[0].x = xlow.x;
  globalBoundingBox[0].y = xlow.y;


  globalBoundingBox[1].x = xmax.x;
  globalBoundingBox[1].y = xmax.y;

  if (dim == 3) {
    if (xmax.z <= xlow.z) {
      throw OPSException(OPS_INVALID_ARGUMENT,
                         "Error: Defined global bounding box of non-positive volume.");
    }
    globalBoundingBox[0].z = xlow.z;
    globalBoundingBox[1].z = xmax.z;
  }
}

void BoundingBox::setBoundingBoxGlobalBound(double* xlow,  double* xmax) {
  if (xmax[0] <= xlow[0] || xmax[1] <= xlow[1]) {
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: Defined global bounding box of non-positive volume.");
  }
  globalBoundingBox[0].x = xlow[0];
  globalBoundingBox[0].y = xlow[1];


  globalBoundingBox[1].x = xmax[0];
  globalBoundingBox[1].y = xmax[1];

  if (dim == 3) {
    if (xmax[2] <= xlow[2]) {
      throw OPSException(OPS_INVALID_ARGUMENT,
                         "Error: Defined bounding box of non-positive volume.");
    }
    globalBoundingBox[0].z = xlow[2];
    globalBoundingBox[1].z = xmax[2];
  }
}


/*-----------------------------------------------------------------*/
/* Function used to vrf that a particle is within the local box
 * NOTE: We need to ask for ownership
 */
bool BoundingBox::isCoordinateInBoundingBox(ops_point &point) {

  if (!owned) {
  //  printf("Box not owned\n");
    return false;
  }
  if (boundingBox[0].x > point.x || boundingBox[1].x <= point.x)
    return false;

  if (boundingBox[0].y > point.y || boundingBox[1].y <= point.y)
    return false;

  if (dim == 3) {
    if (boundingBox[0].z > point.z || boundingBox[1].z <= point.z)
      return false;
  }

  return true;
}

bool BoundingBox::isCoordinateInBoundingBox(const double *point) {
  if (!owned)
    return false;

  if (boundingBox[0].x > point[0] || boundingBox[1].x <= point[0])
    return false;

  if (boundingBox[0].y > point[1] || boundingBox[1].y <= point[1])
    return false;

  if (dim == 3) {
    if (boundingBox[0].z > point[2] || boundingBox[1].z <= point[2])
      return false;
  }

  return true;

}

/*-----------------------------------------------------------------------*/
/* Checks for particle within the BoundingBox (particle-can be marked for
 * block deletion)
 */
bool BoundingBox::isCoordinateInGlobalBoundingBox(const ops_point &point) {

  if (!owned)
    return false;

  if (globalBoundingBox[0].x >= point.x || globalBoundingBox[1].x <= point.x)
    return false;

  if (globalBoundingBox[0].y >= point.y || globalBoundingBox[1].y <= point.y)
    return false;

  if (dim == 3)
    if (globalBoundingBox[0].z >= point.z || globalBoundingBox[1].z <= point.z)
      return false;

  return true;
}

double BoundingBox::getMinCoordDir(int dir) {

  switch (dir) {
  case 0:
    return boundingBox[0].x;
    break;
  case 1:
    return boundingBox[0].y;
    break;
  case 2:
    return boundingBox[0].z;
    break;
  default:
    return -1;
  }

  return -1;
}

double BoundingBox::getGlobalMin(int idir) {
  switch (idir) {
  case 0:
    return globalBoundingBox[0].x;
    break;
  case 1:
    return globalBoundingBox[0].y;
    break;
  case 2:
    return globalBoundingBox[0].z;
    break;
  default:
    return -1;
  }

  return -1;
}

double BoundingBox::getGlobalMax(int idir) {
  switch (idir) {
  case 0:
    return globalBoundingBox[1].x;
    break;
  case 1:
    return globalBoundingBox[1].y;
    break;
  case 2:
    return globalBoundingBox[1].z;
    break;
  default:
    return -1;
  }

  return -1;
}

double BoundingBox::getMaxCoordDir(int dir) {
  switch (dir) {
  case 0:
    return boundingBox[1].x;
    break;
  case 1:
    return boundingBox[1].y;
    break;
  case 2:
    return boundingBox[1].z;
    break;
  default:
    return -1;
  }

  return -1;
}

/*-----------------------------------------------------------------------------*
 * Local functions
 *-----------------------------------------------------------------------------*/

/* Computes the intersection in a given direction
 *
 * return 1 for intersection or 0 when no overlap is found in the given
 *          direction
 */

static int _compute_intersection_region(double xbox1_low, double xbox1_hi,
                                        double xbox2_low, double xbox2_hi,
                                        double *xbox_int_low, double *xbox_int_hi) {

  //TODO: intersections
  double hA{0.5 * (xbox1_low + xbox1_hi)};
  double hB{0.5 * (xbox2_low + xbox2_hi)};
  double rA{0.5 * (xbox1_hi - xbox1_low)};
  double rB{0.5 * (xbox2_hi - xbox2_low)};

  if (fabs(hB - hA) <= rA + rB) {
    *xbox_int_low=MAX(xbox1_low, xbox2_low);
    *xbox_int_hi = MIN(xbox1_hi, xbox2_hi);

    double dx = *xbox_int_hi - *xbox_int_low;
    if (dx < 1.e-14) return 0; //no overlap
    //Check box sixe
    return 1;
  }

  return 0;
}


static int _check_box_intersection(const int dim, double *xbox1_lo, double *xbox1_hi,
                                   double *xbox2_lo, double* xbox2_hi) {
  int a1{1};

  int i{0};

  while (a1 == 1 && i < dim) {
    double hA = 0.5 * (xbox1_lo[i] + xbox1_hi[i]);
    double hB = 0.5 * (xbox2_lo[i] + xbox1_lo[i]);

    double rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    double rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);

    if (fabs(hB-hA) > rA + rB) a1 = 0;
  }

  return a1;
}

static void _ops_particle_export_data_point(FILE *fp, int i, ops_dat dat) {

  if (!dat->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR,
                       "Error: ops_dat not related to particle data\n");
  int dim = dat->dim;
  if (strcmp(dat->type, "double") == 0 ||
      strcmp(dat->type, "real(8)") == 0 ||
      strcmp(dat->type, "real(kind = 8)") == 0 ||
      strcmp(dat->type, "double precision") == 0) {
    for (int isou = 0; isou < dim; isou++)
      if(fprintf(fp, "%16.10e ", ((double *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file");
  }
  else if (strcmp(dat->type, "float") == 0 ||
           strcmp(dat->type, "real") == 0 ||
           strcmp(dat->type, "real(4)") == 0 ||
           strcmp(dat->type, "real(kind=4)") == 0) {
    for (int isou = 0; isou < dim; isou++)
      if (fprintf(fp, "%16.10e ", ((float *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file");
  }
  else if (strcmp(dat->type, "int") == 0 ||
           strcmp(dat->type, "int(4)") == 0 ||
           strcmp(dat->type, "integer") == 0 ||
           strcmp(dat->type, "integer(4)") == 0 ||
           strcmp(dat->type, "integer(kind=4)") == 0 /* ||
           strcmp(dat->type, "long") == 0 ||
           strcmp(dat->type, "long long") == 0 ||
           st rcmp(dat->type, "ll") == 0 ||
           strcmp(dat->type, "short") == 0 ||
           strcmp(dat->type, "char") == 0*/) {
    for (int isou = 0; isou < dim; isou++)
      if (fprintf(fp, "%16d ", ((int *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file");
  }
  else {
   OPSException ex(OPS_RUNTIME_ERROR);
   ex<<"Error: This type " << dat->type <<" is not supported for output of particle data to txt files";
   throw ex;
  }

}

/*-------------------------------------------------------------------------------------*/
/*  Computes the grid cell on which particle is projected                              *
/--------------------------------------------------------------------------------------*/

int _ops_coord_to_bin(const int dim, const ops_point xmin,const  ops_point xmax,
                      const double *dx, const int *Ngrid, const double *xp) {
  int ix{-1}, iy{-1}, iz{-1};


  double epsilon = 1.e-12;
  int within = 0;
  double inv_dx = 1. / dx[0];
  double inv_dy = 1. / dx[1];


  if ((xp[0] >= xmin.x - epsilon) && (xp[0] <= xmax.x + epsilon)) {
    ix = (int) floor((xp[0] - xmin.x) * inv_dx);
    within = 1;
  }

  if ((ix < 0) && within) {
    ix = (int) floor((xp[0] - xmin.x) * inv_dx + epsilon);
  }



  within = 0;
  if ((xp[1] >= xmin.y -epsilon) && (xp[1] <= xmax.y + epsilon )) {
    within = 1;
    iy = (int) floor((xp[1] - xmin.y) * inv_dy);
  }

  if (iy < 0 && within) {
    iy = (int) floor((xp[1] - xmin.y) * inv_dy + epsilon);
  }

  if (dim == 3) {
    double inv_dz = 1. / dx[2];
    within = 0;
    if ((xp[2] >= xmin.z - epsilon) && (xp[2] <= xmax.z + epsilon)) {
      iz =(int) floor((xp[2] - xmin.z) * inv_dz);
      within = 1;
    }
    if (iz == -1 && within)
      iz = (int) floor((xp[2] - xmin.z) * inv_dz + epsilon);
  }
  else
    iz = 0;

  if (ix == -1 || iy == -1 || iz == -1)
    return -1;

  return (dim == 2) ? ix + iy * Ngrid[0] : ix + iy * Ngrid[0] + iz * Ngrid[0] * Ngrid[1];

}

//Temporary stored herein
int _ops_particle_check_for_deletion(int ipart, int bin_part[], int dim, int rmv_limits[],
                                     double *xpos, BoundingBox *box) {

  int check = 0;


  for (int i = 0; i < dim; i++) {
    if ( bin_part[i] < rmv_limits[2 *i] ||
         bin_part[i] > rmv_limits[2 * i + 1]) {
      check = 1; break;
    }
  }



  if (check == 1) return 1;

  ops_point xpoint;
  xpoint.x = xpos[0];
  xpoint.y = xpos[1];
  xpoint.z = (dim == 3) ? xpos[2] : 0.0;

  bool flag = box->isCoordinateInBoundingBox(xpoint);

  return (int) (!flag);
}

bool _ops_particle_moved_to_exchange_zone(int bin_old[], int bin_new[], int rmv_limits[],
                                          int dim) {

  // Check if old point is in the crtical points for rebuilding the list
  for (int i = 0; i < dim; i++) {
    if ( bin_old[i] == bin_new[i]) continue; //No-need to check this direction remains fixed

    //Swap : 0 Check negative direction
    if ( ((bin_old[i] >= rmv_limits[2 *i]) && (bin_new[i] < rmv_limits[2 * i]))
        || ((bin_old[i] < rmv_limits[2 * i]) && (bin_new[i] >= rmv_limits[2 * i])) )
        return true;



    //Swap 1: Check positive direction
    if (  ((bin_old[i] >= rmv_limits[2 * i + 1]) && (bin_new[i] < rmv_limits[2 * i + 1]))
        ||((bin_old[i] < rmv_limits[2 * i + 1]) && (bin_new[i] >= rmv_limits[2 *i + 1])))
      return true;

  }

  return false;
}


/******************************************************************************
 * Bounding Box API functions
 *****************************************************************************/

BoundingBox* ops_create_bounding_box(int dim) {
  BoundingBox* box = new BoundingBox{dim}; //TODO: ops_malloc



  return box;
}

BoundingBox* ops_create_bounding_box(const ops_block block, int dim,
                                    ops_point &point_low, ops_point &point_max) {
  BoundingBox* box = new BoundingBox{block, dim, point_low, point_max}; //todo: ops_malloc add to block list

  return box;
}

BoundingBox* ops_create_bounding_box(const ops_block block, const ops_dat crds,
                                     int dim, double *dx) {



  if (strcmp(block->name, crds->block->name) != 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "The block associated with the "
                       "ops_dat structure not linked to the input block");
  if (block->dims != crds->dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Number of data per grid point "
        "differ from the size of physical space.");

  if (block->dims != dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Input size of physical space "
                                             "differs from the size of the physical space as given in block");

  BoundingBox *box = new BoundingBox{crds, dx, dim};

  return box;
}


/*-----------------------------------------------------------------------------*/
/* Set local and global bounding box based on
 */

void ops_set_bounding_box_local(BoundingBox* box, ops_block block,
                                const ops_point &xlow, const ops_point &xmax) {

  box->setBoundingBoxLocalBound(xlow, xmax);

  double x_min[3], x_max[3], xglb_min[3], xglb_max[3];
  x_min[0] = xlow.x;
  x_min[1] = xlow.y;
  x_max[0] = xmax.x;
  x_max[1] = xmax.y;
  if (box->getDim() == 3) {
    x_min[2] = xlow.z;
    x_max[2] = xmax.z;
  }
  bool owned = ops_get_bounding_box_local_to_global(block, x_min, x_max, xglb_min, xglb_max);

  box->setBoundingBoxGlobalBound(xglb_min, xglb_max);
  box->setOwnership(owned);
}

void ops_set_bounding_box_local(BoundingBox* box, ops_block block,
                                double* xlow, double* xmax) {
  box->setBoundingBoxLocalBound(xlow, xmax);
  double xglb_min[3], xglb_max[3];
  bool owned = ops_get_bounding_box_local_to_global(block, xlow, xmax, xglb_min, xglb_max);

  box->setBoundingBoxGlobalBound(xglb_min, xglb_max);
  box->setOwnership(owned);
}

/*-----------------------------------------------------------------------------*/
/* Set global Bounding Box but not redefine the local bound
 *-----------------------------------------------------------------------------*/

void ops_set_bounding_box_global(BoundingBox* box, ops_block block, const ops_point &xlow,
                                 const ops_point &xmax)
{
  box->setBoundingBoxGlobalBound(xlow, xmax);
}

void ops_set_bounding_box_global(BoundingBox* box, ops_block block, double*  xlow,
                                 double* xmax)
{
  box->setBoundingBoxGlobalBound(xlow, xmax);
}

/*---------------------------------------------------------------------------------*/
/*! Defines the geometry of the BoundingBox for a given block                      *
 *---------------------------------------------------------------------------------*/
void ops_set_bounding_box_from_dat(BoundingBox* box,const ops_dat coords, double *dx,
                                   int dim) {
  if (box == NULL) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty BoundingBox structure");
  }

  if (dim < 2 && dim > 3) {
    throw OPSException(OPS_INVALID_ARGUMENT, "A bounding box is defined only "
                                             "for 2D and 3D spaces\n");
  }

  /* For generating an object with the given function requires domain
   * partition
   */

  if (!ops_partitioned()) {
    throw OPSException(OPS_RUNTIME_ERROR, "Bounding box cannot be defined "
                       "by a ops_dat structure before partition");
  }

  double xmin[3]  ,xmax[3];

  for (int i = 0; i < 3; i++) {
    xmin[i] = INFINITY_double;
    xmax[i] = -INFINITY_double;
  }

  _ops_construct_local_box_from_dat(coords, dx, dim, xmin, xmax);

  box->setBoundingBoxLocalBound(xmin, xmax);

  double xglb_max[3], xglb_min[3];
  bool owned = ops_get_bounding_box_local_to_global(coords->block, xmin, xmax,
                                                    xglb_min, xglb_max);

  box->setBoundingBoxGlobalBound(xglb_min, xglb_max);
  box->setOwnership(owned);
}



BoundingBox* ops_find_send_box_projection(int idef,int dim,double *xbox1_low,
                                          double * xbox1_hi, double *xbox2_min,
                                          double *xbox2_max) {


  double xinter = 0.5 * (xbox2_min[idef] + xbox2_max[idef]);

  if (xinter < xbox1_low[idef] || xinter > xbox1_hi[idef])
    return nullptr;

  double xproj_low[dim], xproj_hi[dim];

  xproj_low[idef] = xbox1_low[idef];
  xproj_hi[idef] = xbox1_hi[idef];

  /* Find intersection regions */
  int imin = (idef == 0) ? 1 : 0;
  int imax = (idef == dim - 1) ? dim - 1 : dim;

  int a1{1};
  for (int i = imin; i < imax; i++) {
    if (i != idef)
     a1 = _compute_intersection_region(xbox1_low[i], xbox1_hi[i], xbox2_min[i],
                                       xbox2_max[i], xproj_low + i, xproj_hi + i);
    if (a1 == 0) break;
  }

  if (a1 == 0) return nullptr;

  BoundingBox *box = new BoundingBox(dim);



  box->setBoundingBoxLocalBound(xproj_low, xproj_hi);

  return box;


}

/*-------------------------------------------------------------------------------------*/
/* Search for overlapping between the region of a bounding box and a user defined
 * region given as [xmin xmax ymin ymax zmin zmax]. The intersection is stored in
 * intersection as [xmin xmax ymin ymax zmin zmax]
 *
 * return 0: simulation region equivalent to size
 *        1: overlap equal to region owned by box
 *        2: no overlap found
 */
/*-------------------------------------------------------------------------------------*/

BoundingBox* ops_find_intersection_region(BoundingBox *box, double *region, int &a1) {

  int dim = box->getDim();
  a1 = 0;

  ops_point xmin = box->getLocalMin();
  ops_point xmax = box->getLocalMax();

  double xbox_hi[dim], xbox_lo[dim];
  double xreg_lo[dim], xreg_hi[dim];

  /* Get elements TODO: Shift into also into points */
  xbox_hi[0] = xmax.x; xbox_hi[1] = xmax.y;
  xbox_lo[0] = xmin.x; xbox_lo[1] =xmin.y;


  if (dim == 3) {
    xbox_hi[2] = xmax.z; xbox_lo[2] = xmin.z;
  }

  for (int i = 0; i < dim; i++) {
    xreg_lo[i] = region[2 * i];
    xreg_hi[i] = region[2 * i + 1];
  }

//  printf("Checking boxes [%f %f %f]x[%f %f %f] and [%f %f %f]x[%f %f %f]\n",
//         xreg_lo[0], xreg_lo[1], xreg_lo[2], xreg_hi[0], xreg_hi[1], xreg_hi[2],
//         xbox_lo[0], xbox_lo[1], xbox_lo[2], xbox_hi[0], xbox_hi[1], xbox_hi[2]);


  double inters[2 * dim];
  for (int i = 0; i < dim; i++) {
    inters[2 * i] = xbox_lo[i];
    inters[2 * i + 1] = xbox_hi[i];
  }
  int overlp{0};
  for (int i = 0; i < dim; i++) {
    int overlp = _compute_intersection_region(xbox_lo[i], xbox_hi[i], xreg_lo[i],
                                              xreg_hi[i],  inters + 2 * i,
                                              inters + 2 * i + 1);

    if (overlp == 0) {
      a1 = 2;
      return nullptr;
    } //return immediately no-overlapping
  }
  //Check that box bound not equal to  intersection region


  BoundingBox* intersection = new BoundingBox(dim);
  intersection->setBoundingBoxLocalBound(inters);

 /* printf("Intersection Box [%f %f %f]x[%f %f %f] (%d)\n", intersection->getLocalMin().x,
         intersection->getLocalMin().y, intersection->getLocalMin().z,
         intersection->getLocalMax().x, intersection->getLocalMax().y,
         intersection->getLocalMax().z, overlp); */

  int iel =0;
  for (int i = 0; i < dim;i++)  {
    iel = 0;
    iel +=  (xbox_lo[i] == intersection->getMinCoordDir(i)) ? 0 : 1;
    iel +=  (xbox_hi[i] == intersection->getMaxCoordDir(i)) ? 0 : 1;
    //printf("iel = %d\n",iel);
    if (iel > 0) {a1 = 1; return intersection; }

  }

  a1 = 0;
  return intersection;

}

/*-------------------------------------------------------------------------------*/

int ops_check_box_intersection(int dim, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double* xbox2_hi) {
  int a1{1};

  int i{0};

  while (a1 == 1 && i < dim) {
    double hA = 0.5 * (xbox1_lo[i] + xbox1_hi[i]);
    double hB = 0.5 * (xbox2_lo[i] + xbox2_hi[i]);

    double rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    double rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);

    if (fabs(hB-hA) > rA + rB) a1 = 0;

    i++;
  }

  return a1;
}

int ops_check_box_intersection(int dim, int idir, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double *xbox2_hi) {
  int a1 = 1;

  for (int i = 0; i < dim; i++) {

    double hA = 0.5 *  (xbox1_lo[i] + xbox1_hi[i]);
    double hB = 0.5 * (xbox2_lo[i] + xbox2_hi[i]);

    double rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    double rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);

    if (ops_get_proc()==3) {
      printf("Dir = %d (crit %d) |hb-ha| - (ra + rb) = %12.9e\n", i, idir, fabs(hA-hB)- (rA + rB));
    }
    if (i == idir) {
      if (fabs(hB-hA) > rA + rB) return 0;
    }
    else
      if (fabs(hB-hA) >= rA + rB) return 0;




  }

  return 1;
}


//TODO: Remove
int ops_check_box_intersections(int dim, double *xbox1_lo,
                                double *xbox1_hi, double *xbox2_lo,
                                double *xbox2_hi) {

  double eps =1.e-9;

  int touches[3] = {0, 0, 0};
  //X
  double hA = 0.5 * (xbox1_lo[0] + xbox1_hi[0]);
  double hB = 0.5 * (xbox2_hi[0] + xbox2_lo[0]);
  double dx = fabs(hB-hA);
  double RAx = 0.5 * (xbox1_hi[0] - xbox1_lo[0]);
  double RBx = 0.5 * (xbox2_hi[0] - xbox2_lo[0]);
  if (dx > RAx + RBx + eps) return 0; //no intersection
  touches[0] = (fabs(dx - RAx - RBx ) <= eps);

  hA = 0.5 * (xbox1_lo[1] + xbox1_hi[1]);
  hB = 0.5 * (xbox2_hi[1] + xbox2_lo[1]);
  double dy = fabs(hB-hA);
  double RAy = 0.5 * (xbox1_hi[1] - xbox1_lo[1]);
  double RBy = 0.5 * (xbox2_hi[1] - xbox2_lo[1]);

  if (dy > RAy + RBy + eps) return 0;
  touches[1] = (fabs(dy - RAy - RBy) <= eps);

  if (dim == 3) {
    hA = 0.5 * (xbox1_lo[2] + xbox1_hi[2]);
    hB = 0.5 * (xbox2_hi[2] + xbox2_lo[2]);
    double dz = fabs(hB-hA);
    double RAz = 0.5 * (xbox1_hi[2] - xbox1_lo[2]);
    double RBz = 0.5 * (xbox2_hi[2] - xbox2_lo[2]);
    if (dz > RAz + RBz + eps) return 0;
    touches[2] = (fabs(dz - RAz - RBz) <= eps);
  }

  int num_touches = 0;
  for (int i = 0; i < dim; i++)
    if (touches[i]) num_touches++;

  if (num_touches == dim) return 0;

  if (num_touches == 2 && dim == 3) return 0;

  return 1;



}

static inline int feq(double a, double b) {
    const double abs_eps = 1e-9;
    const double rel_eps = 1e-9;

    double diff = fabs(a - b);
    if (diff <= abs_eps)
        return 1;

    return diff <= fmax(fabs(a), fabs(b)) * rel_eps;
}

static inline int fgt(double a, double b) {
    // a > b with tolerance
    const double abs_eps = 1e-9;
    return (a - b) > abs_eps;
}

static inline int fge(double a, double b) {
    return fgt(a, b) || feq(a, b);
}

int ops_check_box_intersections2(int dim,
                                 double *xbox1_lo, double *xbox1_hi,
                                 double *xbox2_lo, double *xbox2_hi)
{
    int touches[3] = {0, 0, 0};

    // -------------------------------
    // X-axis
    // -------------------------------
    double hA = 0.5 * (xbox1_lo[0] + xbox1_hi[0]);
    double hB = 0.5 * (xbox2_lo[0] + xbox2_hi[0]);

    double dx  = fabs(hA - hB);
    double RAx = 0.5 * (xbox1_hi[0] - xbox1_lo[0]);
    double RBx = 0.5 * (xbox2_hi[0] - xbox2_lo[0]);
    double sumX = RAx + RBx;

    // No intersection if distance exceeds combined half widths
    if (fgt(dx, sumX)) return 0;

    // Touch if exactly equal within tolerance
    touches[0] = feq(dx, sumX);

    // -------------------------------
    // Y-axis
    // -------------------------------
    hA = 0.5 * (xbox1_lo[1] + xbox1_hi[1]);
    hB = 0.5 * (xbox2_lo[1] + xbox2_hi[1]);

    double dy  = fabs(hA - hB);
    double RAy = 0.5 * (xbox1_hi[1] - xbox1_lo[1]);
    double RBy = 0.5 * (xbox2_hi[1] - xbox2_lo[1]);
    double sumY = RAy + RBy;

    if (fgt(dy, sumY)) return 0;
    touches[1] = feq(dy, sumY);

    // -------------------------------
    // Z-axis (if 3D)
    // -------------------------------
    if (dim == 3) {
        hA = 0.5 * (xbox1_lo[2] + xbox1_hi[2]);
        hB = 0.5 * (xbox2_lo[2] + xbox2_hi[2]);

        double dz  = fabs(hA - hB);
        double RAz = 0.5 * (xbox1_hi[2] - xbox1_lo[2]);
        double RBz = 0.5 * (xbox2_hi[2] - xbox2_lo[2]);
        double sumZ = RAz + RBz;

        if (fgt(dz, sumZ)) return 0;
        touches[2] = feq(dz, sumZ);
    }

    // -------------------------------
    // Count how many axes are exact touches
    // -------------------------------
    int num_touches = 0;
    for (int i = 0; i < dim; i++)
        if (touches[i]) num_touches++;

    // Case: touching on all axes → corner or line contact but no overlap
    if (num_touches == dim) return 0;

    // Case: 3D edge-only touching (touch on 2 axes)
    if (num_touches == 2 && dim == 3) return 0;

    // Otherwise boxes overlap
    return 1;
}

int ops_check_dir_intersection(double xreg1_min, double xreg1_max,
                           double xreg2_min, double xreg2_max) {

  double ha = .5 * (xreg1_min + xreg1_max);
  double hb = .5 * (xreg2_max + xreg2_min);

  double r= 0.5 * (xreg1_max - xreg1_min + xreg2_max - xreg1_min);

  if (fabs(hb - ha) <= r) return 1;
  return 0;
}



int ops_check_box_intersection2(int dim, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double* xbox2_hi) {
  int a1{1};

  int i{0};

  while (a1 == 1 && i < dim) {
    double hA = 0.5 * (xbox1_lo[i] + xbox1_hi[i]);
    double hB = 0.5 * (xbox2_lo[i] + xbox2_hi[i]);

    double rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    double rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);
    double d = fabs(hB - hA);

    if (d -  (rA + rB) > 0) a1 = 0;

    i++;
  }

  return a1;
}

/*******************************************************************************/
/* Particle hanlding functions                                                 *
 ******************************************************************************/

/*-----------------------------------------------------------------------------*
 * ops_arg for particle data structures                                        *
 *-----------------------------------------------------------------------------*/

ops_arg ops_arg_dat_particle(ops_dat dat, int dim, char const *type,
                             ops_particle particle, ops_particle_mapping map,
                             ops_access acc, ops_stencil stencil) {
  (void) type;
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  (&temp)->argtype = OPS_ARG_DAT_PARTICLE;

  (&temp)->part_index = particle->index;
  (&temp)->map_index = map->index; //TODO: Shift into element

  return temp;
}

/*------------------------------------------------------------------------------
 * Function declares and defines a particle data list                           *
 *------------------------------------------------------------------------------*/
ops_particle  _ops_decl_particle(OPS_instance *instance, ops_block block,
                                 BoundingBox *box, char const* name) {

  /* Create ops_particle */
  ops_particle particle = new  ops_particle_core;//(ops_particle)ops_calloc(1, sizeof(ops_particle_core));

  /* Set default values */

  particle->no_particles = 0;
  particle->no_virtual = 0;
  particle->global_particles = particle->no_particles;
  particle->block = block;
  particle->box_block = box;
  particle->Nmax = OPS_MAX_PART;
  particle->name = copy_str(name);

  particle->particle_envelope = nullptr;
  particle->particle_dat_index = 0;
  particle->particle_dat_max = 10;
  particle->particle_dat = (ops_dat *)ops_malloc(sizeof(ops_dat) * particle->particle_dat_max);

  particle->particle_map_index = 0;
  particle->particle_map_max = 10;
  particle->map_list = (ops_particle_mapping *)ops_malloc(sizeof(ops_particle_mapping) *
                                                         particle->particle_map_max);

  particle->mark_deletion  =(int *) ops_malloc(sizeof(int) * particle->Nmax);

  particle->particle_pos_dat = nullptr;
  /* Assign particle to the correct list */

  instance->OPS_block_list[block->index].no_particle_structures++;
  instance->OPS_block_list[block->index].particle
     = (ops_particle_core **)ops_realloc(instance->OPS_block_list[block->index].particle,
                                         instance->OPS_block_list[block->index].no_particle_structures
                                              * sizeof(ops_particle_core *));


  int index = instance->OPS_block_list[block->index].no_particle_structures - 1;
  instance->OPS_block_list[block->index].particle[index] = particle; //TODO: single particle
  particle->index = index;
  return particle;
}

/*---------------------------------------------------------------------------
 * Function deletes particle list                                             *
 *----------------------------------------------------------------------------*/
ops_particle _ops_free_particle(ops_particle particle) {

  if (particle == NULL)
    return NULL;

  /* Clear vector list */
  particle->no_particles = 0;
  particle->global_particles = 0;
  particle->Nmax = 0;

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    ops_free(map);
  }
  ops_free(particle->map_list);



  ops_free(particle->mark_deletion);
  ops_free((char *)particle->name);
  ops_free(particle->particle_dat);
//  if (particle->box_block != NULL)
//    ops_free(particle->box_block);

  delete particle->box_block;
  delete particle;
  return NULL;
}

void ops_particle_realloc_list( ops_particle particle) {
  if (particle->particle_dat_index > particle->particle_dat_max) {
    particle->particle_dat = (ops_dat *) ops_realloc(particle->particle_dat,
                                                     (particle->particle_dat_index + 10)
                                                      *sizeof(ops_dat));
    particle->particle_dat_max = particle->particle_dat_index + 10;
  }
}

/*---------------------------------------------------------------------------------------*
 * Function for re-allocating ops_dat structures associated with     particles           *
 *                                                                                       *
 * \param[in]   noalloc      smallest size if Nmax                                       *
 *                                                                                       *
 *---------------------------------------------------------------------------------------*/
void ops_particle_realloc_data(ops_particle particle, int noalloc) {

  if (particle->Nmax > noalloc)
    throw OPSException(OPS_INVALID_ARGUMENT, "Number of requested allocated particles "
          "is smaller than maximum allocated particles");

  /* Reallocate the ops_dat structures */
  particle->Nmax = noalloc + OPS_MAX_PART;


  particle->mark_deletion = (int *) ops_realloc(particle->mark_deletion,
                                                particle->Nmax * sizeof(int));
  /* Reallocate ops_dat structures */
 // particle->particle_data[0] =
      ops_dat_realloc_core(particle->particle_pos_dat, particle->Nmax);
  //TODO: Add shape directly

  if (particle->particle_envelope != NULL) {
       ops_dat_realloc_core(particle->particle_envelope, particle->Nmax);
  }
  //for (ops_dat &dat : particle->particle_data) {
   for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat = particle->particle_dat[i];
    ops_dat_realloc_core(dat, particle->Nmax);
  }

//   printf("Particle reallocation (Nmax %d)\n", particle->Nmax);

}


int _ops_particle_owned_dat(ops_particle particle, ops_dat dat) {

  if (particle == NULL)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_particle structure");

  if (dat == NULL)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_dat structure");

  if (strcmp(particle->particle_pos_dat->name, dat->name) == 0)
    return 1;

  if (particle->particle_envelope != NULL) {
    if (strcmp(particle->particle_envelope->name, dat->name) ==0)
      return 1;
  }

  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat_elem = particle->particle_dat[i];
    if (strcmp(dat_elem->name, dat->name) == 0)
      return 1;
  }

  return 0;
}


void ops_exit_particles(OPS_instance *instance) {

  /* Get block descriptor */
  ops_block_descriptor *block_list = instance->OPS_block_list;

  for (int i = 0; i < instance->OPS_block_index; i++) {
    for (int index = 0; index < block_list[i].no_particle_structures;index++) {
      block_list[i].particle[index] =
          _ops_free_particle( block_list[i].particle[index]);
    }


    ops_free(block_list[i].particle);
  }


}
/*******************************************************************************/
/*  Particle API  Functions
 *******************************************************************************/

ops_particle ops_decl_particle(ops_block block, char const* name, BoundingBox *Box) {
   return _ops_decl_particle(OPS_instance::getOPSInstance(), block, Box, name);
}


/*-------------------------------------------------------------------------------*/
/* \brief Marking particles for deletion and forward exchange                    *
 *
 * \param[in]  particle an ops_particle structure
 */
/*-------------------------------------------------------------------------------*/

void ops_particle_mark_for_del(ops_particle particle) {

  if (particle == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_particle structure");

  BoundingBox *box = particle->box_block;
  if (box == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty BoundingBox structure");


  int dim = particle->block->dims;

  long int no_particles = (long int)particle->no_particles;
  double *xpos = (double *) particle->particle_pos_dat->data;
  for (long int i = 0; i < no_particles; i++) {
    particle->mark_deletion[i] = 0; //TODO: Check for removing

    ops_point xlocal{xpos[dim * i], xpos[dim * i + 1],
                     (dim == 3) ? xpos[dim *i + 2] : 0.0};
    bool is_in = box->isCoordinateInBoundingBox(xlocal);
    if (!is_in) particle->mark_deletion[i] = 1; //particle marked for deletion and exchange
  }

}


/* Removing marked particles from the system */


void ops_particle_remove_marked(ops_particle particle) {

  if (particle->no_particles == 0)
    return;

  int dim = particle->block->dims;
  int Nlocal = particle->no_particles;

  for (int i =0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] > 0) {
      if (i == Nlocal-1) {
        Nlocal--;
        break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      //swap deletion as well
      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      Nlocal--;
      if (Nlocal == 0) break;

    }
  }


  particle->no_particles = Nlocal;
}


void ops_particle_remove_marked_flag(ops_particle particle, int flag) {

  if (particle->no_particles == 0)
    return;

  int dim = particle->block->dims;
  int Nlocal = particle->no_particles;

  for (int i =0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] == flag) {
      if (i == Nlocal-1) {
        Nlocal--;
        break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      //swap deletion as well
      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      Nlocal--;
      if (Nlocal == 0) break;

    }
  }


  particle->no_particles = Nlocal;

}

void ops_particle_remove_marked_with_maps(ops_particle particle ) {
  if (particle->no_particles == 0)
    return;

  int dim = particle->block->dims;
  int Nlocal = particle->no_particles;

  for (int i = 0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] > 0) {

      if (i == Nlocal-1) {
        Nlocal--; break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal - 1, dat->elem_size);
      }


      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));


      //Update maps


      for (int imaps = 0; imaps < particle->particle_map_index; imaps++) {
        ops_particle_mapping map = particle->map_list[imaps];
   //     printf("Nlocal = %d nmax = %d\n", Nlocal, particle->Nmax);
   //     printf("iPart = %d, Nlocal - 1: %d\n", i, Nlocal - 1);
        _ops_particle_copy_mapping_data_to(map, i, Nlocal - 1);
      }

      Nlocal--;
      if (Nlocal == 0) break;

    }

  }

  particle->no_particles = Nlocal;
}


//TODO: Need further testing
void  ops_particle_reset_virtual_particles(ops_particle particle) {

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->mapping_type != OPS_WITH_VIRTUAL) continue;

    int *binhead = (int *)map->binhead->data;
    int *bins = (int *)map->bin->data;
    int *bin2grid = (int *)map->parts_to_grid->data;



    for (int  i = particle->no_particles;
              i < particle->no_particles + particle->no_virtual; i++) {

      int address = bin2grid[i];

    //  printf("Entered here for %d\n", i);
      if (address > -1) binhead[address] = -1; //bins[i];

      bins[i] = -1;
      bin2grid[i] = -1;
    }

  }



  particle->no_virtual = 0;
}



void  ops_particle_rearrange_particles_for_removal(ops_particle particle) {
  if (particle->no_particles == 0) return;

  int dim = particle->block->dims;
  int Nlocal = particle->no_particles;

//  exit(-1);
  double *xpos = (double *)particle->particle_pos_dat->data;

  int *part2bins = (int *)particle->map_list[0]->parts_to_grid->data;

// for (int i = 0; i < Nlocal; i++)
//    printf("x[%d] = [%f %f] Marking = %d bin = %d\n", i,xpos[2 * i], xpos[2 * i + 1], particle->mark_deletion[i], part2bins[i]);

  for (int i = 0; i < Nlocal; i++) {
    while(particle->mark_deletion[i] > 0) {
      //no need to remove
      if (i == Nlocal - 1) {
        Nlocal--; break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal - 1, dat->elem_size);
      }


      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));


////      printf("Mapping index = %d\n", particle->particle_map_index);

      for (int imaps = 0; imaps < particle->particle_map_index; imaps++) {
        ops_particle_mapping map = particle->map_list[imaps];

      //  _ops_particle_swap_mapping_data(map, i, Nlocal - 1);
        _ops_particle_copy_mapping_data_to(map, i, Nlocal - 1);
      }

      Nlocal--;
      if (Nlocal == 0) break;
    }
  }

 // printf("Nlocal = %d\n", Nlocal);
  particle->no_particles = Nlocal;

}

/*void ops_particle_remove_marked(ops_particle particle) {

  if (particle->no_particles == 0)
    return;

  int dim = particle->block->dims;
  long int Nlocal = particle->no_particles;
  long int Ndel{0};

  int i = 0;
  while ( i < Nlocal) {
    int imark = particle->mark_deletion[i];
    if (imark > 0) {
      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                             particle->particle_pos_dat->elem_size);
      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      //for (ops_dat &dat : particle->particle_data) {
      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      //swap deletion as well
      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      Nlocal--;
    }
    else i++;
  }

  particle->no_particles = Nlocal;
}
*/

/*------------------------------------------------------*/
/*! Reset particles for deletion                       */
/*-----------------------------------------------------*/

void ops_particle_reset_marked(ops_particle particle) {

  int nlocal = particle->no_particles;
  for (int i = 0; i < nlocal; i++)
    particle->mark_deletion[i] = 0;
}

void ops_particle_reset_flags(ops_particle particle, bool decide) {

  if (!decide) return;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    map->decide = false;
  }
}


/***************************************************************************************/
/*  Definitions of particle halo and halo groups                                       *
 ***************************************************************************************/

ops_particle_halo_exchange _ops_particle_init_halo_info() {

  ops_particle_halo_exchange halo_info = nullptr;

  halo_info = (ops_particle_halo_exchange)
      ops_calloc(1, sizeof(OPS_particle_halo_exchange_info_core));


  halo_info->nsend = 0;
  halo_info->nrecv =0;
  halo_info->nmax = OPS_MAX_PART;
  halo_info->sendlist = (int *)ops_calloc(halo_info->nmax, sizeof(int));
  halo_info->firstrecv = 0;

  return halo_info;
}

ops_particle_halo_data _ops_particle_decl_halo_data_core(OPS_instance *instance,
                                                         ops_dat from, ops_dat to,
                                                                                                                ops_part_orient orient_flag) {

  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {//TODO:
    instance->OPS_particle_halo_data_max += 10;
    instance->OPS_particle_halo_data_list =
        (ops_particle_halo_data *) ops_realloc(instance->OPS_particle_halo_data_list,
                                               instance->OPS_particle_halo_data_max
                                                * sizeof(ops_particle_halo_data));
    if (instance->OPS_particle_halo_data_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_particle_halo_core--"
                                            "Error allocation memory");
  }
  ops_particle_halo_data halo
  = (ops_particle_halo_data)ops_calloc(1, sizeof(ops_particle_halo_data_core));

  /* Set halos */
  if (from->dim != to->dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "To and from ops_dat must have "
                                             "the same size of elements per point.");

  if (from->type_size != to->type_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "Different types of the sending and receiving "
                                             "ops_dat structures");

  if (!from->is_particle || !to->is_particle)
    throw OPSException(OPS_INVALID_ARGUMENT, "Sending or receiving ops_dat structure not related"
                                              "to particle data structures");


  halo->from = from;
  halo->to = to;
  halo->orient = orient_flag;

  instance->OPS_particle_halo_data_list[instance->OPS_particle_halo_data_index] = halo;
  instance->OPS_particle_halo_data_index++;

  return halo;
}

ops_particle_halo _ops_free_particle_halo(ops_particle_halo halo) {

  if (halo == nullptr) {
    return nullptr;
  }

  delete halo->sendBox;

  ops_free(halo->dat);

 // if (halo->sendBox != nullptr)


  ops_free(halo);

  return NULL;
}

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, ops_particle_halo_data halos[],
                                          int nhalos, double *critical_length,
                                          int *dir_from, int *dir_to,
                                          double *translate) {

  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {
    instance->OPS_particle_halo_max += 10;
    instance->OPS_particle_halo_list = (ops_particle_halo *)ops_realloc(
        instance->OPS_particle_halo_list, instance->OPS_particle_halo_max
                                                * sizeof(ops_particle_halo));
    if (instance->OPS_particle_halo_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  ops_particle_halo grp =
      (ops_particle_halo)ops_calloc(1, sizeof(ops_particle_halo_core));

  if (nhalos <= 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error non-positive size of ops_particle_halos");

  grp->nhalos = nhalos;
  if (to == NULL || from == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structures");

  grp->particle_from = from;
  grp->particle_to = to;



  for (int i = 0; i < OPS_MAX_DIM; i++) {
    if (i < to->block->dims) grp->dx[i] = critical_length[i];
    else grp->dx[i] = 0.0;
  }

  for (int idir = 0; idir < OPS_MAX_DIM; idir++) {
    grp->translate[idir] = translate[idir];
    grp->dir_from[idir] = dir_from[idir];
    grp->dir_to[idir] = dir_to[idir];

  }

  grp->dat  = (ops_particle_halo_data *)ops_calloc(nhalos, sizeof(ops_particle_halo_data));
  for (int i = 0; i < nhalos; i++) {
   int ip1_owned = _ops_particle_owned_dat(from, halos[i]->from);
   int ip2_owned = _ops_particle_owned_dat(from, halos[i]->to);

   if (ip1_owned == 0 || ip2_owned == 0)
     throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure not related to given ops_particle"
                                              " structures");
   grp->dat[i] = halos[i];
  }

  grp->sendBox = nullptr;

  grp->instance = instance;
  grp->nPoints = 0;

  int nbites = 0;
  for (int i = 0; i < nhalos; i++)
    nbites += halos[i]->from->elem_size;
  grp->nbites = nbites;

  instance->OPS_particle_halo_list[instance->OPS_particle_halo_index] = grp;
  grp->index = instance->OPS_particle_halo_index++;

  return grp;
}

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, int nhalos,
                                          ops_particle_halo_data halos[],
                                          double sending_region[],
                                          int *dir_from, int *dir_to,
                                          double *translate)
{
  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {
    instance->OPS_particle_halo_max += 10;
    instance->OPS_particle_halo_list = (ops_particle_halo *)ops_realloc(
        instance->OPS_particle_halo_list, instance->OPS_particle_halo_max
                                                * sizeof(ops_particle_halo));
    if (instance->OPS_particle_halo_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  ops_particle_halo grp =
      (ops_particle_halo)ops_calloc(1, sizeof(ops_particle_halo_core));

  if (nhalos <= 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error non-positive size of ops_particle_halos");

  grp->nhalos = nhalos;
  if (to == NULL || from == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structures");

  grp->particle_from = from;
  grp->particle_to = to;


  for (int idir = 0; idir < OPS_MAX_DIM; idir++) {
    grp->translate[idir] = translate[idir];
    grp->dir_from[idir] = dir_from[idir];
    grp->dir_to[idir] = dir_to[idir];
    grp->dx[idir] = 0.0;
  }

  //Create boundingBox;

  double xmax[OPS_MAX_DIM], xmin[OPS_MAX_DIM];
  for (int i = 0; i < to->block->dims; i++) {
    xmin[i] = sending_region[2 * i];
    xmax[i] = sending_region[2 * i + 1];
  }
  grp->sendBox = new BoundingBox(to->block->dims,xmin, xmax);


  grp->dat  = (ops_particle_halo_data *)ops_calloc(nhalos, sizeof(ops_particle_halo_data));
  for (int i = 0; i < nhalos; i++) {
   int ip1_owned = _ops_particle_owned_dat(from, halos[i]->from);
   int ip2_owned = _ops_particle_owned_dat(from, halos[i]->to);

   if (ip1_owned == 0 || ip2_owned == 0)
     throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure not related to given ops_particle"
                                              " structures");
   grp->dat[i] = halos[i];
  }

  grp->instance = instance;
  grp->nPoints = 0;

  int nbites = 0;
  for (int i = 0; i < nhalos; i++)
    nbites += halos[i]->from->elem_size;
  grp->nbites = nbites;

  instance->OPS_particle_halo_list[instance->OPS_particle_halo_index] = grp;
  grp->index = instance->OPS_particle_halo_index++;

  return grp;
}


ops_particle_halo_group _ops_particle_decl_halo_group(OPS_instance *instance,
                                                      ops_particle_halo particle_halos[],
                                                      int nhalos,
                                                      ops_part_halo_grp_type halo_type,
                                                      ops_with_virtual with_virtual,
                                                      ops_particle_halo_group master) {

  if (instance->OPS_particle_halo_group_index == instance->OPS_particle_halo_group_max) {
    instance->OPS_particle_halo_group_max += 10;
    instance->OPS_particle_halo_group_list
      = (ops_particle_halo_group *) ops_realloc(instance->OPS_particle_halo_group_list,
                                                instance->OPS_particle_halo_group_max *
                                                  sizeof(ops_particle_halo_group));
    if (instance->OPS_particle_halo_group_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  if (nhalos <= 0) return nullptr;

  ops_particle_halo_group halo_grp
    = (ops_particle_halo_group)ops_calloc(1, sizeof(ops_particle_halo_group_core));

  halo_grp->halo_list = (ops_particle_halo *)ops_calloc(nhalos, sizeof(ops_particle_halo));
  halo_grp->nhalos = nhalos;

  for (int i = 0; i < nhalos; i++)
    halo_grp->halo_list[i] = particle_halos[i];


  /* Allocate definition of exchange info */
  halo_grp->halo_info =
      (ops_particle_halo_exchange *)ops_calloc(nhalos, sizeof(ops_particle_halo_exchange));



  halo_grp->halo_type = halo_type;

  halo_grp->with_virtual = with_virtual;

  if (halo_type == OPS_HALO_GRP_FORWARD || halo_type == OPS_HALO_GRP_BACKWARD) {
    halo_grp->halo_master = master; //Yes if we shift bites to a different locaton
  }
  else { //Initialize halo infos
    for (int i = 0; i < nhalos; i++)
      halo_grp->halo_info[i] = _ops_particle_init_halo_info();
  }


  if (halo_grp == OPS_HALO_GRP_EXCHANGE && halo_grp->with_virtual == OPS_WITH_VIRTUAL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Exchange-halos to operate on actual and "
                                             "virtual particles");


  instance->OPS_particle_halo_group_list[instance->OPS_particle_halo_group_index] = halo_grp;
  halo_grp->index = instance->OPS_particle_halo_group_index;
  instance->OPS_particle_halo_group_index++;

  return halo_grp;
}

ops_particle_halo_group _ops_free_particle_halo_group(ops_particle_halo_group halo_grp) {

  if (halo_grp == NULL) return NULL;


  //Free
  if (  halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE
       || halo_grp->halo_type == OPS_HALO_GRP_BORDER) {
    for (int i = 0; i < halo_grp->nhalos; i++) {
      ops_free(halo_grp->halo_info[i]->sendlist);
      ops_free(halo_grp->halo_info[i]);
    }
  }
  ops_free(halo_grp->halo_info);

  ops_free(halo_grp->halo_list);

  ops_free(halo_grp);

  return NULL;
}

/*-------------------------------------------------------------------------------------*/
/* Assignment of particle data exchanged in particle halos
 *
 * \param[in] from       ops_dat structure from which data will be copied
 * \param[in] to         ops_dat structure to which data will be copied
 * \param[in] dir_from   Orientation of local coordinate system in block of from (Used
 *                       only to rotate data in the new frame)
 * \param[in] to_dir     Orientation  of local coordinate system to the of the to ops_dat
 *                       structure
 * \param[in] translate  Translation of particle data to a new location (Used only in
 *                       association with particle position ops_dat structures)
 */
/*-------------------------------------------------------------------------------------*/

ops_particle_halo_data ops_particle_decl_data_halo(ops_dat from, ops_dat to,
                                                   ops_part_orient orient_flag) {

  return _ops_particle_decl_halo_data_core(OPS_instance::getOPSInstance(),from, to,
                                           orient_flag);
}

/*-------------------------------------------------------------------------------------*/
/* Declaration of particle-halos
 *
 * \param[in] from  pointer to an ops_particle structure from which data will be send
 * \param[in] to    pointer to an ops_particle structure to which data will be send
 * \param[in] particle_halos  array of ops_particle_halo_data exchanged within this
 *                            halo
 * \param[in] nhalos  number of halo-data send within this halo structure
 * \param[in] critical_length length for definition of sending region
 * \param[in] dir_from        direction from where to send
 * \param[in] dir_to          direction to send for vector and tensorial fields
 * \param[in] translate       vector for translating bounding box of the sending
 *                            ops_particle structure
 */
/*-------------------------------------------------------------------------------------*/

ops_particle_halo ops_particle_decl_halo(ops_particle from, ops_particle to,
                                         ops_particle_halo_data particle_halos[],
                                         int  nhalos, double* critical_length,
                                         int *dir_from, int *dir_to,
                                         double *translate) {
  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 particle_halos, nhalos, critical_length,
                                 dir_from, dir_to, translate);

}

ops_particle_halo ops_particle_decl_halo2(ops_particle from, ops_particle to,
                                          ops_particle_halo_data particle_halos[],
                                          int nhalos, double critical_length[],
                                          int *dir_from, int *dir_to,
                                          double *translate){

  int nhalos1 = nhalos + 1;

  ops_dat pos_to = to->particle_pos_dat;
  ops_dat pos_from = from->particle_pos_dat;

  ops_particle_halo_data  *halo_data
  = (ops_particle_halo_data *) ops_malloc(sizeof(ops_particle_halo_data) * nhalos1);

  halo_data[0] = ops_particle_decl_data_halo(pos_from, pos_to, OPS_PART_POSITION);

  for (int i = 0; i < nhalos; i++)
    halo_data[i + 1] = particle_halos[i];


  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 halo_data, nhalos1, critical_length,
                                 dir_from, dir_to, translate);

}

ops_particle_halo ops_particle_decl_halo(ops_particle from, ops_particle to,
                                         int nhalos, ops_particle_halo_data particle_halos[],
                                         double sending_region[],
                                         int *dir_from, int *dir_to,
                                         double *translate) {


  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 nhalos, particle_halos, sending_region, dir_from,
                                 dir_to, translate);

}


ops_particle_halo_group ops_particle_decl_halo_group(ops_particle_halo particle_halos[],
                                                     int nhalos,
                                                     ops_part_halo_grp_type halo_type,
                                                     ops_with_virtual with_virtual,
                                                     ops_particle_halo_group master) {

  return _ops_particle_decl_halo_group(OPS_instance::getOPSInstance(), particle_halos,
                                       nhalos, halo_type, with_virtual, master);
}

void _ops_particle_pack_halo_data(char *buf, ops_particle_halo_data* halo_data, int ndata,
                                  int ipart) {

  int size_data = 0;
  for (int i = 0; i < ndata; i++) {
    ops_dat dat = halo_data[i]->from;

    double *data = (double *)dat->data;

    int nbites = dat->elem_size;
    memcpy(buf + size_data, dat->data + ipart * nbites, nbites);

    size_data += dat->elem_size;
  }

}

void _ops_particle_unpack_halo_data(char *buff, ops_particle_halo_data* halo_data, int ndata,
                                    int iloc, int *dir_to, int *dir_from, double *translate) {

  char *temp = nullptr;

  int size_loc = 0;
  for (int i = 0; i < ndata; i++) {
    ops_dat data_to = halo_data[i]->to;
    ops_part_orient orient = halo_data[i]->orient;

    int bites = iloc * data_to->elem_size;
    if (orient == OPS_PART_ORIENT_OFF) {
      memcpy(data_to->data + bites, buff + size_loc, data_to->elem_size);
    }
    else {

      temp = (char *)ops_realloc(temp, data_to->elem_size);

      int dim = data_to->dim;
      int dims = data_to->block->dims;

      //translate = halo_data[i]->translate;
      //dir_to = halo_data[i]->to_dir;
      //dir_from = halo_data[i]->from_dir;


      double *shifted = (double *)ops_calloc(dims, sizeof(double));
      if (dim != dims || data_to->type_size != sizeof(double))
        throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat with orient"
                                                           " must be a vector (of size dim per particle point) and of "
                                                           "type double");

      memcpy(temp, buff + size_loc, data_to->elem_size);

      double *tmp = (double *)temp;

     // printf("Translate = [%f %f %f]\n", translate[0], translate[1], translate[2]);

      for (int isou = 0; isou < dim; isou++) {
        shifted[dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }

      memcpy(data_to->data + bites, (char *)shifted,  data_to->elem_size);
      ops_free(shifted);

    }
    size_loc += data_to->elem_size;

  }

  ops_free(temp);
}

void _ops_particle_unpack_reverse_halo_data(char *buff, ops_particle_halo_data *halo_data, int ndata,
                                            int iloc, int *dir_to, int *dir_from) {
  int nsizemax = 0;
  for (int i = 0; i < ndata; i++) {
    if (halo_data[i]->to->type_size != sizeof(double) ||
        halo_data[i]->to->type_size != sizeof(float))
      throw OPSException(OPS_RUNTIME_ERROR," Reverse communications only for float or real"
                         "data structures");

    nsizemax = MAX(nsizemax, halo_data[i]->to->elem_size);

    if (halo_data[i]->orient == OPS_PART_POSITION)
      throw OPSException(OPS_RUNTIME_ERROR, " Reverse communications only for orientation");
  }

  char *tmp = (char *)ops_malloc(nsizemax);
  char *shifted = (char *)ops_malloc(nsizemax);

  int size_loc = 0;
  for (int i = 0; i < ndata;i++) {

    ops_dat data_to = halo_data[i]->from;
    memcpy(tmp, buff + size_loc, data_to->elem_size);
    char *enter = nullptr;
    if (halo_data[i]->orient == OPS_PART_ORIENT_OFF)
      enter = tmp;
    else {
      for (int isou = 0; isou < data_to->dim; isou++)
        shifted[dir_to[isou]] = tmp[dir_from[isou]];
    }


    if (strcmp(halo_data[i]->to->type, "double")== 0) {

      if (halo_data[i]->orient != OPS_PART_ORIENT_OFF) {
        double *shift = (double *)shifted;
        double *temp = (double *)tmp;

        for (int isou = 0; i < halo_data[i]->to->dim; isou++)
          shift[dir_to[isou]] = temp[dir_from[isou]];
        enter = shifted;

      }
      else
        enter = tmp;
      _ops_inc_element((double *)(data_to->data + data_to->elem_size * iloc), (double *)enter,
                     halo_data[i]->to->dim, OPS_INC);
    }
    else if (strcmp(halo_data[i]->to->type, "float") == 0){
      if (halo_data[i]->orient != OPS_PART_ORIENT_OFF) {
         float *shift = (float *)shifted;
         float *temp = (float *)tmp;

         for (int isou = 0; i < halo_data[i]->to->dim; isou++)
           shift[dir_to[isou]] = temp[dir_from[isou]];
         enter = shifted;

       }
       else
         enter = tmp;
      _ops_inc_element((float *)(data_to->data + data_to->elem_size * iloc), (float *) tmp,
                       halo_data[i]->to->dim, OPS_INC);
    }
    else if (strcmp(halo_data[i]->to->type, "int") == 0) {
      if (halo_data[i]->orient != OPS_PART_ORIENT_OFF) {
         float *shift = (float *)shifted;
         float *temp = (float *)tmp;

         for (int isou = 0; i < halo_data[i]->to->dim; isou++)
           shift[dir_to[isou]] = temp[dir_from[isou]];
         enter = shifted;

       }
       else
         enter = tmp;
      _ops_inc_element((float *)(data_to->data + data_to->elem_size * iloc), (float *) tmp,
                       halo_data[i]->to->dim, OPS_INC);
    }
  }


  ops_free(tmp);


}


/*---------------------------------------------------------------------------------------*/
/*  Setup the exchange structure according to exchange type                              */
/*---------------------------------------------------------------------------------------*/
void ops_particle_set_halo_group(ops_particle_halo_group halo_grp) {

  /* Get ops instance */
  OPS_instance *instance = OPS_instance::getOPSInstance();

  // TMP Definition: Later on shift within in class
  switch(halo_grp->halo_type) {
  case  OPS_HALO_GRP_EXCHANGE:
//    printf("Halo type exchange\n");
    _ops_particle_setup_exchange_comm(instance, halo_grp); //TODO:
    break;
  case  OPS_HALO_GRP_BORDER:
  case OPS_HALO_GRP_DEFAULT:
//    printf("Halo type border\n");
    _ops_particle_setup_border_comm(instance, halo_grp);
    break;
  case OPS_HALO_GRP_FORWARD:
  case OPS_HALO_GRP_BACKWARD:
    _ops_particle_setup_for_rev_comm(instance, halo_grp);
    break;
  default:
    break;
  }



}


void ops_particle_halo_transfer_group_grid(ops_part_halo_grp_type exchange_type,
                                           bool exchange) {
  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {
    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];

    switch(exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange)
          _ops_particle_halo_exchange_transfer(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT ) {
        if (exchange)
          _ops_particle_halo_border_transfer_vs(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_forward_transfer(instance, halo_grp);
      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_reverse_transfer(instance, halo_grp);
      break;
    default:
      _ops_particle_halo_forward_transfer(instance, halo_grp);

    }
  }
}

//TODO: Make it later on the default algorithm
void ops_particle_halo_transfer_group_hybrid(ops_part_halo_grp_type exchange_type,
                                             bool exchange) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {
    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];

    switch(exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange)
          _ops_particle_halo_exchange_transfer_map(instance, halo_grp); //TODO
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {
        if (exchange)
         _ops_particle_halo_border_transfer_vs(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        if (!exchange)
          _ops_particle_halo_forward_map(instance, halo_grp); //TODO
      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_reverse_transfer(instance, halo_grp);
      break;
    default:
      _ops_particle_halo_forward_map(instance, halo_grp);
    }
  }

}

//TODO: Modify as the BORDER to be able to perform forward and backward
//comms
void ops_particle_halo_transfer_group(ops_part_halo_grp_type exchange_type,
                                      bool exchange) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {

    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];

/*    if (halo_grp->halo_type != exchange_type) continue;
    switch (halo_grp->halo_type)
    case OPS_HALO_GRP_EXCHANGE: {
      if (exchange)
        _ops_particle_halo_exchange_transfer(instance, halo_grp); //TODO
      break;
    case OPS_HALO_GRP_BORDER:
      if (exchange) {
        _ops_particle_halo_border_transfer(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
   //   printf("Forward exchange\n");
      _ops_particle_halo_forward_transfer(instance, halo_grp);
      break;
    case OPS_HALO_GRP_BACKWARD:
      _ops_particle_halo_reverse_transfer(instance, halo_grp);
      break;
    case OPS_HALO_GRP_DEFAULT:
//      _ops_particle_halo_default_transfer(instance, halo_grp); //TODO
      break;
    default:
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Exchange type not defined.");
    }
    */

    switch (exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange) {
          _ops_particle_halo_exchange_transfer(instance, halo_grp);
        }
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
            halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {

        if (exchange)
          _ops_particle_halo_border_transfer(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (//halo_grp->halo_type != OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_forward_transfer(instance, halo_grp);

      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD)
        _ops_particle_halo_reverse_transfer(instance, halo_grp);
      break;
    default:
      _ops_particle_halo_forward_transfer(instance, halo_grp);
    }
  }
}

void ops_particle_halo_transfer_positions_grp(ops_part_halo_grp_type exchange_type,
                                              bool exchange) {

  if (  exchange_type == OPS_HALO_GRP_EXCHANGE ||
        exchange_type == OPS_HALO_GRP_BACKWARD ||
        exchange_type == OPS_HALO_GRP_DEFAULT)
    throw OPSException(OPS_RUNTIME_ERROR, "This exchange type is not supported for "
        "positional data\n");

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {
    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];
    if (halo_grp->halo_type != OPS_HALO_GRP_DEFAULT) continue;
    switch(exchange_type) {
    case OPS_HALO_GRP_BORDER:
      if (exchange)
        _ops_particle_halo_border_pos_transfer(instance, halo_grp);
      break;
    case OPS_HALO_GRP_FORWARD:
      _ops_particle_halo_forward_pos_transfer(instance, halo_grp);
      break;
    default:
      throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"Error: Invalid exchange type");
    }
  }

}


void ops_particle_halo_transfer(ops_particle_halo_group halo_grp,
                                ops_part_halo_grp_type exchange_type,
                                bool exchange) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  switch(exchange_type) {

      //halo_grp->halo_type)
  case OPS_HALO_GRP_EXCHANGE:
    if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE)
      if (exchange)
        _ops_particle_halo_exchange_transfer(instance, halo_grp); //TODO
    break;
  case OPS_HALO_GRP_BORDER:
    if (  halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
      if (exchange)
        _ops_particle_halo_border_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_FORWARD:
    if ( halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
         halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
      _ops_particle_halo_forward_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_BACKWARD:
    if ( halo_grp->halo_type == OPS_HALO_GRP_BACKWARD)
      _ops_particle_halo_reverse_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_DEFAULT:
    if ( halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
      _ops_particle_halo_border_transfer(instance, halo_grp);
    break;
  default:
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Exchange type not defined.");
  }
}



/**************************************************************************************************/
/* Mapping functions                                                                              */
/**************************************************************************************************/

//TODO: Split based on the different needs-This will become the core function

ops_particle_mapping  ops_decl_mapping_core(ops_particle particle, ops_dat grid,
                                            ops_dat radius, int size[],
                                            int d_m[], int d_p[], int base[],
                                            ops_stencil stencil,
                                            ops_with_virtual include_virtual,
                                            ops_shape_evolve particle_changes,
                                            ops_grid_type grid_type,
                                            double skin, int Ng) {
  /* Create a new structure */
  ops_particle_mapping map = (ops_particle_mapping)
       ops_malloc(sizeof(ops_particle_mapping_core));

  map->mapping_type = include_virtual;

  map->grid_type = grid_type;

  map->grid = grid;

  map->Ngrids = 1;
  if (map->grid_type != OPS_UNIFORM_STAG || map->grid_type != OPS_UNIFORM_COLL) {
    map->Ngrids = (Ng > 1) ? Ng : 1;
  }

  printf("Map grid = %s", map->grid->name);
  double* nulld{nullptr};
  int*    nulli{nullptr};

  map->parts_to_grid = ops_decl_particle_dat(particle, 1, base, nulli,
                                             "int", "particle_to_map", false);
  map->bin = ops_decl_particle_dat(particle, 1, base, nulli, "int", "bin_to_do", false);
  map->pos_old = ops_decl_particle_dat(particle, 3, base, nulld, "double", "ps_old_name", false);


  map->binhead = ops_decl_dat(particle->block, 1, size, base, d_m, d_p, nulld,"double", "binhead");
  /* Set changes of particle and define Rp_old*/
  map->particle_changes = particle_changes;;
  if (map->particle_changes == OPS_EVOLV_SHAPE) {
    map->Rp_old = ops_decl_particle_dat(particle, 1, base, nulld, "double", "ADD",false); //TODO-Deassociate
  }
  map->decide = true;

  /* Copy Rp dat structure */
  map->Rp = radius;


  /* Set remapping options */
  if (skin < 0.0) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-positive flag");
  }


  map->skin = skin;
  /* Define uniform grid mapping */
  // printf("Skin is set to %f\n", map->skin);
  /* Set stencil for building maps for virtual particles */
  map->mapping_stencil = stencil;

  particle->particle_map_index++;
  if (particle->particle_map_index > particle->particle_map_max) {
    particle->particle_map_max = particle->particle_map_index + 10;
    particle->map_list = (ops_particle_mapping *)ops_realloc(particle->map_list,
                     particle->particle_map_max * sizeof(ops_particle_mapping));
  }
  particle->map_list[particle->particle_map_index - 1] = map;

  map->index = particle->particle_map_index - 1;

  //map->index = particle->mapping_list.size() - 1;

  return map;

}

ops_particle_mapping ops_decl_mapping(ops_particle particle, ops_dat grid, ops_dat radius,
                                      ops_stencil stencil,
                                      ops_with_virtual include_virtual,
                                      ops_shape_evolve particle_changes,
                                      ops_grid_type grid_type,
                                      double skin, int Ng) {

  int size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], base[OPS_MAX_DIM];

  ops_mapping_def_core(particle, grid, stencil,
                       include_virtual, size, base, d_m, d_p);

  return ops_decl_mapping_core(particle, grid, radius, size,
                                d_m, d_p, base,
                                stencil, include_virtual,
                                particle_changes, grid_type,
                                skin, Ng);

}


//TODO: Remove
void ops_particle_map_decide_and_build(ops_particle particle, ops_particle_mapping map,
                                       bool enforce)
{
  /* Decide if list requires rebuild */
  int decide =_ops_particle_mapping_decide(map, particle, enforce);


  //TODO: See how to set up elements-Insert and remove herein


  /* Build new neighbor list */
  if (decide) {
    //TODO: Remove or exchange particles prior to neighbor build
    //TODO: Uniform
    if (map->grid_type == OPS_UNIFORM_STAG ||
        map->grid_type == OPS_UNIFORM_COLL)
      _ops_particle_build_local_uniform(map, particle); //TODO
    else if (map->grid_type == OPS_NON_UNI_STAG  ||
             map->grid_type == OPS_NON_UNI_COLL)
      _ops_particle_build_local_non_uniform(map, particle); //TODO
   //NON-Uniform grid

    map->decide = false;
  }
}

/*------------------------------------------------------------------------------------*/
/* \brief Building maps of particles to uniform or non-uniform structured grids
 *
 * \param[in] particle pointer to an ops_particle structure
 * \param[in] map      pointer to an ops_particle_mapping structure
 */
/*-----------------------------------------------------------------------------------*/

void ops_particle_map_build(ops_particle particle, ops_particle_mapping map) {
//  if (map->decide) {
    if (map->grid_type == OPS_UNIFORM_STAG ||
        map->grid_type == OPS_UNIFORM_COLL)
      _ops_particle_build_local_uniform(map, particle);
    else if (map->grid_type == OPS_NON_UNI_STAG ||
             map->grid_type == OPS_NON_UNI_COLL)
      _ops_particle_build_local_non_uniform(map, particle);

    map->decide = false;
 // }
}

/**
 * Maps new & virtual particles only. The map is updated previously for existing particles
 *
 * @@ particle pointer to an ops_particle_structure
 * @@ map      pointer to an ops_particle_mapping structure
 */
void ops_particle_map_build_testing(ops_particle particle, ops_particle_mapping map) {
  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_create_local_uniform(map, particle);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New build option not currently "
                                            "supported for non-uniform grids\n");
  map->decide = false;
}

void  ops_particle_update_map_testing(ops_particle particle, ops_particle_mapping map) {
  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_update_local_uniform(map, particle);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New build option not currently "
                                            "supported for non-uniform grids\n");
}

void ops_particle_map_update_halo_maps(ops_particle particle, ops_particle_mapping map) {

  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_update_maps_due_to_halos(particle, map);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New build option not supported "
                                            " for non-uniform grids ");
}


void ops_particle_map_setup_actual_particles(ops_particle particle,
                                             ops_particle_mapping map) {

  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_setup_map(particle, map);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New mapping algorithm does not support "
                                            "non-uniform grids");

}

void ops_particle_map_setup_virtual_particles(ops_particle particle,
                                              ops_particle_mapping map) {

  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_setup_map_virtual(particle, map);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New mapping algorithm does not support "
                                            "non-uniform grids");


}


/*-------------------------------------------------------------------------------------*/
/* \brief Function to decide if a given particle will be build
 *
 * \param[in]  particle      pointer to an ops_particle structure
 * \param[in]  map           pointer to an ops_particle_mapping structure
 * \param[in]  enforce       flag for enforcing particle build list
 */
/*-----------------------------------------------------------------------------------*/
void ops_particle_map_decide(ops_particle particle, ops_particle_mapping map,
                             bool enforce) {

  if (enforce) {
    map->decide = true;
    return;
  }

  int decide = _ops_particle_mapping_decide(map, particle, enforce);

  if (decide > 0)
    map->decide = true;
}

/*-----------------------------------------------------------------------------------*/
/* \brief the map for sequential code
 *
 * \return a flag that at least a single list is build
 */

bool ops_particle_update_map_lists(ops_particle particle) {

  bool decide_global=false;

  for (int index = 0; index < particle->particle_map_index; index++) {
  //for (auto &map : particle->mapping_list) {
    ops_particle_mapping map = particle->map_list[index];
    ops_particle_map_decide(particle, map);

//    if (!decide_global)

      decide_global = (map->decide) ? true : false;

      if (decide_global) break;

  }

  /* Mark particles for exchange or add them into an array */
  if (decide_global) {
   ops_particle_mark_for_del(particle);

   _ops_particle_exchange(particle);

   particle->no_virtual = 0; //Reset virtuals
 //   ops_particle_create_forward_comms(particle);
  //Exchange particles for MPI and build comms for forward communications

   _ops_particle_build_border(particle); //TODO
 }

  return decide_global;

}


int ops_particle_update_map_lists_actual_parts(ops_particle particle) {

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    if (map->grid_type == OPS_UNIFORM_STAG ||
        map->grid_type == OPS_UNIFORM_COLL)
      int a =  _ops_particle_decide_build_local_uniform(map, particle);
    else if (map->grid_type == OPS_NON_UNI_STAG ||
             map->grid_type == OPS_NON_UNI_COLL)
      throw OPSException(OPS_NOT_IMPLEMENTED, "Concurrent decide and update is not"
                         "implemented for non uniform grids");
    else
      throw OPSException(OPS_NOT_IMPLEMENTED, "This type of grid is not supported\n");

  }

  int flag{0};
  for (int index = 0; index < particle->particle_map_index; index++)
    if (particle->map_list[index]->decide) {flag = 1; break;}

  //printf("Exit from mapping (flag = %d\n", flag);
  //Herein resort particles prior to deletion

  if (flag) { //TODO: Second sanity check for reseting the list.
    ops_particle_reset_virtual_particles(particle);
//    ops_particle_rearrange_particles_for_removal(particle);
  }

  return flag;
}

int ops_particle_update_map_lists_actual_hybrid(ops_particle particle) {
  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    if (map->grid_type == OPS_UNIFORM_STAG ||
          map->grid_type == OPS_UNIFORM_COLL)
        int a =  _ops_particle_decide_build_only_local_uniform(map, particle);
      else if (map->grid_type == OPS_NON_UNI_STAG ||
               map->grid_type == OPS_NON_UNI_COLL)
        throw OPSException(OPS_NOT_IMPLEMENTED, "Concurrent decide and update is not"
                           "implemented for non uniform grids");
      else
        throw OPSException(OPS_NOT_IMPLEMENTED, "This type of grid is not supported\n");

    map->decide = ops_particle_global_rebuild(map->decide);

  }

  int flag{0};

  for (int index = 0; index < particle->particle_map_index; index++)
   if (particle->map_list[index]->decide) {flag = 1; break;}

  if (flag == 1) {
    ops_particle_reset_virtual_particles(particle);

    _ops_particle_exchange_map_update(particle); //TODO

    _ops_particle_build_border_maps(particle);

    //TODO: Rebuild maps for virtual intra-block
  }
  else {
    _ops_particle_forward_intra_maps(particle);

  }
    return flag;
}

void ops_particle_remove_particles(ops_particle particle, bool flag) {

  if (!flag) return;

  ops_particle_remove_marked(particle);

  ops_particle_reset_marked(particle);
}

/*--------------------------------------------------------------------------------------*
 * Function removes particles and update particle maps in the lists
 *
 * @param particle pointer to an ops_particle structure
 *--------------------------------------------------------------------------------------*/

void ops_particle_remove_delete_maps(ops_particle particle, int decide) {

  if (!decide)
    return;

  ops_particle_remove_marked_with_maps(particle);


  //TODO: Check if last needs removal
  ops_particle_reset_marked(particle);
}


void ops_particle_build_maps(ops_particle particle, bool flag) {


  //Prior to building expand the lists

  if (!flag)
    return;

  // Remove and delete particle
//  ops_particle_remove_marked(particle); //TODO: Shift to different location

//  ops_particle_reset_marked(particle);
  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    ops_particle_map_build(particle, map);
  }

}

void ops_particle_build_maps_testing(ops_particle particle) {

  bool flag = false;
  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->decide) {flag = true; break;}
  }

 // printf("Map to build is %d\n", (int)flag);

  if (flag) {
    for (int imap = 0; imap < particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      map->decide = (bool) flag;
      if (flag)
        ops_particle_map_build_testing(particle, map);
      else
        ops_particle_update_map_testing(particle, map);

    }
  }


}

void ops_particle_build_maps_testing_v2(ops_particle particle, int flag) {

  if (!flag) return;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->mapping_type != OPS_WITH_VIRTUAL) continue;

    ops_particle_map_update_halo_maps(particle, map);
  }
}

void ops_particle_setup_map_grid(ops_particle particle) {

  //Set up the maps

  for (int i = 0; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_setup_actual_particles(particle, map);
  }

  //TODO: Exchange intra-block halos
  if (particle->particle_map_index > 0) {
    printf("Entering to build border maps\n");
    _ops_particle_build_border_maps(particle);
  }
  else {

  }
}

void ops_particle_setup_map(ops_particle particle) {

  //First set actual particle non in deletion state
  for (int i = 0; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  //Exchange particles first between processes
  _ops_particle_build_border(particle);

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_build(particle, map);
  }
}


void ops_particle_setup_virtual_particles(ops_particle particle) {

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->mapping_type != OPS_WITH_VIRTUAL) continue;
    ops_particle_map_setup_virtual_particles(particle, map);
  }
}

/*----------------------------------------------------------------------------------------*/
/* Auxiliarry functions to be moved
 * ---------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------*
 *  Swap data between points in particle lists
 *--------------------------------------------------------------------------------------*/
void _ops_particle_swap_data(char *data, int i, int j, int elems) {

  for (int ielem = 0; ielem < elems; ielem++) {
    char ctmp = data[elems * i + ielem];
    data[elems * i + ielem] = data[elems * j + ielem];
    data[elems * j + ielem] = ctmp;
  }
}


void _ops_particle_swap_mapping_data(ops_particle_mapping map, int from, int to) {

  //Get map structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;

  int *part2bin = (int *)map->parts_to_grid->data;
  double *xold = (double *)map->pos_old->data;
  //Part I: Swap binheads;

  int address_to = part2bin[to];
  int address_from = part2bin[from];

  //Swap part2bins
  part2bin[from] = part2bin[to];
  part2bin[to] = address_from;

  //Part II: Swap bins

  int bin_to = bins[to];
  int bin_from = bins[from];



  int a1 = binhead[address_to];
  int a2 = binhead[address_from];

//  printf("Binhead[to] = %d binhead[from] = %d\n", a1, a2);

  if (address_to != -1) {
    if (binhead[address_to] == to) {
      binhead[address_to] = from;
    }
    else {
      int iPart = binhead[address_to];
      int iPrev;
      while (iPart != to) {
        iPrev = iPart;
        iPart = bins[iPart];
      }

      bins[iPrev] = from;
    }

  }

  if (address_from != - 1) {
    if (binhead[address_from] == from) {
      binhead[address_from] = to;
    }
    else {
      int iPart = binhead[address_from];
      int iPrev;
      while (iPart != from) {
        iPrev = iPart;
        iPart = bins[iPart];
      }

      bins[iPrev] = to;
    }
  }


  bins[from] = bins[to];
  bins[to] = bin_from;

}

/*--------------------------------------------------------------------------------------*
 *  Function that copies mapping data from particle from to particle to
 */


//        _ops_particle_copy_mapping_data_to(map, i, nactual - 1);


void _ops_particle_copy_mapping_data_to(ops_particle_mapping map, int to, int from) {

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2bin = (int *)map->parts_to_grid->data;


  //Get addresses from and to
  int address_from = part2bin[from];





  int address = part2bin[from];
 // printf("Address = %d\n");

  if (address == -1) return;


  part2bin[from] = part2bin[to];

  part2bin[to] = address;

  //Replace data to address bin

  if (binhead[address] == from) {
    binhead[address] = to;
    bins[to] = bins[from];

  }
  else {
    int iPart = binhead[address];
    int iPrev;
    while (iPart != from ) {
      iPrev = iPart;
      iPart = bins[iPart];
    }

    bins[iPrev] = to;
    bins[to] = bins[from];
  }

 bins[from] = -1;

}


/*--------------------------------------------------------------------------------------*
 *  Build bounding boxes for ops_particle structures
 *--------------------------------------------------------------------------------------*/


void ops_particle_setup_partition() {
  //TODO: Set the bounding box based on particles

  OPS_instance *instance = OPS_instance::getOPSInstance();


  for (int index = 0; index < instance->OPS_block_index; index++) {

    int nparticles = instance->OPS_block_list[index].no_particle_structures;
    for (int ipart = 0; ipart < nparticles; ipart++) {
      ops_particle particle = instance->OPS_block_list[index].particle[ipart];
      ops_build_bounding_box(particle);
      ops_particle_init_maps(particle);

      ops_particle_setup_intrablock_comms(particle);

    }
  }


}


void ops_particle_init_maps(ops_particle particle) {

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    ops_particle_init_map(map);

  }
}


/*-------------------------------------------------------------------------------------*
 * Core function to output particle data to txt files. The given function flushed all
 * particle ops_dat structures to txt files
 */

bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2);

void ops_particle_print_data_to_txtfile_core(ops_particle particle,const char *file_name_in) {
  std::string file_name, ignored;
  ops_checkpointing_filename(file_name_in, file_name, ignored);

  FILE *fp;
  if (fopen_s(&fp, file_name.c_str(), "a") != 0) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Can't open file\n");
  }
  if (fprintf(fp, "Particle: %s\n", particle->name)<0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

#ifdef OPS_MPI
  if (fprintf(fp, "Number of particles: %d   Block: [%f %f] x [%f %f]", particle->nglobal,
              particle->box_block->getGlobalMin().x, particle->box_block->getGlobalMax().x,
              particle->box_block->getGlobalMin().y, particle->box_block->getGlobalMax().y) < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->block->dims == 3) {
    if (fprintf(fp, "x [%f %f]\n", particle->box_block->getGlobalMin().z,
                 particle->box_block->getGlobalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file");
  }
  else
    if(fprintfp(fp,"\n") < 0)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file");
#else
  if (fprintf(fp, "Number of particles: %d   Block: [%f %f] x [%f %f]",
              particle->no_particles, particle->box_block->getLocalMin().x,
              particle->box_block->getLocalMax().x, particle->box_block->getLocalMin().y,
              particle->box_block->getLocalMax().y) < 0) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  }

  if (particle->block->dims == 3) {
    if (fprintf(fp, "x [%f %f]\n", particle->box_block->getLocalMin().z,
                particle->box_block->getLocalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  }
  else
    if (fprintf(fp, "\n") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
#endif

  if (fprintf(fp, "               x                y ") < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->block->dims == 3)
    if(fprintf(fp,"               z ") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->particle_envelope != nullptr)
    if (fprintf(fp, "%16s ", particle->particle_envelope->name) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");


  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat = particle->particle_dat[i];
    switch (dat->dim) {
    case 1:
      if (fprintf(fp, "%16s ", dat->name) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    case 2:
      if (fprintf(fp, "%14s_x %14s_y " ,dat->name, dat->name)<0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    case 3:
      if (fprintf(fp, "%14s_x %14s_y %14s_z ",dat->name, dat->name, dat->name)<0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    default:
      for (int isou = 0; isou < dat->dim; isou++)
        if (fprintf(fp, "%13s[%d] ", dat->name, isou) < 0)
          throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    }
  }
  if (fprintf(fp,"\n") < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  /* Export particles data to txt file */
  for (int i = 0; i < particle->no_particles; i++) {
    _ops_particle_export_data_point(fp, i, particle->particle_pos_dat);
    if (particle->particle_envelope != nullptr)
     _ops_particle_export_data_point(fp, i, particle->particle_envelope);
    for (int idefs = 0; idefs < particle->particle_dat_index; idefs++)
     _ops_particle_export_data_point(fp, i, particle->particle_dat[idefs]);

    if (fprintf(fp, "\n")<0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  }



  fclose(fp);

}

void ops_particle_print_dat_to_txtfile_core(ops_dat dat, ops_particle particle,
                                            const char *file_name_in) {
  if (!dat->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR,
                       "Error: ops_dat not associated with particle data");

  std::string file_name, ignored;
  ops_checkpointing_filename(file_name_in, file_name, ignored);
  FILE *fp;
  if (fopen_s(&fp,file_name.c_str(), "a") != 0) {
   OPSException ex(OPS_RUNTIME_ERROR);
   ex << "Error: can't open file " << file_name;
   throw ex;
  }

  if (fprintf(fp, "ops_dat:    %s\n",dat->name) < 0)
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file\n");

#ifdef OPS_MPI
  if (fprintf(fp, "Block: %s   [%f %f] x [%f %f]", dat->block->name,
              particle->box_block->getGlobalMin().x, particle->box_block->getGlobalMax().x,
              particle->box_block->getGlobalMin().y, particle->box_block->getGlocalMax().y) < 0)
   throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (dat->block->dims == 3)
    if (fprintf(fp, " x[%f %f]\n", particle->box_block->getGlobalMin().z,
                particle->box_block->getGlobalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
#else
  if (fprintf(fp, "Block:   %s   [%f %f] x [%f %f]", dat->block->name,
              particle->box_block->getLocalMin().x, particle->box_block->getLocalMax().x,
              particle->box_block->getLocalMin().y, particle->box_block->getLocalMax().y) < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  if (dat->block->dims == 3)
    if (fprintf(fp, " x[%f %f]\n", particle->box_block->getLocalMin().z,
                particle->box_block->getLocalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
#endif

  if (fprintf(fp, "Dim = %d Size = %ld\n", dat->dim, particle->no_particles) < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  for(int isou = 0; isou < dat->dim; isou++)
    if (fprintf(fp,"             [%d] ", isou) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (fprintf(fp,"\n") < 0)
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file\n");

  for (int i = 0; i < particle->no_particles;i++) {
  //Send data
    if (strcmp(dat->type, "double") == 0 ||
        strcmp(dat->type, "real(8)") == 0 ||
        strcmp(dat->type, "real(4)") == 0 ||
        strcmp(dat->type, "double precision") == 0) {
      for (int isou = 0; isou < dat->dim; isou++)
        if (fprintf(fp, "%16.10lf ", ((double *)dat->data)[i * dat->dim + isou]) < 0)
          throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file\n");
    }
    else if (strcmp(dat->type, "float") == 0 ||
             strcmp(dat->type, "real") == 0 ||
             strcmp(dat->type, "real(4)") == 0 ||
             strcmp(dat->type, "real(kind=4)") == 0) {

      for (int isou = 0; isou < dat->dim; isou++)
        if (fprintf(fp, "%16.10e ", ((float *)dat->data)[i * dat->dim + isou]) < 9)
          throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file\n");
    }
    else if (strcmp(dat->type, "int") == 0 ||
             strcmp(dat->type, "int(4)") == 0 ||
             strcmp(dat->type, "integer") == 0 ||
             strcmp(dat->type, "integer(4)") == 0 ||
             strcmp(dat->type, "integer(kind=4)") == 0 ) {

      for (int isou = 0; isou < dat->dim; isou++)
        if (fprintf(fp,"%16d ", ((int *)dat->data)[dat->dim * i + isou]) < 0)
          throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else {
      OPSException ex(OPS_NOT_IMPLEMENTED);
      ex<<"Error: "<<dat->type<<" is not currently supported"
          "by the OPS-Lagrangian tracking module\n";
    }


    if (fprintf(fp, "\n") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  }

  fclose(fp);
}

ops_arg ops_arg_idp() {
  ops_arg arg;
  memset(&arg, 0, sizeof(ops_arg));

  arg.argtype = OPS_ARG_IDP; //TODO
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;
}

ops_arg ops_arg_idx_map() {
  ops_arg arg;
  memset(&arg, 0, sizeof(ops_arg));

  arg.argtype = OPS_ARG_IDX_MAP;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;

}


