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

  if (dim < 2 && dim > 3) {
    ops_printf("Error: OPS-bounding box is defined only for 2D and 3D "
               "geometries.");
    exit(-1);
  }

  /* Sanity check for box */
  if (minCrd.x >= maxCrd.x) {
    ops_printf("Error: The size of the Bounding box is non-positive\n");
    exit(-1);
  }

  globalBoundingBox[0].x = minCrd.x;
  globalBoundingBox[1].x = maxCrd.x;

  if (minCrd.y >= maxCrd.y) {
    ops_printf("Error: The size of the bounding box in y-dir is "
               "non-positive");
    exit(-1);
  }

  globalBoundingBox[0].y = minCrd.y;
  globalBoundingBox[1].y = maxCrd.y;

  if (dim == 3) {
    if (minCrd.z >= maxCrd.z) {
      printf("Error: The size of the bounding box in the z-dir is non"
             "positive.");
      exit(-1);
    }


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
  if (dim < 2 && dim > 3) {
    ops_printf("Bounding box defined only for 2D and 3D spaces\n");
    exit(-1);
  }
}

BoundingBox::BoundingBox(const ops_dat coords, const double *Deltax,
                         int dim) : dim(dim) , coords(coords) {

  if (dim < 2 && dim > 3) {
    ops_printf("A bounding box is defined only for 2D and 3D spaces\n");
    exit(-1);
  }

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

const ops_point& BoundingBox::getLocalMin() const {
  return this->boundingBox[0];
}

const ops_point& BoundingBox::getLocalMax() const {
  return this->boundingBox[1];
}

const ops_point& BoundingBox::getGlobalMax() const {
  return this->globalBoundingBox[0];
}

const ops_point& BoundingBox::getGlobalMin() const {
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

  if (xm.x <= xl.x || xm.y <= xl.y) {
    ops_printf("Error: Defined bounding box of non-positive volume.");
    exit(-1);
  }
  boundingBox[0].x = xl.x;
  boundingBox[0].y = xl.y;

  boundingBox[1].x = xm.x;
  boundingBox[1].y = xm.y;

  if (dim == 3) {
    if (xm.z <= xl.z) {
      ops_printf("Error: Defined bounding box of non-positive volume.");
      exit(-1);
    }

    boundingBox[0].z = xl.z;
    boundingBox[1].z = xm.z;
  }
}
void BoundingBox::setBoundingBoxLocalBound(const double* xlow, const double* xmax) {

  if (xmax[0] <= xlow[0] || xmax[1] <= xlow[1]) {
    ops_printf("Error: Defined bounding box of non-positive volume.");
    exit(-1);
  }
  boundingBox[0].x = xlow[0];
  boundingBox[0].y = xlow[1];


  boundingBox[1].x = xmax[0];
  boundingBox[1].y = xmax[1];

  if (dim == 3) {
    if (xmax[2] <= xlow[2]) {
      ops_printf("Error: Defined bounding box of non-positive volume.");
      exit(-1);
    }
    boundingBox[0].z = xlow[2];
    boundingBox[1].z = xmax[2];
  }
}

void BoundingBox::setBoundingBoxLocalBound(const double *boxregion) {

  if (boxregion[0] > boxregion[1] || boxregion[2] > boxregion[3]) {
    ops_printf("Error: Defined bounding box of non-positive volume");
    exit(-1);
  }

  boundingBox[0].x = boxregion[0];
  boundingBox[0].y = boxregion[2];

  boundingBox[1].x = boxregion[1];
  boundingBox[1].y = boxregion[3];

  if (dim == 3) {
    if (boxregion[4] > boxregion[5]) {
      ops_printf("Error: Defined bounding of non-positive volume");
      exit(-1);
    }

    boundingBox[0].z = boxregion[4];
    boundingBox[1].z = boxregion[5];
  }

}


void BoundingBox::setBoundingBoxGlobalBound(const ops_point &xlow, const ops_point &xmax) {
  if (xmax.x <= xlow.x || xmax.y <= xlow.y) {
    ops_printf("Error: Defined global bounding box of non-positive volume.");
    exit(-1);
  }
  globalBoundingBox[0].x = xlow.x;
  globalBoundingBox[0].y = xlow.y;


  globalBoundingBox[1].x = xmax.x;
  globalBoundingBox[1].y = xmax.y;

  if (dim == 3) {
    if (xmax.z <= xlow.z) {
      ops_printf("Error: Defined global bounding box of non-positive volume.");
      exit(-1);
    }
    globalBoundingBox[0].z = xlow.z;
    globalBoundingBox[1].z = xmax.z;
  }
}

void BoundingBox::setBoundingBoxGlobalBound(double* xlow,  double* xmax) {
  if (xmax[0] <= xlow[0] || xmax[1] <= xlow[1]) {
    ops_printf("Error: Defined global bounding box of non-positive volume.");
    exit(-1);
  }
  globalBoundingBox[0].x = xlow[0];
  globalBoundingBox[0].y = xlow[1];


  globalBoundingBox[1].x = xmax[0];
  globalBoundingBox[1].y = xmax[1];

  if (dim == 3) {
    if (xmax[2] <= xlow[2]) {
      ops_printf("Error: Defined global bounding box of non-positive volume.");
      exit(-1);
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

  if (!owned)
    return false;

  if (boundingBox[0].x > point.x || boundingBox[1].x < point.x)
    return false;

  if (boundingBox[0].y > point.y || boundingBox[1].y < point.y)
    return false;

  if (dim == 3) {
    if (boundingBox[0].z > point.z || boundingBox[1].z < point.z)
      return false;
  }

  return true;
}

bool BoundingBox::isCoordinateInBoundingBox(const double *point) {
  if (!owned)
    false;

  if (boundingBox[0].x > point[0] || boundingBox[1].x < point[0])
    return false;

  if (boundingBox[0].y > point[1] || boundingBox[1].y < point[1])
    return false;

  if (dim == 3) {
    if (boundingBox[0].z > point[2] || boundingBox[1].z < point[2])
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

  if (globalBoundingBox[0].x > point.x || globalBoundingBox[1].x < point.x)
    return false;

  if (globalBoundingBox[0].y > point.y || globalBoundingBox[1].y < point.y)
    return false;

  if (dim == 3)
    if (globalBoundingBox[0].z > point.z || globalBoundingBox[1].z < point.z)
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

  if (fabs(hB - hA) < rA + rB) {
    *xbox_int_low=MAX(xbox1_low, xbox2_low);
    *xbox_int_hi = MIN(xbox1_hi, xbox2_hi);
    return 1;
  }

  printf("Enerting into computing intersection");
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

/******************************************************************************
 * Bounding Box API functions
 *****************************************************************************/

BoundingBox* ops_create_bounding_box(int dim) {
  BoundingBox* box = new BoundingBox{dim};

  return box;
}

BoundingBox* ops_create_bounding_box(const ops_block block, int dim,
                                    ops_point &point_low, ops_point &point_max) {
  BoundingBox* box = new BoundingBox{block, dim, point_low, point_max};

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
void ops_set_bounding_box_from_dat(BoundingBox* box,const ops_dat coords, double dx,
                                   int dim) {
  if (box == NULL) {
    ops_printf("Please define a BoundingBox first\n");
    exit(-1);
  }

  if (dim < 2 && dim > 3) {
    ops_printf("A bounding box is defined only for 2D and 3D spaces\n");
    exit(-1);
  }

  /* For generating an object with the given function requires domain
   * partition
   */

  if (!ops_partitioned()) {
    ops_printf("Bounding box cannot be defined via a ops_dat structure.");
    exit(-1);
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
 * return 0: region overlap
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

  int iel =0;
  for (int i = 0; i < dim;i++)  {
    iel +=  (xbox_lo[i] == intersection->getMinCoordDir(i)) ? 0 : 1;
    iel +=  (xbox_hi[i] == intersection->getMaxCoordDir(i)) ? 0 : 1;
    if (iel < 2 * dim) return intersection;

  }

  a1 = 1;
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

/*******************************************************************************/
/* Particle hanlding functions                                                 *
 ******************************************************************************/

/*-----------------------------------------------------------------------------*
 * ops_arg for particle data structures                                        *
 *-----------------------------------------------------------------------------*/

ops_arg ops_arg_dat_particle(ops_dat dat, int dim, char const *type, ops_access acc) {
  (void) type;
  ops_arg temp = ops_arg_dat_core(dat, nullptr, acc);
  (&temp)->dim = dim;
  (&temp)->argtype = OPS_ARG_DAT_PARTICLE;

  return temp;
}

/*------------------------------------------------------------------------------
 * Function declares and defines a particle data list                           *
 *------------------------------------------------------------------------------*/
ops_particle  _ops_decl_particle(OPS_instance *instance, ops_block block,
                                 BoundingBox *box) {

  /* Create ops_particle */
  ops_particle particle = new ops_particle_core; //(ops_particle)ops_calloc(1, sizeof(ops_particle_core));

  /* Set default values */
  particle->no_particles = 0;
  particle->no_virtual = 0;
  particle->global_particles = particle->no_particles;
  particle->block = block;
  particle->box_block = box;
  particle->Nmax = 10;

  particle->mark_deletion  =(int *) ops_malloc(sizeof(int) * particle->Nmax);

  particle->particle_pos_dat = NULL;
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

  /* Remove ops pointers from list */
  if (!particle->particle_data.empty())
    particle->particle_data.clear();

  /* Remove ops_dat (pointer) from list */

  if (!particle->mapping_to_grid.empty())
    particle->mapping_to_grid.empty();

//  if (particle->box_block != NULL)
//    ops_free(particle->box_block);

  delete particle;
  return NULL;
}

/*---------------------------------------------------------------------------------------*/
/* Function for the allocation of ops_dat structures associated with particles           */
/*---------------------------------------------------------------------------------------*/
void ops_particle_realloc_data(ops_particle particle, int noalloc) {

  if (particle->no_particles + noalloc < particle->Nmax)
    return;

  /* Reallocate the ops_dat structures */
  particle->Nmax += noalloc + 100;


  particle->mark_deletion = (int *) ops_realloc(particle->mark_deletion,
                                                particle->Nmax * sizeof(int));
  /* Reallocate ops_dat structures */
 // particle->particle_data[0] =
      ops_dat_realloc_core(particle->particle_pos_dat,particle->Nmax);
  //TODO: Add shape directly

  if (particle->particle_envelope != NULL) {
       ops_dat_realloc_core(particle->particle_envelope, particle->Nmax);
  }
  for (ops_dat &dat : particle->particle_data) {
    ops_dat_realloc_core(dat, particle->Nmax);
  }

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

  for (ops_dat &dat_elem : particle->particle_data) {
    if (strcmp(dat_elem->name, dat->name) == 0)
      return 1;
  }

  return 0;
}


void ops_exit_particles(OPS_instance *instance) {

  /* Get block descriptor */
  ops_block_descriptor *block_list = instance->OPS_block_list;

  for (int i = 0; i < instance->OPS_block_index; i++) {
    ops_particle particle= block_list[i].particle;
    particle = _ops_free_particle(particle);
  }

}
/*******************************************************************************/
/*  Particle API  Functions
 *******************************************************************************/

ops_particle ops_decl_particle(ops_block block, BoundingBox *Box) {
   return _ops_decl_particle(OPS_instance::getOPSInstance(), block, Box);
}

/*------------------------------------------------------------------------------*/
/* The function inserts Np particles into the block that owns the ops_particle_t
 * structure (particle). It is assumed that the user has performed a sanity check
 * for introduced particles in the block.
 */
/*------------------------------------------------------------------------------*/

void ops_particles_insert(ops_particle particle, int Np) {

  /* Update number of particles */
  particle->no_particles += Np;


  int re_alloc{0};
  if (particle->no_particles > particle->Nmax)
    re_alloc = 1;

  if (re_alloc) {
    particle->Nmax = particle->no_particles + 100;
    //TODO: See if I can shift into & x : pos_dta
 //   for (std::size_t i = 0; i < particle->particle_pos_dat.size(); i++)
 //     ops_dat_realloc_core(particle->particle_data[i], particle->Nmax);
  }

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

  for (long int i = 0; i < no_particles; i++) {
    particle->mark_deletion[i] = 0;
  }



  double *xpos = (double *) particle->particle_pos_dat->data;
  for (long int i = 0; i < no_particles; i++) {
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
  long int Nlocal = particle->no_particles;
  long int Ndel{0};

  int i = 0;
  while ( i < Nlocal) {
    int imark = particle->mark_deletion[i];

    if (imark == 1) {
      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              dim * sizeof(double));
      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      for (ops_dat &dat : particle->particle_data) {
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      Nlocal--;
    }
    else i++;
  }

  particle->no_particles = Nlocal;
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
  halo_info->nmax = 10;
  halo_info->sendlist = (int *)ops_calloc(halo_info->nmax, sizeof(int));
  halo_info->firstrecv = 0;

  return halo_info;
}

ops_particle_halo_data _ops_particle_decl_halo_data_core(OPS_instance *instance,
                                                         ops_dat from, ops_dat to,
                                                         int *dir_from, int *dir_to,
                                                         double *translate,
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
                                             "the same size per element.");

  if (from->type_size != to->type_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "Different types of to and from ops_dat"
                       "structures");

  if (!from->is_particle || !to->is_particle)
    throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure not associated with"
                       "particle data structures");


  halo->from = from;
  halo->to = to;
  halo->orient = orient_flag;
  for (int i = 0; i < from->block->dims; i++) {
    halo->from_dir[i] = (orient_flag == OPS_PART_ORIENT_ON) ? dir_from[i] : i;
    halo->to_dir[i] = (orient_flag == OPS_PART_ORIENT_ON) ? dir_to[i] : i;
    halo->translate[i] =
        (orient_flag == OPS_PART_ORIENT_OFF || translate == nullptr) ?  0.  :translate[i];
  }
  for (int i = from->block->dims; i < OPS_MAX_DIM; i++) {
    halo->from_dir[i] = 0;
    halo->to_dir[i] = 0;
    halo->translate[i] = 0.0;
  }

  instance->OPS_particle_halo_data_list[instance->OPS_particle_halo_data_index] = halo;
  instance->OPS_particle_halo_data_index++;


  return halo;
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

  grp->instance = instance;
  grp->nPoints = 0;

  int nbites = 0;
  for (int i = 0; i < nhalos; i++)
    nbites += halos[i]->from->elem_size;
  grp->nbites = nbites;

  instance->OPS_particle_halo_list[instance->OPS_particle_halo_group_index] = grp;
  grp->index = instance->OPS_particle_halo_index++;

  return grp;
}

ops_particle_halo_group _ops_particle_decl_halo_group(OPS_instance *instance,
                                                      ops_particle_halo particle_halos[],
                                                      int nhalos,
                                                      ops_part_halo_grp_type halo_type,
                                                      ops_part_loop_type loop_type,
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

  for (int i = 0; i < nhalos; i++)
    halo_grp->halo_info[i] = _ops_particle_init_halo_info();

  halo_grp->halo_type = halo_type;

  halo_grp->loop_type = loop_type;

  if (halo_type == OPS_HALO_GRP_FORWARD || halo_type == OPS_HALO_GRP_BACKWARD) {
    halo_grp->halo_master = master; //Yes if we shift bites to a different locaton
  }

  instance->OPS_particle_halo_group_list[instance->OPS_particle_halo_group_index] = halo_grp;
  instance->OPS_particle_halo_group_index++;

  return halo_grp;
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
                                                   int *dir_from, int *dir_to,
                                                   double *translate,
                                                   ops_part_orient orient_flag) {

  return _ops_particle_decl_halo_data_core(OPS_instance::getOPSInstance(),from, to,
                                           dir_from, dir_to, translate, orient_flag);
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


ops_particle_halo_group ops_particle_decl_halo_group(ops_particle_halo particle_halos[],
                                                     int nhalos,
                                                     ops_part_halo_grp_type halo_type,
                                                     ops_part_loop_type loop_type,
                                                     ops_particle_halo_group master) {

  return _ops_particle_decl_halo_group(OPS_instance::getOPSInstance(), particle_halos,
                                       nhalos, halo_type, loop_type, master);
}

void _ops_particle_pack_halo_data(char *buf, ops_particle_halo_data* halo_data, int ndata,
                                  int ipart) {

  int size_data = 0;
  for (int i = 0; i < ndata; i++) {
    ops_dat dat = halo_data[i]->from;
    int nbites = dat->elem_size;
    memcpy(buf + size_data, dat->data + ipart * nbites, nbites);
    size_data += dat->elem_size;
  }

}

void _ops_particle_unpack_halo_data(char *buff, ops_particle_halo_data* halo_data, int ndata,
                                    int iloc) {

  char *temp;

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
      int *dir_to, *dir_from;
      double *translate;

      translate = halo_data[i]->translate;
      dir_to = halo_data[i]->to_dir;
      dir_from = halo_data[i]->from_dir;


      double *shifted = (double *)ops_calloc(dims, sizeof(double));
      if (dim != dims || data_to->type_size != sizeof(double))
        throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat should with orient"
                                                           " should be a vector");

      memcpy(temp, buff + size_loc, data_to->elem_size);

      double *tmp = (double *)temp;

      for (int isou = 0; isou < dim; isou++) {
        shifted[dir_to[isou]] = tmp[dir_from[isou]] - translate[dir_from[isou]];
      }

      memcpy(data_to->data + bites, (char *)shifted, data_to->elem_size);
      ops_free(shifted);
    }
    size_loc += data_to->elem_size;
  }

  ops_free(temp);
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
    printf("Halo type exchange\n");
    _ops_particle_setup_exchange_comm(instance, halo_grp); //TODO:
    break;
  case  OPS_HALO_GRP_BORDER:
    printf("Halo type border\n");
    _ops_particle_setup_border_comm(instance, halo_grp);
    break;
  case OPS_HALO_GRP_FORWARD:
  case OPS_HALO_GRP_BACKWARD:
    _ops_particle_setup_for_rev_comm(instance, halo_grp);
    break;
  case OPS_HALO_GRP_DEFAULT:
    _ops_particle_setup_default_comm(instance, halo_grp); //TODO:
    //tODO
    break;
  default:
    break;
  }



}

void ops_particle_halo_transfer_group(ops_part_halo_grp_type exchange_type) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {
    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];
    switch (halo_grp->halo_type)
    case OPS_HALO_GRP_EXCHANGE: {
      _ops_particle_halo_exchange_transfer(instance, halo_grp); //TODO
      break;
    case OPS_HALO_GRP_BORDER:
      _ops_particle_halo_border_transfer(instance, halo_grp);
      break;
    case OPS_HALO_GRP_FORWARD:
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
  }
}

void ops_particle_halo_transfer(ops_particle_halo_group halo_grp) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  switch(halo_grp->halo_type) {
  case OPS_HALO_GRP_EXCHANGE:
    _ops_particle_halo_exchange_transfer(instance, halo_grp); //TODO
    break;
  case OPS_HALO_GRP_BORDER:
    _ops_particle_halo_border_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_FORWARD:
    _ops_particle_halo_forward_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_BACKWARD:
    _ops_particle_halo_reverse_transfer(instance, halo_grp);
    break;
  case OPS_HALO_GRP_DEFAULT:
//    _ops_particle_halo_default_transfer(instance, halo_grp); //TODO
    break;
  default:
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Exchange type not defined.");
  }
}



/**************************************************************************************************/
/* Mapping functions                                                                              */
/**************************************************************************************************/


ops_particle_mapping ops_decl_mapping(ops_particle particle, ops_dat grid, ops_dat radius,
                                      ops_stencil stencil,
                                      ops_with_virtual include_virtual,
                                      ops_shape_evolve particle_changes,
                                      ops_grid_type grid_type,
                                      double epsilon,
                                      double crit_length, int Ng) {

  if (particle == NULL) {
    ops_printf("Particle structure is empty. Generate mapping structures after particle "
               "definition.\n");
    exit(-1);
  }

  if (grid == NULL) {
    ops_printf("Empty ops_dat structure. Mappings must be set after grid ops_dat"
               "definitions\n");
    exit(-1);
  }

  /* TODO: Sanity check to verify that the two structures are owned by the same block */

  /* Assure that particle block is the same with the grid block */
  ops_block block_particle = particle->block;
  ops_block block_grid = grid->block;
  if (strcmp(block_particle->name, block_grid->name) != 0) {
    ops_printf("Error: Try to generate mapping for different blocks. Please check your "
               "settings.\n");
    exit(-1);
  }

  /* create new structure */
  ops_particle_mapping map = new ops_particle_mapping_core; //Try to shift to the other structure

  map->mapping_type = include_virtual;

  if (!(map->mapping_type =  OPS_WITH_VIRTUAL || map->mapping_type == OPS_NO_VIRTUAL)) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Mapping supports only mapping of actual "
                                             "and with virtual particles.");
  }

  /* Set uniform or non uniform grid */

  if (grid_type < 0 || grid_type > 1) {
    ops_printf("Error: Only uniform and non-uniform structured grids are supported.\n");
    exit(-1);
  }
  map->grid_type = grid_type;
  if (map->grid_type == OPS_UNIFORM_GRID) {
    map->Ngrids = (Ng > 1) ? Ng : 1;
  }


  double* nulld{nullptr};
  int*    nulli{nullptr};

  /* Define structures */
  int dim = block_grid->dims;
  int size[dim], base[dim], d_m[dim], d_p[dim];

  //TODO: Verify that we can keep that way. Check size
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i];
    base[i] = grid->base[i];
    d_p[i] = grid->d_p[i];
    d_m[i] = grid->d_m[i];
  }

  //Find if stencil has more points to access the list
  for (int i = 0; i < dim; i++) {
    for (int p = 0; p < stencil->points; p++) {
      d_p[i] = MAX(d_p[i], stencil->stencil[particle->block->dims * p + i]);
      d_m[i] = MIN(d_m[i], stencil->stencil[particle->block->dims * p + i]);
    }
  }

  /* Create a copy */
  map->grid = grid;
  /* Set mapping structures */
  map->binhead = ops_decl_dat(particle->block, 1, size, base, d_m, d_p,
                              nulli, "int", "name_to_do");


  /* Decleration of particle structures */
  map->nParticles = particle->no_particles;
  map->Nmax = 10;

  map->parts_to_grid = ops_decl_particle_dat(particle, 1, base, nulli,
                                             "int", "particle_to_map");

  //TODO: Modify structure */
  map->bin = ops_decl_particle_dat(particle, 1, base, nulli, "int", "name_to_do");

  map->pos_old = ops_decl_particle_dat(particle, 3, base, nulld, "double", "name_to_do");

  /* Set changes of particle and define Rp_old*/
  map->particle_changes = particle_changes;;
  if (map->particle_changes == OPS_EVOLV_SHAPE) {
    ops_decl_particle_dat(particle, 1, base, nulld, "double", "ADD");
  }
  map->decide = true;

  /* Copy Rp dat structure */
  map->Rp = radius;


  /* Set remapping options */
  if (epsilon < 0.0) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-positive flag");
  }

  if (crit_length < 0.0)  {
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-positive length");
    crit_length = 1.0;
  }

  map->skin = epsilon * crit_length;
  /* Define uniform grid mapping */

  /* Set stencil for building maps for virtual particles */
  map->mapping_stencil = stencil;


  /* Add the mapping structure to the particle list */
  particle->mapping_list.push_back(map);

  return map;
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
    if (map->grid_type == OPS_UNIFORM_GRID)
      _ops_particle_build_local_uniform(map, particle); //TODO
    else if (map->grid_type == OPS_NON_UNIFORM_GRID)
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
  if (map->decide) {
    if (map->grid_type == OPS_UNIFORM_GRID)
      _ops_particle_build_local_uniform(map, particle);
    else if (map->grid_type == OPS_NON_UNIFORM_GRID)
      _ops_particle_build_local_non_uniform(map, particle);

    map->decide = false;
  }
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

  if (decide > 1)
    decide = true;
}

/*-----------------------------------------------------------------------------------*/
/* \brief the map for sequential code
 *
 * \return a flag that at least a single list is build
 */

bool ops_particle_update_map_lists(ops_particle particle) {


  bool decide_global{false};
  for (auto &map : particle->mapping_list) {
    ops_particle_map_decide(particle, map);

    if (!decide_global)
      decide_global = (map->decide) ? true : false;

  }

  /* Mark particles for exchange or add them into an array */
  if (decide_global) {
    ops_particle_mark_for_del(particle);

    _ops_particle_exchange(particle);

    ops_particle_create_forward_comms(particle);
  //Exchange particles for MPI and build comms for forward communications

  }
  return decide_global;

}

void ops_particle_build_maps(ops_particle particle) {


  //Prior to building expand the lists
  ops_particle_remove_marked(particle);

  for (auto & map : particle->mapping_list)
    ops_particle_map_build(particle, map);

}

/*----------------------------------------------------------------------------------------*/
/* Auxiliarry functions to be moved
 * ---------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------*
 *  Swap data between points in particle lists
 *--------------------------------------------------------------------------------------*/
void _ops_particle_swap_data(char *data, int i, int j, int elems) {

  for (int ielem = 0; ielem < elems; i++) {
    char ctmp = data[elems * i + ielem];
    data[elems * i + ielem] = data[elems * j + ielem];
    data[elems * j + ielem] = ctmp;
  }
}

/*--------------------------------------------------------------------------------------*
 *  Build bounding boxes for ops_particle structures
 *--------------------------------------------------------------------------------------*/
void ops_build_bounding_box(ops_particle particle ) {

  particle->box_block->partitionBoundingBox(particle->block);
}

