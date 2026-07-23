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

/** @brief  Box block definitions
  * @author Valantis Tsinginos
  * @details Definitions of BoxBlock functions */

#ifndef __OPS_BOUNDING_BOX_DEFS_H_
#define __OPS_BOUNDING_BOX_DEFS_H_


#include "limits.h"
#include "ops_util.h"
//Structure that defines
template<typename T>
struct ops_point {
    ops_point(T _x, T _y, T _z) {
        x = _x;
        y = _y;
        z = _z;
        sizeT = sizeof(T);
    };

    ops_point() { };

    T x = 0.0;
    T y = 0.0;
    T z = 0.0;
    int sizeT;
};




/**
 * Structure for handling the simulation domain
 */
template<typename T>
class BoundingBox {
  public:
   BoundingBox(const ops_block block, int dim, ops_point<T> minCrd,
               ops_point<T> maxCrd);
   BoundingBox(const ops_dat coords, const T *grid_size, int dim);
   BoundingBox(int dim);
   BoundingBox(int dim, T *xmin, T *xmax);
   ~BoundingBox();

   const ops_point<T> getLocalMin() const;
   const ops_point<T> getLocalMax() const;
   const ops_point<T> getGlobalMax() const;
   const ops_point<T> getGlobalMin() const;
   T getDx(int idir) { if (idir < dim) return dx[idir];
                            return 0.0;}
   void getLocalMaxMin(T *xmin, T* xmax) const;
   T getGlobalMax(int idir);
   T getGlobalMin(int idir);
   bool isCoordinateInBoundingBox(ops_point<T>& point);
   bool isCoordinateInBoundingBox(const T *point) const;
   bool isCoordinateInGlobalBoundingBox(const ops_point<T>& point);
   void setBoundingBoxLocalBound(const ops_point<T> &xlow, const ops_point<T> &xmax);
   void setBoundingBoxLocalBound(const T* xlow, const T* xmax);
   void setBoundingBoxGlobalBound(const ops_point<T> &xlow, const ops_point<T> &xmax);
   void setBoundingBoxGlobalBound(T* xlow, T* xmax);
   void setBoundingBoxLocalBound(const T *boxregion);

   void partitionBoundingBox(ops_block block, ops_dat map_bin = nullptr, T *dx = nullptr);
   T getBlockVolume() { return volume;}

   void setOwnership(bool flag) {owned = flag;}
   bool getOwnership() {return owned;};
   inline int getDim() const { return dim;}
   T getMinCoordDir(int dir) const;
   T getMaxCoordDir(int dir) const;
#ifdef OPS_MPI
   void generateLocalBoundingBox(/*TODO: */);
#endif
  private:
   void generateGlobalBoundingBox(int count);
   int dim = 0; /**< Size of the spatial space */
   bool owned = true; /**< Ownership of the BoundaryBlock on this rank */

   T volume;
   int sizeT;
   std::array<ops_point<T>,2> boundingBox; /**< Part of the simulation box owned by the given rank */
   std::array<ops_point<T>,2> globalBoundingBox; /**< The simulation box of the given block  */
   T dx[OPS_MAX_DIM]; /**< Expansion of the local simulation box (Linked to staggered grids) */
   ops_dat coords; /** < Pointer to an ops_dat structure  */
};

template<typename T>
BoundingBox<T>::BoundingBox(const ops_block block, int dim,
                         ops_point<T> minCrd, ops_point<T> maxCrd) :
                         dim(dim), coords(nullptr) {
  if (dim < 2 && dim > 3) {
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                        "for 2D or 3D geometries\n");
  }

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

  volume = (dim == 3) ?
      (maxCrd.x - minCrd.x) * (maxCrd.y - minCrd.y) * (maxCrd.z - minCrd.z) :
      (maxCrd.x - minCrd.x) * (maxCrd.y - minCrd.y);

  for (int i = 0; i < dim; i++)
    dx[i] = 0.0;

  sizeT = sizeof(T);
  owned = false;

}

template<typename T>
BoundingBox<T>::BoundingBox(int dim) : dim(dim) , coords(nullptr) {
  if (dim < 2 && dim > 3)
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                        "for 2D or 3D geometries\n");

  volume = 0.;

  owned = false;

  sizeT = sizeof(T);
}

template<typename T>
BoundingBox<T>::BoundingBox(const ops_dat coords, const T *Dx,
                         int dim) : dim(dim) , coords(coords) {

  if (dim < 2 && dim > 3) {
    throw OPSException(OPS_INVALID_ARGUMENT, "OPS-bound boxes defined only"
                       "for 2D or 3D geometries\n");
  }

  owned = false;

  for (int i = 0; i < dim; i++)
    dx[i] = Dx[i];

  volume = 0.0;

  sizeT = sizeof(T);

}

template<typename T>
BoundingBox<T>::BoundingBox(int dim, T *xmin, T *xmax) : dim{dim} , coords(nullptr) {

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

  volume = 1.0;
  for (int i = 0; i < dim; i++)
    volume  *= (xmax[i] - xmin[i]);

  owned = false;

  sizeT = sizeof(T);

}

template<typename T>
BoundingBox<T>::~BoundingBox() { }

template<typename T>
const ops_point<T> BoundingBox<T>::getLocalMin() const {

  return this->boundingBox[0];
}

template<typename T>
const ops_point<T> BoundingBox<T>::getLocalMax() const {
  return this->boundingBox[1];
}

template<typename T>
const ops_point<T> BoundingBox<T>::getGlobalMin() const {
  return this->globalBoundingBox[0];
}

template<typename T>
const ops_point<T> BoundingBox<T>::getGlobalMax() const {
  return this->globalBoundingBox[1];
}

template<typename T>
void BoundingBox<T>::getLocalMaxMin(T *xmin, T* xmax) const {
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

template<typename T>
void BoundingBox<T>::setBoundingBoxLocalBound(const ops_point<T> &xl,
                                              const ops_point<T> &xm) {

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

template<typename T>
void BoundingBox<T>::setBoundingBoxLocalBound(const T *xlow, const T *xmax) {
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

template<typename T>
void BoundingBox<T>::setBoundingBoxLocalBound(const T *boxregion) {

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

  //TODO: What volume computes
}

template<typename T>
void BoundingBox<T>::setBoundingBoxGlobalBound(const ops_point<T> &xlow,
                                               const ops_point<T> &xmax) {
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

template<typename T>
void BoundingBox<T>::setBoundingBoxGlobalBound(T* xlow,  T* xmax) {
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

template<typename T>
bool BoundingBox<T>::isCoordinateInBoundingBox(ops_point<T> &point) {
  if (!owned)
    return false;


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

template<typename T>
bool BoundingBox<T>::isCoordinateInBoundingBox(const T* point) const {
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

template<typename T>
bool BoundingBox<T>::isCoordinateInGlobalBoundingBox(const ops_point<T> &point) {

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

template<typename T>
T BoundingBox<T>::getMinCoordDir(int dir) const {

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
    return -1.;
  }

  return -1.;
}

template<typename T>
T BoundingBox<T>::getMaxCoordDir(int dir) const {
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

template<typename T>
T BoundingBox<T>::getGlobalMin(int idir) {
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

template<typename T>
T BoundingBox<T>::getGlobalMax(int idir) {
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



/*=============================================================================*/
/*  Functions for handing BoundingBoxes
 *============================================================================ */
template<typename T>
inline int _check_box_intersection(const int dim, T *xbox1_lo, T *xbox1_hi,
                                   T *xbox2_lo, T * xbox2_hi) {
  int a1{1};

  int i{0};

  while (a1 == 1 && i < dim) {
    T hA = 0.5 * (xbox1_lo[i] + xbox1_hi[i]);
    T hB = 0.5 * (xbox2_lo[i] + xbox1_lo[i]);

    T rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    T rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);

    if (fabs(hB-hA) > rA + rB) a1 = 0;
  }

  return a1;
}

template<typename T>
inline int _compute_intersection_region(T xbox1_low, T xbox1_hi,
                                        T xbox2_low, T xbox2_hi,
                                        T *xbox_int_low,
                                        T *xbox_int_hi) {

  //TODO: intersections
  double hA{0.5 * (xbox1_low + xbox1_hi)};
  double hB{0.5 * (xbox2_low + xbox2_hi)};
  double rA{0.5 * (xbox1_hi - xbox1_low)};
  double rB{0.5 * (xbox2_hi - xbox2_low)};

  if (fabs(hB - hA) <= rA + rB) {
    *xbox_int_low=MAX(xbox1_low, xbox2_low);
    *xbox_int_hi = MIN(xbox1_hi, xbox2_hi);

    double dx = *xbox_int_hi - *xbox_int_low;
    if (dx < std::numeric_limits<T>::epsilon()) return 0; //no overlap
    //Check box sixe
    return 1;
  }

  return 0;
}

template<typename T>
BoundingBox<T>* ops_find_intersection_region(BoundingBox<T> *box, T *region, int &a1) {

  int dim = box->getDim();
  a1 = 0;

  ops_point<T> xmin = box->getLocalMin();
  ops_point<T> xmax = box->getLocalMax();

  T xbox_hi[OPS_MAX_DIM], xbox_lo[OPS_MAX_DIM];
  T xreg_lo[OPS_MAX_DIM], xreg_hi[OPS_MAX_DIM];

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

  T inters[2 * dim];
  for (int i = 0; i < dim; i++) {
    inters[2 * i] = xbox_lo[i];
    inters[2 * i + 1] = xbox_hi[i];
  }
  int overlp{0};
  for (int i = 0; i < dim; i++) {
    int overlp = _compute_intersection_region(xbox_lo[i], xbox_hi[i], xreg_lo[i],
                                              xreg_hi[i],  inters + 2 * i,
                                              inters + 2 * i + 1); //TODO

    if (overlp == 0) {
      a1 = 2;
      return nullptr;
    } //return immediately no-overlapping
  }
  //Check that box bound not equal to  intersection region


  BoundingBox<T>* intersection = new BoundingBox<T>(dim);
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

//TODO: Check if we need

template<typename T>
int ops_check_box_intersection(int dim, int idir, T *xbox1_lo,
                               T *xbox1_hi,  T *xbox2_lo,
                               T *xbox2_hi) {
  int a1 = 1;

  for (int i = 0; i < dim; i++) {

    T hA = 0.5 *  (xbox1_lo[i] + xbox1_hi[i]);
    T hB = 0.5 * (xbox2_lo[i] + xbox2_hi[i]);

    T rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    T rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);


    if (i == idir) {
      if (fabs(hB-hA) > rA + rB) return 0;
    }
    else
      if (fabs(hB-hA) >= rA + rB) return 0;

  }

  return 1;
}


template<typename T>
int ops_check_box_intersection(int dim, T *xbox1_lo,
                               T* xbox1_hi, T *xbox2_lo,
                               T* xbox2_hi) {
  int a1{1};

  int i{0};

  while (a1 == 1 && i < dim) {
    T hA = 0.5 * (xbox1_lo[i] + xbox1_hi[i]);
    T hB = 0.5 * (xbox2_lo[i] + xbox2_hi[i]);

    T rA = 0.5 * fabs(xbox1_hi[i] - xbox1_lo[i]);
    T rB = 0.5 * fabs(xbox2_hi[i] - xbox2_lo[i]);

    if (fabs(hB-hA) > rA + rB) a1 = 0;

    i++;
  }

  return a1;
}

//Checking for overall-No touch is permitted

template<typename T>
int ops_check_box_intersections2(int dim,
                                 T *xbox1_lo, T *xbox1_hi,
                                 T *xbox2_lo, T *xbox2_hi)
{
  int touches[3] = {0, 0, 0};

  // -------------------------------
  // X-axis
  // -------------------------------
  T hA = 0.5 * (xbox1_lo[0] + xbox1_hi[0]);
  T hB = 0.5 * (xbox2_lo[0] + xbox2_hi[0]);

  T dx  = ops_abs(hA - hB);
  T RAx = 0.5 * (xbox1_hi[0] - xbox1_lo[0]);
  T RBx = 0.5 * (xbox2_hi[0] - xbox2_lo[0]);
  T sumX = RAx + RBx;

  // No intersection if distance exceeds combined half widths
  if (fgt(dx, sumX)) return 0;

  // Touch if exactly equal within tolerance
  touches[0] = feq(dx, sumX);

  // -------------------------------
  // Y-axis
  // -------------------------------
  hA = 0.5 * (xbox1_lo[1] + xbox1_hi[1]);
  hB = 0.5 * (xbox2_lo[1] + xbox2_hi[1]);

  T dy  = fabs(hA - hB);
  T RAy = 0.5 * (xbox1_hi[1] - xbox1_lo[1]);
  T RBy = 0.5 * (xbox2_hi[1] - xbox2_lo[1]);
  T sumY = RAy + RBy;

  if (fgt(dy, sumY)) return 0;

  touches[1] = feq(dy, sumY);

  // -------------------------------
  // Z-axis (if 3D)
  // -------------------------------
  if (dim == 3) {
    hA = 0.5 * (xbox1_lo[2] + xbox1_hi[2]);
    hB = 0.5 * (xbox2_lo[2] + xbox2_hi[2]);

    T dz  = fabs(hA - hB);
    T RAz = 0.5 * (xbox1_hi[2] - xbox1_lo[2]);
    T RBz = 0.5 * (xbox2_hi[2] - xbox2_lo[2]);
        T sumZ = RAz + RBz;

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

template<typename T>
int ops_check_box_intersection2(int dim, T *xbox1_lo,
                                T *xbox1_hi, T *xbox2_lo,
                                T* xbox2_hi) {
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

template<typename T>
inline void _ops_compute_map_grid_size_dir(int i, char *dx, char *box_block, int size) {

  BoundingBox<T> *box = (BoundingBox<T> *)box_block;
  T *dxs = (T *)dx;

  dxs[i] = (box->getMaxCoordDir(i) - box->getMinCoordDir(i)) / static_cast<T>(size);

}
template<typename T>
void _ops_particle_number_of_particles_in_range(BoundingBox<T> *box, int dim, T *xcrd,
                                                int noParticles, int *nsend) {
  int nwithin = 0;

  for (int i = 0; i < noParticles; i++) {
    bool isin = box->isCoordinateInBoundingBox(xcrd + dim * i);
    if (isin) nwithin++;
  }

  (*nsend) = nwithin;
}

template<typename T>
void _ops_particle_mapped_into_region(BoundingBox<T> *box, int dim, const T* xcrd,
                                      const int noParticles, int *sendlist) {
  int nwithin = 0;

  for (int iPart = 0; iPart < noParticles; iPart++) {
    bool isin = box->isCoordinateInBoundingBox(xcrd + dim * iPart);
    if (isin) {
      sendlist[nwithin] = iPart;
      nwithin++;
    }
  }
}

template<typename T>
void _ops_particle_remove_from_region(BoundingBox<T> *box, T *env,
                                      const T *xcrds, int  *mark_del,
                                      const size_t noParticles, const int dim,
                                      int *sendlist) {
  int nwithin = 0;

  for (size_t ipart = 0; ipart < noParticles; ipart++) {
    if (env != nullptr) {
      if (env[ipart] < 0.) continue; //TODO: Threhold
    }

    if (mark_del[ipart] != 1) continue;

    if (box->isCoordinateInBoundingBox(xcrds + dim * ipart)) {
      mark_del[ipart] = 2;
      sendlist[nwithin] = ipart;
      nwithin++;
    }
  }
}

template<typename T>
void _ops_particle_mark_for_removal(BoundingBox<T> *box, T *xcrds, int *mark_del,
                                    const int dim, const int ifirst, const int ilast) {

  for (int ipart = ifirst; ipart < ilast; ipart++) {
    mark_del[ipart] = 0;
    if (!box->isCoordinateInBoundingBox(xcrds + dim * ipart))
      mark_del[ipart] = 1;
  }
}

template<typename T>
int _ops_particle_check_for_deletion(int ipart, int bin_part[], int dim, int rmv_limits[],
                                     T *xpos, BoundingBox<T> *box) {

  int check = 0;

  for (int i = 0; i < dim; i++) {
    if ( bin_part[i] < rmv_limits[2 *i] ||
      bin_part[i] > rmv_limits[2 * i + 1]) {
      check = 1; break;
    }
  }

  if (check == 1) return 1;

  ops_point<T> xpoint;
  xpoint.x = xpos[0];
  xpoint.y = xpos[1];
  xpoint.z = (dim == 3) ? xpos[2] : 0.0;

  bool flag = box->isCoordinateInBoundingBox(xpoint);

  return (int) (!flag);
}

template<typename T>
char * _ops_particle_create_box(T* sending_region, const int dims) {

  T xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  for (int i = 0; i < dims; i++) {
    xmin[i] = sending_region[2 * i];
    xmax[i] = sending_region[2 * i + 1];
  }


  return (char *) new BoundingBox<T>(dims, xmin, xmax);
}

#endif /* OPS_C_INCLUDE_OPS_BOUNDING_BOX_DEFS_H_ */
