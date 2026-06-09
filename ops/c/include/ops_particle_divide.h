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
  * @brief Functionalities for generating N particles based on
  *        user defined weights \sum_wi = 1.
  * @author Valantis Tsinginos
  * @details Generating a quasi-uniform number of particles per
  *          rank based on user defined weights.
  */

#ifndef _OPS_PARTICLE_DIVIDE_H_
#define _OPS_PARTICLE_DIVIDE_H_

#define EPS_LIMIT 1e-12

#include <ops_exceptions.h>
#include <ops_lib_core.h>

template<typename T>
struct weightIndex{
  T w;   //weight
  int index; //original index
};

template<typename T>
inline void _swap_WI(weightIndex<T> *a, weightIndex<T> *b) {

  weightIndex<T> t = *a;  *a = *b; *b = t;
}

template <typename T>
int _ops_part_decv(weightIndex<T> *work, int left,int right, int pivotIndex) {
  T pv = work[pivotIndex].w;

  _swap_WI(&work[pivotIndex], &work[right]);
  int store  = left;

  for (int i = left; i < right; i++) {
      if (work[i].w > pv) {           /* strictly greater */
        _swap_WI(&work[i], &work[store]);
          ++store;
      }
  }

  _swap_WI(&work[store] , &work[right]);
  return store;

}

template <typename T>
void _ops_select_k_desc(weightIndex<T> *work, int left, int right, int k) {

  while (left < right) {
    int p = left + (right - left) / 2;
    int m = _ops_part_decv(work, left, right, p);
    if (m == k) return;
    if (k < m) right = m - 1;
    else left = m + 1;
  }
}


template <typename T>
void _ops_particle_partition_selection(const T *weights,const int Ninsert,
                                       const int nranks, int *nparts_out) {

  for (int i = 0; i < nranks; i++)
    nparts_out[i] = 0;

  //Sanity check
  T sum = 0;
  for (int i = 0; i < nranks; i++) {
    if (weights[i] < 0.)
      throw OPSException(OPS_RUNTIME_ERROR,
                         "ERROR: Non-positive weights for particle insertion");
    sum += weights[i];
  }

  if (sum < EPS_LIMIT)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Sum of weights is non-positive");

  T *t = (T *) ops_malloc(sizeof(T) * nranks);
  weightIndex<T> *weight_inds = (weightIndex<T> *) ops_malloc(sizeof(weightIndex<T>) * nranks);

  int sum_floor = 0;
  for (int i = 0; i < nranks; i++) {
    t[i] = Ninsert * weights[i];
    int ni = (int) ops_floor(t[i]);
    sum_floor += ni;
    nparts_out[i] = ni;

    weight_inds[i].w =  (weights[i] - (T) ni > 0) ? weights[i] - (T) ni : 0.0;
    weight_inds[i].index =  i;
  }

  int R = Ninsert - sum_floor;

  if (R == 0) {
    ops_free(t);
    ops_free(weight_inds);
    return;
  }



  if (R > 0) {
    int need = R;
    if (need > nranks) need = nranks;

    weightIndex<T> *work
      = (weightIndex<T> *) ops_malloc(sizeof(weightIndex<T>) * nranks);

    for (int i = 0; i < nranks; i++) work[i] = weight_inds[i];

    int kth = need - 1;
    if (kth < 0) kth = 0;
    if (kth >= nranks) kth = nranks - 1; //TODO: Check

    _ops_select_k_desc(work, 0, nranks-1, kth);//TODO
    T tau = work[kth].w;
    ops_free(work);

    int given = 0;
    for (int i = 0; i < nranks; i++) {
      if (weight_inds[i].w > tau + EPS_LIMIT) {
        nparts_out[i] += 1;
        given++;
      }
    }

    //Seond pass: ties to tau
    for (int i = 0; i < nranks  && given < R; i++) {
      if (fabs(weight_inds[i].w - tau) <= EPS_LIMIT) {
        nparts_out[i] += 1;
        ++given;
      }

    }
  }

  ops_free(t);
  ops_free(weight_inds);
}

template <typename T>
void _ops_particle_partition_less_than_ranks(const T *weights, const int Ninsert,
                                             const int nranks, int *nparts_out) {
  for (int i = 0; i < nranks; i++)
    nparts_out[i] = 0;

  if (Ninsert == 0) return;

  if (Ninsert == 1) {
    nparts_out[0] = 1; return;
  }

  weightIndex<T> *arr = (weightIndex<T> *) ops_malloc(sizeof(weightIndex<T>) * nranks);

  for (int i = 0; i < nranks; i++) {
    arr[i].w = weights[i];
    arr[i].index = i;
  }

  int k = nranks - 1;
  _ops_select_k_desc(arr, 0, nranks-1, k); //TODO

  T threshold = arr[k].w;

  int assigned = 0;
  for (int i = 0; i < nranks; i++)
    if (weights[i] > threshold + EPS_LIMIT) {
      nparts_out[i] = 1;
      assigned++;
    }

  for (int i = 0; i < nranks && assigned < Ninsert; i++) {
     if (fabs(weights[i] - threshold) <= EPS_LIMIT) {
       nparts_out[i] = 1;
       assigned++;
     }
  }

  ops_free(arr);
}


/**
 * Generates the number of particles that must be inserted in different
 * processes based on volume weights
 *
 * @param weights      a pointer to the weights for each rank
 * @param Ninsert      number of particles to insert
 * @param nrank        number of processes
 * @param nparts_out   particles to be generated in each rank
 */

template <typename T>
void ops_weight_particle_partition(const T *weights, const int Ninsert,
                                   const int nranks, int * nparts_out) {


  ops_printf("Particles to insert for weights [ ");
  for (int i = 0; i < nranks; i++)
    ops_printf(" %f ", ((double *)weights)[i]);
  printf(" ] = %d\n", Ninsert);

  //Case I: Ninsert <  nsplit
  if (nranks > Ninsert) {
    _ops_particle_partition_less_than_ranks(weights, Ninsert, nranks,nparts_out);
  }
  else { //Case II: Number of particles larger than the number of given ranks
   ops_printf("Case II");
    _ops_particle_partition_selection(weights, Ninsert, nranks, nparts_out); //TODO
  }
}


#endif /* OPS_C_INCLUDE_OPS_PARTICLE_DIVIDE_H_ */
