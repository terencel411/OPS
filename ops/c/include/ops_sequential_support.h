#ifndef OPS_SEQUENTIAL_SUPPORT_H_
#define OPS_SEQUENTIAL_SUPPORT_H_

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
  *  @brief Header for template support functions for cpu based codes
  *  @author Valantis Tsinginos
  *  @details Include functions for handling backward communications for CPU
  *           backend
  */

#include "ops_lib_core.h"

template <typename T>
void  ops_unpack_halo_seq_backward_data(ops_halo halo, T* buff, int soa,
                                        ops_access access) {
  int dim = halo->from->dim;
  T* tmp = (T*) ops_malloc(sizeof(T));
  T* array = (T* )halo->from->data;
  int ranges[2 * OPS_MAX_DIM] = {0};
  int step[OPS_MAX_DIM] = {0};
  int buf_strides[OPS_MAX_DIM] = {0};

  for (int i = 0; i < OPS_MAX_DIM; i++) {
    if (halo->from_dir[i] > 0) {
      ranges[2 * i] = halo->from_base[i] - halo->from->d_m[i] - halo->from->base[i];
      ranges[2 * i + 1]
             = ranges[2 * i] + halo->iter_size[abs(halo->from_dir[i]) - 1];
      step[i] = 1;
    }
    else {
      ranges[2 * i + 1] = halo->from_base[i] - halo->from->d_m[i] - halo->from->base[i] - 1;
      ranges[2 * i] = ranges[2 * i + 1] + halo->iter_size[abs(halo->from_dir[i])- 1];
      step[i] = -1;
    }

  #if OPS_MAX_DIM > 4
    #if OPS_MAX_DIM == 5
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(5)
    #endif
    #endif
    for (int m = MIN(ranges[8], ranges[9] + 1);
             m < MAX(ranges[8] + 1, ranges[9]); m++) { //2
  #else
    int m = 0; {
  #endif
    #if OPS_MAX_DIM > 3
      #if OPS_MAX_DIM == 4
      #ifdef _OPENMP
      #pragma omp parallel for OMP_COLLAPSE(4)
      #endif
      #endif
      for (int l = MIN(ranges[6], ranges[7]+ 1);
               l < MAX(ranges[6] + 1, ranges[7]); l++) { //3
    #else
      int l = 0; {
    #endif
      #if OPS_MAX_DIM > 2
        #if OPS_MAX_DIM == 3
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(3)
        #endif
        #endif
        for (int k = MIN(ranges[4], ranges[5] + 1);
                 k < MAX(ranges[4] + 1, ranges[5]); k++) { //4
      #else
        int k = 0 {
      #endif
        #if OPS_MAX_DIM > 1
          #if OPS_MAX_DIM == 2
          #ifdef _OPENMP
          #pragma omp parallel for OMP_COLLAPSE(2)
          #endif
          #endif
          for (int j = MIN(ranges[2], ranges[3] + 1);
                   j < MAX(ranges[2] + 1, ranges[3]); j++) { //5
        #else
          int  j = 0; {
        #endif
            for (int i = MIN(ranges[0], ranges[1] + 1);
                     i < MIN(ranges[0] + 1, ranges[1]); i++) {//6
              for (int d = 0; d < halo->from->dim; d++) {

                //Copy the temporary element herein
                memcpy(tmp, buff +  (
                   # if OPS_MAX_DIM > 4
                       (m - ranges[8]) * step[4] * buf_strides[4] +
                   #endif
                   # if OPS_MAX_DIM > 3
                       (l - ranges[6]) * step[3] * buf_strides[3] +
                   #endif
                   # if OPS_MAX_DIM > 2
                       (k - ranges[4]) * step[2] * buf_strides[2] +
                   #endif
                   # if OPS_MAX_DIM > 1
                       (j - ranges[2]) * step[1] * buf_strides[1] +
                   #endif
                       (i - ranges[0]) * step[0] * buf_strides[0]) +
                       d, sizeof(T));

                //Copy to actual location
                int iloc = (soa) ?
                #if OPS_MAX_DIM > 4
                  m * halo->from->size[0] * halo->from->size[1] * halo->from->size[2] * halo->from->size[3] +
                #endif
                #if OPS_MAX_DIM > 3
                  l * halo->from->size[0] * halo->from->size[1] * halo->from->size[2] +
                #endif
                #if OPS_MAX_DIM > 2
                  k * halo->from->size[0] * halo->from->size[1] +
                #endif
                #if OPS_MAX_DIM > 1
                  j * halo->from->size[0] +
                #endif
                  i + d * halo->from->size[0]
                #if OPS_MAX_DIM > 4
                    * halo->from->size[4]
                #endif
                #if OPS_MAX_DIM > 3
                    * halo->from->size[3]
                #endif
                #if OPS_MAX_DIM > 2
                    * halo->from->size[2]
                #endif
                #if OPS_MAX_DIM > 1
                    * halo->from->size[1]
                #endif
                  :(
               #if OPS_MAX_DIM > 4
                 m * halo->from->size[0] * halo->from->size[1] * halo->from->size[2] * halo->from->size[3] +
               #endif
               #if OPS_MAX_DIM > 3
                 l * halo->from->size[0] * halo->from->size[1] * halo->from->size[2] +
               #endif
               #if OPS_MAX_DIM > 2
                 k * halo->from->size[0] * halo->from->size[1] +
               #endif
               #if OPS_MAX_DIM > 1
                 j * halo->from->size[0] +
               #endif
                 i) * dim + d;


                switch (access) {
                case OPS_INC:
                  array[iloc] += *tmp;
                  break;
                case OPS_MIN:
                  array[iloc] = MIN(*tmp, array[iloc]);
                  break;
                case OPS_MAX:
                  array[iloc] = MAX(*tmp, array[iloc]);
                  break;
                default:
                  throw OPSException(OPS_RUNTIME_ERROR,"Invalid reduction type\n");

                }
              }

            }//6

          } //5
        } //4
      }//3
    } //2

  }

  ops_free(tmp);
}




#endif /* OPS_C_INCLUDE_OPS_SEQUENTIAL_SUPPORT_H_ */
