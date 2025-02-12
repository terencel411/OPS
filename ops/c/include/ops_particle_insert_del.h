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

/** @file
  * @brief OPS functionality for inserting and deleting particles from a
  *        block structure
  * @author C. Tsinginso
  * @details OPS functionality for inserting and deleting particles from  a given
  *          block.
  *          Assumptions: Insertion: Particles are assumed to be freshly created
  *                       by user functionalities. The developed functionalities
  *                       do not exchange particles between blocks
  *                       Deletion: Particles are removed from the system any
  *                       possible exchange between particles is not handled herein
  */

#ifndef __OPS_PARTICLE_INSERT_DEL_H_
#define __OPS_PARTICLE_INSERT_DEL_H_

#include "ops_lib_core.h"
#include "ops_particles_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#if __cplusplus >= 201103L
#include <utility>
#if __cplusplus >= 201402L
// after c++14 use built in types.
template <size_t... I> using indices = std::index_sequence<I...>;
template <size_t Num> using build_indices = std::make_index_sequence<Num>;

#else
// pre c++14 we need our own helper structs
template <size_t... Is> struct indices {};
template <size_t N, size_t... Is>
struct build_indices : public build_indices<N - 1, N - 1, Is...> {};
template <size_t... Is> struct build_indices<0, Is...> : indices<Is...> {};
#endif


template <typename T> struct param_remove_cvref {
  using type =
      typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

// helper struct to get the underlying type of pointer kernel parameters
// e.g. const int * to int *
template <typename T> struct param_remove_cvref<T *> {
  using type = typename std::add_pointer<typename std::remove_cv<
      typename std::remove_reference<T>::type>::type>::type; // remove_reference may be wrong here..
};

template <typename T>
using param_remove_cvref_t = typename param_remove_cvref<T>::type;

/*---------------------------------------------------------------------------------------*/
/* Introduce data to particle lists
 */
template<typename ParamT>
struct param_handler {
  static ParamT *get_data(ops_arg &arg, int ientry, int ip, int Nins) {
    if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      ParamT *data =(ParamT *)arg.data;
      int nelems = arg.dim / Nins;
      return (data != nullptr) ? data + ip * nelems : nullptr;
    }
    else if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ParamT *data = (ParamT *)arg.data;
      return (data != nullptr) ? data + ientry * arg.dim : nullptr;
    }

    return nullptr;
  }

};

template <typename... ParamType, typename... OPSARG, size_t... J>
void ops_particle_insert_impl(indices<J...>, int (*kernel_decide)(int , double* ,
                              double* , double*, double *),
                              void (*kernel)(ParamType... ),
                              char const *name, ops_particle particle, int flag, int dim, int Nins,
                              double *xCrds, double *shape, OPSARG... arguments)
{

#ifdef OPS_MPI
  ops_block = particle->block;
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
#endif
  constexpr int N = sizeof...(OPSARG);
  int count[OPS_MAX_DIM] = {0};
  ops_arg args[N] = {arguments...};

  BoundingBox *boxBlock = particle->box_block;

  double xmin[dim], xmax[dim];

  boxBlock->getLocalMaxMin(xmin, xmax);

  /* Loop over all insert particles */
  for (int i = 0; i < Nins; i++) {
    double *x_local = (xCrds + dim * i);
    double *r_shape = (shape != nullptr) ? shape + i : nullptr; //TODO: Expand

    int decide = kernel_decide(dim, xmin, xmax, x_local, r_shape); //TODO:Add decide

    int ielem = 0;
    if (decide) {
      int ientry = particle->no_particles;

      if (flag) {
        particle->no_particles +=1;
        ielem = particle->no_particles - 1;
      }
      else {
        ielem++;
        if (ielem > particle->no_particles)
          particle->no_particles++;
      }

      ops_particle_realloc_data(particle);

      //TODO: Insert functions //Get right data point
      _ops_particle_add_elem(dim, x_local + dim * i, particle->particle_pos_dat[0],
                             (flag) ? particle->no_particles - 1 : ielem - 1);

      /* Shifting particles to entry point */
      //kernel(... );
      kernel((param_handler<param_remove_cvref_t<ParamType>>::get_data(args[J], ielem, i, Nins))...);
    }
  }
}

template <typename... ParamType, typename... OPSPARG>
void ops_particle_insert(void (*kernel)(ParamType...), char const *name,
                         int (*kernel_decide)(int, double*, double*,double *, double *),
                         ops_particle particle, int  dim, int Nins, double *xCrds,
                         double *shape, OPSPARG... args) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSPARG),
           "Number of inserting kernel parameters should match the number ops_parg");
  ops_particle_insert_impl(build_indices<sizeof...(ParamType)>{}, kernel, kernel_decide, particle,
                           dim, Nins, xCrds, shape, args...);
}

void ops_particle_remove(ops_particle particle);
void ops_particle_init_mark_deletion(ops_particle particle);
#endif //C++ 2011
#endif /* OPS_OPS_C_INCLUDE_OPS_PARTICLE_INSERT_H_ */
