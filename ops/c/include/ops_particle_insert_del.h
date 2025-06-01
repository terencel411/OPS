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
/*#if __cplusplus >= 201402L
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
*/

int find_offset(ops_arg arg, int *iprev, int elem, int insert_elem) {
  int offs{0};
  if (   arg.argtype==OPS_ARG_DAT_PARTICLE) {
    ops_dat dat = arg.dat;
    if (!dat->is_particle)
      throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure is not "
                                               "related to a particle structure");
    offs = (elem - (*iprev)) * dat->dim; //TODO: Check if I need bytes

  //  printf("elem = %d Offs = %d iprev = %d dim = %d\n", elem , offs, *iprev, dat->dim);

    (*iprev) = elem;
  }
  else if ( arg.argtype == OPS_ARG_GBL_PARTICLE) {
    offs = (insert_elem - (*iprev)) * arg.dim;
    (*iprev) = elem; //TODO: Vrf that it must be elem
  }

  return offs;
}

template <typename... ParamType, typename... OPSARG, size_t... J>
void ops_particle_insert_impl(indices<J...>,
                              int (*kernel_decide)(int , double* , double* , double*, double *),
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
  BoundingBox *boxBlock = particle->box_block;
  double xmin[dim], xmax[dim];
  boxBlock->getLocalMaxMin(xmin, xmax);

  ops_block block = particle->block;
  int ndim = particle->block->dims;

  /* First loop identify particles to insert */
  int Nmax{10};
  int Ninsert{0};
  int *inserting_particles = nullptr;
  int ilast{0};
  inserting_particles = (int *) ops_malloc(sizeof(int) * Nmax);

  for (int i = 0; i < Nins; i++) {
    double *x_loc = (xCrds + dim * i);
    double *rShape = (shape != nullptr) ? shape + i : nullptr;

    int decide = kernel_decide(dim, xmin, xmax, x_loc, rShape);
    if (decide) {
      Ninsert++;
      if (Ninsert > Nmax) {
        Nmax += 10;
        inserting_particles = (int *)ops_realloc(inserting_particles, sizeof(int) * Nmax);
      }

      inserting_particles[Ninsert-1] = i;
      ilast = i;
    }
  }


  /* Inserting at the front or at the back */
  if (flag) particle->no_particles = 0;

  /* Update number of particles */
  particle->no_particles += Ninsert;
  ops_particle_realloc_data(particle);

  ilast = 0;

  int iprev[N];
  for (int i = 0; i < N; i++)
    iprev[i] = 0;

  /* Get additional elements to access data structures */

  ops_arg args[N] = {arguments...};

  char *p_a[N] =
    {particle_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim, ndim, block,
                                                                        NULL, particle->no_particles)...};

  /* Inserting particles in the loop */
  int elem = (flag) ? 0 : particle->no_particles - Ninsert;
  for (int i = 0; i < Ninsert; i++) {
    double *x_local = (xCrds + dim * inserting_particles[i]);
    _ops_particle_add_elem(dim, x_local, particle->particle_pos_dat, elem);
    elem++;

    int offs[N] = {find_offset(arguments, &iprev[J], i, inserting_particles[i])...};

    (void) std::initializer_list<int>{(
        particle_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], offs[J]), 0)...}; //TODO-Debug but before add

    (void) std::initializer_list<int>{(
        particle_param_handler<param_remove_cvref_t<ParamType>>::shift_address(arguments, p_a[J], offs[J]), 0)...};

    kernel((particle_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
  }

  /* Enforce rebuild on particle lists */
  if (Ninsert > 0)
   for (auto &mapping : particle->mapping_list) {
     mapping->decide = true;
   }

}
template <typename... ParamType, typename... OPSARG, size_t... J>
void   ops_particle_user_delete_impl(indices<J...>,  int (*kernel)(ParamType... ),
                                     char const *name, ops_particle particle,
                                     int dim, OPSARG... arguments) {
#ifdef OPS_MPI
  ops_block = particle->block;
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
#endif
  constexpr int N = sizeof...(OPSARG);
  int *markForDeletion = particle->mark_deletion;

  ops_block block = particle->block;
  int ndim = particle->block->dims;

  ops_arg args[N] = {arguments...};

  char *p_a[N] =
    {particle_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim, ndim, block,
                                                                        NULL, particle->no_particles)...};

  int iprev[N];
  for (int i = 0; i < N; i++)
   iprev[i] = 0;

  size_t nParticles = particle->no_particles;
  int imarked{0};
  for (size_t i = 0; i < nParticles; i++) {
    int offs[N] = {find_offset(arguments, &iprev[J], i,  i)...};

    (void) std::initializer_list<int>{(
        particle_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], offs[J]), 0)...}; //TODO-Debug but before add

    (void) std::initializer_list<int>{(
        particle_param_handler<param_remove_cvref_t<ParamType>>::shift_address(arguments, p_a[J], offs[J]), 0)...};

    if (markForDeletion[i] == 1) continue;

    markForDeletion[i] = kernel((particle_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);

  }

// Enforce list build if not requested
  for (auto &map : particle->mapping_list)
    if (map->decide) return;

  int enforce{0};
  for (size_t i = 0; i < nParticles; i++) {
    if (markForDeletion[i]==1) {
      enforce = 1; break;
    }

  }

  for (auto &map : particle->mapping_list)
    map->decide = true;

  //TODO: Shift into MPI

}

/*------------------------------------------------------------------------------------------------*/
/* \brief Function for inserting particles in the simulation
 *
 * \param[in] kernel           Pointer to function for adding/initializing particle data structures
 * \param[in] kernel_decide    Pointer to function for deciding if a particle will be inserted in
 *                             the simulation
 * \param[in] flag             Flag for inserting particles at the front (1) or at the back (0) of
 *                             the particle list
 * \param[in] dim              size of the spatial space
 * \param[in] Nins             number of particles to be inserted in the list
 * \param[in] xcrds            array containing the coordinates of the particles to be inserted
 * \param[in] shape            array containing the shape of the particles to be inserted
 * \param[in] args             user defined arguments for inserting particle data into the lists
 */

template <typename... ParamType, typename... OPSPARG>
void ops_particle_insert(void (*kernel)(ParamType...), char const *name,
                         int (*kernel_decide)(int, double*, double*,double *, double *),
                         ops_particle particle, int flag, int  dim, int Nins, double *xCrds,
                         double *shape, OPSPARG... args) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSPARG),
           "Number of inserting kernel parameters should match the number ops_parg");
  ops_particle_insert_impl(build_indices<sizeof...(ParamType)>{}, kernel_decide,  kernel, name,
                           particle, flag, dim, Nins, xCrds, shape, args...);
}

/*---------------------------------------------------------------------------------------------*/
/* \brief Function for removing particles based on user defined criteria
 *
 * \param[in] kernel         Pointer to function for removing particles from the block
 *                           The particular function must be called after
 *                           ops_particle_update_map_lists
 * \param[in] name           Name of the kernel function
 * \param[in] particle       ops_particle_structure of a given block
 * \param[in] dim            size of the spatial space
 * \param[in] args           used defined arguments for deleting particles for the list
 *
 */
/*---------------------------------------------------------------------------------------------*/

template <typename... ParamType, typename...OPSPARG>
void ops_particle_user_delete(int (*kernel)( ParamType...), char const *name,
                              ops_particle particle, int dim,
                              OPSPARG... args) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSPARG),
                "Number of user kernel parameters do not match the number of ops_parg params");

  ops_particle_user_delete_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                                particle, dim,  args...);

}
void ops_particle_remove(ops_particle particle);
void ops_particle_init_mark_deletion(ops_particle particle);
#endif //C++ 2011
#endif /* OPS_OPS_C_INCLUDE_OPS_PARTICLE_INSERT_H_ */
