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
  * @brief Sequential handling of building neighbor histories
  * @author Valantis Tsinginos
  * @details Implements the generation of neighbor history list for sequential
  *          implementation
  */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>

namespace ops_histories {

//TODO: Modify always store for the smallest element in the list. It may be safer to access data
//In other terms add elements to the list & remove later on as new element is accessed

  inline int _get_address(const int  *cell_center, const int *point,
                          const int *size, const int dim) {

    int address = 0;
    int prod = 1;
    for (int isou = 0; isou < dim; isou++) {
      int ipoint = point[isou] + cell_center[isou];
      if (ipoint < 0 || ipoint > size[isou] - 1) return -1;
      address += ipoint * prod;
      prod *= size[isou];
    }

    return address;

}

  inline void _get_cell_from_address(const int address, const int dim,
                                     const int size[], const int d_m[],
                                     int cell[]) {

    int addr = address;

    int max_address = 1;
    for (int i = 0; i < dim; i++) max_address *= size[i];

    if (address > max_address) {
      throw OPSException(OPS_RUNTIME_ERROR," Error: The address exceeds max  size");
    }
    for (int i = dim - 1; i >= 0; i--) {
      int prod = 1;
      for (int j = 0; j < i;j++)
        prod *= size[j];

      cell[i] = addr / prod;
      addr -= cell[i] * prod;
    }
  }


  inline int _get_point_to_mapJ(const int address,ops_dat binheadI,
                                ops_dat binheadJ, ops_stencil stencil,
                                const int dim, int local_cell[]) {

    int *stride = stencil->mgrid_stride;
    int celli[OPS_MAX_DIM];

    _get_cell_from_address(address, dim, binheadI->size, binheadI->d_m, celli);

    for (int i = 0; i < dim; i++) celli[i] += binheadI->d_m[i];

    switch (stencil->type) {
    case 0:
      for (int i = 0; i < dim; i++) local_cell[i] = celli[i] -binheadJ->d_m[i];
      break;
    case 1:
      for (int i = 0; i < dim; i++)
        local_cell[i] = celli[i] / stencil->mgrid_stride[i] - binheadJ->d_m[i];
      break;
    case 2:
      for (int i = 0; i < dim; i++) local_cell[i] =
          celli[i] * stencil->mgrid_stride[i] - binheadJ->d_m[i];
      break;
    }

    int point_zero[OPS_MAX_DIM] = {};

    return _get_address(local_cell, point_zero, binheadJ->size, dim);

  }

}


template<typename T, typename... ParamType, typename... args>
void ops_particle_update_neighbor_history(void (*kernel)(T *, ParamType...), char const *name,
                                          ops_neighbor_history history, ops_particle_mapping mapI,
                                          ops_particle_mapping mapJ, ops_stencil stencil,
                                          int dim, args... argument) {


  if (sizeof...(ParamType) != sizeof...(args))
    throw OPSException(OPS_RUNTIME_ERROR,"Error: The number of the kernel parameters varies "
                                         "from the number of OPS args\n");

  if (mapI->particle->index != history->particleI->index) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapI is not associated to particles of type I");
  }
  if ( mapJ->particle->index != history->particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapJ is not associated to particles of type J");

  if ( history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
       history->particleJ->is_wall >= 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: History of particle/wall interaction require "
                                          " particleI to be defined as wall");

  if (!history->history_active)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: For non-active history structures. In all other cases "
                                          "Verlet list can be reconstructed from \n");

  if (history->particleJ->ids == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Particle tags are required for particle J when setting "
                                          "contact histories\n");

  if (!history->particleI->is_wall)
    if (history->particleI->ids == nullptr)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Fpr non-rigid wall histories ids must be defined"
                                            "for particleI\n");

  if (!history->flag_update) return;

  //TODO: Fix histories for no particles!!
  int no_particlesI = (history->particleI->is_wall == 1)
          ? history->particleI->no_particles
          : history->particleI->no_particles + history->particleI->no_virtual;

  int no_particlesJ = (history->particleI->is_wall == 0) ?
                       history->particleJ->no_particles
                    + history->particleJ->no_virtual : history->particleJ->no_particles;

  if (no_particlesI == 0 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    for (int jp = 0; jp < no_particlesJ; jp++)
      ((int *) history->n_partnersJ->data)[jp] = 0;
  }

  if (no_particlesJ == 0) {
    for (int ip = 0; ip < no_particlesI; ip++)
      ((int *) history->n_partnersI->data)[ip] = 0;
  }

  //Reset pairs and return
  if (no_particlesI == 0 || no_particlesJ == 0) {

    for (int i = 0; i < history->nconts; i++) {
      ((int *) history->indexing_local->data)[2 * i] = -1;
      ((int *) history->indexing_local->data)[2 * i + 1] = -1;

      if (_check_integer_type_for_int(history->indexing->type)){
        ((int *) history->indexing->data)[2 * i] = 0;
        ((int *) history->indexing->data)[2 * i + 1 ] = 0;
      }
      else if (_check_integer_type_for_int64(history->indexing->type)) {
        ((uint64_t *) history->indexing->data)[2 * i] = 0;
        ((uint64_t *) history->indexing->data)[2 * i + 1] = 0;
      }
    }

    memset(history->data->data, 0, history->nconts * history->data->elem_size);

    history->nconts = 0;
    return;
  }

  //TODO: Replace with the one from the ops_particle_seq
  int *stencil_act = nullptr;
  int iprod = stencil->points;

  if (history->particleI->is_wall == 1) {
    int address = ((int *)mapI->parts_to_grid->data)[0];
    int cell_to_j[OPS_MAX_DIM];
    ops_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_to_j);

    //Get normal direction
    int inorm;
    int idir_norm;
    for (int i = 0; i < dim; i++) {
      switch(history->particleI->type_box) {
      case sizeof(float):
        inorm = static_cast<int>(((float *) history->particleI->normal_vector->data)[i]);
      break;
      case sizeof(double):
        inorm = static_cast<int>(((double *) history->particleI->normal_vector->data)[i]);
      break;
      case sizeof(long double):
        inorm = static_cast<int>(((long double *) history->particleI->normal_vector->data)[i]);
      break;
      }

      if (inorm == 1 || inorm == -1) {
        idir_norm = i; break;
      }
    }

    //Find points
    int imin = INT_MAX;
    int imax = INT_MIN;

    for (int i = 0; i < stencil->points; i++) {
      imin = MIN(imin, stencil->stencil[dim * i + idir_norm]);
      imax = MAX(imax, stencil->stencil[dim * i + idir_norm]);
    }


    //Set stencil_points;
    iprod = 1;
    int size_dir[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++) {

      if (i == idir_norm) {
       imax = (inorm == 1) ? imax : 0;
       imin = (inorm == 1) ? 0 : imin;
       size_dir[i] = imax - imin + 1;
      }
      else size_dir[i] = mapJ->binhead->size[i];
      iprod *=  size_dir[i];
    }

    stencil_act = (int *) ops_malloc(dim * iprod * sizeof(int));
    int isten = 0;

    for (int i = dim; i < OPS_MAX_DIM; i++) size_dir[i] = 1;

    for (int k = 0; k < size_dir[2]; k++) {
      for (int j = 0; j < size_dir[1]; j++) {
        for (int i = 0; i < size_dir[0]; i++) {
          stencil_act[isten * dim] = (idir_norm == 0) ? imin + i : i - cell_to_j[0];
          stencil_act[isten * dim + 1] = (idir_norm == 1) ? imin + j : j - cell_to_j[1];
          if (dim == 3)
            stencil_act[isten * dim + 2] = (idir_norm == 2) ? imin + k : k- cell_to_j[2];

          isten++;
        }
      }
    }
  } //TODO

  int *stencils = (history->particleI->is_wall == 1) ? stencil_act : stencil->stencil;
  int npoints = (history->particleI->is_wall == 1) ? iprod : stencil->points;

  int flag = 0;

  for (int idim = 0; idim < dim; idim++) {
    int flag_pos = 0;
    int flag_neg = 0;
    for (int i = 0; i < stencil->points; i++) {
       flag_pos += (stencil->stencil[i * dim + idim] > 0) ? 1 : 0;
       flag_neg += (stencil->stencil[i * dim + idim] < 0) ? 1 : 0;
    }

    flag += (flag_pos > 0 && flag_neg > 0) ? 1 : 0;

    flag_pos = -1;
    flag_neg = 1;
    for (int i = 0; i < npoints; i++) {
      flag_pos = MAX(flag_pos, stencils[i * dim + idim]);
      flag_neg = MIN(flag_neg, stencils[i * dim + idim]);

      flag += (flag_pos == -flag_neg) ? 0 : -1;
    }
  }

  if (   history->history_type == OPS_HISTORY_NEWTON_OFF
      && flag < dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Option OPS_HISTORY_NEWTON_OFF "
                                             " requires a stencil that is symmetric");

  if (   history->history_type == OPS_HISTORY_NEWTON_ON
      && flag == dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Option OPS_HISTORY_NEWTON_ON requires"
                                             "not symmetric stencils\n");


  int *part2bin = (int *) mapI->parts_to_grid->data;
  int *binhead = (int *) mapJ->binhead->data;
  int *bins = (int *) mapJ->bin->data;

  int new_pairs = 0;

  //Part I: Reset particle flags
  for (int i = 0; i < no_particlesI; i++) {
    for (int j = 0; j <  ((int *) history->n_partnersI->data)[i]; j++)
      ((int *) history->flagI->data)[i * history->num_neighsI + j] = 0;
  }

  if ( history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    int no_particlesJ = history->particleJ->no_particles + history->particleJ->no_virtual;
    for (int i = 0; i < no_particlesJ; i++)
      for (int j = 0; j <  ((int *) history->n_partnersJ->data)[i]; j++)
      ((int *) history->flagJ->data)[i * history->num_neighsJ + j] = 0;
  }

  //Set all flags of history to zero
  for (int i = 0; i < history->nconts; i++) {
    ((int *)history->flag->data)[i] = 0;
  }

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) d_m[isou] = mapJ->binhead->d_m[isou];

  //By-pass stencil with wall-free build

  if (no_particlesI == 0) return;
  //Part II: Loop over particles to create structures
  for (int i = 0; i < no_particlesI; i++) {
    //Reset flags for all neighbors of I
    int neighs_old = ((int *)history->n_partnersI->data)[i];
    int n_curI = 0;
    int bin_cell = part2bin[i];

    //Get cell projection into mapII
    int cell_center[OPS_MAX_DIM] = {};
    ops_histories::_get_point_to_mapJ(bin_cell, mapI->binhead, mapJ->binhead, stencil,
                                      dim, cell_center); //TODO

    //Loop over user-defined stencil point
    int index;
    int index_dim = history->indexing->dim;
    for (int is = 0; is < npoints; is++) {

      //Part I Get-particle in list
      int addressJ = ops_histories::_get_address(cell_center, stencils + is * dim,
                                                 mapJ->binhead->size, dim); //TODO

      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      //Loop against all possible contacts for the particle in the cell
      while (partJ != - 1) {
        //Check for possible contacts

        if (history->particleI->index == history->particleJ->index &&
            partJ == i) {
          partJ = ((int *) mapJ->bin->data)[partJ];
          continue;
        }

        //Second check if particleI->index == particleJ->index and j > i update and continuel
        if (history->particleI->index == history->particleJ->index &&
           ((int *) mapI->parts_to_grid->data)[i] == ((int *) mapI->parts_to_grid->data)[partJ]) {
          if (partJ > i) {
            partJ = ((int *) mapJ->bin->data)[partJ];
            continue;
          }
        }


        int found = 0;

        //Part I: Search in existing pairs
        for (int in = 0; in < neighs_old; in++) {

          int a1;
          if (_check_integer_type_for_int(history->particleJ->ids->type)) {


            a1 = _pair_exists_in_list(((int *) history->partnersI->data)[i * history->num_neighsI + in],
                                      ((int *) history->particleJ->ids->data)[partJ]);
          }
          else if (_check_integer_type_for_int64(history->particleJ->ids->type))
            a1 = _pair_exists_in_list(((uint64_t *) history->partnersI->data)[i * history->num_neighsI + in],
                                      ((uint64_t *) history->particleJ->ids->data)[partJ]);

          if (a1 == 1 ) {

            found = 1;

            //Update both particle & global flags
            ((int *) history->flagI->data)[i * history->num_neighsI + in] = 1;
            index = ((int *)history->indexI->data)[i * history->num_neighsI + in];

            ((int *) history->flag->data)[index] = 1;

            ((int *)history->indexing_local->data)[2 * index] = i;
            ((int *)history->indexing_local->data)[2 * index + 1] = partJ;

            ((int *) history->indexing->data)[2 * index] = (history->particleI->is_wall) ? i :
                ((int *) history->particleI->ids->data)[i];
            ((int *) history->indexing->data)[2 * index + 1] = ((int *) history->particleJ->ids->data)[partJ];

            break;
          }

        }

        //Part IIa: Contact found but due to structure need to update the other history: Contact exists
        int flagJ = 0;
        if (found == 1 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {

            if (history->particleI->is_wall || _check_integer_type_for_int(history->particleI->ids->type)) {
              int tagI = ( history->particleI->is_wall == 1) ? i : ((int *) history->particleI->ids->data)[i];
              if (_pair_exists_in_list(((int *) history->partnersJ->data)[history->num_neighsJ * partJ + jp],
                                       tagI)) {
                flagJ = 1;
                ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
                break;
              }
            }
            else if (_check_integer_type_for_int64(history->particleI->ids->type)) {
              int64_t tagI = ((int64_t *) history->particleI->ids->data)[i];
              if (_pair_exists_in_list(((int64_t *) history->partnersJ->data)[history->num_neighsJ * partJ + jp],
                                        tagI)) {
                flagJ = 1;
               ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
               break;
              }
            }
            else
              throw OPSException(OPS_RUNTIME_ERROR, "Error: This integer type is not supported"
                                                    " for particle ids\n");
          }

          //Part IIb: Contact not found increase the number of elements
          if (flagJ == 0) {
            int nlocal = ((int *) history->n_partnersJ->data)[partJ];

            if (nlocal + 1 > history->num_neighsJ)
              throw OPSException(OPS_RUNTIME_ERROR, "Error: Number may exceed the number of user defined pairs\n");

            ((int *) history->n_partnersJ->data)[partJ]++;

            if (history->particleI->is_wall ||
                _check_integer_type_for_int(history->particleI->ids->type) ) {
              int tagI = (history->particleI->is_wall) ? i :
                  ((int *) history->particleI->ids->data)[i];
              ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + nlocal] = tagI;
            }
            else if (_check_integer_type_for_int64(history->particleI->ids->type) ) {
              int64_t tagI = (history->particleI->is_wall) ? (int64_t) i :
                  ((int64_t *) history->particleI->ids->data)[i];
              ((int64_t *) history->partnersJ->data)[partJ * history->num_neighsJ + nlocal] = tagI;
            }
            else
               throw OPSException(OPS_RUNTIME_ERROR, "Error: This integer type is not supported"
                                                     " for particle ids\n");


            ((int *) history->flagJ->data)[partJ * history->num_neighsJ + nlocal] = 1;
            //Copy index and continue
            if (history->history_type == OPS_HISTORY_NEWTON_ON) {
              ((int *) history->indexJ->data)[partJ * history->num_neighsJ + nlocal] = index;
              ((int *) history->flagJ->data)[partJ * history->num_neighsJ + nlocal] = 1;
            }
          }
        }

        //Part III: Update contact it may exist on the other particle due to particle moving around
        //add type as well for safety but only if flag = 0 to be updated
        if (found == 0 && history->history_type == OPS_HISTORY_NEWTON_ON) {

          int jfound = 0;
          for (int jpart = 0; jpart < ((int *) history->n_partnersJ->data)[partJ]; jpart++) {

            int pair_found = 0;

            if (history->particleI->is_wall ||
                _check_integer_type_for_int(history->particleI->ids->type)) {
              int tagI = (history->particleI->is_wall == 1) ? i :
                         ((int *) history->particleI->ids->data)[i];
              pair_found = _pair_exists_in_list(((int *) history->partnersJ->data)[partJ * history->num_neighsJ + jpart],
                                                 tagI);
            }
            else if (_check_integer_type_for_int64(history->particleI->ids->type)) {
              int64_t tagI= ((int64_t *) history->particleI->ids->data)[i];
              pair_found = _pair_exists_in_list(((int64_t *) history->partnersJ->data)[partJ * history->num_neighsJ + jpart],
                                                tagI);
            }
            else
              throw OPSException(OPS_RUNTIME_ERROR,"Error: This integer type is not supported "
                                                   "for particle ids\n");


            printf("Found pair %d (%d %d)\n", pair_found, i, partJ);

            if (pair_found) {
              found = 1;

              if (history->particleI->is_wall!= 1 && neighs_old + 1 > history->num_neighsI)
                throw OPSException(OPS_RUNTIME_ERROR, "Error: Maximum number of candidate neighbors for "
                                                       "neighbor list is reached");

              if (history->particleI->is_wall== 1 && neighs_old + 1 > history->num_neighsI) {
                //TODO: Realloc
                history->num_neighsI += OPS_MAX_PART;
                //Re-alloc structures
                history->partnersI->data = (char *) ops_realloc(history->partnersI->data,
                                                                history->partnersI->elem_size
                                                                * history->num_neighsI);

                history->flagI->data = (char *) ops_realloc(history->flagI->data,
                                                            history->flagI->elem_size
                                                            * history->num_neighsI);

                history->indexI->data = (char *) ops_realloc(history->indexI->data,
                                                             history->indexI->elem_size
                                                             * history->num_neighsI);

              }


             //Check if this pair is already set for the same bin
             if (history->particleI->index == history->particleJ->index) { //For the same bin address
               if (((int *) mapI->parts_to_grid->data)[i] == ((int *) mapJ->parts_to_grid->data)[partJ])
               if (((int *) history->flagJ->data)[partJ * history->num_neighsJ + jpart] == 1)
                 break; //No-need for nada
             }

              int index = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jpart];

              //Set particle structures

              if (_check_integer_type_for_int(history->partnersI->type)) {
                ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old] =
                    ((int *) history->particleJ->ids->data)[partJ];
              }
              else if (_check_integer_type_for_int64(history->partnersJ->type)) {
                ((int64_t *) history->partnersI->data)[i * history->num_neighsI + neighs_old] =
                    ((int64_t *) history->particleJ->ids->data)[partJ];
              }
              else
                throw OPSException(OPS_RUNTIME_ERROR,"Error: Particle type not supported for ids\n");

              ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old] = index;
              ((int *) history->flagI->data)[i * history->num_neighsI + neighs_old] = 1;

              neighs_old++;
              //Also set the flag for the jth particle to 1
              if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS)
                ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jpart] = 1;

              ((int *) history->indexing_local->data)[2 * index] = i; //TODO: It was i not index
              ((int *) history->indexing_local->data)[2 * index + 1] = partJ;



              //Flip indices
              if (_check_integer_type_for_int64(history->partnersI->type)) {
                int64_t index_tmp = ((int64_t *) history->indexing->data)[2 * index];
                ((int64_t *) history->indexing->data)[2 * index] = ((int64_t *) history->indexing->data)[2 * index +1];
                ((int64_t *) history->indexing->data)[2 * index + 1] = index_tmp;
              }
              else if (_check_integer_type_for_int(history->partnersI->type)) {
                int index_tmp = ((int *) history->indexing->data)[2 * index];
                ((int *) history->indexing->data)[2 * index] = ((int *) history->indexing->data)[2 * index + 1];
                ((int *) history->indexing->data)[2 * index + 1] = index_tmp;
              }

              //Finally update flag of contact storage
              ((int *)history->flag->data)[index] = 1;
              break;
            }
          } //TODO-Check if flag needs to be updated

        }

        //Part IV: Contact do not found-Update lists-This exist because function is called
        if (found == 0) {
          //Just add to the list for new elements
          if (new_pairs + 1 > history->nmax_new) {
            int nmax = MAX(history->num_neighsI, history->num_neighsJ);
            history->new_neighbors->data = (char *) ops_realloc(history->new_neighbors->data,
                                                                history->new_neighbors->elem_size *
                                                                (nmax * OPS_MAX_PART + history->nmax_new));
            history->nmax_new += OPS_MAX_PART * nmax;
          }

          ((int *) history->new_neighbors->data)[2 * new_pairs ] = i;
          ((int *) history->new_neighbors->data)[2 * new_pairs + 1] = partJ; //TODO shift to IDS


           found = 1;

//           if (history->particleI->index == history->particleJ->index) {
//
//           }

           //Re-allocate structures

           /*
           if (history->particleI->is_wall != 1 && neighs_old + 1 > history->num_neighsI)
             throw OPSException(OPS_RUNTIME_ERROR,"Error: The number of particle candidates exceed the user "
                                                  "defined size\n");

           if (history->particleI->is_wall == 1 && neighs_old + 1 > history->num_neighsI) {
             history->num_neighsI = neighs_old + 100;
             history->partnersI->data = (char *) ops_realloc(history->partnersI->data,
                                                             history->partnersI->elem_size
                                                             * history->num_neighsI);
             history->flagI->data = (char *) ops_realloc(history->flagI->data,
                                                         history->flagI->elem_size
                                                       * history->num_neighsI);
             history->indexI->data = (char *) ops_realloc(history->indexI->data,
                                                          history->indexI->elem_size
                                                       *  history->num_neighsI);
           }

           //Add the element to the list if new pair

           if (_check_integer_type_for_int(history->partnersI->type)) {
             ((int *) history->partnersI->data)[history->num_neighsI * i + neighs_old] =
                     ((int *) history->particleJ->ids->data)[partJ];
           }
           else if (_check_integer_type_for_int64(history->partnersI->type)) {
             ((int64_t *) history->partnersI->data)[history->num_neighsI * i + neighs_old] =
                     ((int64_t *) history->particleJ->ids->data)[partJ];
           }

           ((int *) history->flagI->data)[history->num_neighsI * i + neighs_old] = 1;
           ((int *) history->indexI->data)[history->num_neighsI * i + neighs_old] = new_pairs + history->nconts;

           neighs_old++;

           if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
             int num_neighsJ_old = ((int *) history->n_partnersJ->data)[partJ];


             if (num_neighsJ_old + 1 > history->num_neighsJ)
               throw OPSException(OPS_RUNTIME_ERROR,"Error: Maximum number of candidate neighbors for "
                                                    " neighbor list is reached\n");

             if (history->particleI->is_wall ||
                 _check_integer_type_for_int(history->particleI->ids->type)) {
               int tagI = (history->particleI->is_wall) ? i :
                          ((int *) history->particleI->ids->data)[i];
               ((int *) history->partnersJ->data)[history->num_neighsJ * partJ
                                                  + num_neighsJ_old] = tagI;
             }
             else if (_check_integer_type_for_int64(history->particleI->ids->type)) {
               int64_t tagI = ((int64_t *) history->particleI->ids->data)[i];
               ((int64_t *) history->partnersJ->data)[history->num_neighsJ * partJ
                                                      + num_neighsJ_old] = tagI;
             }

             ((int *) history->flagJ->data)[history->num_neighsJ * partJ + num_neighsJ_old] = 1;
             ((int *) history->indexJ->data)[history->num_neighsJ * partJ + num_neighsJ_old] = new_pairs + history->nconts;
             ((int *) history->n_partnersJ->data)[partJ]++;
           }

*/
           new_pairs++;
        }

        partJ = ((int *) mapJ->bin->data)[partJ];

      }

    }


    ((int *) history->n_partnersI->data)[i] = neighs_old;

  }
/*
  //Part III: Remove old pairs from local cells
  for (int i = 0; i < history->particleI->no_particles; i++) {
    printf("Prior removal :Particle %d has %d partners: ", ((int *) history->particleI->ids->data)[i],
           ((int *) history->n_partnersI->data)[i]);
    for (int ic = 0; ic < ((int *)history->n_partnersI->data)[i]; ic++)
    printf("[%d status = %d index = %d]", ((int *) history->partnersI->data)[history->num_neighsI * i + ic],
           ((int *) history->flagI->data)[history->num_neighsI * i + ic], ((int *) history->indexI->data)[history->num_neighsI * i + ic]);
    printf("\n");
  }
*/


  int *flagI = (int *) history->flagI->data;
  for (int i = 0; i < history->particleI->no_particles; i++) {
    int neighs_old = ((int *) history->n_partnersI->data)[i];
    for (int j =  0; j < neighs_old; ) {
      //Removing condition
      if (flagI[i * history->num_neighsI + j] == 0) {

        if (j == neighs_old - 1) {
          neighs_old--;
          break;
        }

        if (_check_integer_type_for_int(history->partnersI->type))
          ((int *) history->partnersI->data)[i * history->num_neighsI + j] =
           ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old - 1];
        else if (_check_integer_type_for_int64(history->partnersI->type))
          ((int64_t *) history->partnersI->data)[i * history->num_neighsI + j] =
          ((int64_t *) history->partnersI->data)[i * history->num_neighsI + neighs_old - 1];

        //Copy index and flag
        flagI[i * history->num_neighsI + j] =  flagI[i *history->num_neighsI + neighs_old - 1];
        ((int *) history->indexI->data)[i * history->num_neighsI + j] =
           ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old - 1];

        neighs_old--;
      }
      else {
        j++;
      }

    }

    ((int *) history->n_partnersI->data)[i] = neighs_old;
  }

  /*
  printf("\n");
  for (int i = 0; i < history->particleI->no_particles; i++) {
    printf("After removal :Particle %d has %d partners: ", ((int *) history->particleI->ids->data)[i],
           ((int *) history->n_partnersI->data)[i]);
    for (int ic = 0; ic < ((int *)history->n_partnersI->data)[i]; ic++)
    printf("[%d status = %d index = %d]", ((int *) history->partnersI->data)[history->num_neighsI * i + ic],
           ((int *) history->flagI->data)[history->num_neighsI * i + ic], ((int *) history->indexI->data)[history->num_neighsI * i + ic]);
    printf("\n");
  }
  */

  //Part IIIa: Removing elements from the list if the opposite requires
  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    for (int i = 0; i < history->particleJ->no_particles + history->particleJ->no_virtual; i++) {
       int neighs_old  = ((int *) history->n_partnersJ->data)[i];
       for (int j =  0; j < neighs_old; ) {

         if (((int *) history->flagJ->data)[i * history->num_neighsJ + j] == 0) {

           if (j == neighs_old - 1) {
             neighs_old--;
             break;
           }

           if (history->particleI->is_wall ||
               _check_integer_type_for_int(history->partnersJ->type))
             ((int *) history->partnersJ->data)[i * history->num_neighsJ + j] =
                ((int *) history->partnersJ->data)[i * history->num_neighsJ + neighs_old - 1];
           else if (_check_integer_type_for_int64(history->partnersJ->type))
             ((int64_t *) history->partnersJ->data)[i * history->num_neighsJ + j] =
                ((int64_t *) history->partnersJ->data)[i * history->num_neighsJ + neighs_old - 1];


           ((int *) history->indexJ->data)[i * history->num_neighsJ + j] =
               ((int *) history->indexJ->data)[i * history->num_neighsJ + neighs_old - 1];
           ((int *) history->flagJ->data)[i * history->num_neighsJ + j] =
               ((int *) history->flagJ->data)[i * history->num_neighsJ + neighs_old - 1];
           neighs_old--;
         }
         else j++;
       }

       ((int *) history->n_partnersJ->data)[i] = neighs_old;
    }
  }


  //Part IV: Remove pairs from the list to save memory for insertion
  int n_neighs = history->nconts;
  for (int i = n_neighs-1; i >= 0;) {
    if (((int *) history->flag->data)[i] == 0) {

      //Case I: Pair is last
      if (i == n_neighs - 1) { n_neighs--; i--; continue;}

      //Shift pair data
      else  {
      memcpy(history->flag->data + i * history->flag->elem_size,
             history->flag->data + (n_neighs - 1) * history->flag->elem_size,
             history->flag->elem_size);

      memcpy(history->indexing->data + i * history->indexing->elem_size,
             history->indexing->data + (n_neighs - 1) * history->indexing->elem_size,
             history->indexing->elem_size);

      memcpy(history->indexing_local->data + i * history->indexing_local->elem_size,
             history->indexing_local->data + (n_neighs - 1) * history->indexing_local->elem_size,
             history->indexing_local->elem_size);

      if (history->history_active) {
        memcpy(history->data->data + i * history->data->elem_size,
               history->data->data + (n_neighs - 1) * history->data->elem_size,
               history->data->elem_size);
      }
    }

      n_neighs--;

      //Perform local operations to update indexI and indexJ if necessary as pair shifted
      int ipart = ((int *) history->indexing_local->data)[2 * i];
      int jpart = ((int *) history->indexing_local->data)[2 * i + 1];
      if (_check_integer_type_for_int(history->indexing->type)) {
        int tagJ = ((int *) history->indexing->data)[2 * i + 1];
        for (int ic = 0; ic < ((int *) history->n_partnersI->data)[ipart]; ic++) {
          if (((int *) history->partnersI->data)[ipart * history->num_neighsI + ic] == tagJ) {
            ((int *) history->indexI->data)[ipart * history->num_neighsI + ic] = i;
            break;
          }
        }
      }
      else  if (_check_integer_type_for_int64(history->indexing->type)) {
        int64_t tagJ = ((int64_t *) history->indexing->data)[2 * i + 1];
        for (int ic = 0; ic < ((int64_t *) history->n_partnersI->data)[ipart]; ic++) {
          if (((int64_t *) history->partnersI->data)[ipart * history->num_neighsJ + ic] == tagJ) {
            ((int *) history->partnersI->data)[ipart * history->num_neighsI + ic] = i;
            break;
          }
        }
      }

      //Check second particle if necessary
      if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
               history->history_type == OPS_HISTORY_NEWTON_ON) {
        for (int ip = 0; ip < ((int *) history->n_partnersJ->data)[jpart]; ip++) {
          if (_check_integer_type_for_int(history->indexing->type)) {
            int tagI = (history->particleI->is_wall) ? ipart : ((int *) history->indexing->data)[2 * i];
            if (((int *) history->partnersJ->data)[jpart * history->num_neighsJ + ip] == tagI) {
              ((int *) history->indexJ->data)[jpart * history->num_neighsJ + ip] = i;
              break;
            }
          }
          else if (_check_integer_type_for_int64(history->indexing->type)) {
            int64_t tagI = (history->particleI->is_wall) ? ipart : ((int64_t *) history->indexing->data)[2 * i];
            if (((int64_t *) history->partnersJ->data)[jpart * history->num_neighsJ + ip] == tagI) {
              ((int *) history->indexJ->data)[jpart * history->num_neighsJ + ip] = i;
              break;
            }
          }
        }
      }


    }
    else {i--;};
  }

  history->nconts = n_neighs;
  //TODO: Modify the insertion
  //Add new elements to list
  if (history->nconts +  new_pairs > history->nmax_cont) {
    int n_upgrade = history->nconts + new_pairs + OPS_MAX_PART * MAX(history->num_neighsI,
                                                                     history->num_neighsJ);
    history->indexing->data = (char *) ops_realloc(history->indexing->data,
                                                   history->indexing->elem_size * n_upgrade);
    history->flag->data = (char *) ops_realloc(history->flag->data,
                                               history->flag->elem_size * n_upgrade);


    history->data->data = (char *) ops_realloc(history->data->data, history->data->elem_size
                                               * n_upgrade);

    history->indexing_local->data = (char *) ops_realloc(history->indexing_local->data,
                                                         history->indexing_local->elem_size * n_upgrade);

    history->nmax_cont = n_upgrade;
  }

  //Allocate walls is neccessary
  if (history->particleI->is_wall == 1 && history->num_neighsI + new_pairs > history->num_neighsI) {
    history->num_neighsI += new_pairs + 100;

    history->partnersI->data = (char *) ops_realloc(history->partnersI->data,
                                                    history->partnersI->elem_size
                                                    * history->num_neighsI);
    history->flagI->data = (char *) ops_realloc(history->flagI->data,
                                                history->flagI->elem_size
                                                * history->num_neighsI);
    history->indexI->data = (char *) ops_realloc(history->indexI->data,
                                                 history->indexI->elem_size
                                                 *  history->num_neighsI);
  }

/*
  for (int i = 0; i < history->nconts; i++)
  printf("Before new generation: Pair %d [%d %d] (%d %d) with flag %d at maps [%d %d]\n",
         i, ((int *) history->indexing_local->data)[2 * i],
         ((int *) history->indexing_local->data)[2 * i + 1],
         ((int *) history->indexing->data)[2 * i], ((int *) history->indexing->data)[2 * i + 1],
         ((int *) history->flag->data)[i], ((int *) mapI->parts_to_grid->data)[((int *) history->indexing_local->data)[2 * i]],
         ((int *) mapI->parts_to_grid->data)[((int *) history->indexing_local->data)[2 * i+1]]);
*/

  //Part V: Adding forcing terms

  //printf("Inserting %d pairs at list\n", new_pairs);

  int *flag_data = (int *) history->flag->data;
  for (int i = 0; i < new_pairs; i++) {
    int cont_number = history->nconts + i;
    flag_data[cont_number] = 1;

    int ielem = ((int *)history->new_neighbors->data)[2 * i];
    int jelem = ((int *)history->new_neighbors->data)[2 * i + 1];

    if (_check_integer_type_for_int(history->indexing->type)) {
      int tagI = (history->particleI->is_wall != 1) ?
                 ((int *) history->particleI->ids->data)[ielem] : ielem;
      ((int *) history->indexing->data)[2 * cont_number] = tagI;
      ((int *) history->indexing->data)[2 * cont_number + 1] =
          ((int *) history->particleJ->ids->data)[jelem];
    }
    else if (_check_integer_type_for_int64(history->indexing->type)) {
      int64_t tagI;
      if (history->particleI->is_wall)
        tagI = (int64_t) ielem;
      else if (history->particleI->ids->type_size == sizeof(int)) {
        int tag = ((int *) history->particleI->ids->data)[ielem];
        tagI = (int64_t) tag;
      }
      else if (history->particleI->ids->type_size == sizeof(int64_t))
        tagI = ((int64_t *) history->particleI->ids->data)[ielem];
      else
        throw OPSException(OPS_RUNTIME_ERROR,"Error: Invalid type for tags");

      ((int64_t *) history->indexing->data)[2 * cont_number] = tagI;

      int64_t tagJ;
      if (history->particleJ->ids->type_size == sizeof(int)) {
        tagJ = (int64_t) ((int *) history->particleJ->ids->data)[jelem];
      }
      else if (history->particleJ->ids->type_size == sizeof(int64_t))
        tagJ = ((int64_t *) history->particleJ->ids->data)[jelem];

      ((int64_t *) history->indexing->data)[2 * cont_number + 1] =
          tagJ;
    }

    ((int *) history->indexing_local->data)[2 * cont_number] = ielem;
    ((int *) history->indexing_local->data)[2 * cont_number + 1] = jelem;


    //TODO: Add a check
    if (history->history_active)
      kernel((T*) history->data->data + history->data->dim * cont_number, argument...);

    //Add pair to particle lists
    int ninsI = ((int *) history->n_partnersI->data)[ielem];


    if (ninsI + 1 > history->num_neighsI) {//TODO: Expand it if necessary
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "Error: Pairs for particle "<<ielem<<" exceeds max number of pairs ("<<ninsI + 1<<", "<< history->num_neighsI<<"\n";
      ex << "map = "<< mapI->binhead->size[0]<< ", "<<mapI->binhead->size[1];
      throw ex;
    }
    int index = ielem * history->num_neighsI + ninsI;
    if (history->particleJ->ids->type_size == sizeof(int))
       ((int *) history->partnersI->data)[index]
                                     = ((int *) history->particleJ->ids->data)[jelem];
    else if (history->particleJ->ids->type_size == sizeof(int64_t)) {
      ((int64_t *) history->partnersI->data)[index] =
          ((int64_t *) history->particleJ->ids->data)[jelem];
    }

    ((int *) history->flagI->data)[index] = 1;
    ((int *) history->indexI->data)[index] = cont_number;
    ((int *) history->n_partnersI->data)[ielem]++;

    //TODO: Add it for the second particle
    if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
        history->history_type == OPS_HISTORY_NEWTON_ON) {
      int ninsJ = ((int *) history->n_partnersJ->data)[jelem];
      if (history->particleI->is_wall) {

      }
      int indexJ = jelem * history->num_neighsJ + ninsJ;
      if (history->particleI->is_wall ||
          history->particleI->ids->type_size == sizeof(int)) {
        int tagI = (history->particleI->is_wall) ? ielem : ((int *) history->particleI->ids->data)[ielem];
        ((int  *) history->partnersJ->data)[indexJ] = tagI;
      }
      else if (history->particleI->ids->type_size == sizeof(int64_t)) {
        int64_t tagI = ((int64_t *) history->particleI->ids->data)[ielem];
        ((int64_t *) history->partnersJ->data)[indexJ] = tagI;
      }
      ((int *) history->flagJ->data)[indexJ] = 1;
      ((int *) history->indexJ->data)[indexJ] = cont_number;

      ((int *) history->n_partnersJ->data)[jelem]++;
    }
  }

  history->nconts += new_pairs;

  //Remove pairs from list-j.

//  if (history->particleI->is_wall == 0)
//    printf("Prior removal I have %d pairs\n", history->nconts);


  /*
  for (int i = 0; i < history->nconts; i++) {
    printf("Pair %d [%d %d] with [%d %d] flag %d with data = [%12.9e %12.9e] in grid [%d %d]\n", i, ((int *) history->indexing_local->data)[2 * i],
           ((int *) history->indexing_local->data)[2 * i + 1], ((int *) history->indexing->data)[2 * i],
           ((int *) history->indexing->data)[2 * i + 1], ((int *) history->flag->data)[i],
           ((double *) history->data->data)[2 * i], ((double *) history->data->data)[2 * i + 1],
           ((int *) mapI->parts_to_grid->data)[((int *) history->indexing_local->data)[2 * i]],
           ((int *) mapJ->parts_to_grid->data)[((int *) history->indexing_local->data)[2 * i + 1]]);
  }

  for (int ip = 0; ip < history->particleJ->no_particles; ip++) {
    printf("Particle %d has %d partners: ", ((int *) history->particleJ->ids->data)[ip],
           ((int *) history->n_partnersJ->data)[ip]);
    for (int ic = 0; ic < ((int *) history->n_partnersJ->data)[ip]; ic++)
      printf(" [%d (in %d)] ", ((int *) history->partnersJ->data)[ip * history->num_neighsJ + ic],
             ((int *) history->indexJ->data)[ip * history->num_neighsJ + ic]);
    printf("\n");
  }
  */

  history->flag_update = false;
  ops_free(stencil_act);

}


//TODO: Update local pair- on the fly=do we need that
template<typename T, typename... ParamType, typename... args>
void ops_particle_setup_neighbor_history(void (*kernel)(T *, ParamType...), char const *name,
                                         ops_neighbor_history history, ops_particle_mapping mapI,
                                         ops_particle_mapping mapJ, ops_stencil stencil,
                                         int dim, args... argument) {

  //Sanity checks similar as above
  if (sizeof...(ParamType) != sizeof...(args))
    throw OPSException(OPS_RUNTIME_ERROR,"Error: The number of the kernel parameters varies "
                                         "from the number of OPS args\n");

  if (mapI->particle->index != history->particleI->index) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapI is not associated to particles of type I");
  }
  if ( mapJ->particle->index != history->particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapJ is not associated to particles of type J");

  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
      history->particleJ->is_wall == 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: For wall-particle interactions particleJ must "
                                          "be an actual particle\n");

  if (!history->flag_update) return;

  int *stencil_act = nullptr;
  int iprod = stencil->points;


  int idir_norm;

  if (history->particleI->is_wall == 1) {
    int address = ((int *)mapI->parts_to_grid->data)[0];


    int cell_to_j[OPS_MAX_DIM];
    ops_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_to_j);

    //Get normal direction
    int inorm;
    for (int i = 0; i < dim; i++) {
      switch(history->particleI->type_box) {
      case sizeof(float):
        inorm = static_cast<int>(((float *) history->particleI->normal_vector->data)[i]);
      break;
      case sizeof(double):
        inorm = static_cast<int>(((double *) history->particleI->normal_vector->data)[i]);
      break;
      case sizeof(long double):
        inorm = static_cast<int>(((long double *) history->particleI->normal_vector->data)[i]);
      break;
      }

      if (inorm == 1 || inorm == -1) {
        idir_norm = i; break;
      }
    }

    //Find points
    int imin = INT_MAX;
    int imax = INT_MIN;

    for (int i = 0; i < stencil->points; i++) {
      imin = MIN(imin, stencil->stencil[dim * i + idir_norm]);
      imax = MAX(imax, stencil->stencil[dim * i + idir_norm]);
    }

    //Set stencil_points;
    iprod = 1;
    int size_dir[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++) {

      if (i == idir_norm) {
       imax = (inorm == 1) ? imax : 0;
       imin = (inorm == 1) ? 0 : imin;
       size_dir[i] = imax - imin + 1;
      }
      else size_dir[i] = mapJ->binhead->size[i];
      iprod *=  size_dir[i];
    }

    stencil_act = (int *) ops_malloc(dim * iprod * sizeof(int));
    int isten = 0;

    for (int i = dim; i < OPS_MAX_DIM; i++) size_dir[i] = 1;


    for (int k = 0; k < size_dir[2]; k++) {
      for (int j = 0; j < size_dir[1]; j++) {
        for (int i = 0; i < size_dir[0]; i++) {
          stencil_act[isten * dim] = (idir_norm == 0) ? imin + i : i - cell_to_j[0];
          stencil_act[isten * dim + 1] = (idir_norm == 1) ? imin + j : j - cell_to_j[1];
          if (dim == 3)
            stencil_act[isten * dim + 2] = (idir_norm == 2) ? imin + k : k- cell_to_j[2];

          isten++;
        }
      }
    }
  }


  int *stencils = (history->particleI->is_wall == 1) ? stencil_act : stencil->stencil;
  int npoints = (history->particleI->is_wall == 1) ? iprod : stencil->points;

  int flag = 0;

  for (int idim = 0; idim < dim; idim++) {
    int flag_pos = 0;
    int flag_neg = 0;
    for (int i = 0; i < npoints; i++) {
       flag_pos += (stencils[i * dim + idim] > 0) ? 1 : 0;
       flag_neg += (stencils[i * dim + idim] < 0) ? 1 : 0;
    }

    flag += (flag_pos > 0 && flag_neg > 0) ? 1 : 0;

    flag_pos = -1;
    flag_neg = 1;
    for (int i = 0; i < stencil->points; i++) {
      flag_pos = MAX(flag_pos, stencil->stencil[i * dim + idim]);
      flag_neg = MIN(flag_neg, stencil->stencil[i * dim + idim]);

      flag += (flag_pos == -flag_neg) ? 0 : -1;
    }
  }



  if (   history->history_type == OPS_HISTORY_NEWTON_OFF
      && flag < dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Option OPS_HISTORY_NEWTON_OFF "
                                             " requires a stencil that is symmetric");

  if (   history->history_type == OPS_HISTORY_NEWTON_ON
      && flag == dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Option OPS_HISTORY_NEWTON_ON requires"
                                             "not symmetric stencils\n");

  int nconts = 0;
  int *part2bin = (int *)mapI->parts_to_grid->data;

  int *binheadJ = (int *)mapJ->binhead->data;
  int *binsJ = (int *)mapJ->bin->data;

  int nlocal = (history->particleI->is_wall) ? history->particleI->no_particles :
                     history->particleI->no_particles + history->particleI->no_virtual;

  if (nlocal == 0) return;

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) d_m[isou] = mapJ->binhead->d_m[isou];

  //No need for tag history!!
  for (int i = 0; i < nlocal; i++) {

    int address = part2bin[i];

    int n_neighs = 0;
    int cell_center[OPS_MAX_DIM];

    ops_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,stencil,
                                      dim, cell_center); //TODO

    //Loop over user-defined stencils
    for (int is = 0; is < npoints; is++) {


      int addressJ = ops_histories::_get_address(cell_center, stencils + is * dim,
                                                 mapJ->binhead->size,  dim);
      if (addressJ < 0) continue;
      int partJ = binheadJ[addressJ];
      while (partJ != - 1) {

        if (history->particleI->index == history->particleJ->index  &&
            partJ == i) {partJ = binsJ[partJ]; continue;}

        n_neighs++;
        partJ = binsJ[partJ];
      }
    }

    if (history->particleI->is_wall != 1 && n_neighs > history->num_neighsI)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: The number of possible neighbors exceeds "
                                            "the user defined one");

    //Reallocate data for wall structures


    ((int *)history->n_partnersI->data)[i] = n_neighs;

    nconts += n_neighs;
  }

  //Re-allocation for walls
  if (history->particleI->is_wall == 1 && nconts > history->num_neighsI) {
    history->num_neighsI = nconts + OPS_MAX_PART;
    history->partnersI->data = (char *) ops_realloc(history->partnersI->data, history->num_neighsI
                                                                             * history->partnersI->elem_size);
    history->flagI->data = (char *) ops_realloc(history->flagI->data, history->num_neighsI
                                                * history->flagI->elem_size);
    history->indexI->data = (char *) ops_realloc(history->indexI->data, history->num_neighsI
                                                 * history->indexI->elem_size);
  }



  //Reallocation of global structures
  if (nconts >= history->nmax_cont) {
   // int nJ = (history->particleI->is_wall) //TODO: Check definition critical for MPI
    history->nmax_cont = nconts + OPS_MAX_PART * MAX(history->num_neighsI, history->num_neighsJ);

    if (history->history_active)
      history->indexing->data = (char *) ops_realloc(history->indexing->data,
                                                     history->nmax_cont * history->indexing->elem_size);

    history->flag->data = (char *) ops_realloc(history->flag->data,
                                               history->nmax_cont * history->flag->elem_size);

    if (history->history_active)
      history->data->data = (char *) ops_realloc(history->data->data,
                                                 history->nmax_cont * history->data->elem_size);

    history->indexing_local->data = (char *) ops_realloc(history->indexing_local,
                                                         history->indexing_local->elem_size *
                                                         history->nmax_cont);
  }


  //Update the lists
  history->nconts = nconts;
  nconts = 0;


  //Set for both ways the flags to zero
  if (history->history_type == OPS_HISTORY_NEWTON_ON &&
      history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    int nlocalJ = history->particleJ->no_particles
               + history->particleJ->no_virtual;
    for (int i = 0; i < nlocalJ; i++)
      ((int *) history->n_partnersJ->data)[i] = 0;
  }

  //Part III: Populate histories
 // T* hist_data = (T *)history->data->data;
 // uint64_t *indexing = (uint64_t *) history->indexing->data;
  int *flags = (int *) history->flag->data;

  for (int i = 0; i < nlocal; i++) {

    int address = part2bin[i];
    int cell_center[OPS_MAX_DIM];
    ops_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_center);
    int n_neighs = 0;
    //Loop over all cells to generate history
    for (int icell = 0; icell < npoints; icell++) {
      int addressJ = ops_histories::_get_address(cell_center, stencils + icell * dim,
                                                 mapJ->binhead->size, dim);

      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      while (partJ != - 1) {

        if (history->particleI->index == history->particleJ->index  &&
            partJ == i) {partJ = binsJ[partJ]; continue;}

        if (n_neighs + 1 > history->num_neighsI && !history->particleI->is_wall)
          throw OPSException(OPS_RUNTIME_ERROR, "Error: The maximum number of contacts for particle I "
                                                "exceeds the maximum number of pairs defined by the user");



        if (_check_integer_type_for_int(history->partnersI->type)) {
          ((int *)history->partnersI->data)[i * history->num_neighsI + n_neighs] =
              ((int *) history->particleJ->ids->data)[partJ];
        }
        else if (_check_integer_type_for_int64(history->partnersI->type)) {
          ((int64_t *)history->partnersI->data)[i * history->num_neighsI + n_neighs] =
                      ((int64_t *) history->particleJ->ids->data)[partJ];
        }



        ((int *)history->flagI->data)[i * history->num_neighsI + n_neighs] = 1;
        ((int *)history->indexI->data)[i * history->num_neighsI + n_neighs] = nconts;

        //Update if necessary the second particle if
        if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          int nelem = ((int *) history->n_partnersJ->data)[partJ];

          if (nelem + 1 > history->num_neighsJ)
            throw OPSException(OPS_RUNTIME_ERROR, "Error: The maximum number of contacts for particle J "
                                                           "exceeds the maximum number of pairs defined by the user");

          if (history->particleI->is_wall ||
              _check_integer_type_for_int(history->partnersJ->type)) {
            int tagI = (!history->particleI->is_wall) ? ((int *) history->particleI->ids->data)[i] : i;
            ((int *)history->partnersJ->data)[partJ * history->num_neighsJ + nelem] = tagI;
          }
          else if (_check_integer_type_for_int(history->partnersJ->type)) {
            int64_t tagI = (!history->particleI->is_wall) ? ((int64_t *) history->particleI->ids->data)[i] : i;
            ((int64_t *)history->partnersJ->data)[partJ * history->num_neighsJ + nelem] = tagI;
          }
          ((int *)history->flagJ->data)[partJ * history->num_neighsJ + nelem] = 1;
          ((int *)history->indexJ->data)[partJ * history->num_neighsJ + nelem] = nconts;
          ((int *) history->n_partnersJ->data)[partJ]++;

        }

        //update contact history
        ((int *) history->flag->data)[nconts] = 1;

        if (history->history_active) {
          if (_check_integer_type_for_int(history->indexing->type)) {
            int tagI = (history->particleI->is_wall != 1) ? ((int *) history->particleI->ids->data)[i] : i;
            int tagJ = ((int *) history->particleJ->ids->data)[partJ];
            ((int *) history->indexing->data)[2 * nconts] = tagI;
            ((int *) history->indexing->data)[2 * nconts + 1] = tagJ;
          }
          else if (_check_integer_type_for_int64(history->indexing->type)) {
            int64_t tagI{-1};
            if (history->particleI->is_wall)
              tagI = (int64_t) i;
            else if (_check_integer_type_for_int(history->particleI->ids->type))
              tagI = (int64_t) ((int *) history->particleI->ids->data)[i];
            else if (_check_integer_type_for_int64(history->particleI->ids->type))
              tagI =  ((int64_t *) history->particleI->ids->data)[i];
            ((int64_t *) history->indexing->data)[2 * nconts] = tagI;

            int64_t tagJ;
            if (_check_integer_type_for_int(history->particleJ->ids->type))
              tagI = (int64_t) ((int *) history->particleJ->ids->data)[i];
            else if (_check_integer_type_for_int64(history->particleJ->ids->type))
              tagI =  ((int64_t *) history->particleJ->ids->data)[i];
            ((int64_t *) history->indexing->data)[2 * nconts + 1] = tagJ;
          }
        }

        ((int *) history->indexing_local->data)[2 * nconts] = i;
        ((int *) history->indexing_local->data)[2 * nconts + 1] = partJ;

        if (history->history_active)
          kernel((T *) history->data->data + history->data->dim * nconts, argument...);

        nconts++;
        n_neighs++;


        partJ = binsJ[partJ];
      }
    }
  }

  history->flag_update = false;
  history->frequency = 0;

}
