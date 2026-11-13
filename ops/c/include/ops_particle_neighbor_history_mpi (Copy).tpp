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
  * @brief Building neighbor histories for MPI backed
  * @author Valantis Tsinginos
  * @details Implements the generation of neighbor history list for the MPI
  *          implementation of the OPS-Particle.
  */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#include <string>

//#include <ops_mpi_core.h>

namespace ops_mpi_histories {

  inline bool _check_integer_type_for_int(char const *type) {

    return (strcmp(type, "int") == 0 ||
            strcmp(type, "int(4)") == 0 ||
            strcmp(type, "integer") == 0 ||
            strcmp(type, "integer(4)") == 0 ||
            strcmp(type, "integer(kind=4)"));
  }

  inline bool _check_integer_type_for_int64(char const *type) {

    return  (strcmp(type, "int64") == 0 ||
             strcmp(type, "uint64_t") == 0 ||
             strcmp(type, "uint64") == 0 ||
             strcmp(type, "int64_t") == 0);
  }

  inline int _get_address(const int cell_center[], const int point[],
                          const int  size[], const int dim) {



    int address = 0;
    int prod = 1;
    for (int isou = 0; isou < dim; isou++) {  //TODO: Check next line
      int ipoint = point[isou] + cell_center[isou];// + d_m[isou];
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

    if (address > max_address)
      throw OPSException(OPS_RUNTIME_ERROR," Error: The address exceeds max  size");

    for (int i = dim - 1; i >= 0; i--) {
      int prod = 1;
      for (int j = 0; j < i;j++)
        prod *= size[j];

      cell[i] = addr / prod;
      addr -= cell[i] * prod;
    }//OK
  }


  inline  uint64_t pack_pair(int tagI, int tagJ) {

    uint32_t a = (tagI < tagJ) ? tagI : tagJ;
    uint32_t b = (tagI < tagJ) ? tagJ : tagI;

    return ((uint64_t)a << 32) | b;
  }


  inline void unpack_pair_key(uint64_t key, int &tagI, int &tagJ) {

    tagI = (int)(key >> 32);
    tagJ = (int)(key & 0xFFFFFFFFu);

  }


  inline int _get_point_to_mapJ(const int address, ops_dat binheadI, ops_dat binheadJ,
                                ops_stencil stencil,
                                const int dim, int local_cell[]) {


    //sub_block_list sb = OPS_sub_block_list[binheadI->block->index];


    int d_mI[OPS_MAX_DIM], d_mJ[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++) {
      d_mI[i] = binheadI->d_m[i] + OPS_sub_dat_list[binheadI->index]->d_im[i];
      d_mJ[i] = binheadJ->d_m[i] + OPS_sub_dat_list[binheadJ->index]->d_im[i];
    }

    int celli[OPS_MAX_DIM];
    _get_cell_from_address(address, dim, binheadI->size, d_mI, celli);

    for (int i = 0; i < dim; i++) celli[i] += d_mI[i];

    //Shift to local stencil
    switch (stencil->type) {
    case 0:
      for (int i = 0; i < dim; i++) local_cell[i] = celli[i] - d_mJ[i];
      break;
    case 1:
      for (int i = 0; i < dim; i++)
        local_cell[i] = celli[i] / stencil->mgrid_stride[i] - d_mJ[i];
      break;
    case 2:
      for (int i = 0; i < dim; i++)
        local_cell[i] = celli[i] * stencil->mgrid_stride[i] - d_mJ[i];
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

  if (mapI->particle->index != history->particleI->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapI is not associated to particles of type I");

  if ( mapJ->particle->index != history->particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapJ is not associated to particles of type J");

  if ( history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
       history->particleJ->is_wall >= 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: History of particle/wall interaction require "
                                          " particleI to be defined as wall");

  if (history->particleJ->ids == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Particle tags are required for particle J when setting "
                                          "contact histories\n");

  if (!history->history_active)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: For non-active history structures. Verlet list reconstructed "
                                          "from setup\n");

  sub_block *sb = OPS_sub_block_list[history->particleI->block->index];
  if (!sb->owned) return;

  if (!history->flag_update) return;


  //Get structure for rigid wall
  int *stencil_act = nullptr;
  int iprod = stencil->points;

  if (history->particleI->is_wall == 1) {
    int address = ((int *)mapI->parts_to_grid->data)[0];
    int cell_to_j[OPS_MAX_DIM];
    ops_mpi_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,
                                          stencil, dim, cell_to_j);

    //Get normal direction
    int inorm, idir_norm;
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

      if (inorm == 1 || inorm == - 1) {
        idir_norm = i; break;
      }
    }

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


  //TODO: Why do we need the virtual to compute force???
  int no_particlesI = (history->particleI->is_wall == 1) //WHY accessing the virtual
      ? history->particleI->no_particles + history->particleI->no_virtual
      : history->particleI->no_particles;
  int *part2bin = (int *)mapI->parts_to_grid->data;
  int *binhead = (int *) mapJ->binhead->data;
  int *bins = (int *) mapJ->bin->data;

  int new_pairs = 0;

  //Part I: Reset flags
  for (int i = 0; i < no_particlesI; i++) {
    for (int j = 0; j < ((int *)history->n_partnersI->data)[i]; j++) {
      ((int *) history->flagI->data)[i * history->num_neighsI + j] = 0;
    }
  }

  if ( history->history_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    int no_particlesJ = history->particleJ->no_particles + history->particleJ->no_virtual;
    for (int i = 0; i < no_particlesJ; i++)
      for (int j = 0; j <((int *) history->n_partnersJ->data)[i]; j++)
      ((int *) history->flagJ->data)[i * history->num_neighsJ + j] = 0;
  }

  int size_index = history->indexing->dim;
  for (int i = 0; i < history->nconts; i++) {
    ((int *)history->flag->data)[i] = 0;
  }

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++)
    d_m[isou] = mapJ->binhead->d_m[isou] + OPS_sub_dat_list[mapJ->binhead->index]->d_im[isou];


  //Part II: Loop over all particles to find structures
  for (int i = 0; i < no_particlesI; i++) {
    //Get old data
    int neighs_old =((int *)history->n_partnersI->data)[i];
    int n_curI = 0;
    int bin_cell = part2bin[i];

    int cell_center[OPS_MAX_DIM] = {};
    ops_mpi_histories::_get_point_to_mapJ(bin_cell, mapI->binhead, mapJ->binhead,
                                          stencil, dim, cell_center);

    //Loop over user-defined stencil regions
    int index;
    int index_din = history->indexing->dim;
    for (int is = 0; is < npoints; is++) {

      int addressJ = ops_mpi_histories::_get_address(cell_center, stencils + is * dim,
                                                     mapJ->binhead->size, dim);

      if (addressJ < 0) continue;

      int partJ = ((int *)mapJ->binhead->data)[addressJ];

      while (partJ != -1) {
      //Check for possible contact
      int found = 0;
      //Part I: Check existing pairs
      for (int in = 0; in < neighs_old; in++) {

        int a1;
        if (ops_mpi_histories::_check_integer_type_for_int(history->particleJ->ids->type))
           a1 = _pair_exists_in_list(((int *) history->partnersI->data)[i * history->num_neighsI + in],
                                     ((int *) history->particleJ->ids->data)[partJ]);
        else if (ops_mpi_histories::_check_integer_type_for_int64(history->particleJ->ids->type))
          a1 = _pair_exists_in_list(((uint64_t *) history->partnersI->data)[i * history->num_neighsI + in],
                                    ((uint64_t *) history->particleJ->ids->data)[partJ]);

        if (tagsJ[partJ] == ((int *) history->partnersI->data)[i * history->num_neighsI + in]) { //TODO: Check it

          found = 1;

          //Update particle & global flags
          ((int *) history->flagI->data)[i * history->num_neighsI + in] = 1;
          index = ((int *) history->indexI->data)[i * history->num_neighsI + in];
          //Update neighbor flag
          ((int *)history->flag->data)[index] = 1;

          //Update index for virtual particles
          if (i  >= history->particleI->no_particles) {
              int iloc = ((int *)history->indexing_local->data)[2 * index];
              int jloc = ((int *)history->indexing_local->data)[2 * index + 1];
              if (!((i == iloc && partJ == jloc) ||
                    (i == jloc && partJ == iloc))) {
                ((int *)history->indexing_local->data)[2 * index] = i;
                ((int *)history->indexing_local->data)[2 * index + 1] = partJ;
              }
          }

          break;
        }
      }

      //Part IIa: Pair found but need to update the reverse access

      //TODO: Check the flags
      int flagJ = 0;
      if (found == 1 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {

        int tagI = ((int *) history->particleI->is_wall == 1) ? i : tagsI[i];

        for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {
          if (((int *)history->partnersI->data)[history->num_neighsJ * partJ + jp] == tagI) {
            flagJ = 1;
            ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
            break;
          }
        }

        //Part IIb: Contact is not found increase size
        if (flagJ == 0) {
          int nlocal = ((int *) history->n_partnersJ->data)[partJ];

          if (nlocal + 1 > history->num_neighsJ)
            throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Number of possible neighbors pairs exceeds the "
                                                  " maximum number of possible pairs");

          ((int *) history->n_partnersJ->data)[partJ]++;
          ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + nlocal] =
                                      (!history->particleI->is_wall) ? tagsI[i] : i;
          ((int *) history->flagJ->data)[partJ * history->num_neighsJ + nlocal] = 1;

          //Copy index and continue
          if (history->history_type == OPS_HISTORY_NEWTON_ON) {
            ((int *) history->indexJ->data)[partJ * history->num_neighsJ + nlocal] = index;
          }
        }
      }

      //Part III: Check if contact exists as (j, i) && ! UPDATE_BOTH_WAYS
      if (found == 0 && history->history_type == OPS_HISTORY_NEWTON_ON) {
        int jfound = 0;

        for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {
          int tag_neighJ = ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + jp];
          int tagI = (history->particleI->is_wall == 1) ? i : tagsI[i];
          if (tagI == tag_neighJ) {
            found = 1;
            jfound = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jp];
            if (history->particleI->is_wall!=  1 && neighs_old + 1 > history->num_neighsI)
              throw OPSException(OPS_RUNTIME_ERROR, "Error: Maximum number of candidate particles is reached");

            //Reallocate structures
            if (history->particleI->is_wall== 1 && neighs_old + 1 > history->num_neighsI) {
              history->num_neighsI += OPS_MAX_PART;
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

            int index = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jp];

            //Set local histories
            ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old] = tagsJ[partJ];
            ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old] = index;
            ((int *) history->flagI->data)[i * history->num_neighsI + neighs_old] = 1;

            neighs_old++;

            //Set also histories to other particle if accessible from X
            if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS)
              ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;

            //Checking to update if necessary the virtual element
            ((int *) history->indexing_local->data)[2 * index ] = i;
            ((int *) history->indexing_local->data)[2 * index + 1] = partJ;

            //Update flag of pair storage
            ((int *) history->flag->data)[index] = 1;
            break;
          }
        }
      }

      //Part IV: Pair not found-Update list
      if (found == 0) {
        //Reallocate storing pairs
        if (history->history_active) {
          if (new_pairs + 1 > history->nmax_new) {
            int nmax = MAX(history->num_neighsI, history->num_neighsJ);
            history->new_neighbors->data = (char *) ops_realloc(history->new_neighbors->data,
                                                                history->new_neighbors->elem_size
                                                                *  (nmax * OPS_MAX_PART + history->nmax_new));
            history->nmax_new += OPS_MAX_PART * nmax;
          }
        }

        //Copy nessarary elements
        ((int *) history->new_neighbors->data)[2 * new_pairs] = i;
        ((int *) history->new_neighbors->data)[2 * new_pairs + 1] = partJ;
        found = 1;

        //Reallocate structures for rigid walls and check if exceed maximum number of particles for other types
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
                                                       * history->num_neighsI);
        }

          //Add element to the local-particle list
          ((int *) history->partnersI->data)[history->num_neighsI * i + neighs_old] = tagsJ[partJ];
          ((int *) history->flagI->data)[history->num_neighsI * i + neighs_old] = 1;
          ((int *) history->indexI->data)[history->num_neighsI * i + neighs_old] = new_pairs + history->nconts;

          neighs_old++;

          if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
            int num_neighsJ_old = ((int *) history->n_partnersJ->data)[partJ];

            if (num_neighsJ_old + 1 > history->num_neighsJ)
              throw OPSException(OPS_RUNTIME_ERROR,"Error: Maximum number of candidate neighbors for "
                                                   " neighbor list is reached\n");

            ((int *) history->partnersJ->data)[history->num_neighsJ * partJ + num_neighsJ_old]
                     = (!(history->particleI->is_wall)) ? tagsI[i] : i;
            ((int *) history->flagJ->data)[history->num_neighsJ * partJ + num_neighsJ_old] = 1;
            ((int *) history->indexJ->data)[history->num_neighsJ * partJ + num_neighsJ_old] = new_pairs + history->nconts;
            ((int *) history->n_partnersJ->data)[partJ]++;
          }

          new_pairs++;

      }
      n_curI ++;

      if (n_curI > history->num_neighsI && history->particleI->is_wall != 1)
        throw OPSException(OPS_RUNTIME_ERROR, "The number possible pairs assigned to particle I exceeds "
                                              "the user defined limit\n" );

      partJ = ((int *) mapJ->bin->data)[partJ];

      }

    }

    //Part II: Remove old pairs from particle-lists
    for (int ip = 0; ip < neighs_old;) {

      if (((int *) history->flagI->data[i * history->num_neighsI + ip]) == 0) {

        if (ip == neighs_old - 1) {
          neighs_old--;
          break;
        }

        ((int *) history->partnersI->data)[i * history->num_neighsI + ip] =
            ((int *)history->partnersI->data)[i * history->num_neighsI + (neighs_old - 1)];
        ((int *) history->flagI->data)[i * history->num_neighsI + ip]
            = ((int *) history->flagI->data)[i * history->num_neighsI + (neighs_old - 1)];

        ((int *) history->indexI->data)[i * history->num_neighsI + ip] =
            ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old - 1];
        neighs_old--;
      }
      else ip++;
    }

    ((int *)history->n_partnersI->data)[i] = n_curI;
  }

  //Part III: Adding new elements to the list
  if (history->nconts + new_pairs > history->nmax_cont) {
    int n_upgrade = history->nconts + new_pairs + OPS_MAX_PART * MAX(history->num_neighsI,
                                                                     history->num_neighsJ);

    history->indexing->data = (char *) ops_realloc(history->indexing->data,
                                                   history->indexing->elem_size * n_upgrade);
    history->flag->data = (char *) ops_realloc(history->flag->data,
                                               history->flag->elem_size * n_upgrade);

    history->data->data = (char *) ops_realloc(history->data->data,
                                               history->data->elem_size * n_upgrade);

    history->indexing_local->data = (char *) ops_realloc(history->indexing_local->data,
                                                         history->indexing_local->elem_size * n_upgrade);
    history->nmax_cont = n_upgrade;
  }

  //Assing the elements
  uint64_t *indexing = (uint64_t *) history->indexing->data; //We may remove it
  int *flag_data = (int *) history->flag->data;

  for (int i = 0; i < new_pairs; i++) {
    int cont_number = history->nconts + i;
    flag_data[cont_number] = 1;
    int ielem = ((int *)history->new_neighbors->data)[2 * i];
    int jelem = ((int *)history->new_neighbors->data)[2 * i + 1];
    int tagI = (!history->particleI->is_wall != 0) ? tagsI[ielem] : ielem;
    indexing[cont_number] = ops_mpi_histories::pack_pair(tagI, tagsJ[jelem]);

    ((int *) history->indexing_local->data)[2 * cont_number] = ielem;
    ((int *) history->indexing_local->data)[2 * cont_number + 1] = jelem;

    if (history->history_active)
      kernel(((T*) history->data->data) + history->data->dim * cont_number, argument...);
  }

  history->nconts += new_pairs;

  //Remove pairs from list-j.
  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    for (int i = 0; i < history->particleJ->no_particles + history->particleJ->no_virtual; i++) {
       int neighs_old  = ((int *) history->n_partnersJ->data)[i];
       for (int j = 0; j  < neighs_old; ) {

         if (((int *) history->flagJ->data)[i * history->num_neighsJ + j] == 0) {

           if (j == neighs_old - 1) {
             neighs_old--;
             break;
           }


           ((int *) history->partnersJ->data)[i * history->num_neighsJ + j] =
               ((int *) history->partnersJ->data)[i * history->num_neighsJ + neighs_old - 1];
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


  //Part IV: Remove old pairs
  if (history->frequency >= 0) {

    int n_neighs = history->nconts;
    for (int i = n_neighs - 1; i >= 0;) {

      if (flag_data[i] == 0) {

        if (i == n_neighs - 1) { n_neighs--; i--; continue;}

        ((int *)history->flag->data)[i] = ((int *)history->flag->data)[n_neighs-1];
        ((uint64_t *)history->indexing->data)[i] = ((uint64_t *) history->indexing->data)[n_neighs - 1];

        ((int *) history->indexing_local->data)[2 * i]
             =  ((int *) history->indexing_local->data)[2  * ( n_neighs - 1)];
        ((int *) history->indexing_local->data)[2 * i + 1]
             =  ((int *) history->indexing_local->data)[2  * ( n_neighs - 1) + 1];

        if (history->history_active) {
          for (int ielem = 0; ielem < history->data->dim; ielem++)
            ((T *) history->data)[history->data->dim * i + ielem] =
            ((T *) history->data)[history->data->dim *(n_neighs - 1) + ielem];
        }

        //Update indexI and indexJ after shift
        int ipart  = ((int *) history->indexing_local->data)[2 * i];
        int jpart = ((int *) history->indexing_local->data)[2 * i + 1];
        int tagJ = ((int *) history->particleJ->ids->data)[jpart];
        int tagI = (history->particleI->is_wall == 0) ? ((int *) history->particleI->ids->data)[ipart] : ipart;
        for (int jp = 0; jp < ((int *) history->n_partnersI->data)[ipart]; jp++) {
          if (((int *) history->partnersI->data)[ipart * history->num_neighsI + jp] == tagJ) {
            ((int *) history->indexI->data)[ipart * history->num_neighsI + jp] = i;
            break;
          }
        }

        if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
            history->history_type == OPS_HISTORY_NEWTON_ON) {
          for (int ip = 0; ip < ((int *) history->n_partnersJ->data)[jpart]; ip++) {
            if (((int *) history->partnersJ->data)[jpart * history->num_neighsI + ip] == tagI) {
              ((int *) history->indexJ->data)[jpart * history->num_neighsJ + ip] = i;
              break;
            }
          }
        }

        n_neighs--;
      } else i--;
    }
    history->nconts = n_neighs;
    history->frequency = 0;
  }
  else history->frequency++;

  history->flag_update = false;
  ops_free(stencil_act);
}

template<typename T, typename... ParamType, typename... args>
void ops_particle_setup_neighbor_history(void (*kernel)(T*, ParamType...), char const *name,
                                         ops_neighbor_history history, ops_particle_mapping mapI,
                                         ops_particle_mapping mapJ, ops_stencil stencil,
                                         int dim, args... argument) {

  //Sanity checks similar as above
  if (sizeof...(ParamType) != sizeof...(args))
    throw OPSException(OPS_RUNTIME_ERROR,"Error: The number of the kernel parameters varies "
                                         "from the number of OPS args\n");

  if (mapI->particle->index != history->particleI->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapI is not associated to particles of type I");

  if ( mapJ->particle->index != history->particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapJ is not associated to particles of type J");

  if ( !history->flag_update) return;

  sub_block *sb = OPS_sub_block_list[history->particleI->block->index];
  if (!sb->owned) return;

  int flag = 0;
  int *stencil_act = nullptr;
  int iprod = stencil->points;

  if (history->particleI->is_wall == 1) {
    int address = ((int *)mapI->parts_to_grid->data)[0];
    int cell_to_j[OPS_MAX_DIM];
    ops_mpi_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,
                                          stencil, dim, cell_to_j);

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

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++)
    d_m[isou] = mapJ->binhead->d_m[isou] + OPS_sub_dat_list[mapJ->binhead->index]->d_im[0];

  int nconts = 0;
  int *part2bin = (int *)mapI->parts_to_grid->data;

  int *binheadJ = (int *)mapJ->binhead->data;
  int *binsJ = (int *)mapJ->bin->data;

  int nlocal = (history->particleI->is_wall) ?  history->particleI->no_particles :
               history->particleI->no_particles + history->particleI->no_virtual;

  if (nlocal == 0) return;

  int *tagsI = (history->particleI->ids != nullptr) ? (int *) history->particleI->ids->data : nullptr;
  int *tagsJ = (history->particleJ->ids != nullptr) ? (int *) history->particleJ->ids->data : nullptr;

  for (int i = 0; i < nlocal; i++) {
    int address = part2bin[i];

    int n_neighs = 0;
    int cell_center[OPS_MAX_DIM];
    ops_mpi_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,stencil,
                                          dim, cell_center); //TODO

    //Loop over user-defined stencils
    for (int is = 0; is < npoints; is++) {
      int addressJ = ops_mpi_histories::_get_address(cell_center, stencils + is * dim,
                                                     mapJ->binhead->size, dim);

      int partJ = binheadJ[addressJ];
      while (partJ != - 1) {
        n_neighs++;
        partJ = binsJ[partJ];
      }
    }

    if (history->particleI->is_wall != 1 && n_neighs > history->num_neighsI)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: The number of possible neighbors exceeds "
                                              "the user defined one");

    //Data reallocation for wall accessing via a single element ok.
    if (history->particleI->is_wall == 1 && nconts > history->num_neighsI) {
      history->num_neighsI = nconts + OPS_MAX_PART;
      history->partnersI->data = (char *) ops_realloc(history->partnersI->data, history->num_neighsI
                                                                               * history->partnersI->elem_size);
      history->flagI->data = (char *) ops_realloc(history->flagI->data, history->num_neighsI
                                                  * history->flagI->elem_size);
      history->indexI->data = (char *) ops_realloc(history->indexI->data, history->num_neighsI
                                                   * history->indexI->elem_size);
    }

    ((int *)history->n_partnersI->data)[i] = n_neighs;
    nconts += n_neighs;
  }

  //Reallocation of global structures
  if (nconts >= history->nmax_cont) {
    history->nmax_cont = nconts + OPS_MAX_DIM * MAX(history->num_neighsI, history->num_neighsJ);

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

  if (history->history_type == OPS_HISTORY_NEWTON_ON &&
      history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    int nlocalJ = history->particleJ->no_particles
               + history->particleJ->no_virtual;
    for (int i = 0; i < nlocalJ; i++)
      ((int *) history->n_partnersJ->data)[i] = 0;
  }

 // uint64_t *indexing = (uint64_t *) history->indexing->data;
  int *flags = (int *) history->flag->data;
  for (int i = 0; i < nlocal; i++) {

    int address = part2bin[i];
    int cell_center[OPS_MAX_DIM];
    ops_mpi_histories::_get_point_to_mapJ(address, mapI->binhead, mapJ->binhead, stencil,
                                         dim, cell_center);
    int n_neighs = 0;

    //Loop over all cells to generate history
    for (int icell = 0; icell < stencil->points; icell++) {
      int addressJ = ops_mpi_histories::_get_address(cell_center, stencil->stencil + icell * dim,
                                                     mapJ->binhead->size,  dim);
      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      while (partJ != -1) {

        ((int *)history->partnersI->data)[i * history->num_neighsI + n_neighs] = tagsJ[partJ];
        ((int *)history->flagI->data)[i * history->num_neighsI + n_neighs] = 1;
        ((int *)history->indexI->data)[i * history->num_neighsI + n_neighs] = nconts;

        int tagI = (!history->particleI->is_wall) ? ((int *) history->particleI->ids->data)[i] : i;

        //Update if necessary the second particle if
        if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          int nelem = ((int *) history->n_partnersJ->data)[partJ];
          ((int *)history->partnersJ->data)[partJ * history->num_neighsJ + nelem] = tagI;
          ((int *)history->flagJ->data)[partJ * history->num_neighsJ + nelem] = 1;
          ((int *)history->indexJ->data)[partJ * history->num_neighsJ + nelem] = nconts;
          ((int *) history->n_partnersJ->data)[partJ]++;

        }

        //update contact history
        flags[nconts] = 1;
        if (history->history_active)
          ((uint64_t *) history->indexing->data)[nconts] = ops_mpi_histories::pack_pair(tagI, tagsJ[partJ]);

        ((int *) history->indexing_local->data)[2 * nconts] = i;
        ((int *) history->indexing_local->data)[2 * nconts + 1] = partJ;

        if (history->history_active)
          kernel(history->data->data + history->data->dim * nconts, argument...);

        nconts++;
        n_neighs++;


        partJ = binsJ[partJ];
      }


    }
  }

  history->flag_update = false;
  history->frequency = 0;

  //exit(-1);

}
