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


  inline int _get_address(int  *cell_center, int *point, int *size,
                          int dim) {

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

    for (int i = dim - 1; i >= 0; i--) {
      int prod = 1;
      for (int j = 0; j < i;j++)
        prod *= size[j];

      cell[i] = addr / prod;
      addr -= cell[i] * prod;
    }
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
      for (int i = 0; i < dim; i++) local_cell[i] = celli[i] / stencil->mgrid_stride[i]
                                                - binheadJ->d_m[i];
      break;
    case 2:
      for (int i = 0; i < dim; i++) local_cell[i] = celli[i] * stencil->mgrid_stride[i]
                                                  - binheadJ->d_m[i];
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

  int no_particlesI = (history->particleI->is_wall == 1)
          ? history->particleI->no_particles + history->particleI->no_virtual
          : history->particleI->no_particles;
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

  int *tagsI = (history->history_active) ? (int *) history->particleI->ids->data : nullptr;
  int *tagsJ = (history->history_active) ? (int *) history->particleJ->ids->data : nullptr;


  //By-pass stencil with wall-free build

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
        int found = 0;

        //Look if contact found--
        for (int in = 0; in < neighs_old; in++) {

          int compi_tag = tagsJ[partJ];
          //Part I: Search in existing pairs
          if (compi_tag == ((int *)history->partnersI->data)[history->num_neighsI * i + in] ) {

            found = 1;

            //Update both particle & global flags
            ((int *) history->flagI->data)[i * history->num_neighsI + in] = 1;
            index = ((int *)history->indexI->data)[i * history->num_neighsI + in];

            ((int *) history->flag->data)[index] = 1;

            //Update index of local
            if (i >= history->particleI->no_particles) {
              //Get pair key
              int iloc = ((int *)history->indexing_local->data)[2 * index];
              int jloc = ((int *)history->indexing_local->data)[2 * index + 1];
              if (i + partJ != iloc + jloc ) {
                ((int *)history->indexing_local->data)[2 * index] = i;
                ((int *)history->indexing_local->data)[2 * index + 1] = partJ;
              }
            }
            break;
          }

        }
        //Part IIa: Contact found but due to structure need to update the other history: Contact exists
        int flagJ = 0;
        if (found == 1 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {

          int flag_tagI = (history->particleI->is_wall == 1)? i : tagsI[i];
          for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {

            if (((int *)history->partnersJ->data)[history->num_neighsJ * partJ + jp] == flag_tagI) {

              flagJ = 1;
              ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
              break;
            }
          }

          //Part IIb: Contact not found increase the number of elements
          if (flagJ == 0) {
            int nlocal = ((int *) history->n_partnersJ->data)[partJ];

            if (nlocal + 1 > history->num_neighsJ)
              throw OPSException(OPS_RUNTIME_ERROR, "Error: Number may exceed the number of user defined pairs\n");

            ((int *) history->n_partnersJ->data)[partJ]++;
            ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + nlocal] = (!history->particleI->is_wall)? tagsI[i] : i;
            ((int *) history->flagJ->data)[partJ * history->num_neighsJ + nlocal] = 1;

            //Copy index and continue
            if (history->history_type == OPS_HISTORY_NEWTON_ON) {
              ((int *) history->indexJ->data)[partJ * history->num_neighsJ + nlocal] = index;
              ((int *) history->flagJ->data)[partJ + history->num_neighsJ + nlocal] = 1;
            }
          }
        }

        //Part III: Update contact it may exist on the other particle due to particle moving around
        //add type as well for safety
        if (found == 0 && history->history_type == OPS_HISTORY_NEWTON_ON) {
          int jfound = 0;

          for (int jpart = 0; jpart < ((int *) history->n_partnersJ->data)[partJ]; jpart++) {
            int tag_neighJ = ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + jpart];
            int tagI = (history->particleI->is_wall == 1) ? i : tagsI[i];
            if (tagI == tag_neighJ) {
              found = 1;
              jfound = ((int *)history->indexJ->data)[partJ * history->num_neighsJ + jpart];

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

              int index = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jpart];

              //Set particle structures


              ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old] = tagsJ[partJ];
              ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old] = index;
              ((int *) history->flagI->data)[i * history->num_neighsI + neighs_old] = 1;

              neighs_old++;
              //Also set the flag for the jth particle to 1
              if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS)
                ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jpart] = 1;

              ((int *) history->indexing_local->data)[2 * i] = i;
              ((int *) history->indexing_local->data)[2 * i + 1] = partJ;

              //Finally update flag of contact storage
              ((int *)history->flag->data)[index] = 1;
              break;
            }
          }

        }

        //Part IV: Contact do not found-Update lists-This exist because function is called
        if (found == 0) {

          if (history->history_active) {
            if (new_pairs + 1 > history->nmax_new) {
              int nmax = MAX(history->num_neighsI, history->num_neighsJ);
              history->new_neighbors->data = (char *) ops_realloc(history->new_neighbors->data,
                                                                history->new_neighbors->elem_size *
                                                                (nmax * OPS_MAX_PART + history->nmax_new));
              history->nmax_new += OPS_MAX_PART * nmax;
            }
          }


          ((int *) history->new_neighbors->data)[2 * new_pairs ] = i;
          ((int *) history->new_neighbors->data)[2 * new_pairs + 1] = partJ; //TODO shift to IDS


           found = 1;




           //Re-allocate structures

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
           ((int *) history->partnersI->data)[history->num_neighsI * i + neighs_old] =  tagsJ[partJ];
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

             //For wall OK:
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



    //Remove contact pairs from list
    for (int ip = 0; ip < neighs_old;) {

      if ( ((int *) history->flagI->data)[i * history->num_neighsI + ip] == 0) {

        if (ip == neighs_old - 1) {
          neighs_old--;
          continue;
        }

        ((int *) history->partnersI->data)[i * history->num_neighsI + ip] =
            ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old - 1];
        ((int *) history->flagI->data)[i * history->num_neighsI + ip] =
            ((int *) history->flagI->data)[i * history->num_neighsI + (neighs_old - 1)];
        ((int *) history->indexI->data)[i * history->num_neighsI + ip] =
            ((int *) history->indexI->data)[i * history->num_neighsI + (neighs_old - 1)];
        neighs_old--;

      }
      else
        ip++;
    }

    ((int *)history->n_partnersI->data)[i] = n_curI;

  }


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

  uint64_t *indexing = (uint64_t *) history->indexing->data;

  int *flag_data = (int *) history->flag->data;
  for (int i = 0; i < new_pairs; i++) {
    int cont_number = history->nconts + i;
    flag_data[cont_number] = 1;
    int ielem = ((int *)history->new_neighbors->data)[2 * i];
    int jelem = ((int *)history->new_neighbors->data)[2 * i + 1];

    int tagI = (!history->particleI->is_wall == 1) ? tagsI[ielem] : ielem;

    indexing[cont_number] = ops_histories::pack_pair(tagI, tagsJ[ielem]);

    ((int *) history->indexing_local->data)[2 * cont_number] = ielem;
    ((int *) history->indexing_local->data)[2 * cont_number + 1] = jelem;

    //TODO: Add a check
    if (history->history_active)
      kernel((T*) history->data->data + history->data->dim * cont_number, argument...);

  }

  history->nconts += new_pairs;

  //Remove pairs from list-j.
  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    for (int i = 0; i < history->particleJ->no_particles + history->particleJ->no_virtual; i++) {
       int neighs_old  = ((int *) history->n_partnersJ->data)[i];
       for (int j =  0; j < neighs_old; ) {

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

  //Rmv and wrap up history of pairs not updated for a while
  if (history->frequency >= 0) {// history->update_list) {

    int n_neighs = history->nconts;
    for (int i = n_neighs-1; i >= 0;) {

      if (flag_data[i] == 0) {

        if (i == n_neighs - 1) { n_neighs--; i--; continue;}

        ((int *)history->flag->data)[i] = ((int *)history->flag->data)[n_neighs-1];
        ((int *)history->indexing->data)[i] = ((int *) history->indexing->data)[n_neighs - 1];

        ((int *) history->indexing_local->data)[2 * i]
             =  ((int *) history->indexing_local->data)[2  * ( n_neighs - 1)];
        ((int *) history->indexing_local->data)[2 * i + 1]
             =  ((int *) history->indexing_local->data)[2  * ( n_neighs - 1) + 1];

        if (history->history_active)
          for (int ielem = 0; ielem < history->data->dim; ielem++)
            ((T*)history->data)[history->data->dim * i + ielem] =
            ((T*) history->data)[history->data->dim *(n_neighs - 1) + ielem];

        n_neighs--;

        //Update index for received elements
      }
      else i--;
    }

    history->nconts = n_neighs;

    history->frequency = 0;
  }
  else history->frequency++; //Damarcated

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

  if (mapI->particle->index != history->particleI->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapI is not associated to particles of type I");

  if ( mapJ->particle->index != history->particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: mapJ is not associated to particles of type J");


  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS &&
      history->particleJ->is_wall == 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: For wall-particle interactions particleJ must "
                                          "be an actual particle\n");

  if (!history->flag_update) return;

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

  int nlocal = history->particleI->no_particles + history->particleI->no_virtual;

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) d_m[isou] = mapJ->binhead->d_m[isou];

  //No need for tag history!!
  int *tagsI = (history->history_active) ? (int *) history->particleI->ids->data : nullptr;
  int *tagsJ = (history->history_active) ? (int *) history->particleJ->ids->data : nullptr;

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

      int partJ = binheadJ[addressJ];


      while (partJ != - 1) {
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
    for (int icell = 0; icell < stencil->points; icell++) {
      int addressJ = ops_histories::_get_address(cell_center, stencils + icell * dim,
                                                 mapJ->binhead->size, dim);

      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      while (partJ != - 1) {

        ((int *)history->partnersI->data)[i * history->num_neighsI + n_neighs] = tagsJ[partJ];
        ((int *)history->flagI->data)[i * history->num_neighsI + n_neighs] = 1;
        ((int *)history->indexI->data)[i * history->num_neighsI + n_neighs] = nconts;

        //Update if necessary the second particle if
        if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          int nelem = ((int *) history->n_partnersJ->data)[partJ];

          ((int *)history->partnersJ->data)[partJ * history->num_neighsJ + nelem] = i;
          ((int *)history->flagJ->data)[partJ * history->num_neighsJ + nelem] = 1;
          ((int *)history->indexJ->data)[partJ * history->num_neighsJ + nelem] = nconts;
          ((int *) history->n_partnersJ->data)[partJ]++;

        }

        //update contact history
        ((int *) history->flag->data)[nconts] = 1;
        if (history->history_active)
          ((uint64_t *) history->indexing->data)[nconts] = ops_histories::pack_pair(tagsI[i], tagsJ[partJ]);

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
