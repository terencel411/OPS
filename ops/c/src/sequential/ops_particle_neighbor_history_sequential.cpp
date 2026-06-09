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
  *          implementation.
  */

#include <ops_lib_core.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <iostream>



inline int _get_address(int  *cell_center, int *point, int *size,
                        int *d_m, int dim) {

  int address = 0;
  int prod = 1;
  for (int isou = 0; isou < dim; isou++) {
    int ipoint = point[isou] + cell_center[isou] + d_m[isou];
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
    addr -= cell[i] * addr;
  }
}

static inline  uint64_t pack_pair(int tagI, int tagJ) {

  uint32_t a = (tagI < tagJ) ? tagI : tagJ;
  uint32_t b = (tagI < tagJ) ? tagJ : tagI;

  return ((uint64_t)a << 32) | b;
}

static inline void unpack_pair_key(uint64_t key, int &tagI, int &tagJ) {

  tagI = (int)(key >> 32);
  tagJ = (int)(key & 0xFFFFFFFFu);

}

static inline int _get_point_to_mapJ(const int address,ops_dat binheadI,
                                     ops_dat binheadJ, ops_stencil stencil,
                                     const int dim, int local_cell[]) {

  int *stride = stencil->mgrid_stride;
  int celli[OPS_MAX_DIM];
  _get_cell_from_address(address, dim, binheadI->size, binheadI->d_m, celli);

  switch (stencil->type) {
  case 0:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i];
    break;
  case 1:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i] / stencil->mgrid_stride[i];
    break;
  case 2:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i] * stencil->mgrid_stride[i];
    break;
  }
  int point_zero[OPS_MAX_DIM] = {};
  return _get_address(local_cell, point_zero, binheadJ->size, binheadJ->d_m, dim);

}


//TODO: Remove & correct structure-We can have two points for fast access

template<typename T, typename... ParamType, typename... args>
void ops_particle_update_neighbor_history(void (*kernel)(T, ParamType...), char const *name,
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

  int no_particlesI = history->particleI->no_particles + history->particleI->no_virtual;
  int *part2bin = (int *) mapI->parts_to_grid->data;
  int *binhead = (int *) mapJ->binhead->data;
  int *bins = (int *) mapJ->bin->data;

  int new_pairs = 0;


  //Reset flags
  for (int i = 0; i < no_particlesI; i++)
    for (int j = 0; j < ((int *)history->n_partnersI->data)[i]; j++)
      ((int *) history->flagI->data)[i * history->num_neighsI + j] = 0;

  if ( history->history_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    int no_particlesJ = history->particleJ->no_particles + history->particleJ->no_virtual;
    for (int i = 0; i < no_particlesJ; i++)
      for (int j = 0; j < ((int *)history->n_partnersJ->data)[i]; j++)
      ((int *) history->flagJ->data)[i * history->num_neighsJ + j] = 0;
  }

  //Set all flags of history to zero
  uint64_t *indexing =(uint64_t *)history->indexing->data;
  int size_index = history->indexing->dim;
  for (int i = 0; i < history->nconts; i++) {
    ((int *)history->flag->data)[i] = 0;
  }

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) d_m[isou] = mapJ->binhead->d_m[isou];

  int *tagsI = (int *) history->particleI->ids->data;
  int *tagsJ = (int *) history->particleJ->ids->data;


  //Loop over all particles
  for (int i = 0; i < no_particlesI; i++) {
    //Reset flags for all neighbors of I
    int neighs_old = ((int *)history->n_partnersI->data)[i];
    int n_curI = 0;

    int bin_cell = part2bin[i];

    //Get cell projection into mapII
    int cell_center[OPS_MAX_DIM];
    _get_point_to_mapJ(bin_cell, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_center); //TODO


    //Loop over user-defined stencil point

    int index;
    int index_dim = history->indexing->dim;
    for (int is = 0; is < stencil->points; is++) {
      int addressJ = _get_address(cell_center, stencil->stencil + is * dim,
                                  mapJ->binhead->size, d_m, dim); //TODO

      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      //Loop against all possible contacts for the particle in the cell
      while (partJ != - 1) {

        if (tagsJ[partJ] == tagsI[i]) {
          partJ = bins[partJ];
          continue;
        }

        //Identify possible contacts!!
        int found = 0;

        //Look if contact found--
        for (int in = 0; in < neighs_old; in++) {
          if (tagsJ[partJ] == ((int *)history->partnersI->data)[history->num_neighsI * i]) {
            found = 1;
            ((int *) history->flagI)[i * history->num_neighsI + in] = 1;
            //Set flag to 1
            //get intex and update
            index = ((int *)history->indexI->data)[i * history->num_neighsI + in];

            //Set flag that is found
            ((int *) history->flag->data)[index] = 1;

            //Update index of local
            if (i >= history->particleI->no_particles) {
              //Get pair key
              int iloc, jloc;
              unpack_pair_key(((uint64_t *)history->indexing->data)[2 * index + 1], iloc, jloc);
              if (iloc + jloc == i) {
                ((uint64_t *) history->indexing->data)[2 * index + 1] = pack_pair( i, partJ);
              }
            }
            break;
          }
        }

        //Update if necessary the contact for J if both ways storage
        int flagJ = 0;
        if (found == 1 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {
            if (((int *)history->partnersI->data)[history->num_neighsJ * partJ + jp] == tagsI[i]) {
              flagJ = 1;
              ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
              break;
            }
          }

          //NOT-FOUND NEED TO POINT TO ADDRESS
          if (flagJ == 0) {
            int nlocal = ((int *) history->n_partnersJ->data)[partJ];
            if (nlocal + 1 > history->num_neighsJ)
              throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Number of possible neighbors pairs "
                                                   "exceeds the maximum number of possible pairs");
            ((int *) history->n_partnersJ->data)[partJ]++;
            ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + nlocal] = tagsI[i];
            ((int *) history->flagJ->data)[partJ * history->num_neighsJ + nlocal] = 1;

            //Copy index and continue
            if (history->history_type == OPS_HISTORY_NEWTON_ON) {
              ((int *) history->indexJ->data)[partJ * history->num_neighsJ + nlocal] = index;
              ((int *) history->flagJ->data)[partJ + history->num_neighsJ + nlocal] = 1;
            }

          }
        }

        //Case II: Update contact if may exist on the other particle due to particle moving around

        if (found == 0 && history->history_type == OPS_HISTORY_NEWTON_ON) {
          int jfound = 0;

          for (int jpart = 0; jpart < ((int *) history->n_partnersJ->data)[partJ]; jpart++) {
            int tag_neighJ = ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + jpart];
            if (tagsI[i] == tag_neighJ) {
              found = 1;
              jfound = ((int *)history->indexJ->data)[partJ * history->num_neighsJ + jpart];

              if (neighs_old + 1 > history->num_neighsI)
                throw OPSException(OPS_RUNTIME_ERROR, "Error: Maximum number of candidate neighbors for "
                                                       "neighbor list is reached");

              int index = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jpart];

              ((int *) history->partnersI->data)[i * history->num_neighsI + neighs_old] = tagsJ[partJ];
              ((int *) history->indexI->data)[i * history->num_neighsI + neighs_old] = index;
              ((int *) history->flagI->data)[i * history->num_neighsI + neighs_old] = 1;

              neighs_old++;
              //Also set the flag for the jth particle to 1
              if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS)
                ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jpart] = 1;

              //Checking the history
              if (i >= history->particleI->no_particles || partJ >= history->particleJ->no_particles) {
                int ipair, jpair;
                unpack_pair_key(((uint64_t *)history->indexing->data)[2 * index + 1], ipair, jpair);
                if (ipair + jpair == i || jpair + ipair == partJ)
                  ((uint64_t *) history->indexing->data)[2 * index + 1] = pack_pair(i, partJ);

              }

              //Finally update flag of contact storage
              ((int *)history->flag->data)[index] = 1;

              n_curI++;
              break;
            }
          }

        }


        if (found == 0) {

          //HEAR IS WRONG Correct
          if (new_pairs + 1 > history->nmax_new) {
            int nmax = MAX(history->num_neighsI, history->num_neighsJ);
            history->new_neighbors->data = (char *) ops_realloc(history->new_neighbors->data,
                                                                history->new_neighbors->elem_size *
                                                                (nmax * OPS_MAX_PART + history->nmax_new));
            history->nmax_new += OPS_MAX_PART * nmax;
          }


          ((int *) history->new_neighbors->data)[2 * new_pairs ] = i;
          ((int *) history->new_neighbors->data)[2 * new_pairs + 1] = partJ; //TODO shift to IDS
           new_pairs++;

           //Add the element to the list
           ((int *) history->partnersI->data)[history->num_neighsI + neighs_old] = tagsJ[partJ];
           ((int *) history->flagI->data)[history->num_neighsI + neighs_old] = 1;
           //INDEX TO BE ADDED AT THE END
           neighs_old++;

           if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
             int num_neighsJ_old = ((int *) history->n_partnersJ->data)[partJ];
             if (num_neighsJ_old + 1 > history->num_neighsJ)
               throw OPSException(OPS_RUNTIME_ERROR,"Error: Maximum number of candidate neighbors for "
                                                    " neighbor list is reached\n");
             ((int *) history->partnersJ->data)[history->num_neighsJ * partJ + num_neighsJ_old]
                      = tagsI[i];
             ((int *) history->flagJ->data)[history->num_neighsJ * partJ + num_neighsJ_old] = 1;
             ((int *) history->n_partnersJ->data)[history->num_neighsJ]++;

           }

        }

        n_curI += (flag) ? 1 : 0;
        partJ = ((int *) mapJ->bin->data)[partJ];

      }

    }


    //Remove contact pairs from list
    for (int ip = 0; ip < neighs_old;) {

      if (((int *) history->flagI->data[i * history->num_neighsI + ip] == 0)) {
        int size = history->partnersI->elem_size;
        int type_size = history->partnersI->type_size;
        memmove(history->partnersI->data + i * size + ip * type_size,
                history->partnersI->data + i * size + (neighs_old - 1) * type_size,
                type_size);
        ((int *) history->flagI->data)[i * history->num_neighsI + ip] =
            (int *) history->flagI->data[i * history->num_neighsI + (neighs_old - 1)];
        ((int *) history->indexI->data)[i * history->num_neighsI + ip] =
            (int *) history->indexI->data[i * history->num_neighsI + ip];
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
    history->nmax_cont = n_upgrade;
  }


  indexing = (uint64_t *) history->indexing->data;
  T* hist_data = (T *) history->data->data;
  int *flag_data = (int *) history->flag->data;
  for (int i = 0; i < new_pairs; i++) {
    int cont_number = history->nconts + i;
    flag_data[cont_number] = 1;
    int ielem = ((int *)history->new_neighbors->data)[2 * i];
    int jelem = ((int *)history->new_neighbors->data)[2 * i + 1];
    indexing[2 * cont_number] = pack_pair(tagsI[ielem], tagsJ[ielem]);
    indexing[2 * cont_number + 1] = pack_pair(ielem, jelem);

    kernel(hist_data + history->data->dim * cont_number, argument...);

  }

  history->nconts += new_pairs;


  //Rmv and wrap up history of pairs not updated for a while
  if (history->frequency == history->update_list) {

    int n_neighs = history->nconts;
    for (int i = n_neighs-1; i >= 0;) {
      if (flag_data[i] == 0) {
        flag_data[i] = flag_data[n_neighs-1];
        indexing[2 * i] = indexing[2 * (n_neighs-1)];
        indexing[2 * i + 1] = indexing[2 * (n_neighs-1) + 1];

        for (int ielem = 0; ielem < history->data->dim; ielem++)
          hist_data[history->data->dim * i + ielem] =
              hist_data[history->data->dim *(n_neighs - 1) + ielem];

        n_neighs--;

        //Update index for received elements

        if (flag_data[i] == 1) {

          int ip, jp;
          unpack_pair_key(indexing[2 * i + 1], ip, jp);

          if (history->history_type == OPS_HISTORY_NEWTON_ON) {
            int flag_his = 0; //Assume history works only for particle to particle contacts
            for (int ineighs = 0; ineighs < ((int *) history->n_partnersI->data)[ip]; ineighs++) {
              if (tagsJ[jp] == ((int *)history->partnersI->data)[ip * history->num_neighsI + ineighs]) {
                flag_his = 1;
                ((int *) history->indexI->data)[ip * history->num_neighsI + ineighs] = i;
                break;
              }
            }

            if (flag_his) continue;

            //if not look for particle J to be in the list first
            for (int ineighs = 0; ineighs < ((int *) history->n_partnersI->data)[jp]; ineighs++) {
              if (tagsJ[ip] == ((int *)history->partnersI->data)[jp * history->num_neighsI + ineighs]) {
                ((int *) history->indexI->data)[jp * history->num_neighsI + ineighs] = i;

              }
            }
          }
          else {
            int flag_i = 0;
            for (int ineighs = 0; ineighs < ((int *) history->n_partnersI->data)[ip]; ineighs++) {
              if (tagsJ[jp] == ((int *) history->partnersI->data)[ip * history->num_neighsI + ineighs]) {
                if (((int *) history->indexI->data)[ip * history->num_neighsI + ineighs] == n_neighs) {
                  flag_i = 1;
                  ((int *) history->indexI->data)[ip * history->num_neighsI + ineighs] = i;
                }
              }
            }

            if (flag_i == 0) {
              for (int ineighs = 0; ineighs < ((int *) history->n_partnersI->data)[ip]; ineighs++) {
                if (tagsJ[ip] == ((int *) history->partnersI->data)[jp * history->num_neighsI + ineighs])
                  ((int *) history->indexI->data)[jp * history->num_neighsI + ineighs] = i;
              }
            }
          }
        }
      }
      else i--;

    }

    history->frequency = 0;
  }
  else history->frequency++;

}


//TODO: Update local pair- on the fly=do we need that
template<typename T, typename... ParamType, typename... args>
void ops_particle_setup_neighbor_history(void (*kernel)(T, ParamType...), char const *name,
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

  int *tagsI = (int *) history->particleI->ids->data;
  int *tagsJ = (int *) history->particleJ->ids->data;

  for (int i = 0; i < nlocal; i++) {
    int address = part2bin[i];

    int n_neighs = 0;
    int cell_center[OPS_MAX_DIM];
    _get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,stencil,
                       dim, cell_center); //TODO

    //Loop over user-defined stencils
    for (int is = 0; is < stencil->points; is++) {
      int addressJ = _get_address(cell_center, stencil->stencil + is * dim,
                                 mapJ->binhead->size, d_m, dim);

      int partJ = binheadJ[addressJ];
      while (partJ != - 1) {
        if (tagsI[i] != tagsJ[i]) n_neighs++;
        partJ = binsJ[partJ];
      }
    }

    if (n_neighs > history->num_neighsI)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: The number of possible neighbors exceeds "
                                            "the user defined one");
    ((int *)history->n_partnersI->data)[i] = n_neighs;

    nconts += n_neighs;
  }

  //Reallocation of global structures
  if (nconts >= history->nmax_cont) {
    history->nmax_cont = nconts + OPS_MAX_DIM * MAX(history->num_neighsI, history->num_neighsJ);
    history->indexing->data = (char *) ops_realloc(history->indexing->data,
                                                   history->nmax_cont * history->indexing->elem_size);

    history->flag->data = (char *) ops_realloc(history->flag->data,
                                               history->nmax_cont * history->flag->elem_size);

    history->data->data = (char *) ops_realloc(history->data->data,
                                               history->nmax_cont * history->data->elem_size);
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


  T* hist_data = (T *)history->data->data;
  uint64_t *indexing = (uint64_t *) history->indexing->data;
  int *flags = (int *) history->flag->data;
  for (int i = 0; i < nlocal; i++) {

    int address = part2bin[i];
    int cell_center[OPS_MAX_DIM];
    _get_point_to_mapJ(address, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_center);
    int n_neighs = 0;
    //Loop over all cells to generate history
    for (int icell = 0; icell < stencil->points; icell++) {
      int addressJ = _get_address(cell_center, stencil->stencil + icell * dim,
                                  mapJ->binhead->size, d_m, dim);

      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

      while (partJ != - 1) {

        if (tagsJ[partJ] == tagsI[i]) {
          partJ= binsJ[partJ];
          continue;
        }


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
        flags[nconts] = 1;
        indexing[2 * nconts] = pack_pair(tagsI[i], tagsJ[partJ]);
        indexing[2 * nconts + 1] = pack_pair(i, partJ);

        kernel(hist_data + history->data->dim * nconts, argument...);

        nconts++;
        n_neighs++;


        partJ = binsJ[partJ];
      }
    }
  }

  history->frequency = 0;

}


