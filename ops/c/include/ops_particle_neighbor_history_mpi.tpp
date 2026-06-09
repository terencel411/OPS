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

//#include <ops_mpi_core.h>

inline int _get_address(const int cell_center[], int point[], int  size[],
                        const int dim) {
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

  for (int i = dim - 1; i >= 0; i--) {
    int prod = 1;
    for (int j = 0; j < i;j++)
      prod *= size[j];

    cell[i] = addr / prod;
    addr -= cell[i] * prod;
  }//OK
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

static inline int _get_point_to_mapJ(const int address, ops_dat binheadI, ops_dat binheadJ,
                                     ops_stencil stencil,
                                     const int dim, int local_cell[]) {

  //sub_block_list sb = OPS_sub_block_list[binheadI->block->index];

  int d_mI[OPS_MAX_DIM], d_mJ[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    d_mI[i] = binheadI->d_m[i] + OPS_sub_dat_list[binheadI->index]->d_im[i];
    d_mJ[i] = binheadJ->d_m[i] + OPS_sub_dat_list[binheadJ->index]->d_im[i];
  }

  //printf("d_m = [%d %d] d_im = [%d %d]\n",  binheadI->d_m[0], binheadI->d_m[1],
  //       OPS_sub_dat_list[binheadJ->index]->d_im[0], OPS_sub_dat_list[binheadJ->index]->d_im[1]);

  int celli[OPS_MAX_DIM];
  _get_cell_from_address(address, dim, binheadI->size, d_mI, celli);

  //printf("Address = %d Cell = [%d %d]\n", address, celli[0], celli[1]);

  for (int i = 0; i < dim; i++) celli[i] += d_mI[i];
  //printf("Address = %d Cell without virtual = [%d %d]\n", address, celli[0], celli[1]);


  switch (stencil->type) {
  case 0:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i] - d_mJ[i];
    break;
  case 1:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i] / stencil->mgrid_stride[i] - d_mJ[i];
    break;
  case 2:
    for (int i = 0; i < dim; i++) local_cell[i] = celli[i] * stencil->mgrid_stride[i] - d_mJ[i];
    break;
  }

 // printf("Local cell in J [%d %d]\n", local_cell[0], local_cell[1]);

  int point_zero[OPS_MAX_DIM] = {};
  return _get_address(local_cell, point_zero, binheadJ->size, dim);
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

  sub_block *sb = OPS_sub_block_list[history->particleI->block->index];
  if (!sb->owned) return;

  int no_particlesI = history->particleI->no_particles + history->particleI->no_virtual;
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

  uint64_t *indexing = (uint64_t *)history->indexing->data;
  int size_index = history->indexing->dim;
  for (int i = 0; i < history->nconts; i++) {
    ((int *)history->flag->data)[i] = 0;
  }

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++)
    d_m[isou] = mapJ->binhead->d_m[isou] + OPS_sub_dat_list[mapJ->binhead->index]->d_im[0];

  int *tagsI = (int *) history->particleI->ids->data;
  int *tagsJ = (int *) history->particleJ->ids->data;

  //Part II: Loop over all particles
  for (int i = 0; i < no_particlesI; i++) {
    //Get old data
    int neighs_old =((int *)history->n_partnersI->data)[i];
    int n_curI = 0;
    int bin_cell = part2bin[i];

    int cell_center[OPS_MAX_DIM];
    _get_point_to_mapJ(bin_cell, mapI->binhead, mapJ->binhead, stencil,
                       dim, cell_center);

    int index;
    int index_din = history->indexing->dim;
    for (int is = 0; is < stencil->points; is++) {
      int addressJ = _get_address(cell_center, stencil->stencil + is * dim,
                                  mapJ->binhead->size, dim);

      if (addressJ < 0) continue;

      int partJ = ((int *)mapJ->binhead->data)[addressJ];

      while (partJ != -1) {

        if (tagsJ[partJ] == tagsI[i]) {
          partJ = bins[partJ];
          continue;
        }

        //Check for possible contact
        int found = 0;
        printf("Neighs_old = %d for particle %d\n", neighs_old, i);
        for (int in = 0; in < neighs_old; in++) {
          if (tagsJ[partJ] == ((int *) history->partnersI->data)[i * history->num_neighsI + in]) {
            found = 1;
            ((int *) history->flagI)[i * history->num_neighsI + in] = 1;

            index = ((int *) history->indexI->data)[i * history->num_neighsI + in];

            //Update neighbor flag
            ((int *)history->flag->data)[index] = 1;

            //Update index for virtual particles
            if (i  >= history->particleI->no_particles) {
              int iloc, jloc;
              unpack_pair_key(((uint64_t *)history->indexing->data)[2 * index + 1], iloc, jloc);
              if (iloc + jloc == i)
                ((uint64_t *) history->indexing->data)[2 * index + 1] = pack_pair(i, partJ);
            }

            break;
          }
        }


        printf("Contact between particles %d and %d found %d\n", i, partJ, found);

        //Update external contacts
        int flagJ = 0;
        if (found == 1 && history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
          for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ]; jp++) {
            if (((int *)history->partnersI->data)[history->num_neighsJ * partJ + jp] == tagsI[i]) {
              flagJ = 1;
              ((int *) history->flagJ->data)[partJ * history->num_neighsJ + jp] = 1;
              break;
            }
          }

          if (flagJ == 0) {
            int nlocal = ((int *) history->n_partnersJ->data)[partJ];
            if (nlocal + 1 > history->num_neighsJ) {
              throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Number of possible neighbors pairs exceeds the "
                                                    " maximum number of possible pairs");
            }

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

        //Case II: Contact may exist but recorded as pair (J,I) previously
        if (found == 0 && history->history_type == OPS_HISTORY_NEWTON_ON) {
          int jfound = 0;

          for (int jp = 0; jp < ((int *)history->n_partnersJ->data)[partJ];jp++) {
            int tag_neighJ = ((int *) history->partnersJ->data)[partJ * history->num_neighsJ + jp];
            if (tagsI[i] == tag_neighJ) {
              found = 1;
              jfound = ((int *) history->indexJ->data)[partJ * history->num_neighsJ + jp];

              if (neighs_old + 1 > history->num_neighsI)
                throw OPSException(OPS_RUNTIME_ERROR, "Error: Maximum number of candidate particles is reached");
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
              if (i >= history->particleI->no_particles || partJ >= history->particleJ->no_particles) {
                int ipair, jpair;
                unpack_pair_key(((uint64_t *)history->indexing->data)[2 * index + 1], ipair, jpair);
                if (ipair + jpair == i || jpair + ipair == partJ)
                  ((uint64_t *) history->indexing->data)[2 * index + 1] = pack_pair(i, partJ);

              }

              ((int *) history->flag->data)[index] = 1;
              n_curI++;

              break;
            }
          }
        }

        if (found == 0) {

          //Reallocate storing pairs
          if (new_pairs + 1 > history->nmax_new) {
            int nmax = MAX(history->num_neighsI, history->num_neighsJ);
            history->new_neighbors->data = (char *) ops_realloc(history->new_neighbors->data,
                                                                  history->new_neighbors->elem_size
                                                                *  (nmax * OPS_MAX_PART + history->nmax_new));
            history->nmax_new += OPS_MAX_PART * nmax;
          }

          //Copy nessarary elements
          ((int *) history->new_neighbors->data)[2 * new_pairs] = i;
          ((int *) history->new_neighbors->data)[2 * new_pairs + 1] = partJ;
          new_pairs++;

          //Add element to the local-particle list
          ((int *) history->partnersI->data)[history->num_neighsI + neighs_old] = tagsJ[partJ];
          ((int *) history->flagI->data)[history->num_neighsI + neighs_old] = 1;
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

    //Part II: Remove old pairs from particle-lists
    for (int ip = 0; ip < neighs_old;) {
      if (((int *) history->flagI->data[i * history->num_neighsI + ip]) == 0) {

        ((int *) history->partnersI->data)[i * history->num_neighsI + ip] =
            ((int *)history->partnersI->data)[i * history->num_neighsI + (neighs_old - 1)];
        ((int *) history->flagI->data)[i * history->num_neighsI + ip]
            = ((int *) history->flagI->data)[i * history->num_neighsI + (neighs_old - 1)];

        ((int *) history->indexI->data)[i * history->num_neighsI + ip] =
            ((int *) history->indexI->data)[i * history->num_neighsI + ip];
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
    history->nmax_cont = n_upgrade;
  }

  //Assing the elements
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

  //Part IV: Remove old pairs
  if (history->frequency == history->update_list) {

    int n_neighs = history->nconts;
    for (int i = n_neighs - 1; i >= 0;) {
      if (flag_data[i] == 0) {
        flag_data[i] = flag_data[n_neighs - 1];
        indexing[2 * i] = indexing[2 * (n_neighs - 1)];
        indexing[2 * i + 1] = indexing[2 * (n_neighs-1) + 1];

        for (int ielem = 0; ielem < history->data->dim; ielem++)
          hist_data[history->data->dim * i + ielem] =
              hist_data[history->data->dim *(n_neighs - 1) + ielem];

        //Reindex receive data
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
        else i--;
      }
    }
    history->frequency = 0;
  }
  else history->frequency++;

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

  sub_block *sb = OPS_sub_block_list[history->particleI->block->index];
  if (!sb->owned) return;

  int d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++)
    d_m[isou] = mapJ->binhead->d_m[isou] + OPS_sub_dat_list[mapJ->binhead->index]->d_im[0];

  int nconts = 0;
  int *part2bin = (int *)mapI->parts_to_grid->data;

  int *binheadJ = (int *)mapJ->binhead->data;
  int *binsJ = (int *)mapJ->bin->data;

  int nlocal = history->particleI->no_particles + history->particleI->no_virtual;

  int *tagsI = (int *)history->particleI->ids->data;
  int *tagsJ = (int *)history->particleJ->ids->data;

  for (int i = 0; i < nlocal; i++) {
    int address = part2bin[i];

    int n_neighs = 0;
    int cell_center[OPS_MAX_DIM];
    _get_point_to_mapJ(address, mapI->binhead, mapJ->binhead,stencil,
                       dim, cell_center); //TODO

    //Loop over user-defined stencils
    for (int is = 0; is < stencil->points; is++) {
      int addressJ = _get_address(cell_center, stencil->stencil + is * dim,
                                 mapJ->binhead->size, dim);

      int partJ = binheadJ[addressJ];
      while (partJ != - 1) {
        if (tagsI[i] != tagsJ[partJ]) n_neighs++;
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
    printf("Address = %d Cell = [%d %d] binsize = [%d %d] block = [%f %f]x[%f %f] dx = %12.9e\n", address, cell_center[0], cell_center[1],
           mapI->binhead->size[0], mapI->binhead->size[1], history->particleI->box_block->getLocalMin().x,
           history->particleI->box_block->getLocalMax().x, history->particleI->box_block->getLocalMin().y,
           history->particleI->box_block->getLocalMax().y, mapI->dx[0]);
    //Loop over all cells to generate history
    for (int icell = 0; icell < stencil->points; icell++) {
      int addressJ = _get_address(cell_center, stencil->stencil + icell * dim,
                                  mapJ->binhead->size,  dim);
      //printf("Address is %d\n", addressJ);
      if (addressJ < 0) continue;

      int partJ = ((int *) mapJ->binhead->data)[addressJ];

   //   printf("partJ = %d for addressJ = %d size = [%d %d]\n", partJ, addressJ,
   //          m);

      while (partJ != -1) {

       // printf("Entering herein\n");

      //  printf("Tags for [%d %d] are [%d %d] with %d possible neighbors\n", i, partJ,
      //         tagsI[i], tagsJ[partJ], nconts);


        if (tagsJ[partJ] == tagsI[i]) {
          partJ = binsJ[partJ];
       //   printf("For same tag we get new part = %d\n", partJ);
          continue;
        }

      //  printf("Tags for [%d %d] are [%d %d] with %d possible neighbors\n", i, partJ,
      //         tagsI[i], tagsJ[partJ], nconts);

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

  //exit(-1);

}
