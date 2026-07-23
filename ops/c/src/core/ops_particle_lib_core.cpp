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
  * @brief Functions for handling particles
  * @author Gihan Mudalige
  * @details Implements dummy functions from the MPI backend for the sequential
  * cpu backend (OpenMP and Sequential)
  */

#include <float.h>
#include <stdlib.h>
#include <iostream>

#include <ops_lib_core.h>
#include <ops_exceptions.h>
#include "ops_util.h"
#include "ops_bounding_box.h"

static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)ops_calloc(len, sizeof(char));
  snprintf(dest, len, "%s", src);
  return dest;
}

static char* append_str(char const *array1, char const *array2) {

  char *newArray = new char[std::strlen(array1) + std::strlen(array2)];

  std::strcpy(newArray, array1);
  return std::strcpy(newArray, array2);
}

/*-----------------------------------------------------------------------------*
 * Local functions
 *-----------------------------------------------------------------------------*/


void _ops_particle_append_dat_point(ops_dat dat, char *&buff, size_t &len,
                                    size_t &cap, const int ip) {

  if (!dat->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR,
                       "ERROR: ops_dat not associated with particle data\n");

  int dim = dat->dim;

  if (strcmp(dat->type, "double") == 0 ||
      strcmp(dat->type, "real(8)") == 0 ||
      strcmp(dat->type, "real(kind = 8)") == 0 ||
      strcmp(dat->type, "double precision") == 0) {
    for (int isou = 0; isou < dim; isou++) {
      _ops_append_char(buff, len, cap,"%16.10e   ",
                       ((double *)dat->data)[ip *dim +  isou]);
    }
  }
  else if (strcmp(dat->type, "float") == 0 ||
           strcmp(dat->type, "real") == 0 ||
           strcmp(dat->type, "real(4)") == 0 ||
           strcmp(dat->type, "real(kind=4)") == 0) {
    for (int isou = 0; isou < dim; isou++)
      _ops_append_char(buff, len, cap, "%16.10e   ",
                       ((float *)dat->data)[ip * dim + isou]);

  }
  else if (strcmp(dat->type, "int") == 0 ||
      strcmp(dat->type, "int(4)") == 0 ||
      strcmp(dat->type, "integer") == 0 ||
      strcmp(dat->type, "integer(4)") == 0 ||
      strcmp(dat->type, "integer(kind=4)") == 0  ||
      strcmp(dat->type, "long") == 0 ||
      strcmp(dat->type, "long long") == 0 ||
      strcmp(dat->type, "ll") == 0 ||
      strcmp(dat->type, "short") == 0) {

    for (int isou = 0; isou < dim; isou ++)
      _ops_append_char(buff, len, cap, "%16d   ",
                       ((int *)dat->data)[ip * dim + isou]);
  }

}


/*-------------------------------------------------------------------------------------*/
/*  Computes the grid cell on which particle is projected                              *
/--------------------------------------------------------------------------------------*/

bool _ops_particle_moved_to_exchange_zone(int bin_old[], int bin_new[], int rmv_limits[],
                                          int dim) {

  // Check if old point is in the crtical points for rebuilding the list
  for (int i = 0; i < dim; i++) {
    if ( bin_old[i] == bin_new[i]) continue; //No-need to check this direction remains fixed

    //Swap : 0 Check negative direction
    if ( ((bin_old[i] > rmv_limits[2 *i]) && (bin_new[i] <= rmv_limits[2 * i]))
        || ((bin_old[i] <= rmv_limits[2 * i]) && (bin_new[i] > rmv_limits[2 * i])) )
        return true;


  //TODO: Check afterwards
    //Swap 1: Check positive direction
    if (  ((bin_old[i] >= rmv_limits[2 * i + 1]) && (bin_new[i] < rmv_limits[2 * i + 1]))
        ||((bin_old[i] < rmv_limits[2 * i + 1]) && (bin_new[i] >= rmv_limits[2 *i + 1])))
      return true;

  }

  return false;
}


/*******************************************************************************/
/* Particle hanlding functions                                                 *
 ******************************************************************************/

/*-----------------------------------------------------------------------------*
 * ops_arg for particle data structures                                        *
 *-----------------------------------------------------------------------------*/

ops_arg ops_arg_dat_particle(ops_dat dat, int dim, char const *type,
                             ops_particle particle, ops_particle_mapping map,
                             ops_access acc, ops_stencil stencil) {
  (void) type;
  ops_arg temp = ops_arg_dat_core(dat, stencil, acc);
  (&temp)->dim = dim;
  (&temp)->argtype = OPS_ARG_DAT_PARTICLE;

  (&temp)->part_index = particle->index;
  (&temp)->map_index = map->index; //TODO: Shift into element

  return temp;
}

ops_arg ops_arg_dat_particleJ(ops_dat dat, int dim, char const *type,
                              ops_particle particle, ops_access acc) {

  (void) type;
  ops_arg temp = ops_arg_dat_core(dat, nullptr, acc);
  (&temp)->dim = dim;
  (&temp)->argtype = OPS_ARG_DAT_PARTICLE_J;
  (&temp)->part_index = particle->index;

  return temp;

}

ops_arg ops_arg_dat_history(ops_neighbor_history history, int dim, char const *type,
                            ops_access acc) {
  (void) type;
  ops_arg temp = ops_arg_dat_core(history->data, nullptr, acc);
  (&temp)->dim = dim;
  (&temp)->argtype = OPS_ARG_DAT_HISTORY;
  (&temp)->hist_index = history->index;
  (&temp)->history = history;

  return temp;
}

/*------------------------------------------------------------------------------
 * Function declares and defines a particle data list                           *
 *------------------------------------------------------------------------------*/

ops_particle  _ops_decl_particle_core(OPS_instance *instance, ops_block block,
                                      char *box, int type_size, char const* name) {

  /* Create ops_particle */
  ops_particle particle = new  ops_particle_core;//(ops_particle)ops_calloc(1, sizeof(ops_particle_core));

  /* Set default values */

  particle->no_particles = 0;
  particle->no_virtual = 0;
  particle->global_particles = particle->no_particles;
  particle->block = block;
  particle->box_block = box;
  particle->type_box = type_size;

  particle->Nmax = OPS_MAX_PART;
  particle->name = copy_str(name);

  particle->particle_envelope = nullptr;
  particle->ids = nullptr;
  particle->normal_vector = nullptr;

  particle->particle_dat_index = 0;
  particle->particle_dat_max = 10;
  particle->particle_dat = (ops_dat *)ops_malloc(sizeof(ops_dat) * particle->particle_dat_max);


  particle->particle_map_index = 0;
  particle->particle_map_max = 10;
  particle->map_list = (ops_particle_mapping *)ops_malloc(sizeof(ops_particle_mapping) *
                                                         particle->particle_map_max);

  particle->mark_deletion  =(int *) ops_malloc(sizeof(int) * particle->Nmax);

  particle->particle_pos_dat = nullptr;

  particle->histories = nullptr;
  particle->nhistories = 0;

  particle->histories = (ops_neighbor_history *) ops_malloc(particle->nhistories_max *
                                                            sizeof(ops_neighbor_history *));

  /* Assign particle to the correct list */

  instance->OPS_block_list[block->index].no_particle_structures++;
  instance->OPS_block_list[block->index].particle
     = (ops_particle_core **)ops_realloc(instance->OPS_block_list[block->index].particle,
                                         instance->OPS_block_list[block->index].no_particle_structures
                                              * sizeof(ops_particle_core *));


  int index = instance->OPS_block_list[block->index].no_particle_structures - 1;

  instance->OPS_block_list[block->index].particle[index] = particle; //TODO: single particle
  particle->index = index;

  return particle;
}

void ops_set_bounding_box_to_block(ops_block block, char * box) {

  OPS_instance *instance = OPS_instance::getOPSInstance();
  ops_block_descriptor block_list = instance->OPS_block_list[block->index];

  if (block_list.box == nullptr)
    block_list.box = box;

}



/*---------------------------------------------------------------------------
 * Function deletes particle list                                             *
 *----------------------------------------------------------------------------*/
ops_particle _ops_free_particle(ops_particle particle) {

  if (particle == NULL)
    return NULL;

  /* Clear vector list */
  particle->no_particles = 0;
  particle->global_particles = 0;
  particle->Nmax = 0;

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];

    ops_free(map->dx);
    ops_free(map);
  }
  ops_free(particle->map_list);

  ops_free(particle->histories);

  ops_free(particle->mark_deletion);
  ops_free((char *)particle->name);
  ops_free(particle->particle_dat);

  if (particle->is_wall) {
    ops_free(particle->xcm);
    ops_free(particle->nx);
  }
//  if (particle->box_block != NULL)
//    ops_free(particle->box_block);

  delete particle;
  return NULL;
}

ops_neighbor_history _ops_free_histories(ops_neighbor_history history) {

  if (history == NULL) return NULL;

 // ops_free(history->n_partnersI);
 // ops_free(history->partnersI);
 // ops_free(history->flagI);

 // ops_free(history->indexing);
 // ops_free(history->data);
 // ops_free(history->flag);

 // if (history->history_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
 //   ops_free(history->n_partnersJ);
 //   ops_free(history->partnersJ);
 //   ops_free(history->flagJ);
 // }

  ops_free(history->name);

  delete history;
  return NULL;

}

void ops_particle_realloc_list( ops_particle particle) {
  if (particle->particle_dat_index > particle->particle_dat_max) {
    particle->particle_dat = (ops_dat *) ops_realloc(particle->particle_dat,
                                                     (particle->particle_dat_index + 10)
                                                      *sizeof(ops_dat));
    particle->particle_dat_max = particle->particle_dat_index + 10;
  }
}

/*---------------------------------------------------------------------------------------*
 * Function for re-allocating ops_dat structures associated with     particles           *
 *                                                                                       *
 * \param[in]   noalloc      smallest size if Nmax                                       *
 *                                                                                       *
 *---------------------------------------------------------------------------------------*/
void ops_particle_realloc_data(ops_particle particle, int noalloc) {

  if (particle->Nmax >(size_t) noalloc) //TODO: Shift into size_t
    throw OPSException(OPS_INVALID_ARGUMENT, "Number of requested allocated particles "
          "is smaller than maximum allocated particles");

  /* Reallocate the ops_dat structures */
  particle->Nmax = noalloc + OPS_MAX_PART;


  particle->mark_deletion = (int *) ops_realloc(particle->mark_deletion,
                                                particle->Nmax * sizeof(int));

  ops_dat_realloc_core(particle->particle_pos_dat, particle->Nmax);

  if (particle->particle_envelope != NULL) {
       ops_dat_realloc_core(particle->particle_envelope, particle->Nmax);

  }

  if (particle->ids != NULL)
    ops_dat_realloc_core(particle->ids, particle->Nmax);

  if (particle->normal_vector != NULL)
    ops_dat_realloc_core(particle->normal_vector, particle->Nmax);

  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat = particle->particle_dat[i];
    ops_dat_realloc_core(dat, particle->Nmax);
  }

}


int _ops_particle_owned_dat(ops_particle particle, ops_dat dat) {

  if (particle == NULL)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_particle structure");

  if (dat == NULL)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_dat structure");

  if (strcmp(particle->particle_pos_dat->name, dat->name) == 0)
    return 1;

  if (particle->particle_envelope != NULL) {
    if (strcmp(particle->particle_envelope->name, dat->name) ==0)
      return 1;
  }

  if (particle->ids != NULL)
    if (strcmp(particle->ids->name, dat->name) == 0)
      return 1;

  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat_elem = particle->particle_dat[i];
    if (strcmp(dat_elem->name, dat->name) == 0)
      return 1;
  }

  return 0;
}


void ops_exit_particles(OPS_instance *instance) {

  /* Get block descriptor */
  ops_block_descriptor *block_list = instance->OPS_block_list;

  for (int i = 0; i < instance->OPS_block_index; i++) {
    for (int index = 0; index < block_list[i].no_particle_structures;index++) {
      block_list[i].particle[index] =
          _ops_free_particle( block_list[i].particle[index]);
    }


    ops_free(block_list[i].particle);
  }

}

void ops_exit_histories(OPS_instance *instance) {

  ops_block_descriptor *block_list = instance->OPS_block_list;

  for (int i = 0; i < instance->OPS_block_index; i++) {
    for (int index = 0; index < block_list[i].no_history_structures; index++) {
      block_list[i].histories[index] =
          _ops_free_histories(block_list[i].histories[index]);
    }

    ops_free(block_list[i].histories);
  }
}

//TODO: Check for exiting particle halos

/*******************************************************************************/
/*  Particle API  Functions
 *******************************************************************************/

ops_particle ops_decl_particle_char(ops_block block, char const* name,
                                    char *Box, int type_size) {
  return _ops_decl_particle_core(OPS_instance::getOPSInstance(), block, Box,
                                 type_size, name);
}

void ops_particle_set_wall_cm_normal(ops_particle particle, int dim,
                                     const char *xcm, const char *np) {

  particle->xcm = (char *) ops_malloc(dim * particle->type_box);
  particle->nx = (char *) ops_malloc(dim * particle->type_box);

  memcpy(particle->xcm, xcm, particle->type_box * dim);
  memcpy(particle->nx, np, particle->type_box * dim);
}



ops_neighbor_history _ops_decl_neigh_history_char(ops_particle particleI,
                                                  ops_particle particleJ,
                                                  const int dofs, const int num_neighsI,
                                                  const int num_neighsJ, int update_type,
                                                  const int type_hist, char* data,
                                                  int size_elem, const char *type,
                                                  const char *name) {

  ops_neighbor_history history = new ops_neighbor_history_core;

  //Sanity checks

  if (type_hist < 0 && type_hist > 1)
    throw OPSException(OPS_INVALID_ARGUMENT,"ERROR: Neighbor history takes "
                                            "only the following values\n"
                                            "0: (I, J) pair access the same neighbor history point\n"
                                            "1: (I, J) and (J, I) access different neighbor history "
                                            "point\n "
                                            "Please check your settings.\n");


  if (particleI != NULL && particleJ != NULL) {
    if (particleI->block->index != particleJ->block->index)
      throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Neighbor history can be defined only for "
                                               "particle types owned by the same block");
  }

  if (update_type < 0 && update_type > 2)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Neighbor update histpry can take only "
                                             "the following values:\n"
                                             "0: Contact history linked to a single type of particle"
                                             "1: Contact history can be reset by particle I only"
                                             "2: Both particles can reset the list\n");

  if (update_type == 1 && type_hist == 1)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Update list from a single type requires to"
                                             "access list in a unique point for (i,j) as accessed "
                                             "only for particle i\n");
  if (particleI == NULL )
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Neighbor history cannot be set "
                                             " for empty particle structures\n");

  if (num_neighsI < 1 || num_neighsJ < 1)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: A non-positive number of max. neighbors "
                                             "is set. Please check your settings\n");


  if ((dofs > 0 && particleI->ids == NULL) || (dofs > 0 && particleJ->ids == NULL))
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Neighbor history requires tags\n");

  //TODO: For check for particleI = particleJ and set the update flag to unique

  history->particleI = particleI;
  history->particleJ = particleJ;


  history->history_type = type_hist;
  history->num_neighsI = num_neighsI;
  history->num_neighsJ = (update_type == OPS_HISTORY_UPDATE_UNIQUE) ? num_neighsI : num_neighsJ;
  history->update_type = update_type;

  history->name = copy_str(name);
  history->flag_update = false;

  history->history_active = (dofs > 0) ? 1 : 0;

  //Generate data structures
  int block_size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], stride[OPS_MAX_DIM];
  int base[OPS_MAX_DIM] = {};

  d_m[0] = d_p[0] = 0;
  stride[0] = 1;
  block_size[0] =  particleI->Nmax;
  for (int i = 1; i < particleI->block->dims; i++) {
    d_m[i] = 0;
    d_p[i] = 0;
    block_size[i] = 1;
    stride[i] = 1;
  }

  //Generate name for the first field
  char *name_loc = append_str(name, "_nPartnersI");
  int *data_int = NULL;
  history->n_partnersI = ops_decl_particle_dat_char(particleI, 1, block_size, base,
                                                    d_m ,d_p, stride, (char *) data_int, sizeof(int),
                                                    "int", name_loc, true, false);

  name_loc = append_str(name, "_partnersI");
  history->partnersI = ops_decl_particle_dat_char(particleI, num_neighsI, block_size, base,
                                                  d_m, d_p, stride, (char *) data_int, sizeof(int),
                                                  "int", name_loc, true, false);

  name_loc = append_str(name, "_flagsI");
  history->flagI = ops_decl_particle_dat_char(particleI, num_neighsI, block_size, base,
                                              d_m, d_p, stride, (char *) data_int, sizeof(int),
                                              "int", name_loc, true, false);

  name_loc = append_str(name, "_indexI");
  history->indexI = ops_decl_particle_dat_char(particleI, num_neighsI, block_size, base,
                                               d_m, d_p, stride, (char *) data_int, sizeof(int),
                                               "int",name_loc, true, false);

  //Set for particle II if necessary

  switch(history->update_type) {
  case OPS_HISTORY_UPDATE_UNIQUE:
    history->partnersJ = history->partnersI;
    history->n_partnersJ = history->partnersJ;
    history->flagJ = history->flagI;
    history->indexJ = history->indexI;
    break;
  case OPS_HISTORY_UPDATE_BOTH_WAYS:
  {
    //Reset of variables

    block_size[0] =  particleJ->Nmax;

    name_loc = append_str(name, "_nPartnersJ");
    history->n_partnersJ = ops_decl_particle_dat_char(particleJ, num_neighsI, block_size, base,
                                                      d_m, d_p, stride, (char *) data_int,
                                                      sizeof(int), "int", name_loc, true, false);
    history->n_partnersJ->is_particle = true;

    name_loc = append_str(name, "_partnersJ");
    history->partnersJ = ops_decl_particle_dat_char(particleJ, num_neighsJ, block_size, base,
                                                    d_m, d_p, stride, (char *) data_int, sizeof(int),
                                                    "int", name_loc, true, false);

    name_loc = append_str(name, "_flagsJ");
    history->flagJ = ops_decl_particle_dat_char(particleJ, num_neighsJ, block_size, base,
                                       d_m, d_p, stride, (char *) data_int, sizeof(int),
                                       "int", name_loc, true, false);

    name_loc = append_str(name, "_indexJ");
    history->indexJ = ops_decl_particle_dat_char(particleJ, num_neighsJ, block_size, base,
                                        d_m, d_p, stride, (char *) data_int, sizeof(int),
                                        "int",name_loc, true, false);

  } break;
  default:
    history->partnersJ = nullptr;
    history->n_partnersJ = nullptr;
    history->flagJ = nullptr;
  }

  //Set the remaining structures
  int nmax = MAX(num_neighsI * particleI->Nmax, num_neighsJ * particleJ->Nmax);

  block_size[0] = nmax;

  if (history->history_active) {
    uint64_t *data_64t = NULL;
    name_loc = append_str(name,"_indexing");
    history->indexing = ops_decl_particle_dat_char(particleI, 1, block_size, base,
                                                   d_m, d_p, stride, (char *)data_64t,
                                                   sizeof(uint64_t), "uint64_t", name_loc,
                                                   false, false); //TODO: Shift into 1
  }

  name_loc = append_str(name,"local_index");
  history->indexing_local = ops_decl_particle_dat_char(particleI, 2, block_size, base,
                                                       d_m, d_p, stride, (char *) data_int, sizeof(int),
                                                       "int", name_loc, false, false);

  name_loc = append_str(name,"_flag");
  history->flag =ops_decl_particle_dat_char(particleI, 1, block_size, base, d_m,
                                            d_p, stride, (char *) data_int, sizeof(int),
                                            "int", name_loc, false, false);
  name_loc = append_str(name, "_data");

  history->data = nullptr;
  if (history->history_active)
    history->data = ops_decl_particle_dat_char(particleI, dofs, block_size, base,
                                               d_m, d_p, stride, data, size_elem, type,
                                               name_loc, false, false);

  history->nmax_cont = nmax; //TODO:
  history->nconts = 0;


  if (history->history_active) {
    name = append_str(name, "_tmp_data");
    history->nmax_new = MAX(history->num_neighsI, history->num_neighsJ) * OPS_MAX_PART;
    block_size[0] = history->nmax_new;
    size_elem = sizeof(int) * 2;
    history->new_neighbors  = ops_decl_particle_dat_char(particleI, dofs, block_size, base,
                                                       d_m, d_p, stride, data, size_elem, "int",
                                                       name_loc, false, false);
  }

  //Set up particle I histories
  particleI->nhistories++;
  particleI->histories = (ops_neighbor_history *) ops_realloc(particleI->histories,
                                                              particleI->nhistories *
                                                              sizeof(ops_neighbor_history));
  particleI->histories[particleI->nhistories - 1] = history;

  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    particleJ->nhistories++;
    particleJ->histories = (ops_neighbor_history *) ops_realloc(particleJ->histories,
                                                                particleJ->nhistories *
                                                                sizeof(ops_neighbor_history));
    particleJ->histories[particleJ->nhistories - 1] = history;
  }

  //Set to overall structure-TODO: May have an issue
  OPS_instance *instance = OPS_instance::getOPSInstance();
  ops_block block = particleI->block;
  instance->OPS_block_list[block->index].no_history_structures++;
  instance->OPS_block_list[block->index].histories =
    (ops_neighbor_history_core **) ops_realloc(instance->OPS_block_list[block->index].histories,
                                               instance->OPS_block_list[block->index].no_history_structures
                                               * sizeof(ops_neighbor_history_core *));


  int index = instance->OPS_block_list[block->index].no_history_structures - 1;
  instance->OPS_block_list[block->index].histories[index] = history; //TODO: single particle
  history->index = index;

  return history;

}

/*-------------------------------------------------------------------------------*/
/* \brief Marking particles for deletion and forward exchange                    *
 *
 * \param[in]  particle an ops_particle structure
 */
/*-------------------------------------------------------------------------------*/

void ops_particle_mark_for_del(ops_particle particle) {

  if (particle == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty ops_particle structure");

  char *box = particle->box_block;
  if (box == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Empty BoundingBox structure");


  int dim = particle->block->dims;
  switch(particle->type_box) {
  case sizeof(float):
    _ops_particle_mark_for_removal((BoundingBox<float> *)box,
                                   (float *) particle->particle_pos_dat->data,
                                   particle->mark_deletion, dim, 0, particle->no_particles);
    break;
  case sizeof(double):
    _ops_particle_mark_for_removal((BoundingBox<double> *)box,
                                   (double *) particle->particle_pos_dat->data,
                                   particle->mark_deletion, dim, 0, particle->no_particles);
    break;
  case sizeof(long double):
    _ops_particle_mark_for_removal((BoundingBox<long double> *)box,
                                  (long double *) particle->particle_pos_dat->data,
                                  particle->mark_deletion, dim, 0, particle->no_particles);
    break;
  }

}


/* Removing marked particles from the system */


void _ops_particle_remove_marked(ops_particle particle) {

  if (particle->no_particles == 0)
    return;

  int Nlocal = particle->no_particles;

  for (int i =0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] > 0) {
      if (i == Nlocal-1) {
        Nlocal--;
        break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      if (particle->ids != nullptr)
        _ops_particle_swap_data(particle->ids->data, i, Nlocal - 1,
                               particle->ids->elem_size);

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      //swap deletion as well
      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      Nlocal--;
      if (Nlocal == 0) break;

    }
  }


  particle->no_particles = Nlocal;
}


void _ops_particle_remove_marked_flag(ops_particle particle, int flag) {

  if (particle->no_particles == 0)
    return;

  int Nlocal = particle->no_particles;

  for (int i =0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] == flag) {
      if (i == Nlocal-1) {
        Nlocal--;
        break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);

      }

      if (particle->ids != nullptr)
        _ops_particle_swap_data(particle->ids->data, i, Nlocal - 1,
                                particle->ids->elem_size);

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal-1, dat->elem_size);
      }

      //swap deletion as well
      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      Nlocal--;
      if (Nlocal == 0) break;

    }
  }


  particle->no_particles = Nlocal;

}

void _ops_particle_remove_marked_with_maps(ops_particle particle ) {

  if (particle->no_particles == 0)
    return;

  int Nlocal = particle->no_particles;

  for (int i = 0; i < Nlocal; i++) {
    while (particle->mark_deletion[i] > 0) {

      if (i == Nlocal-1) {
        Nlocal--; break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      if (particle->ids != nullptr)
        _ops_particle_swap_data(particle->ids->data, i, Nlocal - 1,
                                particle->ids->elem_size);

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal - 1, dat->elem_size);
      }

      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      //Update maps
      for (int imaps = 0; imaps < particle->particle_map_index; imaps++) {
        ops_particle_mapping map = particle->map_list[imaps];
        _ops_particle_copy_mapping_data_to(map, i, Nlocal - 1);
      }

      Nlocal--;
      if (Nlocal == 0) break;

    }

  }

  particle->no_particles = Nlocal;
}

void  ops_particle_reset_virtual_particles(ops_particle particle) {

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->mapping_type != OPS_WITH_VIRTUAL) continue;

    int *binhead = (int *)map->binhead->data;
    int *bins = (int *)map->bin->data;
    int *bin2grid = (int *)map->parts_to_grid->data;



    for (size_t  i = particle->no_particles;
                 i < particle->no_particles + particle->no_virtual; i++) {

      int address = bin2grid[i];

      if (address > -1) binhead[address] = -1; //bins[i];

      bins[i] = -1;
      bin2grid[i] = -1;
    }

  }

  particle->no_virtual = 0;
}



void  ops_particle_rearrange_particles_for_removal(ops_particle particle) {
  if (particle->no_particles == 0) return;

  int Nlocal = particle->no_particles;

  for (int i = 0; i < Nlocal; i++) {
    while(particle->mark_deletion[i] > 0) {
      //no need to remove
      if (i == Nlocal - 1) {
        Nlocal--; break;
      }

      _ops_particle_swap_data(particle->particle_pos_dat->data, i, Nlocal - 1,
                              particle->particle_pos_dat->elem_size);

      if (particle->particle_envelope != nullptr) {
        _ops_particle_swap_data(particle->particle_envelope->data, i, Nlocal - 1,
                                particle->particle_envelope->elem_size);
      }

      if (particle->ids != nullptr)
        _ops_particle_swap_data(particle->ids->data, i, Nlocal - 1,
                                particle->ids->elem_size);

      for (int idat = 0; idat < particle->particle_dat_index; idat++) {
        ops_dat dat = particle->particle_dat[idat];
        _ops_particle_swap_data(dat->data, i, Nlocal - 1, dat->elem_size);
      }


      _ops_particle_swap_data((char *)particle->mark_deletion, i, Nlocal - 1,
                              sizeof(int));

      for (int imaps = 0; imaps < particle->particle_map_index; imaps++) {
        ops_particle_mapping map = particle->map_list[imaps];

      //  _ops_particle_swap_mapping_data(map, i, Nlocal - 1);
        _ops_particle_copy_mapping_data_to(map, i, Nlocal - 1);
      }

      Nlocal--;
      if (Nlocal == 0) break;
    }
  }

  particle->no_particles = Nlocal;

}

/*------------------------------------------------------*/
/*! Reset particles for deletion                       */
/*-----------------------------------------------------*/

void _ops_particle_reset_marked(ops_particle particle) {

  int nlocal = particle->no_particles;
  for (int i = 0; i < nlocal; i++)
    particle->mark_deletion[i] = 0;
}

void ops_particle_reset_flags(ops_particle particle, bool decide) {

  if (!decide) return;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    map->decide = false;
  }
}

void ops_particle_reset_history_flags(ops_neighbor_history *history,
                                      int nhistories) {

  for (int ihis = 0; ihis < nhistories; ihis++)
    history[ihis]->flag_update = false;
}


/***************************************************************************************/
/*  Definitions of particle halo and halo groups                                       *
 ***************************************************************************************/

ops_particle_halo_exchange _ops_particle_init_halo_info() {

  ops_particle_halo_exchange halo_info = nullptr;

  halo_info = (ops_particle_halo_exchange)
      ops_calloc(1, sizeof(OPS_particle_halo_exchange_info_core));


  halo_info->nsend = 0;
  halo_info->nrecv =0;
  halo_info->nmax = OPS_MAX_PART;
  halo_info->sendlist = (int *)ops_calloc(halo_info->nmax, sizeof(int));
  halo_info->firstrecv = 0;

  return halo_info;
}

ops_particle_halo_data _ops_particle_decl_halo_data_core(OPS_instance *instance,
                                                         ops_dat from, ops_dat to,
                                                         ops_part_orient orient_flag) {

  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {//TODO:
    instance->OPS_particle_halo_data_max += 10;
    instance->OPS_particle_halo_data_list =
        (ops_particle_halo_data *) ops_realloc(instance->OPS_particle_halo_data_list,
                                               instance->OPS_particle_halo_data_max
                                                * sizeof(ops_particle_halo_data));
    if (instance->OPS_particle_halo_data_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_particle_halo_core--"
                                            "Error allocation memory");
  }
  ops_particle_halo_data halo
  = (ops_particle_halo_data)ops_calloc(1, sizeof(ops_particle_halo_data_core));

  /* Set halos */
  if (from->dim != to->dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "To and from ops_dat must have "
                                             "the same size of elements per point.");

  if (from->type_size != to->type_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "Different types of the sending and receiving "
                                             "ops_dat structures");

  if (!from->is_particle || !to->is_particle)
    throw OPSException(OPS_INVALID_ARGUMENT, "Sending or receiving ops_dat structure not related"
                                              "to particle data structures");


  halo->from = from;
  halo->to = to;
  halo->orient = orient_flag;

  halo->history_from = nullptr;
  halo->history_to = nullptr;

  halo->halo_type = OPS_EXCHANGE_PARTICLE_DAT;

  instance->OPS_particle_halo_data_list[instance->OPS_particle_halo_data_index] = halo;
  instance->OPS_particle_halo_data_index++;

  return halo;
}

ops_particle_halo_data _ops_particle_decl_history_halo_core(OPS_instance *instance,
                                                            ops_neighbor_history from,
                                                            ops_neighbor_history to) {
  if (to->data->dim != from->data->dim)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Dimensions of Neighbor data structure of exchange "
                                          "histories are not equal\n");

  if (from->data->type_size != to->data->type_size)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Type of neighbor data structure of exchange "
                                          "history differ\n");

  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {//TODO:
    instance->OPS_particle_halo_data_max += 10;
    instance->OPS_particle_halo_data_list =
        (ops_particle_halo_data *) ops_realloc(instance->OPS_particle_halo_data_list,
                                               instance->OPS_particle_halo_data_max
                                                * sizeof(ops_particle_halo_data));
    if (instance->OPS_particle_halo_data_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error, ops_decl_particle_halo_core--"
                                            "Error allocation memory");
  }

  ops_particle_halo_data halo
  = (ops_particle_halo_data)ops_calloc(1, sizeof(ops_particle_halo_data_core));



  halo->from = nullptr;
  halo->to = nullptr;
  halo->orient = OPS_PART_ORIENT_OFF; //Default

  halo->halo_type = OPS_EXCHANGE_HISTORY;
  halo->history_from = from;
  halo->history_to = to;

  instance->OPS_particle_halo_data_list[instance->OPS_particle_halo_data_index] = halo;
  instance->OPS_particle_halo_data_index++;

  return halo;
}

ops_particle_halo _ops_free_particle_halo(ops_particle_halo halo) {

  if (halo == nullptr) {
    return nullptr;
  }

  delete halo->sendBox;

  ops_free(halo->dx);

  ops_free(halo->translate);

  ops_free(halo->dat);

  ops_free(halo);

  return NULL;
}


ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, ops_particle_halo_data halos[],
                                          int nhalos, char *critical_length,
                                          int *dir_from, int *dir_to,
                                          char *translate, int elem_type) {

  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {
    instance->OPS_particle_halo_max += 10;
    instance->OPS_particle_halo_list = (ops_particle_halo *)ops_realloc(
        instance->OPS_particle_halo_list, instance->OPS_particle_halo_max
                                                * sizeof(ops_particle_halo));
    if (instance->OPS_particle_halo_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  ops_particle_halo grp =
      (ops_particle_halo)ops_calloc(1, sizeof(ops_particle_halo_core));

  if (nhalos <= 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error non-positive size of ops_particle_halos");

  grp->nhalos = nhalos;
  if (to == NULL || from == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structures");

  if (from->particle_pos_dat->type_size != elem_type)
    throw OPSException(OPS_INVALID_ARGUMENT, "Non compatible types\n");

  if (to->particle_pos_dat->type_size != elem_type)
    throw OPSException(OPS_INVALID_ARGUMENT, "Non compatible types\n");


  grp->particle_from = from;
  grp->particle_to = to;

  grp->dx = (char *) ops_malloc(elem_type * to->block->dims);
  memcpy(grp->dx, critical_length, elem_type * to->block->dims);

  for (int idir = 0; idir < to->block->dims; idir++) {
    grp->dir_from[idir] = dir_from[idir];
    grp->dir_to[idir] = dir_to[idir];
  }


  grp->translate = (char *) ops_malloc(elem_type * to->block->dims);
  memcpy(grp->translate, translate, elem_type * to->block->dims);


  grp->dat  = (ops_particle_halo_data *)ops_calloc(nhalos, sizeof(ops_particle_halo_data));
  for (int i = 0; i < nhalos; i++) {
   int ip1_owned = _ops_particle_owned_dat(from, halos[i]->from);
   int ip2_owned = _ops_particle_owned_dat(from, halos[i]->to);

   if (ip1_owned == 0 || ip2_owned == 0)
     throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure not related to given ops_particle"
                                              " structures");
   grp->dat[i] = halos[i];
  }

  grp->sendBox = nullptr;

  grp->instance = instance;
  grp->nPoints = 0;

  int nbites = 0;
  for (int i = 0; i < nhalos; i++)
    nbites += halos[i]->from->elem_size;
  grp->nbites = nbites;

  instance->OPS_particle_halo_list[instance->OPS_particle_halo_index] = grp;
  grp->index = instance->OPS_particle_halo_index++;

  return grp;
}

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, int nhalos,
                                          ops_particle_halo_data halos[],
                                          int *dir_from, int *dir_to,
                                          char *sending_region,
                                          char *translate, int type_size)
{
  if (instance->OPS_particle_halo_index == instance->OPS_particle_halo_max) {
    instance->OPS_particle_halo_max += 10;
    instance->OPS_particle_halo_list = (ops_particle_halo *)ops_realloc(
        instance->OPS_particle_halo_list, instance->OPS_particle_halo_max
                                                * sizeof(ops_particle_halo));
    if (instance->OPS_particle_halo_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  ops_particle_halo grp =
      (ops_particle_halo)ops_calloc(1, sizeof(ops_particle_halo_core));

  if (nhalos <= 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error non-positive size of ops_particle_halos");

  grp->nhalos = nhalos;
  if (to == NULL || from == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structures");

  grp->particle_from = from;
  grp->particle_to = to;


  //TODO: Set it within a function
  grp->translate = translate;

  for (int idir = 0; idir < OPS_MAX_DIM; idir++) {
    grp->dir_from[idir] = dir_from[idir];
    grp->dir_to[idir] = dir_to[idir];
  }

  grp->dx = (char *) ops_malloc(type_size * to->block->dims);
  for (int i = 0; i < to->block->dims; i++) {
    switch (type_size) {
    case sizeof(float):
      ((float *)grp->dx)[i] = 0.0;
      break;
    case sizeof(double):
      ((double *)grp->dx)[i] = 0.0;
      break;
    case sizeof(long double):
      ((long double *)grp->dx)[i] = 0.0;
    }
  }

  switch (type_size) {
  case sizeof(float):
    grp->sendBox = _ops_particle_create_box((float *)sending_region, to->block->dims);
    break;
  case sizeof(double):
    grp->sendBox = _ops_particle_create_box((double *)sending_region, to->block->dims);
    break;
  case sizeof(long double):
    grp->sendBox = _ops_particle_create_box((long double *) sending_region, to->block->dims);
    break;
  }
  //Create boundingBox;


  grp->dat  = (ops_particle_halo_data *)ops_calloc(nhalos, sizeof(ops_particle_halo_data));
  for (int i = 0; i < nhalos; i++) {
   int ip1_owned = _ops_particle_owned_dat(from, halos[i]->from);
   int ip2_owned = _ops_particle_owned_dat(from, halos[i]->to);

   if (ip1_owned == 0 || ip2_owned == 0)
     throw OPSException(OPS_INVALID_ARGUMENT, "ops_dat structure not related to given ops_particle"
                                              " structures");
   grp->dat[i] = halos[i];
  }

  grp->instance = instance;
  grp->nPoints = 0;

  int nbites = 0;
  for (int i = 0; i < nhalos; i++) {
    if (halos[i]->halo_type == OPS_EXCHANGE_PARTICLE_DAT)
      nbites += halos[i]->from->elem_size;
    else {
      nbites += halos[i]->history_from->n_partnersI->elem_size
              + halos[i]->history_from->partnersI->elem_size
              + halos[i]->history_from->data->elem_size;
    }
  }
  grp->nbites = nbites;

  instance->OPS_particle_halo_list[instance->OPS_particle_halo_index] = grp;
  grp->index = instance->OPS_particle_halo_index++;

  return grp;
}


ops_particle_halo_group _ops_particle_decl_halo_group(OPS_instance *instance,
                                                      ops_particle_halo particle_halos[],
                                                      int nhalos,
                                                      ops_part_halo_grp_type halo_type,
                                                      ops_with_virtual with_virtual,
                                                      ops_particle_halo_group master) {

  if (instance->OPS_particle_halo_group_index == instance->OPS_particle_halo_group_max) {
    instance->OPS_particle_halo_group_max += 10;
    instance->OPS_particle_halo_group_list
      = (ops_particle_halo_group *) ops_realloc(instance->OPS_particle_halo_group_list,
                                                instance->OPS_particle_halo_group_max *
                                                  sizeof(ops_particle_halo_group));
    if (instance->OPS_particle_halo_group_list == NULL)
      throw OPSException(OPS_RUNTIME_ERROR, "Error ops_particle_decl_halo_group--error"
                                            "reallocating memory");
  }

  if (nhalos <= 0) return nullptr;

  ops_particle_halo_group halo_grp
    = (ops_particle_halo_group)ops_calloc(1, sizeof(ops_particle_halo_group_core));

  halo_grp->halo_list = (ops_particle_halo *)ops_calloc(nhalos, sizeof(ops_particle_halo));
  halo_grp->nhalos = nhalos;

  for (int i = 0; i < nhalos; i++)
    halo_grp->halo_list[i] = particle_halos[i];


  /* Allocate definition of exchange info */
  halo_grp->halo_info =
      (ops_particle_halo_exchange *)ops_calloc(nhalos, sizeof(ops_particle_halo_exchange));



  halo_grp->halo_type = halo_type;

  halo_grp->with_virtual = with_virtual;

  if (halo_type == OPS_HALO_GRP_FORWARD || halo_type == OPS_HALO_GRP_BACKWARD) {
    halo_grp->halo_master = master; //Yes if we shift bites to a different locaton
  }
  else { //Initialize halo infos
    for (int i = 0; i < nhalos; i++)
      halo_grp->halo_info[i] = _ops_particle_init_halo_info();
  }


  if (halo_grp == OPS_HALO_GRP_EXCHANGE && halo_grp->with_virtual == OPS_WITH_VIRTUAL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Exchange-halos to operate on actual and "
                                             "virtual particles");


  instance->OPS_particle_halo_group_list[instance->OPS_particle_halo_group_index] = halo_grp;
  halo_grp->index = instance->OPS_particle_halo_group_index;
  instance->OPS_particle_halo_group_index++;

  return halo_grp;
}

ops_particle_halo_group _ops_free_particle_halo_group(ops_particle_halo_group halo_grp) {

  if (halo_grp == NULL) return NULL;


  //Free
  if (  halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE
       || halo_grp->halo_type == OPS_HALO_GRP_BORDER) {
    for (int i = 0; i < halo_grp->nhalos; i++) {
      ops_free(halo_grp->halo_info[i]->sendlist);
      ops_free(halo_grp->halo_info[i]);
    }
  }
  ops_free(halo_grp->halo_info);

  ops_free(halo_grp->halo_list);

  ops_free(halo_grp);

  return NULL;
}

ops_particle_halo_data ops_particle_decl_data_halo(ops_dat from, ops_dat to,
                                                   ops_part_orient orient_flag) {

  return _ops_particle_decl_halo_data_core(OPS_instance::getOPSInstance(),from, to,
                                           orient_flag);

}

ops_particle_halo_data ops_particle_decl_history_halo(ops_neighbor_history from,
                                                      ops_neighbor_history to) {
  return _ops_particle_decl_history_halo_core(OPS_instance::getOPSInstance(),
                                              from, to);
}

ops_particle_halo ops_particle_decl_halo_char(ops_particle from, ops_particle to,
                                              ops_particle_halo_data particle_halos[],
                                              int  nhalos, char* critical_length,
                                              int *dir_from, int *dir_to,
                                              char *translate, int type_size) {
  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 particle_halos, nhalos, critical_length,
                                 dir_from, dir_to, translate, type_size);

}

ops_particle_halo ops_particle_decl_halo_with_pos_char(ops_particle from, ops_particle to,
                                                       ops_particle_halo_data particle_halos[],
                                                       int nhalos, char* critical_length,
                                                       int *dir_from, int *dir_to,
                                                       char *translate, int type_size){

  int nhalos1 = nhalos + 1;

  ops_dat pos_to = to->particle_pos_dat;
  ops_dat pos_from = from->particle_pos_dat;

  ops_particle_halo_data  *halo_data
  = (ops_particle_halo_data *) ops_malloc(sizeof(ops_particle_halo_data) * nhalos1);

  halo_data[0] = ops_particle_decl_data_halo(pos_from, pos_to, OPS_PART_POSITION);

  for (int i = 0; i < nhalos; i++)
    halo_data[i + 1] = particle_halos[i];


  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 halo_data, nhalos1, critical_length,
                                 dir_from, dir_to, translate, type_size);

}

ops_particle_halo ops_particle_decl_halo_char(ops_particle from, ops_particle to,
                                              int nhalos, ops_particle_halo_data particle_halos[],
                                              int *dir_from, int *dir_to,
                                              char *sending_region,
                                              char *translate, int type_size) {


  return _ops_particle_decl_halo(OPS_instance::getOPSInstance(), from, to,
                                 nhalos, particle_halos, dir_from,
                                 dir_to, sending_region, translate, type_size);

}


ops_particle_halo_group ops_particle_decl_halo_group(ops_particle_halo particle_halos[],
                                                     int nhalos,
                                                     ops_part_halo_grp_type halo_type,
                                                     ops_with_virtual with_virtual,
                                                     ops_particle_halo_group master) {

  return _ops_particle_decl_halo_group(OPS_instance::getOPSInstance(), particle_halos,
                                       nhalos, halo_type, with_virtual, master);
}

/*---------------------------------------------------------------------------------------*/
/*  Setup the exchange structure according to exchange type                              */
/*---------------------------------------------------------------------------------------*/

//TODO: Stopped herein.

void ops_particle_set_halo_group(ops_particle_halo_group halo_grp) {

  /* Get ops instance */
  OPS_instance *instance = OPS_instance::getOPSInstance();

  // TMP Definition: Later on shift within in class
  switch(halo_grp->halo_type) {
  case  OPS_HALO_GRP_EXCHANGE:
    _ops_particle_setup_exchange_comm(instance, halo_grp); //TODO:
    break;
  case  OPS_HALO_GRP_BORDER:
  case OPS_HALO_GRP_DEFAULT:
    _ops_particle_setup_border_comm(instance, halo_grp);
    break;
  case OPS_HALO_GRP_FORWARD:
  case OPS_HALO_GRP_BACKWARD:
    _ops_particle_setup_for_rev_comm(instance, halo_grp);
    break;
  default:
    break;
  }



}


void ops_particle_halo_check_for_periodicity(ops_particle_halo_group *group, int ngroups) {

  for (int igroup = 0; igroup < ngroups; igroup++) {
    for (int ihalos = 0; ihalos < group[igroup]->nhalos; ihalos++) {
      ops_particle_halo halo = group[igroup]->halo_list[ihalos];
      if (halo->particle_to->index != halo->particle_from->index)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Particle halo is not associated with the "
                                              " same particle structure (block and type");
    }
  }
}

//TODO: Make it later on the default algorithm
//TODO: Rename from hybrid to map
//THIS PERFORM MAP UPDATES AS WELL

void ops_particle_halo_transfer_group_map(ops_part_halo_grp_type exchange_type,
                                          bool exchange, ops_access access) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {
    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];

    switch(exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange)
          _ops_particle_halo_exchange_transfer_map(instance, halo_grp); //TODO
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {
        if (exchange)
         _ops_particle_halo_border_transfer_map(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        if (!exchange)
          _ops_particle_halo_forward_map(instance, halo_grp); //TODO
      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_reverse_transfer(instance, halo_grp, access);
      break;
    default:
      _ops_particle_halo_forward_map(instance, halo_grp);
    }
  }

}

void ops_particle_halo_transfer_group_map(ops_particle_halo_group *halo_group, int ngroup,
                                          ops_part_halo_grp_type exchange_type,
                                          bool exchange, ops_access access) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < ngroup; ihalos++) {
    ops_particle_halo_group halo_grp = halo_group[ihalos];

    switch(exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange)
          _ops_particle_halo_exchange_transfer_map(instance, halo_grp); //TODO
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {
        if (exchange)
         _ops_particle_halo_border_transfer_map(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        if (!exchange)
          _ops_particle_halo_forward_map(instance, halo_grp); //TODO
      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_reverse_transfer(instance, halo_grp, access);
      break;
    default:
      _ops_particle_halo_forward_map(instance, halo_grp);
    }
  }
}

void ops_particle_halo_transfer_group(ops_particle_halo_group *halo_group,
                                      int ngroup, ops_part_halo_grp_type exchange_type,
                                      bool exchange, ops_access access) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < ngroup; ihalos++) {
    ops_particle_halo_group halo_grp = halo_group[ihalos];
    switch(exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange) {
          _ops_particle_halo_exchange_transfer(instance, halo_grp);
        }
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
            halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {

        if (exchange)
          _ops_particle_halo_border_transfer(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (//halo_grp->halo_type != OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_forward_transfer(instance, halo_grp);

      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) //TODO: Check if we can add the default as well
        _ops_particle_halo_reverse_transfer(instance, halo_grp, access);
      break;
    default:
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Halo operation not supported. Supported types\n"
                                            "i. OPS_HALO_GRP_EXCHANGE\n"
                                            "ii. OPS_HALO_GRP_BORDER\n"
                                            "iii. OPS_HALO_GRP_FORWARD\n"
                                            "iv. OPS_HALO_GRP_BACKWARD");
    }
  }
}

//TODO: Modify as the BORDER to be able to perform forward and backward
//comms
void ops_particle_halo_transfer_group(ops_part_halo_grp_type exchange_type,
                                      bool exchange, ops_access access) {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  for (int ihalos = 0; ihalos < instance->OPS_particle_halo_group_index; ihalos++) {

    ops_particle_halo_group halo_grp = instance->OPS_particle_halo_group_list[ihalos];
    switch (exchange_type) {
    case OPS_HALO_GRP_EXCHANGE:
      if (halo_grp->halo_type == OPS_HALO_GRP_EXCHANGE) {
        if (exchange) {
          _ops_particle_halo_exchange_transfer(instance, halo_grp);
        }
      }
      break;
    case OPS_HALO_GRP_BORDER:
      if (halo_grp->halo_type == OPS_HALO_GRP_BORDER ||
            halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) {

        if (exchange)
          _ops_particle_halo_border_transfer(instance, halo_grp);
      }
      break;
    case OPS_HALO_GRP_FORWARD:
      if (//halo_grp->halo_type != OPS_HALO_GRP_BORDER ||
          halo_grp->halo_type == OPS_HALO_GRP_FORWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT)
        _ops_particle_halo_forward_transfer(instance, halo_grp);

      break;
    case OPS_HALO_GRP_BACKWARD:
      if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD ||
          halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) //TODO: Check if we can add the default as well
        _ops_particle_halo_reverse_transfer(instance, halo_grp, access);
      break;
    default:
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Halo operation not supported. Supported types\n"
                                            "i. OPS_HALO_GRP_EXCHANGE\n"
                                            "ii. OPS_HALO_GRP_BORDER\n"
                                            "iii. OPS_HALO_GRP_FORWARD\n"
                                            "iv. OPS_HALO_GRP_BACKWARD");
    }
  }
}

/**************************************************************************************************/
/* Mapping functions                                                                              */
/**************************************************************************************************/

ops_particle_mapping  _ops_decl_mapping_core(ops_particle particle, ops_dat grid,
                                             int size[],
                                             int d_m[], int d_p[], int base[],
                                             int stride[],
                                             ops_stencil stencil,
                                             ops_with_virtual include_virtual,
                                             ops_grid_type grid_type, int Ng) {
  /* Create a new structure */
  ops_particle_mapping map = (ops_particle_mapping)
       ops_malloc(sizeof(ops_particle_mapping_core));

  map->mapping_type = include_virtual;

  map->grid_type = grid_type;

  map->grid = grid;

  map->Ngrids = 1;
  if (map->grid_type != OPS_UNIFORM_STAG || map->grid_type != OPS_UNIFORM_COLL) {
    map->Ngrids = (Ng > 1) ? Ng : 1;
  }

  int*    nulli{nullptr};


  //TODO: Here we need to declaire type of bin
  map->parts_to_grid = ops_decl_particle_dat(particle, 1, base, nulli,
                                             "int", "particle_to_map", true, false);
  map->bin = ops_decl_particle_dat(particle, 1, base, nulli, "int", "bin_to_do",
                                     true, false);


//It should be strided

  map->binhead = ops_decl_dat(particle->block, 1, size, base, d_m, d_p,
                                stride, nulli,"int", "binhead");

  //Setting up the particle structure
  int dim = particle->particle_pos_dat->dim;
  void *data = nullptr;
  if (particle->particle_pos_dat->type_size == sizeof(float)) {
    map->pos_old = ops_decl_particle_dat(particle, dim, base, (float *) data,
                                         "float", "xold", true, false);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(double)) {
    map->pos_old = ops_decl_particle_dat(particle, dim, base, (double *) data,
                                         "double", "xold", true, false);
  }
  else if (particle->particle_pos_dat->type_size == sizeof(long double)) {
    map->pos_old = ops_decl_particle_dat(particle, dim, base, (long double *)data,
                                         "long double", "xold", true, false);

  }

  //Setup map->dx;
  int type_size = particle->particle_pos_dat->type_size;
  map->dx = (char *) ops_malloc(particle->block->dims * type_size);



  /* Copy Rp dat structure */
  map->particle = particle;
  map->Nmax = particle->Nmax;

  /* Define uniform grid mapping */
  map->mapping_stencil = stencil;

  particle->particle_map_index++;
  if (particle->particle_map_index > particle->particle_map_max) {
    particle->particle_map_max = particle->particle_map_index + 10;
    particle->map_list = (ops_particle_mapping *)ops_realloc(particle->map_list,
                     particle->particle_map_max * sizeof(ops_particle_mapping));
  }
  particle->map_list[particle->particle_map_index - 1] = map;

  map->index = particle->particle_map_index - 1;

  //map->index = particle->mapping_list.size() - 1;

  return map;

}

ops_particle_mapping _ops_decl_mapping_core(ops_particle particle, char *skin,
                                            int size[], int d_m[],
                                            int d_p[], int base[],
                                            ops_with_virtual with_virtual,
                                            ops_grid_type grid_type,
                                            int Ng) {

  //Allocate mapping structure
  ops_particle_mapping map = (ops_particle_mapping)
           ops_malloc(sizeof(ops_particle_mapping_core));

  //Set mapping type
  map->grid_type = grid_type;
  map->mapping_type = with_virtual;

  map->Ngrids = Ng; //TODO:

  /* Allocation of basic structures */
  double* nulld{nullptr};
  int*    nulli{nullptr};

  map->parts_to_grid = ops_decl_particle_dat(particle, 1, base, nulli,
                                             "int", "parts_to_map", true, false);

  map->bin = ops_decl_particle_dat(particle, 1, base, nulli, "int",
                                   "map_bins", true, false);

  map->pos_old = ops_decl_particle_dat(particle, particle->block->dims,
                                       base, nulld, "double", "ps_old", true, false);

  map->binhead = ops_decl_dat(particle->block, 1, size, base, d_m, d_p,
                              nulld, "double", "binhead");

  map->mapping_stencil = nullptr;

  map->grid = nullptr;

  int type_size = particle->particle_pos_dat->type_size;
  map->dx = (char *) ops_malloc(type_size * particle->block->dims);

  map->particle = particle;
  map->Nmax = particle->Nmax;


  //Assign array to structure
  particle->particle_map_index++;
  if (particle->particle_map_index > particle->particle_map_max) {
    particle->particle_map_max = particle->particle_map_index + 10;
    particle->map_list = (ops_particle_mapping *)ops_realloc(particle->map_list,
                     particle->particle_map_max * sizeof(ops_particle_mapping));
  }
  particle->map_list[particle->particle_map_index - 1] = map;

  map->index = particle->particle_map_index - 1;



  return map;
}


ops_particle_mapping ops_decl_mapping(ops_particle particle, ops_dat grid,
                                      ops_stencil stencil,
                                      ops_with_virtual include_virtual,
                                      ops_grid_type grid_type, int Ng) {

  int size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  int base[OPS_MAX_DIM], stride[OPS_MAX_DIM];

  _ops_mapping_def_core(particle, grid, stencil,
                        include_virtual, size, base, d_m, d_p);

  for (int i = 0; i < OPS_MAX_DIM; i++)
    stride[i] = 1;

  return _ops_decl_mapping_core(particle, grid, size, d_m, d_p, base, stride,
                                stencil, include_virtual, grid_type, Ng);

}

ops_particle_mapping ops_decl_mapping(ops_particle particle, ops_dat grid,
                                      ops_stencil stencil, int stride[] ,
                                      ops_with_virtual include_virtual,
                                      ops_grid_type grid_type,
                                      int Ng) {

  int size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], base[OPS_MAX_DIM];
  int stride_map[OPS_MAX_DIM];

  _ops_mapping_def_core(particle, grid, stencil,
                        include_virtual, size, base, d_m, d_p);


  //Set map size based on stride //Need perfect allocation proportionality to left-overs
  for (int i = 0; i < particle->block->dims; i++) {
    if (size[i] % stride[i] != 0) throw OPSException(OPS_RUNTIME_ERROR,
                                                     "Error: Introduce map do not project properly to block grid");
    size[i] /= stride[i];
    stride_map[i] = stride[i];
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++)
    stride_map[i] = 1;

  return _ops_decl_mapping_core(particle, grid, size, d_m, d_p, base, stride_map,
                                stencil, include_virtual, grid_type, Ng);

}

/*------------------------------------------------------------------------------------*/
/* \brief Building maps of particles to uniform or non-uniform structured grids
 *
 * \param[in] particle pointer to an ops_particle structure
 * \param[in] map      pointer to an ops_particle_mapping structure
 */
/*-----------------------------------------------------------------------------------*/

void ops_particle_map_build(ops_particle particle, ops_particle_mapping map) {
//  if (map->decide) {
    if (map->grid_type == OPS_UNIFORM_STAG ||
        map->grid_type == OPS_UNIFORM_COLL)
      _ops_particle_build_local_uniform(map, particle);
    else if (map->grid_type == OPS_NON_UNI_STAG ||
             map->grid_type == OPS_NON_UNI_COLL)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Non-uniform grids are not "
                                            "currently supported\n");

    map->decide = false;
 // }
}

void ops_particle_map_setup_actual_particles(ops_particle particle,
                                             ops_particle_mapping map) {

  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_setup_map(particle, map);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New mapping algorithm does not support "
                                            "non-uniform grids");

}

void ops_particle_map_setup_virtual_particles(ops_particle particle,
                                              ops_particle_mapping map) {

  if (map->grid_type == OPS_UNIFORM_STAG ||
      map->grid_type == OPS_UNIFORM_COLL)
    _ops_particle_setup_map_virtual(particle, map);
  else if (map->grid_type == OPS_NON_UNI_STAG ||
           map->grid_type == OPS_NON_UNI_COLL)
    throw OPSException(OPS_NOT_IMPLEMENTED, "Error: New mapping algorithm does not support "
                                            "non-uniform grids");


}


/*-------------------------------------------------------------------------------------*/
/* \brief Function to decide if a given particle will be build
 *
 * \param[in]  particle      pointer to an ops_particle structure
 * \param[in]  map           pointer to an ops_particle_mapping structure
 * \param[in]  enforce       flag for enforcing particle build list
 */
/*-----------------------------------------------------------------------------------*/
void ops_particle_map_decide(ops_particle particle, ops_particle_mapping map,
                             bool enforce) {

  if (enforce) {
    map->decide = true;
    return;
  }

  map->decide = false; //By default for searching

  int decide = _ops_particle_mapping_decide(map, particle, enforce);

  if (decide > 0)
    map->decide = true;
}

/*-----------------------------------------------------------------------------------*/
/* \brief the map for sequential code
 *
 * \return a flag that at least a single list is build
 */

bool ops_particle_update_map_lists(ops_particle particle) {

  bool decide_global=false;

  for (int index = 0; index < particle->particle_map_index; index++) {
  //for (auto &map : particle->mapping_list) {
    ops_particle_mapping map = particle->map_list[index];
    ops_particle_map_decide(particle, map);

//    if (!decide_global)

      decide_global = (map->decide) ? true : false;

      if (decide_global) break;

  }

  /* Mark particles for exchange or add them into an array */
  if (decide_global) {

    particle->no_virtual = 0; //Reset virtuals

    ops_particle_mark_for_del(particle);

   _ops_particle_exchange(particle);


 }

  return decide_global;

}


int ops_particle_update_map_lists_actual_parts(ops_particle particle) {

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    if (map->grid_type == OPS_UNIFORM_STAG ||
        map->grid_type == OPS_UNIFORM_COLL)
      int a1 =  _ops_particle_decide_build_local_uniform(map, particle);
    else if (map->grid_type == OPS_NON_UNI_STAG ||
             map->grid_type == OPS_NON_UNI_COLL)
      throw OPSException(OPS_NOT_IMPLEMENTED, "Concurrent decide and update is not"
                         "implemented for non uniform grids");
    else
      throw OPSException(OPS_NOT_IMPLEMENTED, "This type of grid is not supported\n");

  }

  int flag{0};
  for (int index = 0; index < particle->particle_map_index; index++)
    if (particle->map_list[index]->decide) {flag = 1; break;}

  //Herein resort particles prior to deletion

  if (flag) { //TODO: Second sanity check for reseting the list.
    ops_particle_reset_virtual_particles(particle);
    ops_particle_rearrange_particles_for_removal(particle);
  }

  return flag;
}




int ops_particle_update_map_lists_actual_hybrid(ops_particle particle) {
  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    if (map->grid_type == OPS_UNIFORM_STAG ||
          map->grid_type == OPS_UNIFORM_COLL)
        int a1 =  _ops_particle_decide_build_only_local_uniform(map, particle);
      else if (map->grid_type == OPS_NON_UNI_STAG ||
               map->grid_type == OPS_NON_UNI_COLL)
        throw OPSException(OPS_NOT_IMPLEMENTED, "Concurrent decide and update is not"
                           "implemented for non uniform grids");
      else
        throw OPSException(OPS_NOT_IMPLEMENTED, "This type of grid is not supported\n");

    map->decide = ops_particle_global_rebuild(map->decide);

  }

  int flag{0};

  for (int index = 0; index < particle->particle_map_index; index++)
   if (particle->map_list[index]->decide) {flag = 1; break;}

  if (flag == 1) {
    ops_particle_reset_virtual_particles(particle);

    _ops_particle_exchange_map_update(particle); //TODO

    //TODO: Rebuild maps for virtual intra-block
  }

  return flag;
}

void ops_particle_intrablock_border(ops_particle particle, ops_dat *dats,
                                    int ndats, ops_neighbor_history *histories,
                                    int nhistories) {

  _ops_particle_border_dats(particle, dats, ndats, histories, nhistories); //TODO: Shift to different needs

}

void ops_particle_intrablock_border_map_update(ops_particle particle, ops_dat *dats,
                                         int ndats, ops_neighbor_history *histories,
                                         int nhistories) {
  _ops_particle_border_dats_with_maps(particle, dats, ndats, histories, nhistories);
}

void ops_particle_intrablock_forward(ops_particle particle, ops_dat *dats,
                                     int ndats) {
  _ops_particle_forward_dats(particle, dats, ndats);
}


void ops_particle_intrablock_forward_map_update(ops_particle particle, ops_dat *dats,
                                                int ndats) {
  _ops_particle_forward_dats_with_maps(particle, dats, ndats);
}

void ops_particle_intrablock_reverse(ops_particle particle, ops_dat *dats, int ndats,
                                     ops_access access) {
  _ops_particle_reverse_dats(particle, dats, ndats, access);
}




void ops_particle_remove_particles(ops_particle particle, bool flag) {

  if (!flag) return;

  _ops_particle_remove_marked(particle);

  _ops_particle_reset_marked(particle);
}

/*--------------------------------------------------------------------------------------*
 * Function removes particles and update particle maps in the lists
 *
 * @param particle pointer to an ops_particle structure
 *--------------------------------------------------------------------------------------*/

void ops_particle_remove_delete_maps(ops_particle particle, int decide) {

  if (!decide)
    return;

  _ops_particle_remove_marked_with_maps(particle);


  //TODO: Check if last needs removal
  _ops_particle_reset_marked(particle);
}


void ops_particle_build_maps(ops_particle particle, bool flag) {


  //Prior to building expand the lists

  if (!flag)
    return;

  // Remove and delete particle
//  ops_particle_remove_marked(particle); //TODO: Shift to different location

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];
    ops_particle_map_build(particle, map);
  }

}

void ops_particle_setup_map_grid(ops_particle particle) {

  //Set up the maps

  for (size_t i = 0; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_setup_actual_particles(particle, map);
  }

  //TODO: Exchange intra-block halos
  if (particle->particle_map_index > 0) {
    _ops_particle_build_border_maps(particle);
  }

}

void ops_particle_setup_map_grid_dats(ops_particle particle, ops_dat *dats,
                                      int ndats, ops_neighbor_history *histories,
                                      int nhistories) {
  //Part I: Setup mark_deletion array
  for (size_t ip  = 0; ip < particle->no_particles; ip++)
    particle->mark_deletion[ip] = 0;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_setup_actual_particles(particle, map);
  }

  if (particle->particle_map_index > 0)
    _ops_particle_border_dats_with_maps(particle, dats,
                                        ndats, histories, nhistories); //TODO

}

void ops_particle_setup_map(ops_particle particle) {

  //First set actual particle non in deletion state
  for (size_t i = 0; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;


  //Exchange particles first between processes
  _ops_particle_build_border(particle);

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_build(particle, map);
  }
}

void ops_particle_setup_maps_with_dats(ops_particle particle, ops_dat *dats,
                                       int ndats, ops_neighbor_history *histories,
                                       int nhistories) {

  for (size_t ip = 0; ip < particle->no_particles; ip++)
    particle->mark_deletion[ip] = 0;

  for (int ihis = 0; ihis < nhistories; ihis++)
    ops_particle_init_histories(histories[ihis]);

  ops_particle_intrablock_border(particle, dats, ndats, histories, nhistories);
  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    ops_particle_map_build(particle, map);
  }

}


void ops_particle_setup_virtual_particles(ops_particle particle) {

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    if (map->mapping_type != OPS_WITH_VIRTUAL) continue;
    ops_particle_map_setup_virtual_particles(particle, map);
  }
}


/*--------------------------------------------------------------------------------------*
 *  Swap data between points in particle lists
 *--------------------------------------------------------------------------------------*/
void _ops_particle_swap_data(char *data, int i, int j, int elems) {

  for (int ielem = 0; ielem < elems; ielem++) {
    char ctmp = data[elems * i + ielem];
    data[elems * i + ielem] = data[elems * j + ielem];
    data[elems * j + ielem] = ctmp;
  }
}

//TO: What about xolds::
void _ops_particle_swap_mapping_data(ops_particle_mapping map, int from, int to) {

  //Get map structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;

  int *part2bin = (int *)map->parts_to_grid->data;
  //Part I: Swap binheads;

  int address_to = part2bin[to];
  int address_from = part2bin[from];

  //Swap part2bins
  part2bin[from] = part2bin[to];
  part2bin[to] = address_from;

  //Part II: Swap bins

  int bin_to = bins[to];
  int bin_from = bins[from];



 // int a1 = binhead[address_to];
 // int a2 = binhead[address_from];

  if (address_to != -1) {
    if (binhead[address_to] == to) {
      binhead[address_to] = from;
    }
    else {
      int iPart = binhead[address_to];
      int iPrev;
      while (iPart != to) {
        iPrev = iPart;
        iPart = bins[iPart];
      }

      bins[iPrev] = from;
    }

  }

  if (address_from != - 1) {
    if (binhead[address_from] == from) {
      binhead[address_from] = to;
    }
    else {
      int iPart = binhead[address_from];
      int iPrev;
      while (iPart != from) {
        iPrev = iPart;
        iPart = bins[iPart];
      }

      bins[iPrev] = to;
    }
  }


  bins[from] = bins[to];
  bins[to] = bin_from;



}

/*--------------------------------------------------------------------------------------*
 *  Function that copies mapping data from particle from to particle to
 */


//        _ops_particle_copy_mapping_data_to(map, i, nactual - 1);


void _ops_particle_copy_mapping_data_to(ops_particle_mapping map, int to, int from) {

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2bin = (int *)map->parts_to_grid->data;


  int address = part2bin[to];

  if (address < 0) return;

  if (binhead[address] == from) {
    binhead[address] = to;
  }
  else {
    int ipart = binhead[address];
    int iPrev;
    while (ipart != from) {
      iPrev = ipart;
      ipart = bins[ipart];
    }

    bins[iPrev] = to;
  }

  bins[from] = -1;

}


/*--------------------------------------------------------------------------------------*
 *  Build bounding boxes for ops_particle structures
 *--------------------------------------------------------------------------------------*/


void ops_particle_init_maps(ops_particle particle) {

  for (int index = 0; index < particle->particle_map_index; index++) {
    ops_particle_mapping map = particle->map_list[index];

    _ops_particle_map_validation(map);
    _ops_particle_init_map(map);


  }
}

void ops_partition_walls(ops_particle particle) {

  switch(particle->is_wall) {
  case 0:
    return;
    break;
  case 1:
    _ops_partition_flat_wall(particle);
    break;
  case 2:
    throw OPSException(OPS_RUNTIME_ERROR, "Analytical walls are not currently"
                       "supported\n");
    break;
  default:
    throw OPSException(OPS_RUNTIME_ERROR, "This wall type is not supported");
  }
}

void ops_particle_setup_partition() {
  //TODO: Set the bounding box based on particles

  OPS_instance *instance = OPS_instance::getOPSInstance();


  for (int index = 0; index < instance->OPS_block_index; index++) {

    int nparticles = instance->OPS_block_list[index].no_particle_structures;
    for (int ipart = 0; ipart < nparticles; ipart++) {
      ops_particle particle = instance->OPS_block_list[index].particle[ipart];
      ops_build_bounding_box(particle);

      ops_partition_walls(particle);

      ops_particle_init_maps(particle);

      ops_particle_setup_intrablock_comms(particle);

    }
  }


}

void ops_particle_update_map_halo_periodic_clas(ops_particle particle, ops_dat *dat_border,
                                                int nborder, ops_dat *dat_forward, int nforward,
                                                ops_particle_halo_group *halo_group, int nhalos) {

  if (nhalos > 0)
    ops_particle_halo_check_for_periodicity(halo_group, nhalos);

  bool decide_global = ops_particle_update_map_lists(particle);

  ops_particle_halo_transfer_group(halo_group, nhalos,
                                   OPS_HALO_GRP_EXCHANGE, decide_global);

  //Part III: Remove particles
  ops_particle_remove_particles(particle, decide_global);

  if (decide_global) {
    ops_particle_intrablock_border(particle, dat_border, nborder);

    ops_particle_halo_transfer_group(halo_group, nhalos,
                                     OPS_HALO_GRP_BORDER, decide_global);

    ops_particle_build_maps(particle, true);

  }
  else {
    ops_particle_intrablock_forward(particle, dat_forward, nforward);

    ops_particle_halo_transfer_group(halo_group, nhalos,
                                     OPS_HALO_GRP_FORWARD, decide_global);
  }


  ops_particle_reset_flags(particle, decide_global);
}

void ops_particle_update_map_halo_periodic(ops_particle particle, ops_dat *dat_border,
                                           int nborder, ops_dat *dat_forward, int nforward,
                                           ops_particle_halo_group  *halo_group, int n_halos) {

  int decide = ops_particle_update_map_lists_actual_hybrid(particle);

  if (n_halos > 0)
    ops_particle_halo_check_for_periodicity(halo_group, n_halos);

   //Part II: Exchange particles via halos and remove duplicate particles
  ops_particle_halo_transfer_group_map(halo_group, n_halos,
                                       OPS_HALO_GRP_EXCHANGE, decide);
  ops_particle_remove_delete_maps(particle, decide);

  //Part III: Intrablock & interblock communications
  if (decide) {

    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);

    ops_particle_halo_transfer_group_map(halo_group,n_halos,
                                         OPS_HALO_GRP_BORDER, decide); //TODO: Mdf

  }
  else {
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);

    ops_particle_halo_transfer_group_map(halo_group, n_halos,
                                         OPS_HALO_GRP_FORWARD, decide); //TODO: Mdf

  }

  ops_particle_reset_flags(particle, decide);

}


void ops_particle_init_histories(ops_neighbor_history history) {

  for (size_t i = 0; i <   history->particleI->no_particles
                       + history->particleI->no_virtual; i++) {
    ((int *) history->n_partnersI->data)[i] = 0;
  }

  if (history->update_type == OPS_HISTORY_UPDATE_BOTH_WAYS) {
    for (size_t i = 0; i < history->particleJ->no_particles
                      + history->particleJ->no_virtual; i++) {
      ((int *) history->n_partnersJ->data)[i] = 0;

    }
  }
}


ops_arg ops_arg_idp() {
  ops_arg arg;
  memset(&arg, 0, sizeof(ops_arg));

  arg.argtype = OPS_ARG_IDP; //TODO
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;
}

ops_arg ops_arg_idx_map() {
  ops_arg arg;
  memset(&arg, 0, sizeof(ops_arg));

  arg.argtype = OPS_ARG_IDX_MAP;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;

}


