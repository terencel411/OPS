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
  * @brief OPS single-process specific functions for particles
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements runtime support functions applicable to non-MPI backend
  */


#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <string>
#include <assert.h>
#include <array>
#include <limits>
#include <vector>

#include "ops_particle_mapping_functions.h"
#include "ops_particle_box_host_funcs.h"

char *temp = NULL;
int ntmp_size_max = 0;

static inline  uint64_t pack_pair(int tagI, int tagJ) {

  uint32_t a = (tagI < tagJ) ? tagI : tagJ;
  uint32_t b = (tagI < tagJ) ? tagJ : tagI;

  return ((uint64_t)a << 32) | b;
}

static int local_decide_rebuild(const double *xPart, const double *xGrid, const double dx,
                                const int dim) {

  for (int isou = 0; isou < dim; isou++)
    if (fabs(xPart[isou] - xGrid[isou]) > 0.5 * dx + DBL_EPSILON) {
      return 1;
    }


  return 0;
}

static void get_local_point(const int point,const int size[],
                            const int d_m[],const int dim, int grid[]) {
  int address = point;
  for (int i = dim - 1; i >= 0; i--) {
    int prod = 1;
    for (int j = 0; j < i; j++)
      prod *= size[j];
    grid[i] = address / prod;
    address -= grid[i] * prod;
    grid[i] += d_m[i];
  }
}

static bool _ops_box_get_ownership(char *box, int size_type) {

  switch (size_type) {
  case sizeof(float):
    return ((BoundingBox<float> *) box)->getOwnership();
    break;
  case sizeof(double):
    return ((BoundingBox<double> *) box)->getOwnership();
    break;
  case sizeof(long double):
    return ((BoundingBox<long double> *) box)->getOwnership();
    break;
  }

  return false;
}


static void _remove_particle_from_bins(int address, int i, int *binhead, int *bins) {



  int iPart = binhead[address];

  if (iPart == i)  {
    binhead[address] = bins[i];
  }
  else {
    int iPrev;
    while (iPart != i) {
      iPrev = iPart;
      iPart = bins[iPart];
    }

    bins[iPrev] = bins[i];
  }

  bins[i] = -1;
}

char * _ops_create_exchange_block(char * box_from, char *box_to, const int dim,
                                  const char *translate, const int *dir_to,
                                  const int *dir_from, const int type_size) {

  switch(type_size) {
  case sizeof(float):
    return (char *) _ops_setup_exchange_region((BoundingBox<float> *) box_from,
                                                  (BoundingBox<float> *) box_to,
                                                  (float *) translate, dir_to, dir_from,
                                                   dim);
    break;
  case sizeof(double):
    return (char *) _ops_setup_exchange_region((BoundingBox<double> *) box_from,
                                                  (BoundingBox<double> *) box_to,
                                                  (double *) translate, dir_to, dir_from,
                                                  dim);
    break;
  case sizeof(long double):
    return (char *) _ops_setup_exchange_region((BoundingBox<long double> *) box_from,
                                               (BoundingBox<long double> *) box_to,
                                               (long double *) translate, dir_to, dir_from,
                                               dim);
    break;
  }
  return nullptr;
}


char * _ops_set_exchange_border_box(char *box_from, char *box_to, const int dim,
                                    const char *translate, const int *dir_from,
                                    const int *dir_to, const char *dx,
                                    const int type_size) {

  switch(type_size) {
  case sizeof(float):
    return (char *) _ops_setup_border_region((BoundingBox<float> *) box_from,
                                             (BoundingBox<float> *) box_to,
                                             (float *) translate, dir_from,
                                             dir_to, (float *) dx, dim);
    break;
  case sizeof(double):
    return (char *) _ops_setup_border_region((BoundingBox<double> *) box_from,
                                             (BoundingBox<double> *) box_to,
                                             (double *) translate, dir_from,
                                              dir_to, (double *) dx, dim);
    break;
  case sizeof(long double):
    return (char *) _ops_setup_border_region((BoundingBox<long double> *) box_from,
                                             (BoundingBox<long double> *) box_to,
                                             (long double *) translate, dir_from,
                                              dir_to, (long double *) dx, dim);
    break;
  }

  return nullptr;
}

void _ops_particle_find_halo_cells(int *isend, char *box, const char *xmin, const char *xmax,
                                   const int *size, const char *dx, const int dim,
                                   const int type_size) {
  switch (type_size) {
  case sizeof(float):
    _ops_particle_halo_border_cells(isend, (BoundingBox<float> *) box, (float *) xmin,
                                    (float *) xmax, size, (float *) dx, dim);
    break;
  case sizeof(double):
    _ops_particle_halo_border_cells(isend, (BoundingBox<double> *) box, (double *) xmin,
                                    (double *) xmax, size, (double *) dx, dim);
    break;
  case sizeof(long double):
     _ops_particle_halo_border_cells(isend, (BoundingBox<long double> *) box, (long double *) xmin,
                                    (long double *) xmax, size, (long double *) dx, dim);

    break;
  }
}

void _ops_particle_allocate_tmp_array(int size_elem) {

  if (size_elem == 0) return;

  ntmp_size_max = size_elem * OPS_MAX_PART;
  temp = (char *) ops_malloc(ntmp_size_max);

}

void _ops_particle_free_tmp_array() {
  ops_free(temp);
  ntmp_size_max = 0;
}

void _ops_particle_setup_tmp_array(OPS_instance *instance) {

  int nhalo_groups = instance->OPS_particle_halo_data_index;
  ops_particle_halo_data *halos = instance->OPS_particle_halo_data_list;

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Setting up tmp array before domain "
                                          "partition");

  if (nhalo_groups == 0) return;

  int nelem_max = 0;
  for (int ihalo = 0; ihalo < nhalo_groups; ihalo++) {

    if (halos[ihalo]->halo_type !=  OPS_EXCHANGE_PARTICLE_DAT) continue;
    nelem_max = MAX(nelem_max, halos[ihalo]->to->elem_size);
  }

  if (nelem_max > 0)
    _ops_particle_allocate_tmp_array(nelem_max);


}



//MPI VERSION EXISTS
void _ops_particle_halo_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                   int nhalos, ops_particle_halo_exchange info,
                                   int flag, int *ntot_bites) {

  int *sendlist = info->sendlist;
  int nsend = info->nsend;

  int nsend_bites = 0;

  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->from;
      int nbites = dat->elem_size;
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        memcpy(buff + nsend_bites, dat->data + ipart * nbites, nbites);
        nsend_bites += dat->elem_size;
      }
    }
    else if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_HISTORY && flag){
      ops_neighbor_history history = halo_data[ihalo]->history_from;

      //Part I: Shift npartnes only
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        int nbites = history->n_partnersI->elem_size;
        memcpy(buff + nsend_bites, history->n_partnersI->data + ipart * nbites, nbites);
        nsend_bites += nbites;
      }

      //Part II: Shift partnersI only
      for (int i = 0; i < nsend; i++) {
         int ipart = sendlist[i];
         int npartners = ((int *)history->n_partnersI->data)[ipart];
         int ishift = ipart * history->partnersI->elem_size;
         int nbites = npartners * history->partnersI->type_size;
         memcpy(buff + nsend_bites, history->partnersI->data + ishift, nbites);
         nsend_bites += nbites;
      }

      //Part III: Shift contacts to send
      int nbites = history->data->elem_size;
      for (int i = 0; i < nsend; i++) {
        int ipart = sendlist[i];
        for (int j = 0; j < ((int *)history->n_partnersI->data)[ipart]; j++) {
          int index = ((int *) history->indexI->data)[ipart * history->num_neighsI + j];
          memcpy(buff + nsend_bites, history->data->data + index * nbites, nbites);
          nsend_bites += nbites;
        }
      }
    }

  }

  (*ntot_bites) = nsend_bites;
}

//MPI VErsion exists
void _ops_particle_halo_reverse_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                           int nhalos, ops_particle_halo_exchange halo_info,
                                            int *ntot_bites) {

  int nfirst =  halo_info->firstrecv;
  int nsend = halo_info->nsend;

  int nsend_bites = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->to;
      int send_size = dat->elem_size * nsend;
      memcpy(buff + nsend_bites, dat->data + nfirst, send_size);
      nsend_bites += send_size;
    }
  }

  (*ntot_bites) = nsend_bites;
}

//MPI Version exists
//TODO: We need to change the translate
void _ops_particle_halo_copy_from_buff(char *buff, ops_particle_halo_data *halos,
                                       int nhalos, ops_particle_halo_exchange info,
                                       int flag,
                                       int dir_to[], int dir_from[], char* translate,
                                       int *ntot_bites) {

  int nfirst = info->firstrecv;
  int nrecv = info->nrecv;

  int a1 = 0;
  int max_size = 0;

  int nrecv_bites = 0;

  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halos[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halos[ihalo]->to;
      if (halos[ihalo]->orient == OPS_PART_ORIENT_ON ||
          halos[ihalo]->orient == OPS_PART_POSITION) {
        a1 = 1;
        max_size = MAX(max_size, dat->elem_size);
      }
    }
  }

  if (a1 == 1 && nrecv * max_size > ntmp_size_max) {
    temp = (char *) ops_realloc(temp, nrecv * max_size);
    ntmp_size_max = nrecv * max_size;
  }

  //Unpack structures

  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halos[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halos[ihalo]->to;
      ops_part_orient orient = halos[ihalo]->orient;
      int nsize = nrecv * dat->elem_size;
      if (orient == OPS_PART_ORIENT_OFF) {
        int nbite_first = nfirst + dat->elem_size;
        memcpy(dat->data + nbite_first, buff + nrecv_bites, nsize);
      }
      else { //TODO: Work here to modify
        int dim = dat->dim;
        int dims = dat->block->dims;

        //TODO: Need to shift to multipe types
        if (dim != dims || dat->type_size != sizeof(double) ||
            dat->type_size != sizeof(float) ||
            dat->type_size != sizeof(long double))
          throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat with orient"
                                                             " must be a vector (of size dim per particle point) and of "
                                                             "type double");
        memcpy(temp, buff + nrecv_bites, nsize);

        char *data = dat->data;

        for (int i = 0; i < nrecv; i++) {
          int ipart = nfirst + i;
          for (int isou = 0; isou < dim; isou++)
            switch(dat->type_size) {
            case sizeof(float):
              ((float *) data)[ipart * dim + dir_to[isou]] =
                ((float *) temp)[i * dim + dir_from[isou]]
                + ((float  *) translate)[dir_from[isou]];
            break;
            case sizeof(double):
              ((double *) data)[ipart * dim + dir_to[isou]] =
              ((double *) temp)[i * dim + dir_from[isou]]
              + ((double *) translate)[dir_from[isou]];
            break;
            case sizeof(long double):
              ((long double *) data)[ipart * dim + dir_to[isou]] =
              ((long double *) temp)[i * dim + dir_from[isou]]
              + ((long double  *) translate)[dir_from[isou]];
             break;
            }
        }
      }
      nrecv_bites +=nsize;
    }
    else if (flag && halos[ihalo]->halo_type == OPS_EXCHANGE_HISTORY){
      ops_neighbor_history history = halos[ihalo]->history_to;
      ops_dat npartnersI = history->n_partnersI;
      ops_dat partnersI = history->partnersI;
      ops_dat indexI = history->indexI;

      ops_particle particle = history->particleI;
      int *tags = (int *) particle->ids->data;

      int nshift = nfirst * npartnersI->elem_size;
      int nbites = nrecv * npartnersI->elem_size;
      memcpy(npartnersI->data + nshift, buff + nrecv_bites, nbites);

      nrecv_bites += nbites;

      //Part II: Receive partners and update index;
      int new_neighs = 0;
      for (int i = 0; i < nrecv; i++) {
        int ipart = nfirst + i;
        nshift = ipart * partnersI->elem_size;
        nbites = partnersI->type_size * ((int *) npartnersI->data)[ipart];
        memcpy(partnersI->data + nshift, buff + nrecv_bites, nbites);
        nrecv_bites += nbites;

        //Set the index to the proper contact (i,j)
        for (int j = 0; j < ((int *)npartnersI->data)[ipart]; j++) {
          ((int *) indexI->data)[ipart * history->num_neighsI + j]
                               = history->nconts + new_neighs + j;
        }
        new_neighs+= ((int *)npartnersI->data)[i];
      }

      //Part III Reallocate neighbor histories
      if (new_neighs + history->nconts > history->nmax_cont) {
        history->nmax_cont += (new_neighs + OPS_MAX_PART) * MAX(history->num_neighsI,
                                                                history->num_neighsJ);
        int size = history->data->elem_size * history->nmax_cont;
        history->data->data = (char *) ops_realloc(history->data->data, size);

        size = history->indexing->elem_size * history->nmax_cont;
        history->indexing->data = (char *) ops_realloc(history->indexing->data, size);

        size = history->flag->elem_size * history->nmax_cont;
        history->flag->data = (char *) ops_realloc(history->flag->data, size);
      }

      //Part IV: Update history data
      nshift = history->nconts * history->data->elem_size;
      nbites = new_neighs * history->data->elem_size;
      memcpy(history->data->data + nshift, buff + nrecv_bites, nbites);
      nrecv_bites += nbites;

      //UPdate keys
      int ineighs = 0;
      for (int i = 0; i < nrecv; i++) {
        int ipart = nfirst + i;
        int tag = tags[ipart];
        for (int j = 0; j < ((int *) history->n_partnersI->data)[ipart]; j++) {
          int tagJ = ((int *)partnersI->data)[ipart * history->num_neighsI + j];
          int pair = ineighs + history->nconts;
          ((uint64_t *)history->indexing->data)[history->indexing->dim * pair] =
              pack_pair(tag, tagJ);
          ((uint64_t *)history->indexing->data)[history->indexing->dim * pair + 1] =
              pack_pair(ipart, 0);
          ineighs++;
        }
      }

      history->nconts += new_neighs;
    }
  }

  (*ntot_bites) = nrecv_bites;
}

//MPI Version exists
void _ops_particle_halo_reverse_copy_from_buff(char *buff, ops_particle_halo_data *halo_data,
                                               int nhalos, ops_particle_halo_exchange info,
                                               int dir_from[],  int dir_to[], int *ntot_bites,
                                               ops_access access) {

  int *sendlist = info->sendlist;
  int nsend = info->nsend;

  int nmax = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++)
    nmax = MAX(halo_data[ihalo]->to->elem_size, nmax);

  if (nsend * nmax > ntmp_size_max) {
    ntmp_size_max = (nsend + OPS_MAX_PART) * nmax;
    temp = (char *) ops_realloc(temp, ntmp_size_max);
  }

  int ifirst = 0;
  for (int ihalo = 0; ihalo < nhalos; ihalo++) {
    if (halo_data[ihalo]->halo_type == OPS_EXCHANGE_PARTICLE_DAT) {
      ops_dat dat = halo_data[ihalo]->from;
      int size_dat = dat->elem_size;
      memcpy(temp, buff + ifirst, size_dat);
      ifirst += size_dat;

      if (strcmp(dat->type, "float") == 0) {
        float *data = (float *)dat->data;
        float *tmp_fl = (float *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_fl + i * dat->dim,
                           dat->dim, access);
        }
      }
      else if (strcmp(dat->type, "int") == 0) {
        int *data = (int *)dat->data;
        int *tmp_i = (int *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_i + i * dat->dim,
                           dat->dim, access);
        }
      }
      else if (strcmp(dat->type, "double") == 0) {
        double *data = (double *)dat->data;
        double *tmp_dbl = (double *)temp;
        for (int i = 0; i < nsend; i++) {
          int ipart = sendlist[i];
          _ops_inc_element(data + dat->dim * ipart, tmp_dbl + i * dat->dim,
                           dat->dim, access);
        }
      }
      else {
        throw OPSException(OPS_RUNTIME_ERROR, "ERROR: This variable type not supported in reverse "
                                            " halo exchanges" );
      }
    }

  }

  (*ntot_bites) = ifirst;

}

/*-----------------------------------------------------------------------------------------------*/
/* Assert the intersection of a grid cell with a particle center                                 */
/*-----------------------------------------------------------------------------------------------*/

//TODO: RMV
int _ops_check_particle_nunif_grid_inters(const int dim,const double *xGrid, const double *dxGrid,
                                          const double *xp) {
   
  if (xp[0] < xGrid[0] - 0.5 * dxGrid[0] || xp[0] > xGrid[0] + 0.5 * dxGrid[0])
    return 0;

  /* y-direction */
  if (xp[1] > xGrid[1] - 0.5 * dxGrid[1] || xp[1] > xGrid[1] + 0.5 * dxGrid[1])
    return 0;

  if (dim == 3) {
    if (xp[2] > xGrid[2] - 0.5 * dxGrid[2] || xp[2] > xGrid[2] + 0.5 * dxGrid[2])
      return 1;
  }

  return 1;
}

//TODO: RMV
static void _ops_particle_export_data_point(FILE *fp, int i, ops_dat dat) {

  if (!dat->is_particle) {
    throw OPSException(OPS_RUNTIME_ERROR,
                       "Error: ops_dat not related to particle data\n");
  }

  int dim = dat->dim;
  if (strcmp(dat->type, "double") == 0 ||
      strcmp(dat->type, "real(8)") == 0 ||
      strcmp(dat->type, "real(kind = 8)") == 0 ||
      strcmp(dat->type, "double precision") == 0) {
    for (int isou = 0; isou < dim; isou++)
      if(fprintf(fp, "%16.10e ", ((double *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR,"Error: Writing to file");
  }
  else if (strcmp(dat->type, "float") == 0 ||
           strcmp(dat->type, "real") == 0 ||
           strcmp(dat->type, "real(4)") == 0 ||
           strcmp(dat->type, "real(kind=4)") == 0) {
    for (int isou = 0; isou < dim; isou++)
      if (fprintf(fp, "%16.10e ", ((float *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file");
  }
  else if (strcmp(dat->type, "int") == 0 ||
           strcmp(dat->type, "int(4)") == 0 ||
           strcmp(dat->type, "integer") == 0 ||
           strcmp(dat->type, "integer(4)") == 0 ||
           strcmp(dat->type, "integer(kind=4)" )== 0  ||
           strcmp(dat->type, "long") == 0 ||
           strcmp(dat->type, "long long") == 0 ||
           strcmp(dat->type, "ll") == 0 ||
           strcmp(dat->type, "short") == 0 ) {
    for (int isou = 0; isou < dim; isou++)
      if (fprintf(fp, "%8d ", ((int *)dat->data)[i * dim + isou]) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file");
  }
  else {
   OPSException ex(OPS_RUNTIME_ERROR);
   ex<<"Error: This type " << dat->type <<" is not supported for output of particle data to txt files";
   throw ex;
  }

}

void _ops_particle_halo_dat_to_buff(char *buff, ops_dat dat,
                                    ops_particle_halo_exchange halo_info,
                                    int *ntot_bites) {

  int *sendlist  = halo_info->sendlist;
  int nsend = halo_info->nsend;

  int nsend_bites = 0;
  int nbites = dat->elem_size;
  for (int i = 0; i < nsend; i++) {
    int ipart = sendlist[i];
    memcpy(buff + nsend_bites, dat->data + ipart * nbites, nbites);
    nsend_bites += dat->elem_size;
  }

  (*ntot_bites) = nsend_bites;
}

void _ops_particle_dat_copy_from_buff(char *buff, ops_dat dat, ops_part_orient orient,
                                           ops_particle_halo_exchange halo_info,
                                           int dir_to[], int dir_from[], char *translate,
                                           int *ntot_bites) {
  int nfirst = halo_info->firstrecv;
  int nrecv = halo_info->nrecv;

  int nrecv_bites = 0;
  int nsize = dat->elem_size;

  if (orient == OPS_PART_ORIENT_ON) {

    if (nsize * nrecv > ntmp_size_max) {
      temp = (char *) ops_realloc(temp, nsize * (nrecv + OPS_MAX_PART));
      ntmp_size_max = nsize * (nrecv + OPS_MAX_PART);
    }

    memcpy(temp, buff + nrecv_bites, nsize);

    switch(dat->type_size) {
    case sizeof(float): {
      float *tmp = (float *)temp;
      float *data = (float *) dat->data;

      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }

      } break;
    case sizeof(double): {
      double *tmp = (double *) tmp;
      double *data = (double *) dat->data;
      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }
      } break;
    case sizeof(long double): {
      long double *tmp = (long double *)temp;
      long double *data = (long double *) dat->data;

      for (int i = 0; i < nrecv; i++) {
        for (int isou = 0; isou < dat->dim; isou++)
        data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
      }
      } break;
    }
  }
  else {
    int nbite_first = nfirst * dat->elem_size;
    memcpy(dat->data + nbite_first, buff + nrecv_bites, nsize);
  }

  (*ntot_bites) = nsize * nrecv;
}


/*-------------------------------------------------------------------------------------*/
/* Mapping particle functions
 *-------------------------------------------------------------------------------------*/

//MPI Version exists
int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {
  /*Enforce new build */
  if (enforce || map->decide) {
    for (int i = 0; i < particle->nhistories; i++)
      particle->histories[i]->flag_update = true;
    return 1;
  }



  //1. Particle outside structure
 // if (_ops_particle_moved_outside(particle) == 1)
 //   return 1;


  int flag = 0;
  ops_dat xps = particle->particle_pos_dat;
  ops_dat xps_old = map->pos_old;

  /* 2. Particle moved outside previous block */
  int dim = particle->block->dims;
  int nParticles =(int) particle->no_particles;

  for (int i = 0; i < nParticles; i++) {
    switch (xps->type_size) {
    case sizeof(float):
     flag = _ops_check_particle_movement((float *) xps->data + dim * i,
                                         (float *) xps_old->data + dim * i,
                                         (float *) map->dx, dim);
      break;
    case sizeof(double):
      flag = _ops_check_particle_movement((double *) xps->data + dim * i,
                                          (double *) xps_old->data + dim * i,
                                          (double *) map->dx, dim);
      break;
    case sizeof(long double):
      flag = _ops_check_particle_movement((long double *) xps->data + dim * i,
                                         (long double *) xps_old->data + dim * i,
                                         (long double *) map->dx, dim);
      break;
    }

    if (flag == 1) break;
  }


  for (int i = 0; i < particle->nhistories; i++)
    if (!particle->histories[i]->flag_update)
      particle->histories[i]->flag_update = (bool) flag;

  if (flag) map->decide = true;
  else
    map->decide = false;

  return flag;

}

void _ops_build_particle_to_grid(const size_t nParticles, const ops_dat bin,
                                 const ops_dat binhead, ops_dat part_to_grid) {
  if (nParticles < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Non-positive number of particles\n");

  if (!bin->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat bin is not a particle ops_dat"
                                          "structure");

  if (binhead->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat binhead is defined for particle data");

  if (!part_to_grid->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR, "ops_dat part_to_grid must be set"
                       "for grid data");

  int *binheads = (int *)binhead->data;
  int *bins = (int *)bin->data;
  int *par_grid = (int *)part_to_grid->data;

  int dims = bin->block->dims;
  int sizeL{1};
  for (int i = 0; i < dims; i++)
    sizeL*= binhead->size[i];

  for (int  iGrid = 0; iGrid < sizeL; iGrid++) {
    int ip = binheads[iGrid];
    if (ip == -1) continue;

    par_grid[ip] = iGrid;
    while (ip != -1) {
      ip = bins[ip];
      if (ip != -1) par_grid[ip] = iGrid;
    }
  }
}

//MPI-Version exists
void  _ops_particle_build_local_uniform(ops_particle_mapping map, ops_particle particle) {
  //TODO: Realloc all lists as larger size is expected
  size_t Np = (map->mapping_type == OPS_WITH_VIRTUAL) ?
      particle->no_particles + particle->no_virtual : particle->no_particles;

  //TODO: Allocate local structures
  map->nParticles = Np;
  /* Reallocate particle lists */

  //Destroy existing structures and rebuild

  int prod = 1;
  for (int i = 0; i < particle->block->dims; i++) prod *= map->binhead->size[i];
  for (int i = 0; i < prod;i++)
    ((int *)map->binhead->data)[i] = -1;

  for (size_t i = 0; i < Np; i++) {
    ((int *)map->bin->data)[i] = -1;
    ((int *)map->parts_to_grid->data)[i] = -1;
  }

  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->type_box) {
  case sizeof(float): {
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            map->binhead->d_m, map->binhead->d_p, dim);
    ops_point<float> x_min;
    ops_point<float> x_max;
    x_min.x = ((float *)xmin)[0]; x_max.x = ((float *)xmax)[0];
    x_min.y = ((float *)xmin)[1]; x_max.y = ((float *)xmax)[1];
    x_min.z = (dim == 3) ? ((float *)xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((float *)xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (float *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
    } break;
  case sizeof(double): {
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *) dx,
                            map->binhead->d_m, map->binhead->d_p, dim);
    ops_point<double> x_min;
    ops_point<double> x_max;
    x_min.x = ((double *)xmin)[0]; x_max.x = ((double *)xmax)[0];
    x_min.y = ((double *)xmin)[1]; x_max.y = ((double *)xmax)[1];
    x_min.z = (dim == 3) ? ((double *)xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((double *) xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (double *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
    } break;
  case sizeof(long double): {
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *) dx, map->binhead->d_m, map->binhead->d_p,
                            dim);

    ops_point<long double> x_min;
    ops_point<long double> x_max;
    x_min.x = ((long double *) xmin)[0]; x_max.x = ((long double *) xmax)[0];
    x_min.y = ((long double *) xmin)[1]; x_max.y = ((long double *) xmax)[1];
    x_min.z = (dim == 3) ? ((long double *) xmin)[2] : 0.0;
    x_max.z = (dim == 3) ? ((long double *) xmax)[2] : 0.0;
    _ops_build_uniform_dats(1, dim, map->grid, particle->particle_pos_dat,
                            Np, (long double *)dx, x_min, x_max, map->binhead, map->bin,
                            map->parts_to_grid);
    } break;
  }

  int *part_to_bin = (int *) map->parts_to_grid->data;
  char *xps_old = map->pos_old->data;
  int ilocal[OPS_MAX_DIM];
  for (size_t i = 0; i < Np; i++) {
    int address = part_to_bin[i];
    get_local_point(address, map->binhead->size, map->binhead->d_m, particle->block->dims, ilocal);
    switch (particle->type_box) {
    case sizeof(float): {
      ops_point<float> x_min;
      x_min.x = ((float *)xmin)[0];
      x_min.y = ((float *)xmin)[1];
      x_min.z = (dim == 3) ? ((float *)xmin)[2] : 0.0;
      get_coord_point((float *) xps_old + i * dim, ilocal, map->binhead->d_m, x_min, (float *)dx,
                      dim, 1);
      }
      break;
    case sizeof(double): {
      ops_point<double> x_min;
      x_min.x = ((double *)xmin)[0];
      x_min.y = ((double *)xmin)[1];
      x_min.z = (dim == 3) ? ((double *)xmin)[2] : 0.0;
      get_coord_point((double *) xps_old + i * dim, ilocal, map->binhead->d_m, x_min, (double *)dx,
                      dim, 1);
      }
      break;
    case sizeof(long double): {
      ops_point<long double> x_min;
      x_min.x = ((long double *)xmin)[0];
      x_min.y = ((long double *)xmin)[1];
      x_min.z = (dim == 3) ? ((long double *)xmin)[2] : 0.0;
      get_coord_point((long double *) xps_old + i * dim, ilocal, map->binhead->d_m, x_min, (long double *)dx,
                      dim, 1);
      }
      break;
    }
  }

}

//MPI Version exists
//TODO: Stopped herein
void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map) {

  map->nParticles = ifirst + n_to_map;

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int dim = particle->block->dims;
  char *dx = map->dx;
  char xmin[320], xmax[320];
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (int i = ifirst; i < ifirst + n_to_map; i++) {

    int address;
    switch (particle->type_box) {
    case sizeof(float):
      address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                  (float *) xpos + dim * i);
      break;
    case sizeof(double):
      address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                  (double *) xpos + dim * i);
      break;
    case sizeof(long double):
      address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                  map->binhead->size, (long double *) xpos + dim * i);
       break;
    }

    if (address < 0) continue;
    bin2grid[i] = address;

    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);
    switch (particle->type_box) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, map->binhead->d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
      get_coord_point((double *) xold + dim * i, ilocal, map->binhead->d_m, (double *) xmin,
                      (double *)dx, dim, 1);
      break;
    case sizeof(long double):
      get_coord_point((long double *) xold + dim * i, ilocal, map->binhead->d_m,
                      (long double *) xmin, (long double *) dx, dim , 1);
      break;
    }

  }


}

//MPI Version exists
void _ops_particle_setup_map(ops_particle particle, ops_particle_mapping map) {

  if (particle->no_particles == 0) return;

  map->nParticles = particle->no_particles;

  /* Data initialization */
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *) map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  size_t binsize = 1;
  for (int i = 0; i < particle->block->dims; i++)
    binsize *= map->binhead->size[i];

  memset(binhead, -1, sizeof(int) * binsize);
  memset(bin2grid, -1, sizeof(int) * particle->no_particles);
  memset(bins, -1, sizeof(int) * particle->no_particles);

  //Get mapping
  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (size_t i = 0; i < particle->no_particles; i++) {

    int address;
    switch(particle->particle_pos_dat->type_size) {
    case sizeof(float):
       address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                   (float *) xpos + dim * i);
       break;
    case sizeof(double):
       address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                   (double *) xpos + dim * i);
       break;
    case sizeof(long double):
       address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                   map->binhead->size, (long double *) xpos + dim * i);
       break;
    }
    bin2grid[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);

    switch (particle->particle_pos_dat->type_size) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, map->binhead->d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
        get_coord_point((double *) xold + dim * i, ilocal, map->binhead->d_m, (double *) xmin,
                        (double *)dx, dim, 1);
        break;
    case sizeof(long double):
        get_coord_point((long double *) xold + dim * i, ilocal, map->binhead->d_m, (long double *) xmin,
                       (long double *)dx, dim, 1);
        break;
    }

  }

}


//MPI Version exists
void _ops_particle_map_from_exchange(ops_particle_mapping map, ops_particle particle,
                                     int ifirst, int ilast) {

  if ((size_t) ifirst > particle->no_particles || (size_t) ilast > particle->no_particles )
    throw OPSException(OPS_RUNTIME_ERROR, "Mapping particles are not actual particles");

  char xmin[320], xmax[320];
  char *dx = map->dx;
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = ifirst; i < ilast; i++) {

    int address;
    switch(particle->type_box) {
    case sizeof(float): {
      address = _ops_coord_to_bin(dim, (float *)xmin, (float *)xmax,
                                 (float *) dx, map->binhead->size,
                                 (float *)xpos + dim * i);
      if (!((BoundingBox<float> *) particle->box_block)->isCoordinateInBoundingBox((float *) xpos + i * dim)) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        ((int *)map->bin->data)[i] = - 1;
        particle->mark_deletion[i] = 1;
        continue;
      }
      } break;
    case sizeof(double): {
      address = _ops_coord_to_bin(dim, (double *)xmin, (double *)xmax,
                                 (double *) dx, map->binhead->size,
                                 (double *)xpos + dim * i);
      if (!((BoundingBox<double> *) particle->box_block)->isCoordinateInBoundingBox((double *) xpos + i * dim)) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        ((int *)map->bin->data)[i] = - 1;
        particle->mark_deletion[i] = 1;
        continue;
      }
      } break;
    case sizeof(long double): {
      address = _ops_coord_to_bin(dim, (long double *)xmin, (long double *)xmax,
                                 (long double *) dx, map->binhead->size,
                                 (long double *)xpos + dim * i);
      if (!((BoundingBox<long double> *) particle->box_block)->isCoordinateInBoundingBox((long double *) xpos + i * dim)) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        ((int *)map->bin->data)[i] = - 1;
        particle->mark_deletion[i] = 1;
        continue;
      }
      } break;
    }


    ((int *)map->bin->data)[i] = ((int *) map->binhead->data)[address];
    ((int *)map->binhead->data)[address] = i;
    ((int  *) map->parts_to_grid->data)[i] = address;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m, dim, ilocal);
    if (particle->particle_pos_dat->type_size == sizeof(float)) {
      get_coord_point((float *) xold + i * dim, ilocal, map->binhead->d_m, (float *)xmin, (float *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(double)) {
      get_coord_point((double *) xold + i * dim, ilocal, map->binhead->d_m, (double *)xmin, (double *) dx, dim, 1);
    }
    else if (particle->particle_pos_dat->type_size == sizeof(long double))
      get_coord_point((long double *) xold + i * dim, ilocal, map->binhead->d_m, (long double *)xmin,
                      (long double *) dx, dim, 1);
  }

  for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = true;
}



//TODO: Need to generate a version that splits the generation creation and adding halo for
//      the exchange.


//MPI Version exists
//TODO: Stopped herein
void _ops_particle_setup_map_virtual(ops_particle particle, ops_particle_mapping map) {

  if (particle->no_virtual == 0) return;

  map->nParticles += particle->no_virtual;

  //Initialize structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  char *dx = map->dx;
  char xmin[320], xmax[320];
  int dim = particle->block->dims;
  switch (particle->particle_pos_dat->type_size) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                             (BoundingBox<double> *) particle->box_block, (double *)dx,
                              map->binhead->d_m, map->binhead->d_p, dim);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                             map->binhead->d_m, map->binhead->d_p, dim);
    break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (size_t i = particle->no_particles;
           i < particle->no_particles + particle->no_virtual; i++) {
    bins[i] = -1;
    bin2grid[i] = -1;

    int address;
    switch(particle->particle_pos_dat->type_size) {
    case sizeof(float):
       address = _ops_coord_to_bin(dim, (float *) xmin, (float *) xmax, (float *)dx, map->binhead->size,
                                   (float *) xpos + dim * i);
       break;
    case sizeof(double):
       address = _ops_coord_to_bin(dim, (double *) xmin, (double *) xmax, (double *)dx, map->binhead->size,
                                   (double *) xpos + dim * i);
       break;
    case sizeof(long double):
       address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *) xmax, (long double *)dx,
                                   map->binhead->size, (long double *) xpos + dim * i);
       break;
    }


    if (address < 0) continue;
    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);
    switch (particle->particle_pos_dat->type_size) {
    case sizeof(float):
      get_coord_point((float *) xold + dim * i, ilocal, map->binhead->d_m, (float *) xmin,
                      (float *)dx, dim, 1);
      break;
    case sizeof(double):
        get_coord_point((double *) xold + dim * i, ilocal, map->binhead->d_m, (double *) xmin,
                        (double *)dx, dim, 1);
        break;
    case sizeof(long double):
        get_coord_point((long double *) xold + dim * i, ilocal, map->binhead->d_m, (long double *) xmin,
                       (long double *)dx, dim, 1);
        break;
    }

  }

}


/**
 * Function erases and build particle lists directly
 */

//TODO: We need to add herein what it builds (local or all)

//MPI Version exists

//TODO:Stopped herein
int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {

  //TODO: Do we need this to be false by default???
  map->decide = false;

  /* Checking if mapping declaired particles < total */
  int dim = particle->block->dims;

  /*Accessing local particles only */
  /* 1. Check for deletion */
  /* 2. Check if need to move */

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];


 /* Set particles for mapping */
  int nmapping
    = (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                          particle->no_particles + particle->no_virtual;

  char *dx = map->dx;
  char xmin[320], xmax[320];

  switch(particle->type_box) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                            d_m, d_p, particle->block->dims);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block, (long double *)dx,
                            d_m, d_p, particle->block->dims);
    break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data; //Assumed here in that
  //Nparticles in list is build properly

  int exch_limits[2 * OPS_MAX_DIM];
  int rmv_limits[2 * OPS_MAX_DIM];

  for (int i =  0; i < dim; i++) {
    exch_limits[2 *i] = (d_m[i] < 0) ? -d_m[i] : -size[i];
    exch_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - 2 * d_p[i]: 2 * size[i];
    rmv_limits[2 *i ] = 0;
    rmv_limits[2 * i + 1] = size[i] - d_p[i] + d_m[i] - 1;
  }


  size_t nactual = particle->no_particles;

  //TODO: For GPU: Split into two parts
  for (size_t i = 0; i < particle->no_particles; i++) {
    int flag;

    switch(particle->type_box) {
    case sizeof(float):
      flag = local_decide_rebuild((float *)xpos + i * dim, (float *)xold + i * dim,
                                  ((float *)dx)[0], dim);
      break;
    case sizeof(double):
     flag = local_decide_rebuild((double *)xpos + i * dim, (double *)xold + i * dim,
                                 ((double *)dx)[0], dim);
      break;
    case sizeof(long double):
      flag = local_decide_rebuild((long double *)xpos + i * dim, (long double *)xold + i * dim,
                                 ((long double *)dx)[0], dim);
      break;
    }

    if (flag) {

      int address = ((int *)map->parts_to_grid->data)[i];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, (int *)map->binhead->data,
                                 (int *) map->bin->data);

      //Actual particle perform two checks

        //TODO-1: Check for possible removal
      int del_flag;
      switch(particle->type_box) {
      case sizeof(float):
        del_flag=  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (float *)xpos + i * dim,
                                                    (BoundingBox<float> *) particle->box_block);
        break;
      case sizeof(double):
        del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (double *)xpos + i * dim,
                                                    (BoundingBox<double> *) particle->box_block);
        break;
      case sizeof(long double):
        del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (long double *)xpos + i * dim,
                                                    (BoundingBox<long double> *) particle->box_block);
        break;
      }

      if (del_flag) {
        ((int *)map->parts_to_grid->data)[i] = -1;
        map->decide = true;//TODO: Think if it is needed
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
      }

      //Start-mapping
      switch (particle->type_box) {
      case sizeof(float):
         address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                     map->binhead->size, (float *) xpos + dim * i);
        break;
      case sizeof(double):
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
        break;
      case sizeof(long double):
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *) dx,
                                    map->binhead->size, (long double *) xpos + dim * i);
        break;
      }

      if (address < 0) map->decide = true;

      ((int *) map->bin->data)[i] = ((int *) map->binhead->data)[address];
      ((int *) map->binhead->data)[address] = i;

      ((int *)map->parts_to_grid->data)[i] = address;

      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);
      if (particle->particle_pos_dat->type_size == sizeof(float)) {
        get_coord_point((float *) xold + i * dim, ilocal, d_m, (float *) xmin, (float *) dx,
                        dim, 1);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(double)) {
        get_coord_point((double *) xold + i * dim, ilocal, d_m, (double *) xmin, (double *) dx,
                        dim, 1);
      }
      else if (particle->particle_pos_dat->type_size == sizeof(long double))
        get_coord_point((long double *) xold + i * dim, ilocal, d_m, (long double *) xmin, (long double *) dx,
                        dim, 1);
            //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

      if (!map->decide) map->decide = flag_build;

    }
  }

  nactual = particle->no_particles;

  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

  for (int i = ifirst; i < nmapping; i++) {

    int flag;
    switch(particle->type_box) {
    case sizeof(float):
      flag = local_decide_rebuild((float *)xpos + i * dim, (float *)xold + i * dim,
                                  ((float *)dx)[0], dim);
      break;
    case sizeof(double):
     flag = local_decide_rebuild((double *)xpos + i * dim, (double *)xold + i * dim,
                                 ((double *)dx)[0], dim);
      break;
    case sizeof(long double):
      flag = local_decide_rebuild((long double *)xpos + i * dim, (long double *)xold + i * dim,
                                 ((long double *)dx)[0], dim);
      break;
    }

    if (flag) { //To-Rebuild for virtual particle

      int address = ((int *)map->parts_to_grid->data)[i];
      _remove_particle_from_bins(address, i, (int *) map->binhead->data,
                                 (int *) map->bin->data);
      ((int *)map->parts_to_grid->data)[i] = -1;

      //Map particle
      switch (particle->type_box) {
      case sizeof(float):
         address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                     map->binhead->size, (float *) xpos + dim * i);
        break;
      case sizeof(double):
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
        break;
      case sizeof(long double):
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *) dx,
                                    map->binhead->size, (long double *) xpos + dim * i);
        break;
      }

      if (address < 0) {
        map->decide = true; continue;}

      //Virtual still projected'


      get_local_point(address, map->binhead->size, map->binhead->d_m,
                      dim, ilocal);

      int flag_in;
      switch(particle->type_box) {
      case sizeof(float):
        flag_in= virtual_within(ilocal, rmv_limits, (float *)xpos + dim * i,
                                (BoundingBox<float> *)particle->box_block, dim);
        break;
      case sizeof(double):
        flag_in= virtual_within(ilocal, rmv_limits, (double *)xpos + dim * i,
                                    (BoundingBox<double> *)particle->box_block, dim);
        break;
      case sizeof(long double):
        flag_in= virtual_within(ilocal, rmv_limits, (long double *)xpos + dim * i,
                               (BoundingBox<long double> *)particle->box_block, dim);
        break;
      }

      if (flag_in) { //Virtual become actual

        nactual++; //Increase local.
        nvirtual_act--;
        //swap->particle data

        ((int *) map->parts_to_grid->data)[i] = address;
        ((int *) map->bin->data)[i] = ((int *) map->binhead->data)[address];
        ((int *) map->binhead->data)[address] = i;


        _ops_particle_swap_data(particle->particle_pos_dat->data, i, nactual - 1,
                                particle->particle_pos_dat->elem_size);

        //compute new positions and swap: TODO

        if (particle->particle_envelope != nullptr) {
          _ops_particle_swap_data(particle->particle_envelope->data, i, nactual - 1,
                                  particle->particle_envelope->elem_size);
        }

        if (particle->ids != nullptr)
          _ops_particle_swap_data(particle->ids->data, i, nactual - 1,
                                  particle->ids->elem_size);

        for (int idat = 0; idat < particle->particle_dat_index; idat++) {
          ops_dat dat = particle->particle_dat[idat];
          _ops_particle_swap_data(dat->data, i, nactual - 1, dat->elem_size);
        }

        particle->mark_deletion[i] = 0;
        _ops_particle_swap_data((char *)particle->mark_deletion, i, nactual - 1,
                                sizeof(int));

        //swap->map with first virtual

        //set flag map to true

        _ops_particle_swap_mapping_data(map, i, nactual - 1);
        _ops_particle_swap_data(map->pos_old->data, i, nactual -1, \
                                map->pos_old->elem_size);
        map->decide = true;

      }

      if (!map->decide)  {
        ((int *)map->parts_to_grid->data)[i] = address;
        ((int *)map->bin->data)[i] = ((int *) map->binhead->data)[address];
        ((int *)map->binhead->data)[address] = i;


        if (particle->particle_pos_dat->type_size == sizeof(float))
          get_coord_point((float *) xold + i * dim, ilocal, d_m, (float *)xmin, (float *) dx,
                          dim, 1);
        else if (particle->particle_pos_dat->type_size == sizeof(double))
          get_coord_point((double *) xold + i * dim, ilocal, d_m, (double *) xmin, (double *) dx,
                           dim, 1);
        else if (particle->particle_pos_dat->type_size == sizeof(long double))
          get_coord_point((long double *) xold + i * dim, ilocal, d_m, (long double *) xmin, (long double *) dx,
                           dim, 1);
      }

    }

  }



  //Perform action for virtual particles
  if (map->decide)
   for (int ih = 0; ih < particle->nhistories; ih++)
     particle->histories[ih]->flag_update = true;


  //TODO: Add this info only for the mapping list
  if (nactual > particle->no_particles)
    particle->no_particles = nactual;

  particle->no_virtual = nvirtual_act;

  return (int) map->decide;
}


//MPI: Version exists
int _ops_particle_decide_build_only_local_uniform(ops_particle_mapping map,
                                                  ops_particle         particle) {
  map->decide = false; //TODO: Need to rmv-we set directly the flag
  map->flag_history = 0;
 // history_decide = 0;
  int dim = particle->block->dims;

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];

  int nmapping = particle->no_particles;
  char *xmin[320], *xmax[320];
  char *dx = map->dx;
  switch(particle->type_box) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             d_m, d_p, particle->block->dims);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
     break;
  case sizeof(long double):
     _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                             (BoundingBox<long double> *) particle->box_block,
                             (long double *)dx,
                             d_m, d_p, particle->block->dims);
     break;
  }

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  int rmv_limits[2 * OPS_MAX_DIM];
  int exch_limits[2 * OPS_MAX_DIM];
  //TODO: Need adaptation
  for (int i =  0; i < dim; i++) {
    exch_limits[2 * i] = d_m[i] < 0 ?  -d_m[i] : -size[i];
    exch_limits[2 * i + 1] = d_p[i] > 0? size[i] + d_m[i] -  2 * d_p[i] : 2 * size[i];
    rmv_limits[2 * i] = 0;
    rmv_limits[2 * i + 1] = size[i] - d_p[i] + d_m[i] - 1;
  }

  for (int i = 0; i < nmapping; i++) {
    int flag;
    switch(particle->type_box) {
    case sizeof(float):
      flag = local_decide_rebuild((float *)xpos + i * dim, (float *)xold + i * dim,
                                  ((float *)dx)[0], dim);
      break;
    case sizeof(double):
     flag = local_decide_rebuild((double *)xpos + i * dim, (double *)xold + i * dim,
                                 ((double *)dx)[0], dim);
      break;
    case sizeof(long double):
      flag = local_decide_rebuild((long double *)xpos + i * dim, (long double *)xold + i * dim,
                                 ((long double *)dx)[0], dim);
      break;
    }

    if (flag) {
      int address = ((int *) map->parts_to_grid->data)[i];
      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);


      //Remove particle from bin-box
      _remove_particle_from_bins(address, i, (int *) map->binhead->data,
                                (int *) map->bin->data);

      map->flag_history = 1;

      //Check for deletion
      int del_flag;
      switch(particle->type_box) {
      case sizeof(float):
        del_flag=  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (float *)xpos + i * dim,
                                                    (BoundingBox<float> *) particle->box_block);
        break;
      case sizeof(double):
        del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (double *)xpos + i * dim,
                                                    (BoundingBox<double> *) particle->box_block);
        break;
      case sizeof(long double):
        del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                    (long double *)xpos + i * dim,
                                                    (BoundingBox<long double> *) particle->box_block);
        break;
      }

      if (del_flag) {
       ((int *) map->parts_to_grid->data)[i] = -1;
       map->decide = true; //Enforce mapping;
       //nactual--;
       particle->mark_deletion[i] = 1;
       continue;
      }


      //Update map of actual particle
      switch (particle->type_box) {
      case sizeof(float):
         address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                     map->binhead->size, (float *) xpos + dim * i);
        break;
      case sizeof(double):
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *) dx,
                                    map->binhead->size, (double *) xpos + dim * i);
        break;
      case sizeof(long double):
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *) dx,
                                    map->binhead->size, (long double *) xpos + dim * i);
        break;
      }

      ((int *) map->bin->data)[i] = ((int *) map->binhead->data)[address];
      ((int *) map->binhead->data)[address] = i;
      ((int *) map->parts_to_grid->data)[i] = address;

      //Checking for particle moving in or out of exchange zone
      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);
      switch(particle->type_box) {
      case sizeof(float):
        get_coord_point((float *) xold + i * dim, ilocal_new, d_m, (float *)xmin,
                        (float *) dx, dim, 1);
        break;
      case sizeof(double):
        get_coord_point((double *) xold + i * dim, ilocal_new, d_m, (double *)xmin,
                        (double *) dx, dim, 1);
        break;
      case sizeof(long double):
        get_coord_point((long double *) xold + i * dim, ilocal_new, d_m, (long double *) xmin,
                        (long double *) dx,  dim, 1);
        break;
      }

      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

      if (!map->decide) map->decide = flag_build;
//      if (!history_decide ) history_decide = (int) flag_build;

    }
  }

  if (map->flag_history)
    for (int i = 0; i < particle->nhistories; i++)
      if (!particle->histories[i]->flag_update)
        particle->histories[i]->flag_update = true;

  return (int) map->decide;
}


//MPI Version exists
//TODO:
void _ops_particle_remap_virtual(ops_particle_mapping map, ops_particle particle,
                                 int istart, int ilast) {
  //Sanity checks
  if ((size_t) istart < particle->no_particles)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function for mapping virtual "
                                             "called for actual particles");

  if ((size_t) ilast > particle->no_particles + particle->no_virtual)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function called for non-existing "
                                             " particle");

  if (map->mapping_type != OPS_WITH_VIRTUAL) return;

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  char *dx = map->dx;
  char xmin[320], xmax[320];
  switch(particle->type_box) {
  case sizeof(float):
    _ops_points_map_min_max((float *) xmin, (float *) xmax,
                            (BoundingBox<float> *) particle->box_block, (float *)dx,
                             d_m, d_p, particle->block->dims);
    break;
  case sizeof(double):
    _ops_points_map_min_max((double *) xmin, (double *) xmax,
                            (BoundingBox<double> *) particle->box_block, (double *)dx,
                            d_m, d_p, particle->block->dims);
    break;
  case sizeof(long double):
    _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                            (BoundingBox<long double> *) particle->box_block,
                            (long double *)dx,
                            d_m, d_p, particle->block->dims);
    break;
  }

  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int dim = particle->block->dims;

  char *xpos = particle->particle_pos_dat->data;
  char *xold = map->pos_old->data;

  for (int i = istart; i < ilast; i++) {
    int flag;

    switch(particle->type_box) {
    case sizeof(float):
      flag = local_decide_rebuild((float *) xpos + i * dim, (float *) xold + i * dim,
                                  ((float *)dx)[0], dim);
      break;
    case sizeof(double):
      flag = local_decide_rebuild((double *) xpos + i * dim, (double *) xold + i * dim,
                                  ((double *)dx)[0], dim);
      break;
    case sizeof(long double):
      flag = local_decide_rebuild((long double *) xpos + i * dim, (long double *) xold + i * dim,
                                  ((long double *)dx)[0], dim);
      break;
    }


    if (flag) {
      int address = ((int *) map->parts_to_grid->data)[i];
      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);
      _remove_particle_from_bins(address, i, (int *) map->binhead->data,
                                 (int *) map->bin->data);
      switch(particle->type_box) {
      case sizeof(float):
        address = _ops_coord_to_bin(dim, (float *) xmin, (float *)xmax, (float *) dx,
                                    size, (float *) xpos + i * dim);
        break;
      case sizeof(double):
        address = _ops_coord_to_bin(dim, (double *) xmin, (double *)xmax, (double *)dx,
                                    size, (double *) xpos + i * dim);
        break;
      case sizeof(long double):
        address = _ops_coord_to_bin(dim, (long double *) xmin, (long double *)xmax, (long double *)dx,
                                    size, (long double *) xpos + i * dim);
        break;
      }


      ((int *) map->bin->data)[i] = ((int *) map->binhead->data)[address];
      ((int *) map->binhead->data)[address] = i;
      ((int *) map->parts_to_grid->data)[i] = address;

      switch(particle->type_box) {
      case sizeof(float):
        get_coord_point((float *) xold + i * dim, ilocal_new, d_m, (float *)xmin,
                        (float *) dx, dim, 1);
        break;
      case sizeof(double):
         get_coord_point((double *) xold + i * dim, ilocal_new, d_m, (double *) xmin,
                         (double *) dx, dim, 1);
         break;
      case sizeof(long double):
         get_coord_point((long double *) xold + i * dim, ilocal_new, d_m,
                         (long double *)xmin, (long double *) dx, dim, 1);
        break;
      }
    }
  }
}


void _ops_particle_exchange(ops_particle particle) {

}

void _ops_particle_build_border(ops_particle particle) {

}

void _ops_particle_build_border_maps(ops_particle particle) { }

void _ops_particle_border_dats_with_maps(ops_particle particle, ops_dat *dats,
                                         int ndats, ops_neighbor_history *history,
                                         int nhistories) { }

void _ops_particle_forward_dats_with_maps(ops_particle particle, ops_dat *dats,
                                          int ndats) { }

void _ops_particle_forward_intra_maps(ops_particle particle) { }
void _ops_particle_exchange_map_update(ops_particle particle) { }

void _ops_particle_border_dats(ops_particle particle, ops_dat *dats,
                               int ndats, ops_neighbor_history *history,
                               int nhistories) { }

void _ops_particle_forward_dats(ops_particle particle, ops_dat *dats, int ndats) { }

void _ops_particle_reverse_dats(ops_particle particle, ops_dat *dats, int ndats,
                                ops_access access) { }


/*---------------------------------------------------------------------------------------*/
/*  Particle Halos functions                                                             */
/*---------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------*/
/*              Define exchange zone for a border type     halo                          *
 *---------------------------------------------------------------------------------------*/

void _ops_particle_set_exchange_border_zone(OPS_instance *instance,
                                            ops_particle_halo halo) {
  /* Vrf that bounding boxes are defined  */

  if (_ops_box_get_ownership(halo->particle_from->box_block, halo->particle_from->type_box) ||
      _ops_box_get_ownership(halo->particle_to->box_block, halo->particle_to->type_box)) {
    throw OPSException(OPS_RUNTIME_ERROR, "Particle Halo exchange must be set after"
                                          " block boxes are set.");
  }

  int dim = halo->particle_from->block->dims;
  halo->sendBox = _ops_set_exchange_border_box(halo->particle_from->box_block,
                                               halo->particle_to->box_block,
                                               dim, halo->translate, halo->dir_from,
                                               halo->dir_to, halo->particle_from->map_list[0]->dx,
                                               halo->particle_from->type_box);


  if (halo->sendBox == nullptr) {
    for (int i = 0; i < 2 * dim; i++)
      halo->isend[i] = 0;
    return;
  }
  else {

    char xmin[320], xmax[320];
    char *dx = halo->particle_from->map_list[0]->dx;
    int *d_m = halo->particle_from->map_list[0]->binhead->d_m;
    int *d_p = halo->particle_from->map_list[0]->binhead->d_p;

    switch(halo->particle_from->type_box) {
    case sizeof(float):
      _ops_points_map_min_max((float *) xmin, (float *) xmax,
                              (BoundingBox<float> *) halo->particle_from->box_block,
                              (float *)dx,  d_m, d_p, halo->particle_from->block->dims);
      break;
    case sizeof(double):
      _ops_points_map_min_max((double *) xmin, (double *) xmax,
                              (BoundingBox<double> *) halo->particle_from->box_block,
                              (double *)dx, d_m, d_p, halo->particle_from->block->dims);
       break;
    case sizeof(long double):
       _ops_points_map_min_max((long double *) xmin, (long double *) xmax,
                               (BoundingBox<long double> *) halo->particle_from->box_block,
                               (long double *)dx, d_m, d_p, halo->particle_from->block->dims);
       break;
    }





    //Finding grid points
    int size[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++)
      size[i] = (halo->particle_from->map_list[0] != nullptr) ?
          halo->particle_from->map_list[0]->binhead->size[i] : 1;

    _ops_particle_find_halo_cells(halo->isend, halo->sendBox, xmin, xmax, size, dx, dim,
                                  halo->particle_from->type_box);
  }
}

void  _ops_particle_set_exchange_zone(OPS_instance *instance,
                                      ops_particle_halo halo) {


  if (_ops_box_get_ownership(halo->particle_from->box_block, halo->particle_from->type_box) ||
      _ops_box_get_ownership(halo->particle_to->box_block, halo->particle_to->type_box)) {
    throw OPSException(OPS_RUNTIME_ERROR, "Particle Halo exchange must be set after"
                                          " block boxes are set.");
  }

  int dim = halo->particle_from->block->dims;

  halo->sendBox = _ops_create_exchange_block(halo->particle_from->box_block,
                                             halo->particle_to->box_block, dim,
                                             halo->translate, halo->dir_to,
                                             halo->dir_from,
                                             halo->particle_from->type_box);


  //TODO:Add bites for the particle_halo
}

void _ops_particle_verify_exchange_zone(OPS_instance *instance,
                                        ops_particle_halo halo) {

  //Set local Box
  switch (halo->particle_to->type_box) {
  case sizeof(float): {
    BoundingBox<float> *box = (BoundingBox<float> *) halo->sendBox;
    box->setBoundingBoxLocalBound(box->getGlobalMin(), box->getGlobalMax());
  } break;
  case sizeof(double): {
    BoundingBox<double> *box = (BoundingBox<double> *) halo->sendBox;
    box->setBoundingBoxLocalBound(box->getGlobalMin(), box->getGlobalMax());
  } break;
  case sizeof(long double): {
    BoundingBox<long double> *box = (BoundingBox<long double> *) halo->sendBox;
    box->setBoundingBoxLocalBound(box->getGlobalMin(), box->getGlobalMax());
  } break;
  }

}

//TODO:
void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (halo->sendBox == nullptr)
     _ops_particle_set_exchange_zone(instance, halo);
    else
      _ops_particle_verify_exchange_zone(instance, halo);
/*
    if (halo->sendBox != nullptr) {
      switch(halo->particle_from->type_box) {
      case sizeof(float):
        printf("Particle halo %d: SendingBox = [%e %e]x[%e %e]x[%e %e]\n", ihalo,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMin().x,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMax().x,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMin().y,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMax().y,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMin().z,
               ((BoundingBox<float> *)halo->sendBox)->getLocalMax().z);
        break;
      case sizeof(double):
         printf("Particle halo %d: SendingBox = [%e %e]x[%e %e]x[%e %e]\n", ihalo,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().x,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().x,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().y,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().y,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().z,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().z);
        break;
      case sizeof(long double):
        printf("Particle halo %d: SendingBox = [%Le %Le]x[%Le %Le]x[%Le %Le]\n", ihalo,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().x,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().x,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().y,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().y,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().z,
               ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().z);
        break;
      }
    }
    */
  }
}


void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  /* Loop over all halos to define exchange zone */
  for (int ihalos = 0; ihalos < halo_grp->nhalos; ihalos++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalos];
    _ops_particle_set_exchange_border_zone(instance, halo);

    /*
    if (halo->sendBox != nullptr) {
      switch(halo->particle_from->type_box) {
       case sizeof(float):
          printf("Sending Box = [%e %e]x[%e %e] x[%e %e] SendingPoints = [%d %d]x[%d %d]x[%d %d]\n",
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMin().x,
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMax().x,
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMin().y,
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMax().y,
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMin().z,
                 ((BoundingBox<float> *)halo->sendBox)->getLocalMax().z,
                 halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3],
                 halo->isend[4], halo->isend[5]);
       break;
       case sizeof(double):
         printf("Sending Box = [%e %e]x[%e %e] x[%e %e] SendingPoints = [%d %d]x[%d %d]x[%d %d]\n",
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().x,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().x,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().y,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().y,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMin().z,
                ((BoundingBox<double> *)halo->sendBox)->getLocalMax().z,
                halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3],
                halo->isend[4], halo->isend[5]);
       break;
       case sizeof(long double):
         printf("Sending Box = [%Le %Le]x[%Le %Le] x[%Le %Le] SendingPoints = [%d %d]x[%d %d]x[%d %d]\n",
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().x,
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().x,
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().y,
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().y,
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMin().z,
                ((BoundingBox<long double> *)halo->sendBox)->getLocalMax().z,
                halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3],
                halo->isend[4], halo->isend[5]);
       break;
      }
    }
    */
  }
}




  //TODO: Add sanity checks by allocation practically we are using particle_halos to shift only
  //data

  //Sanity 1: Same number of halo_particles
  //Sanity 2: Check for particle_to agrees to particle_from
  //Sanity 3: Check for particle_from agrees with particle_from
  //Sanity 4: Orient from is the same or who cares
  //Sanity 5: Oreint to is the same


void _ops_particle_setup_default_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp)
{

}

//TODO: Add the mark deletion to particle list:
//TODO: Need activation somewhere

//TODO:
void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                        ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    char *xlocal = particle_from->particle_pos_dat->data;
    char *box = halo->sendBox;

    if (box == nullptr) continue;

    info->nsend = 0;

    for (size_t  i = 0; i < particle_from->no_particles; i++) {
      if (particle_from->mark_deletion[i] != 1) continue;
      bool isin;
      switch(halo->particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *)xlocal + dim * i);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *)box)->isCoordinateInBoundingBox((double *)xlocal + dim * i);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *)box)->isCoordinateInBoundingBox((long double *)xlocal + dim * i);
      }
      info->nsend += ((int) isin);
    }

    //Part II: Reallocate sending particles
    if (info->nsend > info->nmax) {
      info->sendlist = (int *) ops_realloc(info->sendlist, sizeof(int) * (info->nsend + 100));
      info->nmax += info->nsend + 100;
    }

    //Part III: Find sending particles
    int nsend = 0;
    for (size_t i = 0; i < particle_from->no_particles; i++) {

      if (particle_from->mark_deletion[i] != 1) continue;

      bool isin;
      switch(halo->particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *)xlocal + dim * i);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *)box)->isCoordinateInBoundingBox((double *)xlocal + dim * i);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *)box)->isCoordinateInBoundingBox((long double *)xlocal + dim * i);
      }

      if (isin) {

        info->sendlist[nsend] = i;
        particle_from->mark_deletion[i] = 2;
        nsend++;
      }
    }

    //Part IV: Re-allocate buffer
    int total_bites = info->nmax * halo->nbites;
    if (total_bites > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer = (char *) ops_realloc(instance->ops_halo_buffer,
                                                       (info->nsend + OPS_MAX_PART)
                                                       * halo->nbites);
      instance->ops_halo_buffer_size = (info->nsend + OPS_MAX_PART)
                                     * halo->nbites;
    }

    //Part V: Copy to buffer
    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 1, &ntot_bites);

    //Part VI: Re-allocate structures
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    info->firstrecv = particle_to->no_particles;

    particle_to->no_particles += nrecv;
    if (particle_to->no_particles > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles);

    //Part VIII: Unpack buffers
    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 1,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate,  &ntot_bites); //TODO:
  }
}

//TODO: Re-organize as to receive data and exchange properly
void _ops_particle_halo_exchange_transfer_map(OPS_instance *instance,
                                              ops_particle_halo_group halo_grp) {

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {

    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    char *xlocal = particle_from->particle_pos_dat->data;
    char *box = particle_from->box_block;
    if (box == nullptr) continue;

    info->nsend = 0;

    for (size_t i = 0; i < particle_from->no_particles; i++) {
      if (particle_from->mark_deletion[i] != 1) continue;
      bool isin;
      switch(particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *) xlocal + i * dim);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *) box)->isCoordinateInBoundingBox((double *) xlocal + i * dim);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *) box)->isCoordinateInBoundingBox((long double *) xlocal + i * dim);
        break;
      }

      info->nsend += ((int) isin);
    }

    if (info->nsend > info->nmax) {
      info->sendlist = (int *) ops_realloc(info->sendlist, sizeof(int) * (info->nsend + 100));
      info->nmax += info->nsend + 100;
    }

    //Identify exchange particles
    int nsend = 0;
    for (size_t i = 0; i < particle_from->no_particles; i++) {
      bool isin;

      switch(particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *) xlocal + i * dim);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *) box)->isCoordinateInBoundingBox((double *) xlocal + i * dim);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *) box)->isCoordinateInBoundingBox((long double *) xlocal + i * dim);
        break;
      }

      if (isin) {
        if (particle_from->mark_deletion[i] != 1) continue;
        info->sendlist[nsend] = i;
        particle_from->mark_deletion[i] = 2;
        nsend++;
      }
    }

    int total_bites = info->nmax * halo->nbites;
    if (total_bites > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer = (char *) ops_realloc(instance->ops_halo_buffer,
                                                       (info->nsend + OPS_MAX_PART)
                                                       * halo->nbites);
      instance->ops_halo_buffer_size = (info->nsend + OPS_MAX_PART)
                                     * halo->nbites;
    }

    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 1, &ntot_bites);

    //Part III: Unpack data
    /* Unpack data */
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    info->firstrecv = particle_to->no_particles;

    //Re-allocate data if necessary
    particle_to->no_particles += nrecv;
    if (particle_to->no_particles > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles);

    //Unpacking data
    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 1,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites);

    for (int imap = 0; imap <particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      _ops_particle_map_from_exchange(map, particle_to, info->firstrecv,
                                      map->nParticles);
    }
  }


}

/*-------------------------------------------------------------------------*/
/* Sequential border halo transfer                                         */
/*-------------------------------------------------------------------------*/

void _ops_particle_halo_border_transfer_map(OPS_instance *instance,
                                           ops_particle_halo_group halo_grp) {

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle from = halo->particle_from;

    int *binhead = (int *)from->map_list[0]->binhead->data;
    int *bins = (int *)from->map_list[0]->bin->data;
    ops_dat binheads = from->map_list[0]->binhead;

    info->nsend = 0;
    for (int k = halo->isend[4]; k < halo->isend[5]; k++) {
      for (int j = halo->isend[2]; j < halo->isend[3]; j++) {
        for (int i = halo->isend[0]; i < halo->isend[1]; i++) {
          int address = i + j * binheads->size[0] + k * binheads->size[1] * binheads->size[0];
          int ipart = binhead[address];
          while (ipart != -1 ) {
            info->nsend++;
            ipart = bins[ipart];
          }
        }
      }
    }

    if (info->nsend >= info->nmax) {
      info->sendlist = (int *) ops_realloc(info->sendlist, (info->nsend + 100) * sizeof(int));
      info->nmax = info->nsend + 100;
    }

    int nsend = 0;
    //Part III: Create sendlist structure
    for (int k = halo->isend[4]; k < halo->isend[5]; k++) {
      for (int j = halo->isend[2]; j < halo->isend[3]; j++) {
        for (int i = halo->isend[0]; i < halo->isend[1]; i++) {
          int address = i + j * binheads->size[0] + k * binheads->size[1] * binheads->size[0];
          int ipart = binhead[address];
          while (ipart != - 1) {
            info->sendlist[nsend] = ipart;
            nsend++;
            ipart = bins[ipart];
          }
        }
      }
    }

    //Part IV: Reallocate buffer
    int total_bites = info->nmax * halo->nbites;
    if (total_bites > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer = (char *) ops_realloc(instance->ops_halo_buffer,
                                                       (info->nsend + OPS_MAX_PART)
                                                       * halo->nbites);
      instance->ops_halo_buffer_size = (info->nsend + OPS_MAX_PART)
                                     * halo->nbites;
    }

    /* Part V: Pack data */
    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 1, &ntot_bites);

    /* Part VI: Reallocate data if necessary */
    ops_particle particle_to = halo->particle_to;

    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;
    particle_to->no_virtual += info->nrecv;

    if (particle_to->no_particles  + particle_to->no_virtual > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles
                                             + particle_to->no_virtual);

    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 1,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites);

    for (int imap = 0; imap < particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      _ops_particle_map_from_exchange(map, particle_to, info->firstrecv,
                                      info->firstrecv + info->nsend);
    }

  }
}

void _ops_particle_halo_border_transfer(OPS_instance *instance,
                                        ops_particle_halo_group halo_grp) {

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    int imin = 0;
    int imax = (halo_grp->with_virtual != OPS_NO_VIRTUAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;

    char *box = halo->sendBox;
    info->nsend = 0;


    //Part I: Find the number of particles with the structure
    char *xlocal = particle_from->particle_pos_dat->data;
    for (int i = imin; i < imax; i++) {

      bool isin;
      switch(particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *) xlocal + i * dim);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *) box)->isCoordinateInBoundingBox((double *) xlocal + i * dim);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *) box)->isCoordinateInBoundingBox((long double *) xlocal + i * dim);
        break;
      }

      info->nsend += ((int) isin);
    }

    if (info->nsend >= info->nmax)  {
      info->sendlist = (int *) ops_realloc(info->sendlist, (info->nmax + 100) * sizeof(int));
      info->nmax = info->nsend + 100;
    }

    //Part II: Find particles to exchange
    int nsend = 0;
    for (int i = imin; i < imax; i++) {

      bool isin;
      switch(particle_from->type_box) {
      case sizeof(float):
        isin = ((BoundingBox<float> *)box)->isCoordinateInBoundingBox((float *) xlocal + i * dim);
        break;
      case sizeof(double):
        isin = ((BoundingBox<double> *) box)->isCoordinateInBoundingBox((double *) xlocal + i * dim);
        break;
      case sizeof(long double):
        isin = ((BoundingBox<long double> *) box)->isCoordinateInBoundingBox((long double *) xlocal + i * dim);
        break;
      }
      if (isin) {
        info->sendlist[nsend] = i;
        nsend++;
      }
    }

    //Part III: Reallocate sending/receiving structures
    int total_bites = info->nmax * halo->nbites;
    if (total_bites > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer = (char *) ops_realloc(instance->ops_halo_buffer,
                                                       (info->nsend + OPS_MAX_PART)
                                                       * halo->nbites);
      instance->ops_halo_buffer_size = (info->nsend + OPS_MAX_PART)
                                     * halo->nbites;
    }

    //Part IV: Pack data
    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 1, &ntot_bites);

    //Part V: Unpack data
    ops_particle particle_to = halo->particle_to;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;

    particle_to->no_virtual += info->nrecv;

    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles
                                           + particle_to->no_virtual);
    }

    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 1,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites); //TODO: Need to hanlde translate
  }
}

void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp) {
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  ops_particle_halo_group main_halo_grp =
      (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    /* Packing data */
    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff=(char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }

    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 0 ,&ntot_bites);

    /* Unpacking data */
    ntot_bites = 0;
    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 0,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites);
  }

}

void _ops_particle_halo_forward_map(OPS_instance   *instance,
                                    ops_particle_halo_group halo_grp) {
  /* Get buffer */
  char *buff = instance->ops_halo_buffer;

  ops_particle_halo_group main_halo_grp =
      (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp : halo_grp->halo_master;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff = (char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }


    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 0 ,&ntot_bites);

    /* Unpacking data  */
    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 0,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites);

    //Remapping particles to grid
    ops_particle particle_to = halo_main->particle_to;
    for (int imap = 0; imap < particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      _ops_particle_map_from_exchange(map, particle_to, info->firstrecv,
                                      info->firstrecv + info->nsend);
    }

  }
}

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp,
                                         ops_access access) {

  ops_particle_halo_group main_halo_grp
    = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];



    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    int nsend = info->nrecv;
    int nbites_tot = nsend * halo->nbites;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      instance->ops_halo_buffer =(char *)ops_realloc( instance->ops_halo_buffer,
                                                      instance->ops_halo_buffer_size);

    }




    int ntot_bites = 0;
    _ops_particle_halo_reverse_copy_tobuf(instance->ops_halo_buffer, halo->dat, halo->nhalos,
                                          info, &ntot_bites);

    ntot_bites = 0;
    _ops_particle_halo_reverse_copy_from_buff(instance->ops_halo_buffer, halo->dat, halo->nhalos,
                                              info, halo_main->dir_from, halo_main->dir_to,
                                              &ntot_bites, access);
  }
}

//MPI Version exists
void  _ops_particle_init_map(ops_particle_mapping map) {

  int size = 1;
  for (int i = 0; i < map->binhead->block->dims; i++)
    size *= map->binhead->size[i];

  for (int i = 0; i <size; i++)
    ((int *)map->binhead->data)[i] = -1;

}

//MPI Version exists
void _ops_particle_map_validation(ops_particle_mapping map) {

  //Compute dx for map only

 // if (map->grid == nullptr) return;

  for (int i = 0; i < map->particle->block->dims; i++) {
    int d_m = map->binhead->d_m[i];
    int d_p = map->binhead->d_p[i];

    //TODO: Check it again please: First part is correct
    int size_loc =  (map->grid == nullptr) ? map->binhead->size[i] + d_m - d_p :// - 1;
                 map->grid->size[i] + d_m - d_p - 1;

    switch (map->particle->particle_pos_dat->type_size) {
    case sizeof(float):
      _ops_compute_map_grid_size_dir<float>(i, map->dx, map->particle->box_block,
                                              size_loc);
      break;
    case sizeof(double):
       _ops_compute_map_grid_size_dir<double>(i, map->dx, map->particle->box_block,
                                              size_loc);
      break;
    case sizeof(long double):
       _ops_compute_map_grid_size_dir<long double>(i, map->dx, map->particle->box_block,
                                                   size_loc);
      break;
    }
  }

}

void ops_build_bounding_box(ops_particle particle ) {

  switch(particle->type_box) {
  case sizeof(float):
    ((BoundingBox<float> *)particle->box_block)->partitionBoundingBox(particle->block);
    break;
  case sizeof(double):
    ((BoundingBox<double> *)particle->box_block)->partitionBoundingBox(particle->block);
    break;
  case sizeof(long double):
    ((BoundingBox<long double > *)particle->box_block)->partitionBoundingBox(particle->block);
    break;
  }

}

void _ops_partition_flat_wall(ops_particle particle) {

  int dim = particle->block->dims;

  if (particle->particle_pos_dat == nullptr || particle->normal_vector == nullptr)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_dat is defined for one of the following: "
                                          "wall position or normal vector\n");

  switch (particle->type_box) {
  case sizeof(float): {
    particle->no_particles = 0;
    bool isin
     = ((BoundingBox<float> *) particle->box_block)->isCoordinateInBoundingBox((float *) particle->xcm);
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((float *) particle->particle_pos_dat->data)[i] = ((float *) particle->xcm)[i];
        ((float *) particle->normal_vector->data)[i] = ((float *) particle->nx)[i]; //TODO:
      }
    }
  } break;
  case sizeof(double): {
    particle->no_particles = 0;
    bool isin
    = ((BoundingBox<double> *) particle->box_block)->isCoordinateInBoundingBox((double *) particle->xcm);
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((double *) particle->particle_pos_dat->data)[i] = ((double *) particle->xcm)[i];
        ((double *) particle->normal_vector->data)[i] = ((double *) particle->nx)[i];
      }
    }
  } break;
  case sizeof(long double): {
    particle->no_particles = 0;
    bool isin
    = ((BoundingBox<long double> *) particle->box_block)->isCoordinateInBoundingBox((long double *) particle->xcm);
    if (isin) {
      particle->no_particles = 1;
      for (int i = 0; i < dim; i++) {
        ((long double *) particle->particle_pos_dat->data)[i] = ((long double *) particle->xcm)[i];
        ((long double *) particle->normal_vector->data)[i] = ((long double *) particle->nx)[i];
      }
    }
  } break;
  }
}

void ops_particle_setup_intrablock_comms(ops_particle particle) {

}


/*-------------------------------------------------------------------------------------*
 *  Functions to output particle data structures to txt files
 *-------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------*
 * Core function to output particle data to txt files. The given function flushed all
 * particle ops_dat structures to txt files
 */

bool ops_checkpointing_filename(const char *file_name, std::string &filename_out,
                                std::string &filename_out2);

void ops_particle_print_data_to_txtfile_core(ops_particle particle,const char *file_name_in) {
  std::string file_name, ignored;
  ops_checkpointing_filename(file_name_in, file_name, ignored);

  FILE *fp;
  if (fopen_s(&fp, file_name.c_str(), "a") != 0) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Can't open file\n");
  }
  if (fprintf(fp, "Particle: %s\n", particle->name)<0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  switch(particle->type_box) {
  case sizeof(float): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%f %f] x [%f %f]",
                particle->no_particles, ((BoundingBox<float> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<float> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<float> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<float> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%f %f]\n",
                  ((BoundingBox<float> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<float> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    } break;
  case sizeof(double): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%f %f] x [%f %f]",
                particle->no_particles, ((BoundingBox<double> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<double> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<double> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<double> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%f %f]\n",
                  ((BoundingBox<double> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<double> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    } break;
  case sizeof(long double): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%Lf %Lf] x [%Lf %Lf]",
                particle->no_particles,
                ((BoundingBox<long double> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%Lf %Lf]\n",
                  ((BoundingBox<long double> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<long double> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    } break;
  }

  if (fprintf(fp, "               x                y ") < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->ids != nullptr)
    if (fprintf(fp, "%16s ", particle->ids->name) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->block->dims == 3)
    if(fprintf(fp,"               z ") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->particle_envelope != nullptr)
    if (fprintf(fp, "%16s ", particle->particle_envelope->name) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");


  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat dat = particle->particle_dat[i];
    switch (dat->dim) {
    case 1:
      if (fprintf(fp, "%16s ", dat->name) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    case 2:
      if (fprintf(fp, "%14s_x %14s_y " ,dat->name, dat->name)<0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    case 3:
      if (fprintf(fp, "%14s_x %14s_y %14s_z ",dat->name, dat->name, dat->name)<0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    default:
      for (int isou = 0; isou < dat->dim; isou++)
        if (fprintf(fp, "%13s[%d] ", dat->name, isou) < 0)
          throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
      break;
    }
  }
  if (fprintf(fp,"\n") < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  /* Export particles data to txt file */
  for (size_t i = 0; i < particle->no_particles; i++) {
    if (particle->ids != nullptr)
      _ops_particle_export_data_point(fp, i, particle->ids);

    _ops_particle_export_data_point(fp, i, particle->particle_pos_dat);
    if (particle->particle_envelope != nullptr)
     _ops_particle_export_data_point(fp, i, particle->particle_envelope);
    for (int idefs = 0; idefs < particle->particle_dat_index; idefs++)
     _ops_particle_export_data_point(fp, i, particle->particle_dat[idefs]);

    if (fprintf(fp, "\n")<0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  }



  fclose(fp);

}

void ops_particle_print_dats_to_txtfile_core(ops_particle particle, ops_dat *dats, int ndats,
                                            const char *file_name_in) {

  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR,
                         "Error: ops_dat not associated with particle data");
  }

  std::string file_name, ignored;
  ops_checkpointing_filename(file_name_in, file_name, ignored);
  FILE *fp;
  if (fopen_s(&fp,file_name.c_str(), "a") != 0) {
   OPSException ex(OPS_RUNTIME_ERROR);
   ex << "Error: can't open file " << file_name;
   throw ex;
  }

  if (fprintf(fp, "Particle: %s\n", particle->name)<0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  switch(particle->type_box) {
  case sizeof(float): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%f %f] x [%f %f]",
                particle->no_particles, ((BoundingBox<float> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<float> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<float> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<float> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%f %f]\n",
                  ((BoundingBox<float> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<float> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    } break;
  case sizeof(double): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%f %f] x [%f %f]",
                particle->no_particles, ((BoundingBox<double> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<double> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<double> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<double> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%f %f]\n",
                  ((BoundingBox<double> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<double> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    } break;
  case sizeof(long double): {
    if (fprintf(fp, "Number of particles: %ld   Block: [%Lf %Lf] x [%Lf %Lf]",
                particle->no_particles,
                ((BoundingBox<long double> *)particle->box_block)->getLocalMin().x,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMax().x,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMin().y,
                ((BoundingBox<long double> *) particle->box_block)->getLocalMax().y) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

    if (particle->block->dims == 3) {
      if (fprintf(fp, "x [%Lf %Lf]\n",
                  ((BoundingBox<long double> *)particle->box_block)->getLocalMin().z,
                  ((BoundingBox<long double> *)particle->box_block)->getLocalMax().z) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
    else
      if (fprintf(fp, "\n") < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    } break;
  }

  //Write titles
  for (int i = 0; i < ndats; i++) {
    for (int isou = 0; isou < dats[i]->dim; isou++) {
      if (fprintf(fp, "%14s[%d] ", dats[i]->name, isou) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
  }

  if (fprintf(fp, "\n")  < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  for (size_t i = 0; i < particle->no_particles; i++) {
    for (int idat = 0; idat < ndats; idat++)
      _ops_particle_export_data_point(fp, i, dats[idat]);

    if (fprintf(fp, "\n") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file");
  }

  fclose(fp);
}


void ops_particle_print_data_to_txtfile(ops_particle particle, const char *file_name) {

  if (particle->ids != nullptr) ops_get_data(particle->ids);

  ops_get_data(particle->particle_pos_dat);

  if (particle->particle_envelope != nullptr) ops_get_data(particle->particle_envelope);

  for (int i = 0; i < particle->particle_dat_index; i++) {
    ops_dat  dat = particle->particle_dat[i];
    ops_get_data(dat);
  }

  ops_particle_print_data_to_txtfile_core(particle, file_name);
}


//TODO: Add a particle pointer within the ops_dat
void ops_particle_print_dats_to_txtfile(ops_particle particle, ops_dat *dats, int ndats,
                                       const char *file_name) {
  for (int idat = 0; idat < ndats; idat++) {
    if (dats[idat]->is_particle )
     ops_get_data(dats[idat]);
    else
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ops_dat structure not related to particle");
  }

  ops_particle_print_dats_to_txtfile_core(particle, dats, ndats, file_name);

}

bool ops_particle_global_rebuild(bool flag) {
  return flag;
}

void _ops_mapping_def_core(ops_particle particle, ops_dat grid, ops_stencil stencil,
                           ops_with_virtual &include_virtual, int size[],
                           int base[], int d_m[], int d_p[]) {

  //Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle structure");

  if (grid == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty grid ops_dat structure");

  ops_block block_particle = particle->block;
  ops_block block_grid = grid->block;
  if (strcmp(block_particle->name, block_grid->name) != 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Non-consistent blocks for grid and particle "
                                             "data structures.");

  int dim = block_particle->dims;
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i] + grid->d_m[i] - grid->d_p[i];
    base[i] = grid->base[i];
    d_m[i] = grid->d_m[i];
    d_p[i] = grid->d_p[i];
    for (int p = 0; p < stencil->points; p++) {
      d_m[i] = MIN(d_m[i], stencil->stencil[particle->block->dims * p + i]);
      d_p[i] = MAX(d_p[i], stencil->stencil[particle->block->dims * p + i]);
    }
  }

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    size[i] = 1;
    base[i] = d_m[i] = d_p[i] = 0;
  }

}

//tODO:
void _ops_mapping_set_structures(ops_particle particle, char *skin, int  d_m[],
                                 int d_p[], int d_mb[], int d_pb[], int size[],
                                 char *dx_map,  ops_with_virtual &include_virtual) {

  switch(particle->type_box) {
  case sizeof(float): {
    if (((BoundingBox<float> *) particle->box_block)->getBlockVolume() < std::numeric_limits<float>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");

    float length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<double> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<double> *) particle->box_block)->getGlobalMin(i);
    _ops_map_compute_dx((float *)skin, length, particle->block->dims, (float *)dx_map, size);
    } break;
  case sizeof(double): {
    if (((BoundingBox<double> *) particle->box_block)->getBlockVolume() < std::numeric_limits<double>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");

    double length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<double> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<double> *) particle->box_block)->getGlobalMin(i);
    _ops_map_compute_dx((double *)skin, length, particle->block->dims, (double *)dx_map, size);
    } break;
  case sizeof(long double): {
    if (((BoundingBox<long double> *) particle->box_block)->getBlockVolume() <
           std::numeric_limits<long double>::epsilon())
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Block of non-positive volume\n");

    long double length[OPS_MAX_DIM];
    for (int i = 0; i < particle->block->dims; i++)
      length[i] = ((BoundingBox<long double> *) particle->box_block)->getGlobalMax(i)
                - ((BoundingBox<long double> *) particle->box_block)->getGlobalMin(i);
    _ops_map_compute_dx((long double *)skin, length, particle->block->dims, (long double *)dx_map,
                        size);
    }break;
  }

  for (int i = 0; i < particle->block->dims; i++) {
    d_mb[i] = (d_m != nullptr) ? MIN(d_m[i], 0) : 0;
    d_pb[i] = (d_p != nullptr) ? MAX(d_p[i], 0) : 0;
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) {
    d_mb[i] = d_pb[i] = 0;
    size[i] = 1;
    switch(particle->type_box) {
    case sizeof(float):
      ((float *)dx_map)[i] = 0.0;
      break;
    case sizeof(double):
      ((double *) dx_map)[i] = 0.0;
      break;
    case sizeof(long double):
      ((long *) dx_map)[i] = 0.0;
    }
  }
}


void _ops_particle_setup_for_rev_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp)
{

  ops_particle_halo_group halo_master = halo_grp->halo_master;

  if (halo_master == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo for forward/backward halo types"
                                                        "has not defined yet");

  if (halo_master->halo_type != OPS_HALO_GRP_BORDER)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo of a forward/backward "
                                                        "halo is not a border type halo");

  if (halo_master->halo_info == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR, "Master halo must be setup before forward/backward"
                                                        "halo");

  if (halo_grp->halo_type == OPS_HALO_GRP_BACKWARD) {

    for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
      ops_particle_halo particle_halo = halo_grp->halo_list[ihalo];
      for (int i = 0; i < particle_halo->nhalos; i++) {
        ops_particle_halo_data halo_data = particle_halo->dat[i];
        if (halo_data->to->type_size != sizeof(double) ||
            halo_data->to->type_size != sizeof(float))
          throw OPSException(OPS_RUNTIME_ERROR," Reverse communications only for float or real"
                             "data structures");

        if (halo_data->orient == OPS_PART_POSITION) {
          throw OPSException(OPS_RUNTIME_ERROR, " Reverse communications only for orientation");

        }
      }
    }
  }
}


ops_dat ops_decl_particle_dat_char(ops_particle particle, int dim, int *dataset_size,
                                   int *base, int *d_m, int *d_p, int *stride,
                                   char *data, int type_size, char const *type,
                                   char const *name, bool assign, bool exchanging) {

  ops_dat dat = ops_decl_dat_temp_core(particle->block, dim, dataset_size, base,
                                       d_m, d_p, stride, data, type_size, type, name);

  dat->user_managed = 0;
  dat->is_hdf5 = 0;

  if (data != NULL && !particle->block->instance->OPS_realloc) {
    dat->user_managed =
        1; // will be reset to 0 if called from ops_decl_dat_hdf5()
    dat->is_hdf5 = 0;
    dat->hdf5_file = "none"; // will be set to an hdf5 file if called from
                             // ops_decl_dat_hdf5()
    size_t bytes = dim * type_size * dataset_size[0];
    dat->mem = bytes;
  } else {
    size_t bytes = type_size * dim * dataset_size[0];

    dat->data = (char *) ops_malloc(bytes);
    dat->user_managed = 0;
    if (data != NULL && particle->block->instance->OPS_realloc) {
          ops_convert_layout(data, dat->data, particle->block, dim,
                             dat->size, dataset_size, type_size, 0);

        } else
          ops_init_zero(dat->data, bytes);
  }
  // Compute offset in bytes to the base index
  dat->base_offset = 0;
  size_t cumsize = 1;
  for (int i = 0; i < particle->block->dims; i++) {
    dat->base_offset +=
        (particle->block->instance->OPS_soa ? dat->type_size : dat->elem_size)
        * cumsize * (-dat->base[i] - dat->d_m[i]);
    cumsize *= dat->size[i];
  }

  //Set up particle details
  dat->is_particle = true;
  dat->is_exchangable = exchanging;

  if (assign) {
    particle->particle_dat_index++;
    ops_particle_realloc_list(particle);
    particle->particle_dat[particle->particle_dat_index - 1] = dat;
  }

  return dat;

}

