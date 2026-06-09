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

char *temp = NULL;
int ntmp_size_max = 0;

static inline  uint64_t pack_pair(int tagI, int tagJ) {

  uint32_t a = (tagI < tagJ) ? tagI : tagJ;
  uint32_t b = (tagI < tagJ) ? tagJ : tagI;

  return ((uint64_t)a << 32) | b;
}

void BoundingBox::partitionBoundingBox(ops_block block, ops_dat map_bin, double *dx) {


  if (owned) {
 //   printf("Bounding box is already set\n");
    return;
  }

  if (coords != nullptr) {
    int dim_dat = coords->block->dims;
    if (dim_dat != dim && coords->dim != dim)
      throw OPSException(OPS_RUNTIME_ERROR, "Dimensions of data structure not consisted "
                         "with dimensions");

    double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];

    _ops_construct_local_box_from_dat(coords, dx, dim, xmin, xmax);

    //Now set local and mininum
    setBoundingBoxLocalBound(xmin, xmax);
    setBoundingBoxGlobalBound(xmin, xmax);

    owned = true;
  }
  else {
    ops_point xmin, xmax;
    xmax = getGlobalMax();
    xmin = getGlobalMin();
    setBoundingBoxLocalBound(xmin, xmax);
    owned = true;
  }

  ops_printf("BoundingBox: [%f %f %f]x[%f %f %f]\n",boundingBox[0].x,
             boundingBox[0].y, boundingBox[0].z,boundingBox[1].x, boundingBox[1].y,
             boundingBox[1].z);
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

static int get_address(const int point[], const int size[], const int d_m[],
                       const int dim) {
  int address = 0;

  for (int isou = 0; isou < dim; isou++) {
    size_t points = 1;
    for (int jsou = 0; jsou < dim; jsou++)
      points *= size[jsou];
    address += (point[isou] - d_m[isou] * points);
  }

  return address;
}

static void get_coord_point(const double *coords, const int size[],const int d_m[],
                            const int ilocal[], const int dim, double *xlocal,
                            double skin, int stag) {

#ifdef OPS_SOA
  if (dim == 2) {
    xlocal[0] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0])
              + 0.5 * static_cast<double>(stag) * skin;
    xlocal[1] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                   + size[0] * size[1]) + 0.5* static_cast<double>(stag) * skin;
  }
  else {
    xlocal[0] = *(  coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                   + (ilocal[2] - d_m[2]) * size[0] * size[1]);
    xlocal[1] = *(  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                   + (ilocal[2] - d_m[2]) * size[0] * size[1] +
                  size[1] * size[2] * size[0]) + 0.5 * skin;
    xlocal[2] = (  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                  + (ilocal[2] - d_m[2]) * size[0] * size[1]
                  + 2 * size[1] * size[2] * size[0]) + 0.5 * static_cast<double>(stag) * skin;
  }
#else
  if (dim == 2) {
    xlocal[0] = *(coords + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0])
              + 0.5 * static_cast<double>(stag) * skin;


    xlocal[1] = *(coords +  dim * (ilocal[0] - d_m[0])
               + dim * (ilocal[1] - d_m[1]) * size[0] + 1)
               + 0.5 * static_cast<double>(stag) * skin;

  }
  else {
    xlocal[0] = *( coords + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
             + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
             + 0.5 * static_cast<double>(stag) * skin;
    xlocal[1] =  *( coords + 1 + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
               + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
               + 0.5 * static_cast<double>(stag) * skin;
    xlocal[2] =  *( coords + 2 + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[1]) * size[0]
              + dim * (ilocal[2] - d_m[2]) * size[0] * size[1])
              + 0.5 * static_cast<double>(stag) * skin;
  }
#endif
}

static void get_coord_point(double *xp, int  *ilocal, int *d_m, ops_point xmin,
                            double *dx, int dim, double skin, int stag) {

  xp[0] = xmin.x + (ilocal[0] - d_m[0]) * dx[0] + 0.5 * static_cast<double>(stag) * dx[0];
  xp[1] = xmin.y + (ilocal[1] - d_m[1]) * dx[1] + 0.5 * static_cast<double>(stag) * dx[1];

  xp[2] = (dim == 3) ?
      xmin.z + (ilocal[2] - d_m[2]) * dx[2] + 0.5 * static_cast<double>(stag) * dx[2] : 0.0;
}



static void get_coord_point_virtual_point(double *coords, int size[], int d_m[],
                                          int ilocal[], int dim, double *xold,
                                          double skin, int stag) {

double xmin[OPS_MAX_DIM];
double dx[OPS_MAX_DIM];

#ifdef OPS_SOA
  if (dim == 2) {
    xmin[0] = *(coords - d_[0] - d_m[1] * size[0]);
    xmin[1] = *( cooords - d_m[0] - d_m[1] * size[0]
                + size[0] * size[1]);

    dx[0] = *(coords + ( -d_[0] + 1) - d_m[1] * size[0])
            - xmin[0];

    dx[1] = *(coords - d[0] + (1 - d_m[1]) * size[0]
               + size[0] * size[1]) - xmin[1];
  }
  else if (dim == 3) {
    xmin[0] = *(    coords - d_m[0] - d_m[1] * size[0]
                  - d_m[2] * size[0] * size[1]);
    xmin[1] = *(    coords - d_m[0] - d_m[1] * size[1]
                  - d_m[2] * size[0] * size[1]
                  + size[0] * size[1] * size[2]);
    xmin[2] = *(    coords - d_m[0] - d_m[1] * size[2]
                  - d_m[2] * size[0] * size[1]
                  + 2 * size[0] * size[1] * size[2]);

    dx[0] = *(    coords + (1 - d_m[0]) - d_m[1] * size[0]
                - d_m[2] * size[0] * size[1]) - xmin[0];
    dx[1] = *(    coords - d_m[0] + (1 - d_m[1]) * size[0]
                - d_m[2] * size[1] * size[2]
                + size[0] * size[1] * size[2]) - xmin[1];
    dx[2] = *(    coords - d_m[0] - d_m[1] * size[1]
                + (1 - d_m[2]) * size[0] * size[1]
                + 2 * size[0] * size[1] * size[2]) - xmin[2];
  }
#else
  if (dim == 2) {
    xmin[0] = *(   coords - dim * d_m[0]  - dim * d_m[1] * size[0]);
    xmin[1] = *(   coords + 1 - dim * d_m[0] - dim * d_m[1] * size[0]);

    dx[0] = *(coords + dim * (1 - d_m[0]) - dim * d_m[1] * size[0]) - xmin[0];
    dx[1] = *(coords + 1 - d_m[0] * dim + (1 - d_m[1] * size[0])) - xmin[1];
  }
  else if (dim == 3) {
    xmin[0] = *(   coords - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);
    xmin[1] = *(   coords + 1 - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);
    xmin[2] = *(   coords + 2 - dim * d_m[0] - dim * d_m[1] * size[0]
                 - d_m[2] * dim * size[0] * size[1]);

    dx[0] = *(    coords + (1 - d_m[0]) * dim - d_m[1] * size[0]
                - d_m[2] * dim * size[0] * size[1]) - xmin[0];
    dx[1] = *(    coords + 1 - d_m[0] * dim + (1 - d_m[1]) * size[0]
                - d_m[2] * dim * size[0] * size[1]) - xmin[1];
    dx[2] = *(    coords + 2  - d_m[0] * dim + ( 1 - d_m[2]) * size[1]
                + (1 - d_m[2]) * dim * size[0] * size[1]) - xmin[2];
  }

  xold[0] = xmin[0] + static_cast<double>(ilocal[0]) * dx[0] + 0.5 * static_cast<double>(stag) * skin;
  xold[1] = xmin[1] + static_cast<double>(ilocal[1]) * dx[1] + 0.5 * static_cast<double>(stag) * skin;
  if (dim == 3)
    xold[2] = xmin[2]  + static_cast<double>(ilocal[2]) * dx[2] + 0.5 * static_cast<double>(stag) * skin;

#endif
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
      ops_dat data = history->data;
      ops_dat npartners = history->n_partnersI;
      ops_dat partnersI = history->partnersI;

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
void _ops_particle_halo_copy_from_buff(char *buff, ops_particle_halo_data *halos,
                                       int nhalos, ops_particle_halo_exchange info,
                                       int flag,
                                       int dir_to[], int dir_from[], double translate[],
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
      else {
        int dim = dat->dim;
        int dims = dat->block->dims;

        //TODO: Need to shift to multipe types
        if (dim != dims || dat->type_size != sizeof(double))
          throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,"ops_dat with orient"
                                                             " must be a vector (of size dim per particle point) and of "
                                                             "type double");
        memcpy(temp, buff + nrecv_bites, nsize);

        double *tmp = (double *)temp;
        double *data = (double *)dat->data;

        for (int i = 0; i < nrecv; i++) {
          int ipart = nfirst + i;
          for (int isou = 0; isou < dim; isou++)
          data[ipart * dim + dir_to[isou]] = tmp[i * dim + dir_from[isou]] + translate[dir_from[isou]];
        }
      }
      nrecv_bites +=nsize;
    }
    else if (flag && halos[ihalo]->halo_type == OPS_EXCHANGE_HISTORY){
      ops_neighbor_history history = halos[ihalo]->history_to;
      ops_dat npartnersI = history->n_partnersI;
      ops_dat partnersI = history->partnersI;
      ops_dat indexing = history->indexing;
      ops_dat indexI = history->indexI;

      ops_dat data = history->data;
      ops_particle particle = history->particleI;
      int *tags = (int *) particle->ids->data;
      int nlocal_elems;

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

  int recv_bites = 0;

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


//TODO: Shift around
int _virtual_within(int bin[], int bin_limits[], double *xpos, BoundingBox *box, int dim) {

  int flag = 0;

  for (int i = 0; i < dim; i++)
    if (bin[i] == bin_limits[2 * i] || bin[i] == bin_limits[2 * i + 1]) {
      flag = 1; break;
    }

  if (flag)
    return (int) box->isCoordinateInBoundingBox(xpos);

  return flag;
}

void _ops_get_max_min(double &minv, double &maxv, const double* dat, const size_t size) {
  
  minv = -1. * std::numeric_limits<double>::max();
  maxv = std::numeric_limits<double>::max();
  for (size_t i = 0; i < size; i++) {
     maxv = MAX(maxv, dat[i]);
     minv = MIN(minv, dat[i]);
  }
}

/*---------------------------------------------------------------------------------------*/
/* Computes bin sizes and grid sizes per direction                                       */
/*---------------------------------------------------------------------------------------*/

void  _ops_compute_bin_size(const ops_point xmin, const ops_point xmax, const int dim,
                            const double dx, int &Nx, double &dx_x, int &Ny,
                            double &dx_y, int &Nz, double &dx_z) {
  
  double box_size_x{xmax.x - xmin.x};
  double box_size_y{xmax.y - xmin.y};
  double box_size_z{0.0};

  if (dim == 3)
    box_size_z = {xmax.z - xmin.z};
  
  Nx = static_cast<int>(box_size_x / (2. * dx));


  Ny = static_cast<int>(box_size_y / (2. * dx));
  if (dim == 3) Nz = static_cast<int>(box_size_z / (2. * dx));

  if (Nx == 0) Nx = 1;
  if (Ny == 0) Ny = 1;
  if (Nz == 0) Nz = 1;
  
  dx_x = box_size_x / Nx;
  dx_y = box_size_y / Ny;
  dx_z = box_size_z / Nz; 
}

int _ops_particle_moved_outside(ops_particle particle) {


  size_t nParticles = particle->no_particles;
  int dim = particle->block->dims;
  BoundingBox *box = particle->box_block;
  double *xlocal = (double *) particle->particle_pos_dat->data;
  for (size_t i = 0; i < nParticles; i++) {
    ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                    (dim == 3) ? xlocal[dim * i + 2] : 0.0};

     bool decide = box->isCoordinateInBoundingBox(point);
     if (!decide)
       return 1;

  }

  return 0;
}


void  _ops_update_old_positions_radii(ops_particle_mapping map, ops_dat coord,
                                      ops_dat shape,size_t Np) {
  int dim = coord->block->dims;

  double *xold = (double *) map->pos_old->data;
  int *part_to_bin =(int *)map->parts_to_grid->data;
  int local_point[OPS_MAX_DIM];

  double dx[OPS_MAX_DIM];

  for (int i = 0; i < map->particle->block->dims; i++) dx[i] = map->dx[i];


  ops_point xmin;
  xmin.x = xmin.x + static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmin.z = (dim == 3) ? xmin.z + static_cast<double>(map->binhead->d_m[2]) * dx[1] : 0.0;

//#ifdef _OPENMP
//# pragma omp parallel for shared (xold, part_to_bin)
//#endif
  for (int i = 0; i <(int) Np; i++) {
    int address = part_to_bin[i];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                    dim, local_point);

    get_coord_point(xold + dim * i, local_point, map->binhead->d_m, xmin,
                    dx, dim, map->skin, 1);
  }
}


/*-------------------------------------------------------------------------------*/
/* Compute the i or j or k of a given point in a computational grid              */
/*-------------------------------------------------------------------------------*/

int _ops_coord_to_bin_dir(const int dir, const int dim, const ops_point xmin,
                          const double *dx, const int *Ngrid, const double *xp) {
  if (dir > dim) {
    ops_printf("Error: Direction %d larger than dimension %d\n", dir,dim);
    exit(-1);
  }
  
  if (dir == 0)
    return static_cast<int>(floor( xp[0] -xmin.x ) / dx[0]);
  else if (dir == 1)
    return static_cast<int>(floor( xp[1] -xmin.y ) / dx[1]);
  else if (dir == 2)
    return static_cast<int>(floor( xp[2] - xmin.z) / dx[2]);

  return -1;
}

/*-------------------------------------------------------------------------------------------*/
/* Construct a bin structure for a uniform grid                                              */
/*-------------------------------------------------------------------------------------------*/

void _ops_uniform_build_map(const int init, const int dim, const size_t Np, const int *Ngrid, 
                            const double* xp, const ops_point xmin, const ops_point xmax, 
                            const double *dx, int *binhead, int *bin) {
  
  /* Initialize structures*/
  size_t grid_size = (dim == 3) ? Ngrid[0] * Ngrid[1] * Ngrid[2] : Ngrid[0] * Ngrid[1];
  
  for (size_t i = 0; i < grid_size; i++)
    binhead[i] = -1;
  if (init) {
    for (size_t i = 0; i < Np; i++)
      bin[i] = -1;
  }
  for (size_t i = Np-1; i >= 0; i++) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, Ngrid, (xp + 3 * i)); //TODO: Need to get elements directly from address
    if (ibin < 0) continue;

    bin[i] = binhead[ibin];
    binhead[ibin] = i;
  }  

}

/*
void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin) {

  int size[dim];
  size_t no_elems{1};
  for (int i = 0; i < dim; i++) {
    size[i] = binhead->size[i];
    no_elems *= size[i];
  }

  int* binhead_data = (int *)binhead->data;
  for (size_t i = 0; i < no_elems; i++)
    binhead_data[i] = -1;



  int* bin_data = (int *)bin->data;
  if (init) {
//#ifdef _OPENMP
//#  pragma omp parallel for shared(bin_data)
//#endif
    for (size_t i = 0; i < Np; i++)
      bin_data[i] = -1;
  }

  double *xp_data = (double *)xp->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared(xp_data)
//#endif
  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
  }
} */


//MPI version exists
void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin, ops_dat part_to_bin) {

  int size[dim];
  size_t no_elems{1};
  for (int i = 0; i < dim; i++) {
    size[i] = binhead->size[i];
    no_elems *= size[i];
  }

  /* Initialize elements */
  memset(binhead->data, -1, sizeof(int) * no_elems);
  int* binhead_data = (int *)binhead->data;
//  for (size_t i = 0; i < no_elems; i++)
//    binhead_data[i] = -1;


  memset(bin->data, -1, sizeof(int) * Np);
  memset(part_to_bin->data, -1, sizeof(int) * Np);
  int *bin_data = (int *)bin->data;
  int *part_2_bin = (int *)part_to_bin->data;

  //if (init) {
//#ifdef _OPENMP
//#  pragma omp parallel for shared(bin_data)
//#endif

// for (size_t i = 0; i < Np; i++) {
//      bin_data[i] = -1;
//      part_2_bin[i] = -1;
//    }
//  }

  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared(xp_data)
//#endif
  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
    part_2_bin[i] = ibin;
  }
}

/*--------------------------------------------------------------------------------------*/
/* Compute grid size for each node of a non-uniform structured grid                     */
/*--------------------------------------------------------------------------------------*/

void _ops_get_grid_size_per_node(const int dims, const ops_dat grid, const ops_point xmin,
                                 const ops_point xmax, double *grid_dx, double *grid_shape)
{
  size_t size[dims], imin[dims], imax[dims];

  for (int i = 0; i < dims; i++)  {
    imin[i] = (size_t) -grid->d_m[i];
    size[i] = (size_t) grid->size[i];
    imax[i] = (size_t) (grid->size[i] - grid->d_p[i]);
  }

  OPS_instance *instance = grid->block->instance;
  double *grid_data = (double *)grid->data;
  if (dims == 2) {
    for (size_t ix = imin[0]; ix < imax[0]; ix++) {
      for (size_t iy = imin[1]; iy < imax[1]; iy++) {
        double dx[2], xm[2], xb[2];
        if (ix == imin[0]) {
          xb[0] = (instance->OPS_soa) ? 2. * xmin.x - *(grid_data + ix + iy * size[0])
                                      : 2. * xmin.x - *(grid_data + 2 * ix + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? *(grid_data + (ix + 1) + iy * size[0])
                                      : *(grid_data + 2*(ix + 1) + 2 * iy * size[0]);
        }
        else if (ix == imax[0] - 1) {
          xb[0] = (instance->OPS_soa) ? *(grid_data + ix - 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix - 1) + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? 2. * xmax.x - *(grid_data + ix+ iy * size[0])
                                      : 2. * xmax.x - *(grid_data + 2 * ix + 2 * iy * size[0]);
        }
        else {
          xb[0] = (instance->OPS_soa) ? *(grid_data + ix - 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix - 1) + 2 * iy * size[0]);
          xm[0] = (instance->OPS_soa) ? *(grid_data + ix + 1 + iy * size[0])
                                      : *(grid_data + 2 * (ix + 1) + 2 * iy * size[0]);
        }

        dx[0] = 0.5 * (xm[0] - xb[0]);

        /* y-direction */
        if (iy == 0) {
          xb[1] = (instance->OPS_soa) ? 2. * xmin.y - *(  grid_data + size[0] * size[1]
                                                        + ix + iy * size[0])
                                      : 2. * xmin.y - *(  grid_data + 1 + 2 * ix
                                                        + 2 * iy * size[0]);
          xm[1] = (instance->OPS_soa) ? *(   grid_data + size[0] * size[1]
                                           + ix + (iy + 1) * size[0])
                                      : *(grid_data + 1 + 2 * ix + 2 * (iy + 1) * size[0]);
        }
        else if (iy == imax[1] - 1) {
          xb[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy - 1) * size[0])
                                      : *(grid_data + 1 + 2 * ix + 2 * (iy -1) * size[0]);
          xm[1] = (instance->OPS_soa) ? 2. * xmax.y - *(  grid_data + size[0] * size[1]
                                                        + ix + iy * size[0])
                                      : 2. * xmax.y - *(  grid_data + 1 + 2 * ix
                                                        + 2 * iy * size[0]);
        }
        else {
          xb[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy - 1) * size[0])
                                      : *(  grid_data + 1 + 2 * ix + 2 * (iy - 1) * size[0]);
          xm[1] = (instance->OPS_soa) ? *(  grid_data + size[0] * size[1] + ix
                                           + (iy + 1) * size[0])
                                      : *(  grid_data + 1 + 2 * ix + 2 * (iy + 1) * size[0]);
        }

        dx[1] = 0.5 * (xm[1] - xb[1]);

        size_t loc = ix + iy * size[0];
        grid_dx[loc] = MAX(dx[0], dx[1]);
        grid_shape[loc * dims] = dx[0];
        grid_shape[loc * dims + 1] = dx[1];

      }
    }
  }
  else if (dims == 3) {
    for (size_t  ix = imin[0]; ix < imax[0]; ix++) {
      for (size_t iy = imin[1]; iy < imax[1]; iy++) {
        for (size_t iz = imin[2]; iz < imax[2]; iz++) {
          double xb[3], xm[3], dx[3];
          if (ix == 0) {
            xb[0] = (instance->OPS_soa) ? 2. * xmin.x - *(  grid_data + ix + size[0] * iy
                                                          + iz * size[0] * size[1])
                                        : 2. * xmin.x - *(  grid_data + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
            xm[0] = (instance->OPS_soa) ? *(  grid_data + (ix + 1) + iy * size[0]
                                            + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix + 1) + 3 * iy * size[0]
                                            + 3 * iz * size[0] * size[1]);
          }
          else if (ix == imax[0] - 1) {
            xb[0] = (instance->OPS_soa) ? *(  grid_data + (ix - 1) + iy * size[0]
                                             + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix - 1) + 3 * iy * size[0]
                                             + 3 * iz * size[0] * size[1]);
            xm[0] = (instance->OPS_soa) ? 2. * xmax.x - *(  grid_data + ix + size[0] * iy
                                                          + iz * size[0] * size[1])
                                        : 2. * xmax.x - *(  grid_data + 3 * ix + 3 * iy *size[0]
                                                          + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[0] = (instance->OPS_soa) ? *(  grid_data + (ix - 1) + iy * size[0]
                                            + iz * size[0] * size[1])
                                        : *(  grid_data + 3 * (ix - 1) + 3 * iy * size[0]
                                            + 3 * iz * size[0] * size[1]);
            xm[0] =  (instance->OPS_soa) ? *(  grid_data + (ix + 1) + iy * size[0]
                                             + iz * size[0] * size[1])
                                         : *(  grid_data + 3 * (ix + 1) + 3 * iy * size[0]
                                              + 3 * iz * size[0] * size[1]);
          }

          dx[0] = 0.5 * (xm[0] - xb[0]);

          /* y-direction */
          if (iy == 0) {
            xb[1] = (instance->OPS_soa) ? 2. * xmin.y - *(grid_data + size[0] * size[1] * size[2]
                                                           + ix + iy * size[0] +
                                                           + iz * size[0] * size[1])
                                        : 2. * xmin.y - *(grid_data + 1 + 3 * ix + 3 * iy * size[0]
                                                            + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy + 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix + 3 * (iy + 1) * size[0]
                                             + 3 * iz * size[0] * size[1]);
          }
          else if (iy == imax[1] - 1) {
            xb[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy - 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix  + 3 * (iy - 1) * size[0]
                                              + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? 2. * xmax.y - *( grid_data + size[0] * size[1] * size[2]
                                                           + ix + iy * size[0]
                                                           + iz * size[0] * size[1])
                                        : 2. * xmax.y - *( grid_data + 1 + 3 * ix
                                                           + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[1] = (instance->OPS_soa) ? *( grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy - 1) * size[0] + iz * size[0] * size[1])
                                        : *( grid_data + 1 + 3 * ix + 3 * (iy-1) * size[0]
                                             + 3 * iz * size[0] * size[1]);
            xm[1] = (instance->OPS_soa) ? *(grid_data + size[0] * size[1] * size[2]
                                             + ix + (iy + 1) * size[0]  + iz * size[0] * size[1])
                                        : *(grid_data + 1 + 3 * ix + 3 * (iy + 1) * size[0]
                                            + 3 * iz * size[0] * size[1]);
          }

          dx[1] = 0.5 * (xm[1] - xb[1]);

          /* z- direction */
          if (iz == imin[2]) {
            xb[2] = (instance->OPS_soa) ? 2. * xmin.z - *( grid_data + 2 * size[0] * size[1] * size[2]
                                                           + ix + iy * size[0]
                                                           + iz * size[0] * size[1])
                                        : 2. * xmin.z - *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? *(  grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz + 1) * size[0] * size[1])
                                        : *( grid_data + 2 +3 * ix + 3 * iy * size[0]
                                             + 3 * (iz + 1) * size[0] * size[1]);
          }
          else if (iz == imax[2] - 1) {
            xb[2] = (instance->OPS_soa) ? *( grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz - 1) * size[0] * size[1])
                                        : *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz - 1) * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? 2. * xmax.z - *( grid_data + 2 * size[0] * size[1] * size[2]
                                                          + ix + iy * size[0]
                                                          + iz * size[0] * size[1])
                                        : 2. * xmax.z - *( grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                                           + 3 * iz * size[0] * size[1]);
          }
          else {
            xb[2] = (instance->OPS_soa) ? *(grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz - 1) * size[0] * size[1])
                                        : *(grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz - 1) * size[0] * size[1]);
            xm[2] = (instance->OPS_soa) ? *(grid_data + 2 * size[0] * size[1] * size[2]
                                             + ix + iy * size[0] + (iz + 1) * size[0] * size[1])
                                        : *(grid_data + 2 + 3 * ix + 3 * iy * size[0]
                                             + 3 * (iz + 1) * size[0] * size[1]);
          }

          dx[2] = 0.5 * (xm[2] - xb[2]);
          size_t loc = ix + iy * size[0] + iz * size[0] * size[1];
          grid_dx[loc] = MAX3(dx[0], dx[1], dx[2]);
          for (int idir = 0; idir < dims; idir++)
            grid_shape[dims * loc + idir] = dx[idir];
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*/
/* Assert the intersection of a grid cell with a particle center                                 */
/*-----------------------------------------------------------------------------------------------*/

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

void _ops_particle_to_non_uniform_grid_intersection(const int init, const int dim, 
                                                    const ops_point xmin,
                                                    const int *binhead_grids, const int *Npoints, 
                                                    const int *bin_grid, const double* xGrid, 
                                                    const double *grid_dx, const int Ngrid,
                                                    const int *binhead_particles, const int *Ng_parts,
                                                    const int *bin_parts, const double *xp, const int Np,
                                                    const double *dx_grid, const double *dx_p_grid, 
                                                    int *binhead, const int *size, int *bin)
{
  /* Initialize structure if necessary */
  if (init == 1) {
    //TODO: Vrf that size[i > dim] = 1
    for (int i = 0; i < size[0] * size[1] * size[2]; i++) {
      binhead[i] = -1;
    }

    for (int i = 0; i < Np; i++)
      bin[i] = -1;
  }

  /* Part II: Identify stencil */
  int Nm[dim]; //TODO: We need to map to smaller particle
  for (int i = 0; i < dim; i++) {
    Nm[i] =static_cast<int>(dx_grid[i] / dx_p_grid[i]);
  }

  /* Loop over coarse*/
  double xelem[dim], dxGrid[dim];
  int ngrid = Npoints[0] * Npoints[1] * Npoints[2];
  for (int i = 0; i < ngrid; i++) {
    /* Get local element */
    int igrid = binhead_grids[i];
    while (igrid > - 1) {
      xelem[0] = xGrid[dim * igrid];
      xelem[1] = xGrid[dim * igrid + 1];
      if (dim == 3)
        xelem[2] = xGrid[dim * igrid + 2];
      
      dxGrid[0] = grid_dx[3 * igrid];
      dxGrid[1] = grid_dx[3 * igrid + 1];
      if (dim == 3)
        dxGrid[2] = grid_dx[3 * igrid + 2];

      /* Get stencil*/
      int ix = _ops_coord_to_bin_dir(0, dim, xmin, dx_p_grid, Ng_parts, xelem);
      int imin = (ix - Nm[0] >= 0 ) ? ix - Nm[0] : 0;
      int imax = (ix + Nm[0] < Ng_parts[0]) ? ix + Nm[0] : Ng_parts[0] - 1;

      int iy = _ops_coord_to_bin_dir(1, dim, xmin, dx_p_grid, Ng_parts, xelem);
      int jmin = (iy - Nm[1] >= 0) ? iy - Nm[1] : 0;
      int jmax = (iy + Nm[1] < Ng_parts[1]) ? iy + Nm[1] : Ng_parts[1] - 1;

      int iz, kmin{0}, kmax{0};
      if (dim == 3) {
        iz = _ops_coord_to_bin_dir(2, dim, xmin, dx_p_grid, Ng_parts, xelem);
        kmin = (iz - Nm[2] >= 0) ? iz - Nm[2] : 0;
        kmax = (iz + Nm[2] < Ng_parts[2]) ? iz + Nm[2] : Ng_parts[2] - 1; 
      }

      /* Search for interaction between grid points and particles */
      for (int ibin = imin; ibin <= imax; ibin++) {
        for (int jbin = jmin; jbin <= jmax; jbin++) {
          for (int kbin = kmin; kbin <= kmax; kbin++) {
            int ipart = binhead_particles[   ibin + jbin *  Ng_parts[0]
                                              + kbin * Ng_parts[0] * Ng_parts[1]];
            while(ipart > - 1) {
              //TODO: Need to verify also for OPS_soa or not
              int inter = _ops_check_particle_nunif_grid_inters(dim, xelem, dxGrid, xp + dim * ipart);
              //onst int dim,const double *xGrid,double *dxGrid, double *xp

              if (inter ) {
                bin[ipart] = binhead[igrid];
                binhead[igrid] = ipart;
              }
              ipart = bin_parts[i];
            }
          }
        }
      }

      igrid = bin_grid[igrid];
    }
  }
}

static void _ops_particle_export_data_point(FILE *fp, int i, ops_dat dat) {

  if (!dat->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR,
                       "Error: ops_dat not related to particle data\n");
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
                                           int dir_to[], int dir_from[], double translate[],
                                           int *ntot_bites) {
  int nfirst = halo_info->firstrecv;
  int nrecv = halo_info->nrecv;
  int max_size = 0;

  int a1 = 0;
  int nrecv_bites = 0;
  int nsize = dat->elem_size;

  if (orient == OPS_PART_ORIENT_ON) {

    if (nsize * nrecv > ntmp_size_max) {
      temp = (char *) ops_realloc(temp, nsize * (nrecv + OPS_MAX_PART));
      ntmp_size_max = nsize * (nrecv + OPS_MAX_PART);
    }

    memcpy(temp, buff + nrecv_bites, nsize);
    double *tmp = (double *)temp;
    double *data = (double *)dat->data;

    for (int i = 0; i < nrecv; i++) {
      for (int isou = 0; isou < dat->dim; isou++)
      data[nfirst * dat->dim + dir_to[isou]] = tmp[dir_from[isou]] + translate[dir_from[isou]];
    }
  }
  else {
    int nbite_first = nfirst * dat->elem_size;
    memcpy(dat->data + nbite_first, buff + nrecv_bites, nsize);
  }

  (*ntot_bites) = nsize * nrecv;
}

/*-----------------------------------------------------------------------------------------------------------*/
/* Finds the box of each process based on an ops_dat structure for non-MPI backend code-assume a uniform grid
 * TODO: Expand to non-uniform
 */
void _ops_construct_local_box_from_dat(ops_dat coords, double *grid_size, int dim, double *xmin, double *xmax) {
  /* Get minimum lowerbound */
  int imin[OPS_MAX_DIM], imax[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];



  /* Get lower and upper bound */
  for (int i = 0; i < dim; i++) {
    imin[i] = -coords->d_m[i];
    imax[i] = -coords->d_m[i] + (coords->size[i] - coords->d_p[i] + coords->d_m[i]) - 1; //TODO: Check
    size[i] = coords->size[i];
  }

  ops_printf("imin = [ ");
  for (int i = 0; i < dim; i++)
    ops_printf("%d ", imin[i]);
  ops_printf("]\n");

  ops_printf("imax = [ ");
  for (int i = 0; i < dim; i++)
    ops_printf("%d ", imax[i]);
  ops_printf("]\n");

  printf("SOA: %d\n", coords->block->instance->OPS_soa);

  double *data = (double *)coords->data; //NEED A Sanity check
  OPS_instance *instance = coords->block->instance;

  if (dim == 2) {
    if (instance->OPS_soa) {
      xmin[0] = *(data + imin[0] + imin[1] * size[0]);
      xmin[1] = *(data + imin[0] + imin[1] * size[0] + size[0] * size[1]);

      /* Getting xmax */
      xmax[0] = *(data + imax[0] + imax[1] * size[0]);
      xmax[1] = *(data + imax[0] + imax[1] * size[0] + size[0] * size[1]);
    }
    else {
      xmin[0] = *(data + imin[0] * coords->dim + imin[1] * coords->dim * size[0]);
      xmin[1] = *(data + 1 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]);

      /* xmax */
      xmax[0] = *(data + imax[0] * coords->dim + imax[1] * coords->dim * size[0]);
      xmax[1] = *(data + 1 + imax[0] * coords->dim + imax[1] * coords->dim * size[0]);
    }
  }
  else if (dim == 3) {
    if (instance->OPS_soa) {
      xmin[0] = *(data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]);
      xmin[1] = *(  data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]
                   + size[0] * size[1] * size[2]);
      xmin[2] = *(  data + imin[0] + imin[1] * size[0] + imin[2] * size[0] * size[1]
                   + 2 * size[0] * size[1] * size[2]);

      //Getting xmax
      xmax[0] = *(data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]);
      xmax[1] = *(  data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]
                   + size[0] * size[1] * size[2]);
      xmax[2] = *(  data + imax[0] + imax[1] * size[0] + imax[2] * size[0] * size[1]
                  + 2 * size[0] * size[1] * size[2]);
    }
    else {
      xmin[0] = *(  data + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);
      xmin[1] = *(  data + 1 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);
      xmin[2] = *(  data + 2 + imin[0] * coords->dim + imin[1] * coords->dim * size[0]
                   + imin[2] * coords->dim * size[0] * size[1]);

      xmax[0] = *( data + imax[0] * coords->dim + imax[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
      xmax[1] = *(  data + 1 + imax[0] * coords->dim + imax[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
      xmax[2] = *(  data + 2 + imax[0] * coords->dim + imin[1] * coords->dim * size[0]
                  + imax[2] * coords->dim * size[0] * size[1]);
     }
  }
  else {
    ops_printf("Error: The size of the spatial domain must be two or zero.\n");
    exit(-1);
  }

  printf("Bounding Box = ");
  for (int i = 0; i < dim; i++)
    printf("[%f %f] ", xmin[i], xmax[i]);
  printf("\n");

  /* Uniform assumption for the moment */
  for (int i = 0; i < dim ; i++) {
    xmin[i] -= 0.5 * grid_size[i];
    xmax[i] += 0.5 *grid_size[i];
  }
}


//MPI VErsion exists
bool ops_get_bounding_box_local_to_global(ops_block block,double* xmin,double *xmax,double* xglb_min, double* xglb_max) {

  for (int i = 0; i < 3; i++) {
     xglb_min[i] = xmin[i];
     xglb_max[i] = xmax[i];
  }

  return true;
}

/* Get the local bounding box for sequential code */
bool ops_bounding_box_global_to_local(const ops_block block, int dim,
                                      std::array<ops_point, 2>& globalBoundingBox,
                                      std::array<ops_point, 2>& boundingBox) {

  boundingBox[0].x = globalBoundingBox[0].x;
  boundingBox[0].y = globalBoundingBox[0].y;

  boundingBox[1].x = globalBoundingBox[1].x;
  boundingBox[1].y = globalBoundingBox[1].y;

  if (dim == 3) {
    boundingBox[1].z = globalBoundingBox[1].z;
    boundingBox[0].z = globalBoundingBox[0].z;
  }
  return true;
}

/*-------------------------------------------------------------------------------------*/
/* Mapping particle functions
 *-------------------------------------------------------------------------------------*/

//MPI Version exists
int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {
  /*Enforce new build */
  if (enforce)
    return 1;

  if (map->decide)
    return 1;

  //1. Particle outside structure
 // if (_ops_particle_moved_outside(particle) == 1)
 //   return 1;


  int flag = 0;
  double *x = (double *)particle->particle_pos_dat->data;
  double *x_old = (double *)map->pos_old->data;

  double dx = map->skin;

  /* 2. Particle moved outside previous block */
  int dim = particle->block->dims;
  int nParticles =(int) particle->no_particles;

  for (int i = 0; i < nParticles; i++) {
    int a1 = 0;
    for (int idir = 0; idir < dim; idir++) {
//      printf("Dx = %f skin = %f\n", fabs(x[i * dim + idir] - x_old[i * dim + idir]), map->skin);
      if (fabs(x[i * dim + idir] - x_old[i * dim + idir]) > 0.5 * dx + DBL_EPSILON) {
        a1 = 1;
        break;
      }
    }
    if (a1 == 1) {
      flag = 1;
      break;
    }
  }

  for (int i = 0; i < particle->nhistories; i++)
    particle->histories[i]->flag_update = (bool) flag;

  if (flag) map->decide = true;
  else
    map->decide = false;

  return flag;

}

/*int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce) {

  // User enforce new map build
  if (enforce)
    return 1;

  if (map->decide)
    return 1;

  //*1.  Particle structure changed
  return _ops_particle_moved_outside(particle);

  int flag = 0;
  //TODO: Expand to three different structures

  //* 2. Particles moved substantially
  double *x = (double *)particle->particle_pos_dat->data;
  double *x_old =(double *)map->pos_old->data;

  double *rad = (double *)map->Rp->data;
  double *rad_old
  = (map->particle_changes== OPS_EVOLV_SHAPE) ?
      (double *) map->Rp_old->data : nullptr;

  int dim = particle->block->dims;
  double dx[dim];
  for (size_t i = 0; i < map->nParticles; i++) {
    double dx_sq{0.0}, dr{0.0};
    for (int isou = 0; isou < dim; isou++) {
      dx[isou] = x[isou + dim * i] - x_old[isou + dim * i];
      dx_sq += dx[isou] * dx[isou];

      if (map->particle_changes== OPS_EVOLV_SHAPE)
        dr = rad[i] - rad_old[i];
    }

    //Case I: Particle of fixed envelope
    if (map->particle_changes != OPS_EVOLV_SHAPE) {
      if (dx_sq > map->skin * map->skin) {
        flag = 1;
        break; //TODO: Check if it works
      }
    }
    else {
      double dr_sq = dr * dr;
      if (dr_sq > map->skin * map->skin || dx_sq > map->skin * map->skin) {
        flag = 1;
        break;
      }
    }
    //TODO: Add radius

  }

  if (flag) map->decide = true;
  else
    map->decide = false;

  if (flag)
    return 1;

  //TODO: Need to verify that also the grid hash't changed as well

  return 0;
}*/

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
  printf("Particle Nmax = %d for map\n", map->Nmax);

  //Destroy existing structures and rebuild

  int prod = 1;
  for (int i = 0; i < particle->block->dims; i++) prod *= map->binhead->size[i];
  for (int i = 0; i < prod;i++)
    ((int *)map->binhead->data)[i] = -1;

  for (int i = 0; i < Np; i++) {
    ((int *)map->bin->data)[i] = -1;
    ((int *)map->parts_to_grid->data)[i] = -1;
  }

 // printf("Passed building local\n");

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];

  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x = xmin.x + static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmin.y = xmin.y + static_cast<double>(map->binhead->d_m[1]) * dx[1];

  xmax.x = xmax.x + static_cast<double>(map->binhead->d_p[0]) * dx[0];
  xmax.y = xmax.y + static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
    xmax.z = xmax.z + static_cast<double>(map->binhead->d_p[2]) * dx[2];
    xmin.z = xmin.z + static_cast<double>(map->binhead->d_m[2]) * dx[2];
  }

//  printf("Mapping list [%f %f %f]x[%f %f %f]\n",xmin.x, xmin.y, xmin.z
 //        ,xmax.x, xmax.y, xmax.z);
 // printf("Dx = [%f %f %f]", dx[0], dx[1], dx[2]);

  /* Build bins based on an ops_dat structure */
  _ops_build_uniform_dats(1, particle->block->dims, map->grid, particle->particle_pos_dat,
                          Np, dx, xmin, xmax, map->binhead, map->bin, map->parts_to_grid);

//  _ops_build_particle_to_grid(map->nParticles, map->bin, map->binhead,
//s                              map->parts_to_grid);


  _ops_update_old_positions_radii(map, particle->particle_pos_dat,
                                  particle->particle_envelope, Np);

 // printf("Mapped is build\n");

}

//MPI Version exists
void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map) {

  map->nParticles = ifirst + n_to_map;

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *) particle->particle_pos_dat->data;
  double *xold = (double *) map->pos_old->data;

  for (int i = ifirst; i < ifirst + n_to_map; i++) {



    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

    if (address < 0) printf("Rank %d: Particle %d address = %dv xpos =[%f %f] dx = %12.9e dy = %12.9e\n",ops_get_proc(), i, address, xpos[i* dim], xpos[i * dim + 1],
                            (xpos[i * dim] - xmin.x)/dx[0], (xpos[i * dim + 1] - xmin.y) / dx[1]);

    if (address < 0) continue;
    bin2grid[i] = address;

    bins[i] = binhead[address];
    binhead[address] = i;

    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);


    //TODO:
    get_coord_point_virtual_point((double *) map->grid->data, map->grid->size,
                                  map->grid->d_m, ilocal, dim, xold + dim * i,
                                  map->skin, 1);

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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = 0; i < particle->no_particles; i++) {
    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    bin2grid[i] = address;
    bins[i] = binhead[address];
    binhead[address] = i;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);

    get_coord_point(xold + dim * i, ilocal, map->binhead->d_m, xmin, dx, dim,
                    map->skin, 1);
  }

}


//MPI Version exists
void _ops_particle_map_from_exchange(ops_particle_mapping map, ops_particle particle,
                                     int ifirst, int ilast) {

  if (ifirst > particle->no_particles || ilast > particle->no_particles )
    throw OPSException(OPS_RUNTIME_ERROR, "Mapping particles are not actual particles");

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *) map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  int *mark_deletion = (int *)particle->mark_deletion;

  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = ifirst; i < ilast; i++) {

    int address = _ops_coord_to_bin(dim , xmin, xmax, dx, map->binhead->size,
                                    xpos + dim * i);

    if (!particle->box_block->isCoordinateInBoundingBox(xpos + dim * i)) {
      bin2grid[i] = -1;
      bins[i] = -1;
      mark_deletion[i] = 1;
      continue;
    }

    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

  }

}



//TODO: Need to generate a version that splits the generation creation and adding halo for
//      the exchange.


//MPI Version exists
void _ops_particle_setup_map_virtual(ops_particle particle, ops_particle_mapping map) {

  if (particle->no_virtual == 0) return;

  map->nParticles += particle->no_virtual;

  //Initialize structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *bin2grid = (int *)map->parts_to_grid->data;

  //Get mapping
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];


  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  int dim = particle->block->dims;
  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = particle->no_particles;
           i < particle->no_particles + particle->no_virtual; i++) {
    bins[i] = -1;
    bin2grid[i] = -1;

    int address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);

    if (address < 0) continue;
    bins[i] = binhead[address];
    binhead[address] = i;
    bin2grid[i] = address;

    // Add particle to list
    int ilocal[OPS_MAX_DIM];
    get_local_point(address, map->binhead->size, map->binhead->d_m,
                     dim, ilocal);

    get_coord_point_virtual_point((double *)map->grid->data, map->grid->size,
                                  map->grid->d_m,
                                   ilocal, dim, xold + dim * i, map->skin, 1);//TODO-Replace function

  }

}


/**
 * Function erases and build particle lists directly
 */

//TODO: We need to add herein what it builds (local or all)

//MPI Version exists
int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {

  //TODO: Do we need this to be false by default???
  map->decide = false;



  int stag = 1; //Staggering is by def 1 for now

 // printf("Entering to check: %d\n");

  /* Checking if mapping declaired particles < total */
  int changed{0};
  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part_to_grid = (int *)map->parts_to_grid->data;
  int *mark_del = particle->mark_deletion;


  /*Accessing local particles only */
  /* 1. Check for deletion */
  /* 2. Check if need to move */

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];


 /* Set particles for mapping */
  int nmapping
    = (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                          particle->no_particles + particle->no_virtual;


  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data; //Assumed here in that
  //Nparticles in list is build properly

  int nlimits = 2 * dim;
  int exch_limits[2 * OPS_MAX_DIM];
  int rmv_limits[2 * OPS_MAX_DIM];

  for (int i =  0; i < dim; i++) {
    exch_limits[2 *i] = (d_m[i] < 0) ? -d_m[i] : -size[i];
    exch_limits[2 * i + 1] = (d_p[i] > 0) ? size[i] + d_m[i] - 2 * d_p[i]: 2 * size[i];
    rmv_limits[2 *i ] = 0;
    rmv_limits[2 * i + 1] = size[i] - d_p[i] + d_m[i] - 1;
  }

//  printf("rmv_limits = [%d %d]x[%d %d]\n", rmv_limits[0], rmv_limits[1], rmv_limits[2], rmv_limits[3]);

/*
  for (int i = 0; i < particle->no_particles + particle->no_virtual; i++) {
    if (i < particle->no_particles)
      printf("Actual particle  %d: x = [ %f %f ] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
             bins[i], part_to_grid[i]);
    else printf("Virtual %d: x = [ %f %f ] xold =[%f %f] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
                 xold[2 * i], xold[2 * i + 1],bins[i], part_to_grid[i]);
  }
*/

  int nactual = particle->no_particles;

  //TODO: For GPU: Split into two parts
  for (int i = 0; i < particle->no_particles; i++) {
    int flag =  local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                     dim);

    if (flag) {

      int address = part_to_grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      //Actual particle perform two checks

        //TODO-1: Check for possible removal
      int del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                       xpos + i * dim, particle->box_block);

      if (del_flag) {
        part_to_grid[i] = -1;
        map->decide = true;//TODO: Think if it is needed
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
        for (int ih = 0; ih < particle->nhistories; ih++) particle->histories[ih]->flag_update = true;
      }

      //Start-mapping
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);
      if (address < 0) map->decide = true;

      bins[i] = binhead[address];
      binhead[address] = i;

      part_to_grid[i] = address;

      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);

      get_coord_point(xold + dim * i, ilocal, map->binhead->d_m, xmin, dx, dim,
                      map->skin, 1);

            //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

 //     if (flag_build) printf("Particle %d to exchange zone\n");

      if (!map->decide) map->decide = flag_build;

    }
  }

  nactual = particle->no_particles;

  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

//  printf("Exit actual mapping: ");

  //printf("Building virtual maps\n");


   //TODO: We need a new version which keeps th

  for (int i = ifirst; i < nmapping; i++) {

    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin, dim);

    if (flag) { //To-Rebuild for virtual particle

      int address = part_to_grid[i];
      _remove_particle_from_bins(address, i, binhead, bins);
      part_to_grid[i] = -1;


      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

      if (address < 0) {
  //      printf("Particle %d non-positive address (%d)\n", i, address);
        map->decide = true; continue;}

      //Virtual still projected'


      get_local_point(address, map->binhead->size, map->binhead->d_m,
                      dim, ilocal);

      int flag_in = _virtual_within(ilocal, rmv_limits, xpos + dim * i, particle->box_block, dim);

      //Need to remove



      if (flag_in) { //Virtual become actual

        nactual++; //Increase local.
        nvirtual_act--;
        //swap->particle data

        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


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
        //TODO: Need to swap positions as well then rebuild
        map->decide = true;

      }

      if (!map->decide)  {
        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


        get_coord_point_virtual_point((double *) map->grid->data, map->grid->size,
                                      map->grid->d_m, ilocal, dim, xold + dim * i,
                                      map->skin, 1);
      }

    }


  }
  //Perform action for virtual particles



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
 // history_decide = 0;
  int changed{0};
  int dim = particle->block->dims;

  //Get mapping structures
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;

  int *mark_del = particle->mark_deletion; //TODO: Need a global factor

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  int nmapping = particle->no_particles;


  /* Compute bounding points for mapping structures */
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();
  double dx[particle->block->dims];
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  int nlimits = 2 * dim;
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
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                    dim);

    //TODO: Add the flag for history!!!!
    if (flag) {
      int address = part2grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);


      //Remove particle from bin-box
      _remove_particle_from_bins(address, i, binhead, bins);
      for (int ih = 0; ih < particle->nhistories; ih++)
        if (!particle->histories[ih]->flag_update) particle->histories[ih]->flag_update = true;
      //Check for deletion
      int del_flag = _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits, xpos + i * dim,
                                                      particle->box_block); //TODO: Check

      if (del_flag) {

       //Add flag for history to be changed
       part2grid[i] = -1;
       map->decide = true; //Enforce mapping;
       //nactual--;
       particle->mark_deletion[i] = 1;
       continue;
      }

      //Update map of actual particle
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;

      //Checking for particle moving in or out of exchange zone
      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);

      get_coord_point(xold + i * dim, ilocal, map->binhead->d_m, xmin, dx, dim,
                      map->skin, 1);

      //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, exch_limits, dim);

      if (!map->decide) map->decide = flag_build;
//      if (!history_decide ) history_decide = (int) flag_build;

    }
  }


  return (int) map->decide;
}


//MPI Version exists
void _ops_particle_remap_virtual(ops_particle_mapping map, ops_particle particle,
                                 int istart, int ilast) {
  //Sanity checks
  if (istart < particle->no_particles)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function for mapping virtual "
                                             "called for actual particles");

  if (ilast > particle->no_particles + particle->no_virtual)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Function called for non-existing "
                                             " particle");

  if (map->mapping_type != OPS_WITH_VIRTUAL) return;

  int stag = 1;

  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part2grid = (int *)map->parts_to_grid->data;

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];

  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  for (int i = 0; i < particle->block->dims; i++) dx[i] = map->dx[i];

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data;

  for (int i = istart; i < ilast; i++) {
    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin,
                                    dim);

    if (flag) {
      int address = part2grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,
                                  xpos + dim * i);

      bins[i] = binhead[address];
      binhead[address] = i;
      part2grid[i] = address;
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
  BoundingBox *sendingBox = halo->particle_from->box_block;
  BoundingBox *recvBox = halo->particle_from->box_block;
  if (!sendingBox->getOwnership() || !recvBox->getOwnership()) {
    throw OPSException(OPS_RUNTIME_ERROR, "Particle halo must be set after the bounding "
                       "sets are defined");
  }

  /* Get local boxes */
  int dim = sendingBox->getDim();
  double xmin[dim], xmax[dim], xsend_min[dim], xsend_max[dim];
  double x_recv_min[dim], x_recv_max[dim];
  recvBox->getLocalMaxMin(x_recv_min, x_recv_max);
  sendingBox->getLocalMaxMin(xmin, xmax);


  double *xtranslate = halo->translate;


  double xrecv_act_min[dim], xrecv_act_max[dim];
  for (int i = 0; i < dim ; i++) {
    int isend_dir =  halo->dir_from[i];
    int irecv_dir = halo->dir_to[i];

    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - xtranslate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - xtranslate[isend_dir];
  }

  /* Search for intersection box */
  for (int i = 0; i < dim; i++) {
    for (int iswap = 0; iswap < 2; iswap++) {
      for (int j = 0; j < dim ; j++) {
        xsend_min[j] = xmin[j];
        xsend_max[j] = xmax[j];
      }

//      xsend_min[i] = (iswap == 0) ? -0.5 * std::numeric_limits<double>::max() : xmax[i] - halo->dx[i];
//      xsend_max[i] = (iswap == 0) ? 0.5 * std::numeric_limits<double>::max() : xmin[i] + halo->dx[i];

      //TODO: We modiftied the table
      xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG : xmax[i] -  halo->dx[i]; //Keep it this way for the mome
      //TODO: Removing the 0.5 consider whole elements
      xsend_max[i] = (iswap == 0) ? xmin[i] + halo->dx[i] : xmax[i] + 0.5 * BIG;


      int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                          xrecv_act_min, xrecv_act_max);

      if (a1 == 1) {
        for (int j = 0; j < dim; j++) {
        //  xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
        //  xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
          if (i != j) {
            xsend_min[j] = (fabs(xsend_min[j] - xrecv_act_min[j]) < 1.e-9) ?
                            -BIG : MAX(xsend_min[j], xrecv_act_min[j]);
            xsend_max[j] = (fabs(xsend_max[j] - xrecv_act_max[j]) < 1.e-9) ?
                             BIG : MIN(xsend_max[j], xrecv_act_max[j]);
           }
        }

        halo->sendBox = new BoundingBox(dim);
        halo->sendBox->setBoundingBoxLocalBound(xsend_min, xsend_max);
        goto endline;
      }

    }
  }


  endline:
  if (halo->sendBox == nullptr) {
    for (int i = 0; i < 2 * dim; i++)
      halo->isend[i] = 0;

    return;
  }
  else { //Finding grid points
    int size[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++)
      size[i] = (halo->particle_from->map_list[0] != nullptr) ?
          halo->particle_from->map_list[0]->binhead->size[i] : 0;

    //Get xmin.x of sending border
    ops_point xmin, xmax, xmin_send, xmax_send;

    double dx[OPS_MAX_DIM];
    for (int i = 0; i < halo->particle_from->block->dims; i++) dx[i] =
        halo->particle_from->map_list[0]->dx[i];

    xmin = halo->particle_from->box_block->getLocalMin();
    xmax = halo->particle_from->box_block->getLocalMax();
    xmin_send = halo->sendBox->getLocalMin();
    xmax_send = halo->sendBox->getLocalMax();

    //TODO: Replacd dx with map dx
    if (halo->particle_from->map_list[0]->mapping_type == OPS_WITH_VIRTUAL) {
      xmin.x += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[0]) * dx[0];
      xmax.x += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_p[0]) * dx[0];
      xmin.y += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[1]) * dx[1];
      xmax.y += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_p[1]) * dx[1];

      if (halo->particle_from->block->dims == 3)  {
        xmin.z += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[2]) * dx[2];
        xmax.z += static_cast<double>(halo->particle_from->map_list[0]->binhead->d_m[2]) * dx[2];
      }
    }

    halo->isend[0] = (xmin_send.x < xmin.x) ? 0 : (int) floor((xmin_send.x - xmin.x) / dx[0]);
    halo->isend[1] = (xmax_send.x > xmax.x) ? size[0] : (int) ceil((xmax_send.x - xmin.x) / dx[0]); //Sanity check

    halo->isend[2] = (xmin_send.y < xmin.y) ? 0 : (int ) floor((xmin_send.y - xmin.y) / dx[1]);
    halo->isend[3] = (xmax_send.y > xmax.y) ? size[1] : (int ) ceil((xmax_send.y - xmin.y) / dx[1]);

    halo->isend[4] = 0;
    halo->isend[5] = 1;
    if (dim == 3) {
      halo->isend[4] = (xmin_send.z < xmin.z) ? 0 : (int ) floor((xmin_send.z - xmin.z) / dx[2]);
      halo->isend[5] = (xmax_send.z > xmax.z) ? size[2] : (int ) ceil((xmax_send.z - xmin.z) / dx[2]);
    }
  }




}

void  _ops_particle_set_exchange_zone(OPS_instance *instance,
                                      ops_particle_halo halo) {

  BoundingBox *sendingBox = halo->particle_from->box_block;
  BoundingBox *recvBox = halo->particle_to->box_block;

  if (!sendingBox->getOwnership() || !recvBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "Particle Halo exchange must be set after"
                                          " block boxes are set.");

  int dim = sendingBox->getDim();
  double xmin[dim], xmax[dim], xsend_min[dim], xsend_max[dim];
  double x_recv_min[dim], x_recv_max[dim];
  recvBox->getLocalMaxMin(x_recv_min, x_recv_max);
  sendingBox->getLocalMaxMin(xmin, xmax);

  /* Shift bounding box of receiving block to the local coordinate system of
   * sending block */
  double xrecv_act_min[dim], xrecv_act_max[dim];
  for (int i = 0; i < dim ; i++) {
    int irecv_dir =  halo->dir_to[i];
    int isend_dir =  halo->dir_from[i];
    xrecv_act_min[isend_dir] = x_recv_min[irecv_dir] - halo->translate[isend_dir];
    xrecv_act_max[isend_dir] = x_recv_max[irecv_dir] - halo->translate[isend_dir];
  }

 /* Checking for intersection region */
  int intersect = ops_check_box_intersection(dim, xmin, xmax,
                                            xrecv_act_min, xrecv_act_max);

  if (intersect == 1) {
    for (int i = 0; i < dim; i++) {
      for (int iswap = 0; iswap < 2; iswap++) {
        for (int j = 0; j < dim ; j++) {
          xsend_min[j] = xmin[j];
          xsend_max[j] = xmax[j];
        }

        xsend_min[i] = (iswap == 0) ? xmin[i] : xmax[i] - 0.5 * std::numeric_limits<double>::max();
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * std::numeric_limits<double>::max();


        xsend_min[i] = (iswap == 0) ? xmin[i] - 0.5 * BIG : xmax[i];
        xsend_max[i] = (iswap == 0) ? xmin[i] : xmax[i] + 0.5 * BIG;

        int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                            xrecv_act_min, xrecv_act_max);


        if (a1 == 1) {

        //Handling multi-block case
          for (int j = 0; j < dim; j++) {
            if (i != j) {
              xsend_min[j] = (fabs(xsend_min[j] - xrecv_act_min[j]) < 1.e-9) ?
                  -BIG : MAX(xsend_min[j], xrecv_act_min[j]);
              xsend_max[j] = (fabs(xsend_max[j] - xrecv_act_max[j]) < 1.e-9) ?
                  BIG : MIN(xsend_max[j], xrecv_act_max[j]);
            }
            //xsend_min[j] = (i == j) ? xsend_min[j] : -BIG;
            //xsend_max[j] = (i == j) ? xsend_max[j] : BIG;
          }

          halo->sendBox = new BoundingBox(dim);
          halo->sendBox->setBoundingBoxLocalBound(xsend_min, xsend_max);
          goto endline;
        }
      }
    }
  }

  endline:
  return;

  //TODO:Add bites for the particle_halo
}

void _ops_particle_verify_exchange_zone(OPS_instance *instance,
                                        ops_particle_halo halo) {

  //Set local Box
  halo->sendBox->setBoundingBoxLocalBound(halo->sendBox->getGlobalMin(), halo->sendBox->getGlobalMax());

  //Ensure that sendBox intersects local box.

}


void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {

//    printf("Halo %d\n", ihalo);
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (halo->sendBox == nullptr)
     _ops_particle_set_exchange_zone(instance, halo);
    else
      _ops_particle_verify_exchange_zone(instance, halo);

    if (halo->sendBox != nullptr) {
      printf("Particle halo %d: SendingBox = [%e %e]x[%e %e]x[%e %e]\n", ihalo,
             halo->sendBox->getLocalMin().x,
             halo->sendBox->getLocalMax().x, halo->sendBox->getLocalMin().y,
             halo->sendBox->getLocalMax().y, halo->sendBox->getLocalMin().z,
             halo->sendBox->getLocalMax().z);
    }
  }
}


void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  /* Loop over all halos to define exchange zone */
  for (int ihalos = 0; ihalos < halo_grp->nhalos; ihalos++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalos];
    _ops_particle_set_exchange_border_zone(instance, halo);

    if (halo->sendBox != nullptr) {
      printf("Sending Box = [%e %e]x[%e %e] x[%e %e] SendingPoints = [%d %d]x[%d %d]x[%d %d]\n", halo->sendBox->getLocalMin().x,
             halo->sendBox->getLocalMax().x, halo->sendBox->getLocalMin().y,
             halo->sendBox->getLocalMax().y, halo->sendBox->getLocalMin().z,
             halo->sendBox->getLocalMax().z, halo->isend[0], halo->isend[1], halo->isend[2], halo->isend[3],
             halo->isend[4], halo->isend[5]);

    }
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


void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                        ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    double *xlocal = (double *) particle_from->particle_pos_dat->data;
    BoundingBox *box = halo->sendBox;

    if (box == nullptr) continue;

    info->nsend = 0;

    for (int  i = 0; i < particle_from->no_particles; i++) {
      if (particle_from->mark_deletion[i] != 1) continue;

      bool isin = box->isCoordinateInBoundingBox(xlocal + dim * i);

      info->nsend += ((int) isin);
    }

    //Part II: Reallocate sending particles
    if (info->nsend > info->nmax) {
      info->sendlist = (int *) ops_realloc(info->sendlist, sizeof(int) * (info->nsend + 100));
      info->nmax += info->nsend + 100;
    }

    //Part III: Find sending particles
    int nsend = 0;
    for (int i = 0; i < particle_from->no_particles; i++) {
      bool isin =  box->isCoordinateInBoundingBox(xlocal + dim * i);
      if (isin) {
        if (particle_from->mark_deletion[i] != 1) continue;
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
                                      halo->translate,  &ntot_bites);
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

    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    BoundingBox *box = halo->sendBox;

    if (box == nullptr) continue;

    info->nsend = 0;

    for (int i = 0; i < particle_from->no_particles; i++) {
      if (particle_from->mark_deletion[i] != 1) continue;

      bool isin = box->isCoordinateInBoundingBox(xlocal + dim * i);
      info->nsend += ((int) isin);
    }

    if (info->nsend > info->nmax) {
      info->sendlist = (int *) ops_realloc(info->sendlist, sizeof(int) * (info->nsend + 100));
      info->nmax += info->nsend + 100;
    }

    //Identify exchange particles
    int nsend = 0;
    for (int i = 0; i < particle_from->no_particles; i++) {
      bool isin = box->isCoordinateInBoundingBox(xlocal + dim * i);
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
    size_t nlocal = from->no_particles;

    int *binhead = (int *)from->map_list[0]->binhead->data;
    int *bins = (int *)from->map_list[0]->bin->data;
    ops_dat binheads = from->map_list[0]->binhead;
    double *xpos = (double *)from->particle_pos_dat->data;

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
    int nrecv = info->nrecv = info->nsend;

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

    BoundingBox *box = halo->sendBox;
    info->nsend = 0;

    //Part I: Find the number of particles with the structure
    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    for (int i = imin; i < imax; i++) {
      bool decide = box->isCoordinateInBoundingBox(xlocal + dim * i);
      info->nsend += ((int) decide);
    }

    if (info->nsend >= info->nmax)  {
      info->sendlist = (int *) ops_realloc(info->sendlist, (info->nmax + 100) * sizeof(int));
      info->nmax = info->nsend + 100;
    }

    //Part II: Find particles to exchange
    int nsend = 0;
    for (int i = imin; i < imax; i++) {
      bool decide = box->isCoordinateInBoundingBox(xlocal + dim * i);
      info->sendlist[nsend] = i;
      nsend++;
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
    int nrecv = info->nrecv = info->nsend;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;

    particle_to->no_virtual += info->nrecv;

    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles
                                           + particle_to->no_virtual);
    }

    _ops_particle_halo_copy_from_buff(instance->ops_halo_buffer, halo->dat,
                                      halo->nhalos, info, 1,
                                      halo->dir_to, halo->dir_from,
                                      halo->translate, &ntot_bites);
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

    ops_particle_halo halo_main = main_halo_grp->halo_list[ihalo];

    ops_particle_halo_exchange info = main_halo_grp->halo_info[ihalo];

    /* Packing data */
    int nsend = info->nsend;
    int nbites_tot = nsend * halo->nbites;
    int *sendlist = info->sendlist;
    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff=(char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }

    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 0 ,&ntot_bites);

    /* Unpacking data */
    int istart = info->firstrecv;
    int nrecv = info->nrecv;

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
    int *sendlist = info->sendlist;

    if (nbites_tot > instance->ops_halo_buffer_size) {
      instance->ops_halo_buffer_size = info->nmax * halo->nbites;
      buff = (char *)ops_realloc(buff, instance->ops_halo_buffer_size);
    }


    int ntot_bites = 0;
    _ops_particle_halo_copy_tobuf(instance->ops_halo_buffer, halo->dat,
                                  halo->nhalos, info, 0 ,&ntot_bites);

    /* Unpacking data  */
    int istart = info->firstrecv;
    int nrecv = info->nrecv;
    int ilast = istart + nrecv;


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

  char *buff = instance->ops_halo_buffer;

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

  if (map->grid == nullptr) return;

  for (int i = 0; i < map->particle->block->dims; i++) {
    int d_m = map->binhead->d_m[i];
    int d_p = map->binhead->d_p[i];
    map->dx[i] =
        (map->particle->box_block->getMaxCoordDir(i) - map->particle->box_block->getMinCoordDir(i))
        / static_cast<double>(map->binhead->size[i] + d_m - d_p);
  }

}

void ops_build_bounding_box(ops_particle particle ) {

  particle->box_block->partitionBoundingBox(particle->block);
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

  if (fprintf(fp, "Number of particles: %d   Block: [%f %f] x [%f %f]",
              particle->no_particles, particle->box_block->getLocalMin().x,
              particle->box_block->getLocalMax().x, particle->box_block->getLocalMin().y,
              particle->box_block->getLocalMax().y) < 0) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
  }

  if (particle->block->dims == 3) {
    if (fprintf(fp, "x [%f %f]\n", particle->box_block->getLocalMin().z,
                particle->box_block->getLocalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  }
  else
    if (fprintf(fp, "\n") < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");


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
  for (int i = 0; i < particle->no_particles; i++) {
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

  if (fprintf(fp, "Block:   %s   [%f %f] x [%f %f]", particle->block->name,
              particle->box_block->getLocalMin().x, particle->box_block->getLocalMax().x,
              particle->box_block->getLocalMin().y, particle->box_block->getLocalMax().y) < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (particle->block->dims == 3)
   if (fprintf(fp, " x[%f %f]\n", particle->box_block->getLocalMin().z,
                particle->box_block->getLocalMax().z) < 0)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  if (fprintf(fp, "\n") < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  //Write titles
  for (int i = 0; i < ndats; i++) {
    for (int isou = 0; isou < dats[i]->dim; isou++) {
      if (fprintf(fp, "%14s[%d] ", dats[i]->name, isou) < 0)
        throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");
    }
  }

  if (fprintf(fp, "\n")  < 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Writing to file\n");

  for (int i = 0; i < particle->no_particles; i++) {
    for (int idat = 0; idat < ndats; idat++)
      _ops_particle_export_data_point(fp, i, dats[i]);

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

void _ops_mapping_set_structures(ops_particle particle, double skin[], int  d_m[],
                                 int d_p[], int d_mb[], int d_pb[], int size[],
                                 double dx_map[],  ops_with_virtual &include_virtual) {


  if (particle->box_block->getBlockVolume() < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle projection\n");

//Sanity checks for skin
  for (int i = 0; i < particle->block->dims; i++)
    if (skin[i] < DBL_EPSILON)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Particles cannot be projected to "
                                          "cells of non-positive dimensions");

  //Finding particle per direction
  for (int i = 0; i < particle->block->dims; i++) {
    double length = particle->box_block->getGlobalMax(i) - particle->box_block->getGlobalMin(i);
    int icells = floor(length / skin[i]);
    size[i] = (icells > 0) ? icells : 1;
    dx_map[i] = (length) / static_cast<double>(size[i]);

    d_mb[i] = (d_m != nullptr) ? MIN(d_m[i], 0) : 0;
    d_pb[i] = (d_p != nullptr) ? MAX(d_m[i], 0) : 0;
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) {
    size[i] = 1;
    dx_map[i] = 0.0;
    d_mb[i] = d_pb[i] = 0;
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

void ops_particle_map_get_dx(ops_particle_mapping map, double dx[]) {

  for (int i = 0; i < map->particle->block->dims; i++) dx[i] = map->dx[i];

}
