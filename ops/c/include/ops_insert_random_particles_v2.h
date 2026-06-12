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
  * @brief Functionalities for inserting random particles into simulations
  * @author Valantis Tsinginos
  * @details Inserting random particles to simulations in CPU. Then elemens
  *          copied to different packends
  */

#ifndef __OPS_INSERT_RANDOM_PARTICLES_V2_H_
#define __OPS_INSERT_RANDOM_PARTICLES_V2_H_

#include "ops_lib_core.h"
#include "ops_distributions.h"
#include <limits>

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include "ops_mpi_particle_core.h"

#include "ops_particle_divide.h"

double *candidate_parts;
int *flag_to_add;

int ncand_actual;
int ncand_virtual;
int ncand_max = OPS_MAX_PART;


extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;
extern int ops_buffer_send_1_size;
extern int ops_buffer_recv_1_size;
extern int ops_buffer_send_2_size;
extern int ops_buffer_recv_2_size;

int  *forward_candids[2 * OPS_MAX_DIM];
int  nforward_candids[2 * OPS_MAX_DIM];
int  irecv_candids[2 * OPS_MAX_DIM];
int  nrecv_candids[2 * OPS_MAX_DIM];

double *xvirtual;
double *rad_virtual;

int *bin_virtual;

int  nforward_max[2 * OPS_MAX_DIM];
#endif


int round_flags =0;

template <template<typename Elem> class Distribution, typename T>
struct OPS_dat_distr {
  OPSDistribution<Distribution, T> *distribution;  /**< OPSDistribution structure for populating
                                                        ops_dat structure */
  ops_dat dat;                                     /**< ops_dat that will be initialized based on
                                                        the OPSDistribution structure */
};

template <template<typename> class Distribution, typename T>
OPS_dat_distr<Distribution, T> ops_dat_distr(ops_dat dat, T* params, T* limit) {

  if (dat == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT,"Empty ops_dat structure");

  if (sizeof(T) != dat->type_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "The size of T differs from the size of the elements "
                                             "of the ops_dat structure\n");
  OPSDistribution<Distribution, T> *distrib =
      ops_declaire_distribution<Distribution, T>(dat->dim, params, limit);

  OPS_dat_distr<Distribution, T> dat_distr;
  dat_distr.distribution = distrib;
  dat_distr.dat = dat;

  return dat_distr;
}

template <template<typename Elem> class Distribution, typename T>
OPS_dat_distr<Distribution, T> ops_dat_distr(ops_dat dat,
                                             OPSDistribution<Distribution, T> *distr) {

  if (dat == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT,"Error: Empty ops_dat structure");

  if (distr == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty OPSDistribution structure");

  OPS_dat_distr<Distribution, T> dat_distr;

  if (dat->dim != distr->dims)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Incompatible sizes of the distribution "
                                              " and ops_dat structures.");

  dat_distr.distribution = distr;
  dat_distr.dat = dat;
  return dat_distr;

}

template <template<typename Elem> class Distribution, typename T>
void insert_random_dat(char *data, OPSDistribution<Distribution, T>  *dist,
                       std::default_random_engine &dre, int i) {

  T* array = (T *)data;
  int dim = dist->dims;

  T local_point[dim];
  ops_generate_random_point(dist, dre, local_point);

  for (int isou = 0; isou < dim; isou++)
    array[isou + i * dim] = local_point[isou];
}

inline double _compute_volume_normal(const int idir, const int dim, const int *flag_need_send,
                                     const double region_insert[], const double dx[]) {

  double volume = 1.0;
  for (int isou = 0; isou < dim; isou++) {
    if (isou == idir) {

      int flag_neg = (flag_need_send[0] > 0) ? 1 : 0;
      int flag_pos = (flag_need_send[1] > 0) ? 1 : 0;
      double flag_need = static_cast<double>(flag_neg + flag_pos);
      volume *= flag_need * dx[isou];
    }
    else {
      if (region_insert[2 * isou] >= region_insert[2 * isou + 1]) return 0.0;
      volume *= (region_insert[2 * isou + 1] - region_insert[2 * isou]);
    }
  }

  return volume;

}
inline int edge_product(const int *size, int dim) {

  int products[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) products[i] = 0;
  int max_prod = INT_MIN;

  int prefix = 1;
  for (int i = 0; i < dim; i++) {
    products[i] = prefix;
    prefix *= size[i];
  }

  int suffix = 1;
  for (int i = dim -1; i>=0; i--) {
    products[i] *= suffix;
    suffix *= size[i];
  }

  for (int i = 0; i < dim; i++)
    max_prod = MAX(max_prod, products[i]);

  return max_prod;
}

#ifdef OPS_MPI
inline void shift_virtual_to_local(sub_block *sb, ops_particle particle, ops_dat envelope,
                                   int  *binhead, int *bin,
                                   const int *size, const int *need_send, const int *need_recv,
                                   int  &nvirtual, int &nvirtual_max, int nexpected, const ops_point xmin,
                                   const double dx[]) {

  int dim = particle->block->dims;

  int send_point_pos[OPS_MAX_DIM * 2];
  int send_point_neg[OPS_MAX_DIM * 2];

  int prod = edge_product(size, dim);

  int *forward_neg = (int *) ops_malloc(sizeof(int) * prod);
  int *forward_pos = (int *) ops_malloc(sizeof(int) * prod);
  int nmax_neg = prod;
  int nmax_pos = prod;

  ops_dat dats[2] = {particle->particle_pos_dat, envelope};
  double *arrays[2] = {xvirtual, rad_virtual};
  int array_size[2] = {dim, 1};

  for (int idirs = 0; idirs < dim; idirs++) {

    //Get limits
    for (int isou = 0; isou < dim; isou++) {
      send_point_neg[2 * isou] = 0; //by default
      send_point_neg[2 * isou + 1] = (idirs == isou) ? 2 : size[isou];

      send_point_pos[2 * isou] = (idirs == isou) ? size[isou] - 2 : 0;
      send_point_pos[2 * isou + 1] = size[isou];
    }
    for (int isou = dim ; isou < OPS_MAX_DIM; isou++) {
      send_point_neg[2 * isou] = 0;
      send_point_neg[2 * isou + 1] = 1;
      send_point_pos[2 * isou ] = 0;
      send_point_pos[2 * isou + 1] = 1;
    }


    int nforward_neg = 0;
    int nforward_pos = 0;

    if (need_send[ 2 * idirs] > -1) {
      for (int iz = send_point_neg[2 * 2]; iz < send_point_neg[2 * 2 + 1]; iz++) {
        for (int iy = send_point_neg[2 * 1]; iy < send_point_neg[2 * 1 + 1]; iy++) {
          for (int ix = send_point_neg[0]; ix < send_point_neg[1]; ix++)  {
            int address = ix + iy * size[0] + iz * size[0] * size[1];

            int ipart = binhead[address];
            while (ipart != -1) {
              nforward_neg++;
              ipart = (ipart >= nexpected) ? bin_virtual[ipart - nexpected] : bin[ipart];
            }
          }
        }
      }
    }



    if (need_send[2 * idirs + 1] > - 1) {
      for (int iz = send_point_pos[4]; iz < send_point_pos[5]; iz++) {
        for (int iy = send_point_pos[2]; iy < send_point_pos[3]; iy++) {
          for (int ix = send_point_pos[0]; ix < send_point_pos[1]; ix++) {
            int address = ix + iy * size[0] + iz * size[0] * size[1];
            int ipart = binhead[address];

            while (ipart != -1) {
              nforward_pos++;
              ipart = (ipart >= nexpected) ? bin_virtual[ipart - nexpected] : bin[ipart];
            }
          }
        }
      }
    }

    //Send and receive number of particles
    int nrecv_pos = 0;
    int nrecv_neg = 0;
    MPI_Status status[4];
    MPI_Sendrecv(&nforward_neg, 1, MPI_INT, need_send[2 * idirs] > -1 ?
                                            sb->id_m[idirs] : MPI_PROC_NULL, 100,
                 &nrecv_pos, 1, MPI_INT, need_recv[2 * idirs + 1] > -1 ?
                                         sb->id_p[idirs] : MPI_PROC_NULL, 100,
                 sb->comm, &status[0]);

    MPI_Sendrecv(&nforward_pos, 1, MPI_INT, need_send[2 * idirs + 1] > -1 ?
                                            sb->id_p[idirs] : MPI_PROC_NULL, 200,
                 &nrecv_neg, 1, MPI_INT, need_recv[2 * idirs] > -1 ?
                                         sb->id_m[idirs] : MPI_PROC_NULL, 200,
                 sb->comm, &status[1]);

    //Generate maps
    if (nforward_neg > nmax_neg) {
      forward_neg = (int *) ops_realloc(forward_neg, (nforward_neg + OPS_MAX_PART) * sizeof(int));
      nmax_neg = nforward_neg + OPS_MAX_PART;

    }

    if (nforward_pos > nmax_pos) {
      forward_pos = (int *) ops_realloc(forward_pos, (nforward_pos + OPS_MAX_PART) * sizeof(int));
      nmax_pos = nforward_pos + OPS_MAX_PART;
    }


    if (need_send[ 2 * idirs] > -1) {
      for (int iz = send_point_neg[2 * 2]; iz < send_point_neg[2 * 2 + 1]; iz++) {
        for (int iy = send_point_neg[2 * 1]; iy < send_point_neg[2 * 1 + 1]; iy++) {
          for (int ix = send_point_neg[0]; ix < send_point_neg[1]; ix++)  {
            int address = ix + iy * size[0] + iz * size[0] * size[1];

            int ipart = binhead[address];
            while (ipart != -1) {
              forward_neg[nforward_neg] = ipart;
              nforward_neg++;
              ipart = (ipart >= nexpected) ? bin_virtual[ipart - nexpected] : bin[ipart];
            }
          }
        }
      }
    }

    if (need_send[2 * idirs + 1] > - 1) {
      for (int iz = send_point_pos[4]; iz < send_point_pos[5]; iz++) {
        for (int iy = send_point_pos[2]; iy < send_point_pos[3]; iy++) {
          for (int ix = send_point_pos[0]; ix < send_point_pos[1]; ix++) {
            int address = ix + iy * size[0] + iz * size[0] * size[1];
            int ipart = binhead[address];

            while (ipart != -1) {
              forward_pos[nforward_pos] = ipart;
              nforward_pos++;
              ipart = (ipart >= nexpected) ? bin_virtual[ipart - nexpected] : bin[ipart];
            }
          }
        }
      }
    }

    int size_send = particle->particle_pos_dat->elem_size +
                    envelope->elem_size;


    if (nforward_neg * size_send >  ops_buffer_send_1_size) {
      ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                           nforward_neg * size_send);
      ops_buffer_send_1_size = nforward_neg * size_send;
    }


    if (nforward_pos * size_send > ops_buffer_send_2_size) {
      ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                           nforward_pos * size_send);
      ops_buffer_send_2_size = nforward_pos * size_send;
    }


    // Pack Data to different buffers
    int ishift_pos_buff = 0;
    int ishift_neg_buff = 0;




    for (int idat = 0; idat < 2; idat++) {

      if (array_size[idat] * sizeof(double) != dats[idat]->elem_size)
        throw OPSException(OPS_RUNTIME_ERROR, "ERROR Size of array differs from dat size\n");

      _ops_particle_intra_dat_array_to_buff(ops_buffer_send_1 + ishift_neg_buff,
                                            dats[idat], arrays[idat], forward_neg, nforward_neg,
                                            array_size[idat], nexpected);
      ishift_neg_buff += nforward_neg * dats[idat]->elem_size ;
      _ops_particle_intra_dat_array_to_buff(ops_buffer_send_2 + ishift_pos_buff,
                                            dats[idat], arrays[idat], forward_pos, nforward_pos,
                                            array_size[idat], nexpected);
      ishift_pos_buff += nforward_pos * dats[idat]->elem_size;
    }

    if (nrecv_neg * size_send > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                                nrecv_neg * size_send);
      ops_buffer_recv_2_size = size_send * nrecv_neg;
    }

    if (nrecv_pos * size_send > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                           nrecv_pos * size_send);
      ops_buffer_recv_1_size = size_send * nrecv_pos;
    }

    //Send and receiving data
    MPI_Request request[4];


    int sending_bytes = nforward_neg * size_send;

    MPI_Isend(ops_buffer_send_1, sending_bytes, MPI_BYTE,
              sending_bytes > 0 ? sb->id_m[idirs] : MPI_PROC_NULL,
              idirs, sb->comm, &request[0]);

    sending_bytes = size_send * nforward_pos;

    MPI_Isend(ops_buffer_send_2, sending_bytes, MPI_BYTE,
              sending_bytes > 0 ? sb->id_p[idirs] : MPI_PROC_NULL,
              idirs + OPS_MAX_DIM, sb->comm, &request[1]);

    int receiving_bytes = size_send * nrecv_pos;

    MPI_Irecv(ops_buffer_recv_1, receiving_bytes, MPI_BYTE,
              receiving_bytes > 0 ? sb->id_p[idirs] : MPI_PROC_NULL,
              idirs, sb->comm, &request[2]);

    receiving_bytes = size_send * nrecv_neg;

    MPI_Irecv(ops_buffer_recv_2, receiving_bytes, MPI_BYTE,
              receiving_bytes > 0 ? sb->id_m[idirs] : MPI_PROC_NULL,
              idirs + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);

    //Set receiving structures
    int irecv_neg = nvirtual;
    int irecv_pos = nvirtual + nrecv_neg;

    if (nvirtual + nrecv_pos + nrecv_neg > nvirtual_max) {
      xvirtual = (double *) ops_realloc(xvirtual, sizeof(double) * dim
                                        * (nvirtual + nrecv_pos + nrecv_neg + OPS_MAX_PART));
      rad_virtual = (double *) ops_realloc(rad_virtual, sizeof(double)
                                        * (nvirtual + nrecv_pos + nrecv_neg + OPS_MAX_PART));
      bin_virtual = (int *) ops_realloc(bin_virtual, sizeof(int) * (nvirtual + nrecv_neg + nrecv_pos + OPS_MAX_PART));
      nvirtual_max = nvirtual + nrecv_pos + nrecv_neg + OPS_MAX_PART;
    }

    if (need_recv[2 * idirs] > -1) {
      _ops_particle_intra_buff_to_array(ops_buffer_recv_2, xvirtual,
                                        irecv_neg, nrecv_neg, dim);
      int irecv_shift_neg  = nrecv_neg * dim * sizeof(double);
      _ops_particle_intra_buff_to_array(ops_buffer_recv_2 + irecv_shift_neg,
                                      rad_virtual,
                                      irecv_neg, nrecv_neg, 1);
    }
    //Unmap positive

    if (need_recv[2 * idirs + 1] > -1) {
       _ops_particle_intra_buff_to_array(ops_buffer_recv_1, xvirtual,
                                         irecv_pos, nrecv_pos, dim);
       int irecv_shift_pos = nrecv_pos * dim * sizeof(double);
       _ops_particle_intra_buff_to_array(ops_buffer_recv_1 + irecv_shift_pos,
                                         rad_virtual, irecv_pos, nrecv_pos, 1);
    }


    nvirtual += nrecv_pos + nrecv_neg;
    MPI_Waitall(2, &request[0], &status[0]);

    int ix[OPS_MAX_DIM];
    int d_m[OPS_MAX_DIM];
    for (int isou = 0;  isou < dim; isou++) d_m[isou] = -1;
    for (int isou = dim; isou < OPS_MAX_DIM; isou++) d_m[isou] = 0;
    for (int ip = irecv_neg; ip < nvirtual; ip++) {
      ix[0] = (int) ops_floor((xvirtual[dim * ip] - xmin.x) / dx[0]);
      ix[1] = (int) ops_floor((xvirtual[dim * ip + 1] - xmin.y) / dx[1]);
      ix[2] = (dim == 3) ? (int) ops_floor((xvirtual[dim * ip + 2] - xmin.z) / dx[2]) : 0;

      int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                  + (ix[2] - d_m[2]) * size[0] * size[1];
      bin_virtual[ip] = binhead[address];
      binhead[address] = ip + nexpected;
    }

  }



  ops_free(forward_neg);
  ops_free(forward_pos);

}

inline int _check_distance_with_bin_border(double *xp, double Rp, double  *candidate_particles,
                                           const int dim, const int nactual) {

  for (int ip = 0; ip < nactual; ip++) {
    double d = 0.0;


    for (int isou = 0; isou < dim; isou++)
      d += (candidate_particles[(dim + 1) * ip + isou] - xp[isou])
         * (candidate_particles[(dim + 1) * ip + isou] - xp[isou]);

    if (d < 1.01 * (Rp + candidate_particles[(dim + 1) * ip + dim])
      * (Rp + candidate_particles[(dim + 1) * ip + dim]))
      return -1;
  }
  return 0;
}

inline int _check_distance_with_actual(const int limits[], const int dim,
                                        const int *binhead, const int size[],
                                        const int dm[], const int *bins,
                                        const ops_dat coords,const ops_dat envelope,
                                        const double xp[], const double Rp) {

#ifdef OPS_3D
  for (int k = limits[4]; k<= limits[5]; k++)
#else
  int k = 0;
#endif
  {
    for (int j = limits[2]; j <= limits[3]; j++) {
      for (int i = limits[0]; i <= limits[1]; i++) {
        int address = (i - dm[0]) + (j - dm[1]) * size[0]
                    + (k - dm[2]) * size[0] * size[1];
        int ineigh = binhead[address];
        while (ineigh != - 1) {
          double d = 0.0;
          for (int isou = 0; isou < dim; isou++)
            d+= (xp[isou] - ((double *) coords->data)[dim * ineigh + isou])
              * (xp[isou] - ((double *) coords->data)[dim * ineigh + isou]);

          double rad = Rp + ((double *)envelope->data)[ineigh];

          if (d < 1.01 * rad * rad)
            return -1;
          ineigh = bins[ineigh];
        }
      }
    }
  }

  return 0;
}

inline void _exchange_candidate_parts(sub_block *sb, ops_dat coords, ops_dat envelope,
                                      const int dim,const int  *flag_need_send,
                                      const int  *flag_need_recv, const ops_point xmin,
                                      const ops_point xmax, const  double dx[],
                                      const int *binhead, const int *size, const int *dm,
                                      const int *dp, const int *bins) {

  double xregion[2 * OPS_MAX_DIM] = {};
  xregion[0] = xmin.x; xregion[1] = xmax.x;
  xregion[2] = xmin.y; xregion[3] = xmax.y;
  xregion[4] = (dim == 3) ? xmin.z : 0;
  xregion[5] = (dim == 3) ? xmax.z : 0;

  for (int idir = 0; idir < dim; idir++) {
    //Start looking for packing in x and y direction
    nforward_candids[2 * idir] = 0;
    nforward_candids[2 * idir + 1]  = 0;

    for (int ip = 0; ip < ncand_actual + ncand_virtual; ip++) {

      if (flag_need_send[2 * idir] > - 1) {
        if (candidate_parts[(dim + 1) * ip + idir] < xregion[2 * idir] + dx[idir]
            && flag_to_add[ip] == 0) {
          nforward_candids[2 * idir]++;
          if (nforward_candids[2 * idir] >= nforward_max[2 * idir]) {
            nforward_max[2 * idir] = nforward_candids[2 * idir] + 10;
            forward_candids[2 * idir] =
                (int *) ops_realloc(forward_candids[2 * idir], sizeof(int)
                                    * nforward_max[2 * idir]);
          }

          forward_candids[2 * idir][nforward_candids[2 * idir]-1] = ip;
        }
      }

      if (flag_need_send[2 * idir + 1] > - 1) {

        if (candidate_parts[(dim + 1) * ip + idir] > xregion[2 * idir + 1] - dx[idir]
            && flag_to_add[ip] == 0) {
          nforward_candids[2 * idir + 1]++;
          //Reallocate if necessary
          if (nforward_candids[2 * idir + 1] >= nforward_max[2 * idir + 1]) {
            nforward_max[2 * idir + 1] = nforward_candids[2 * idir + 1] + 10;
            forward_candids[2 * idir + 1] =
                (int *) ops_realloc(forward_candids[2 * idir + 1], sizeof(int)
                                    * nforward_max[2 * idir + 1]);
          }

          forward_candids[2 * idir + 1][nforward_candids[2 * idir + 1] - 1]  = ip;
        }
      }
    }


    int size_send = sizeof(double) * (dim + 1);
    if (nforward_candids[2 * idir] * size_send >  ops_buffer_send_1_size) {
      ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                           nforward_candids[2 * idir] * size_send);
      ops_buffer_send_1_size = nforward_candids[2 * idir] * size_send;
    }

    if (nforward_candids[2 * idir + 1] * size_send > ops_buffer_send_2_size) {
      ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                           nforward_candids[2 * idir + 1] * size_send);
      ops_buffer_send_2_size = nforward_candids[2 * idir + 1] * size_send;
    }

    //Pack particle before sending TODO:!!!
    ops_particle_intra_array_to_buff(ops_buffer_send_1, candidate_parts, dim + 1,
                                     forward_candids[2 * idir], nforward_candids[2 * idir]);
    ops_particle_intra_array_to_buff(ops_buffer_send_2, candidate_parts, dim + 1,
                                     forward_candids[2 * idir + 1],
                                     nforward_candids[2 * idir + 1]);//TODO

    //Set receive in direction
    nrecv_candids[2 * idir] = 0;
    nrecv_candids[2 * idir + 1] = 0;

    MPI_Status status[4];
    MPI_Sendrecv(&nforward_candids[2 * idir], 1, MPI_INT,
                 flag_need_send[2 * idir] > -1 ? sb->id_m[idir] : MPI_PROC_NULL, 300,
                 &nrecv_candids[2 * idir + 1 ], 1, MPI_INT,
                 flag_need_recv[2 * idir + 1] > -1 ? sb->id_p[idir] : MPI_PROC_NULL,  300,
                 sb->comm, &status[0]);

    MPI_Sendrecv(&nforward_candids[2 * idir + 1], 1, MPI_INT,
                 flag_need_send[2 * idir + 1] > -1 ? sb->id_p[idir] : MPI_PROC_NULL, 400,
                 &nrecv_candids[2 * idir], 1, MPI_INT,
                 flag_need_recv[2 * idir] > -1 ? sb->id_m[idir] : MPI_PROC_NULL, 400,
                 sb->comm, &status[0]);


    //Re allocate structures for receiving data
    if (nrecv_candids[2 * idir] * size_send > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_send_2_size,
                                           nrecv_candids[2 * idir] * size_send);
      ops_buffer_recv_2_size = size_send * nrecv_candids[2 * idir];
    }

    if (nrecv_candids[2 * idir + 1] * size_send > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                           nrecv_candids[2 * idir + 1] * size_send);
      ops_buffer_recv_1_size = size_send * nrecv_candids[2 * idir + 1];
    }

    //Get data structures
    MPI_Request request[4];
    int bytes_send = nforward_candids[2 * idir] * size_send;
    MPI_Isend(ops_buffer_send_1, bytes_send, MPI_BYTE,
              bytes_send > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    bytes_send = nforward_candids[2 * idir + 1] * size_send;
    MPI_Isend(ops_buffer_send_2, bytes_send, MPI_BYTE,
              bytes_send > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    int bytes_recv = nrecv_candids[2 * idir + 1] * size_send;
    int my_rank;

    MPI_Irecv(ops_buffer_recv_1, bytes_recv, MPI_BYTE,
              bytes_recv > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);

    bytes_recv = nrecv_candids[2 * idir] * size_send;

    MPI_Irecv(ops_buffer_recv_2, bytes_recv, MPI_BYTE,
              bytes_recv > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);


    //Set receiving end
    irecv_candids[2 * idir] = ncand_actual + ncand_virtual;
    irecv_candids[2 * idir + 1] = ncand_actual + ncand_virtual
                                + nrecv_candids[2 * idir];

    //Reallocate structures if necessary
    int ntot_cands = nrecv_candids[2 * idir] + nrecv_candids[2 * idir + 1]
                   + ncand_virtual + ncand_actual;
    if (ntot_cands > ncand_max) {
      int bytes_size = (ntot_cands + OPS_MAX_DIM) * (dim + 1) * sizeof(double);
      candidate_parts = (double *) ops_realloc(candidate_parts, bytes_size);
      bytes_size = (ntot_cands + OPS_MAX_DIM) * sizeof(int);
      flag_to_add = (int *) ops_realloc(flag_to_add, bytes_size);
    }

    //Unpack structures
    if (flag_need_recv[2 * idir] > -1) {
      _ops_particle_intra_buff_to_array(ops_buffer_recv_2, candidate_parts,
                                        irecv_candids[2 * idir],
                                        nrecv_candids[2 * idir], dim + 1);
    }

    if (flag_need_recv[2 * idir + 1] > -1) {
      _ops_particle_intra_buff_to_array(ops_buffer_recv_1, candidate_parts,
                                        irecv_candids[2 * idir + 1],
                                        nrecv_candids[2 * idir + 1], dim + 1);
    }


    ncand_virtual += nrecv_candids[2 * idir] + nrecv_candids[2 * idir + 1];
    MPI_Waitall(2, &request[0], &status[0]);


    //Loop to find if in contact
    int ix[OPS_MAX_DIM];
    for (int ip = irecv_candids[2 * idir]; ip < irecv_candids[2 * idir]
                                              + nrecv_candids[2 * idir]; ip++) {

      ix[0] = (int) ops_floor((candidate_parts[(dim + 1) * ip] - xmin.x) / dx[0]);
      ix[1] = (int) ops_floor((candidate_parts[(dim + 1) * ip + 1] - xmin.y) / dx[1]);
      ix[2] = (dim == 3) ?
          (int) ops_floor((candidate_parts[(dim + 1) * ip + 2] - xmin.z) / dx[2]) : 0;


      int limits[2 * OPS_MAX_DIM] = {};
      for (int isou = 0; isou < dim; isou++) {
        limits[2 * isou] = (isou == idir) ? 0 : ix[isou] - 1;
        limits[2 * isou + 1] = ix[isou] + 1;

        if (limits[2 * isou] < 0) limits[2 * isou] = 0;
        if (limits[2 * isou + 1] > size[isou] + dm[isou] - dp[isou] - 1)
          limits[2 * isou + 1] = size[isou] + dm[isou] - dp[isou] - 1;
      }



      double xp[OPS_MAX_DIM], Rp;
      for (int isou = 0; isou < dim; isou++)
        xp[isou] = candidate_parts[(dim + 1) * ip + isou];
      Rp = candidate_parts[(dim + 1) * ip + dim];
      flag_to_add[ip] = _check_distance_with_actual(limits, dim, binhead, size, dm, bins,
                                                    coords, envelope, xp, Rp);

      if (flag_to_add[ip] == 0) {
        //Check against actual in bin
        flag_to_add[ip] = _check_distance_with_bin_border(xp, Rp, candidate_parts,
                                                          dim, ncand_actual);
      }

    }

    //Loop to find if in contact coming in positive direction
    //TODO: Check limits
    for (int ip = irecv_candids[2 * idir + 1]; ip < irecv_candids[2 * idir + 1]
                                                  + nrecv_candids[2 * idir + 1];
         ip++) {
      ix[0] = (int) ops_floor((candidate_parts[(dim + 1) * ip] - xmin.x) / dx[0]);
      ix[1] = (int) ops_floor((candidate_parts[(dim + 1) * ip + 1] - xmin.y) / dx[1]);
      ix[2] = (dim == 3) ?
          (int) ops_floor((candidate_parts[(dim + 1) * ip + 2] - xmin.z) / dx[2]) : 0;

      int limits[2 * OPS_MAX_DIM] = {};
      for (int isou = 0; isou < dim; isou++) {
        limits[2 * isou] = ix[isou] - 1;
        limits[2 * isou + 1] = (isou == idir) ? size[isou] + dm[isou] -dp[isou] - 1 : ix[isou] + 1;

        //Sanity checks
        if (limits[2 * isou] < 0) limits[2 * isou] = 0;
        if (limits[2 * isou + 1] > size[isou] + dm[isou] - dp[isou] - 1)
          limits[2 * isou + 1] = size[isou] + dm[isou] - dp[isou] - 1;
      }



      double xp[OPS_MAX_DIM], Rp;
      for (int isou = 0; isou < dim; isou++)
        xp[isou] = candidate_parts[(dim + 1) * ip + isou];
      Rp = candidate_parts[(dim + 1) * ip + dim];
      flag_to_add[ip] = _check_distance_with_actual(limits, dim, binhead, size, dm, bins,
                                                    coords, envelope, xp, Rp);
      if (flag_to_add[ip] == 0) {
        //Check against actual in bin
        flag_to_add[ip] = _check_distance_with_bin_border(xp, Rp, candidate_parts,
                                                          dim, ncand_actual);
      }


    }


  }
}

inline void _reverse_operations(sub_block *sb, const int dim,int * flag_to_add) {

  //Check for allocation
  int elem_size = sizeof(int) * 1;

  int nmax_pos = 0;
  int nmax_neg = 0;
  int nmaxr_pos = 0;
  int nmaxr_neg = 0;

  for (int idir = 0; idir < dim; idir++) {
    nmaxr_pos = MAX(nmaxr_pos, nforward_candids[2 * idir + 1] );
    nmaxr_neg = MAX(nmaxr_neg, nforward_candids[2 * idir]);
    nmax_pos = MAX(nmax_pos, nrecv_candids[2 * idir + 1]);
    nmax_neg = MAX(nmax_neg, nrecv_candids[2 * idir]);
  }

  //Check for allocations
  if (ops_buffer_send_1_size < nmax_neg * elem_size) {
    ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                         nmax_neg * elem_size);
    ops_buffer_send_1_size = nmax_neg * elem_size;
  }

  if (ops_buffer_send_2_size < nmax_pos * elem_size) {
    ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                         nmax_pos * elem_size);
    ops_buffer_send_2_size = nmax_pos * elem_size;
  }

  if (ops_buffer_recv_2_size < nmaxr_neg * elem_size) {
    ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                         nmaxr_neg * elem_size);
    ops_buffer_recv_2_size = nmaxr_neg* elem_size;
  }

  for (int idir = dim - 1; idir >=0; idir--) {
    _ops_particle_intra_array_to_buff_reverse(ops_buffer_send_1, flag_to_add, 1,
                                              irecv_candids[2 * idir],
                                              nrecv_candids[2 * idir]); //TODO

    _ops_particle_intra_array_to_buff_reverse(ops_buffer_send_2, flag_to_add, 1,
                                              irecv_candids[2 * idir + 1],
                                              nrecv_candids[2 * idir + 1]); //TODO


    MPI_Status status[4];
    MPI_Request requests[4];

    int send_size = nrecv_candids[idir * 2] * sizeof(int);
    MPI_Isend(ops_buffer_send_1, send_size, MPI_BYTE,
              send_size > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &requests[0]);

    send_size = nrecv_candids[idir * 2 + 1] * sizeof(int);
    MPI_Isend(ops_buffer_send_2, send_size, MPI_BYTE,
              send_size > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &requests[1]);

    int recv_size = nforward_candids[idir * 2 + 1] * sizeof(int);
    MPI_Irecv(ops_buffer_recv_1, recv_size, MPI_BYTE,
              recv_size > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &requests[2]);

    recv_size = nforward_candids[idir * 2] * sizeof(int);
    MPI_Irecv(ops_buffer_recv_2, recv_size, MPI_BYTE,
              recv_size > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &requests[3]);

    MPI_Waitall(2, &requests[2], &status[2]);

    //Ready to unpack in the direction
    _ops_particle_unpack_reverse_buff_to_array(ops_buffer_recv_2, flag_to_add, 1,
                                               forward_candids[2 * idir],
                                               nforward_candids[2 * idir],
                                               OPS_INC);

    _ops_particle_unpack_reverse_buff_to_array(ops_buffer_recv_1, flag_to_add, 1,
                                               forward_candids[2 * idir + 1],
                                               nforward_candids[2 * idir + 1],
                                               OPS_INC);

    MPI_Waitall(2, &requests[0], &status[0]);
  }

}



inline void _update_exchange_flags(sub_block *sb,const int dim, int iattempt,  const int nattempts_tot,
                                   const int  nins, const int Nexpected, int *flag_to_send,
                                   int *flag_to_recv) {

  int status = (iattempt == nattempts_tot || nins == Nexpected) ? 1 : 0;
  MPI_Status statusMPI[2];
  for (int i = 0; i < dim; i++) {

    int flag_send_neg_old = flag_to_send[2 * i];
    int flag_send_pos_old = flag_to_send[2 * i + 1];
    int flag_recv_neg_old = flag_to_recv[2 * i];
    int flag_recv_pos_old = flag_to_recv[2 * i + 1];

    if (status) {
      flag_to_send[2 * i] = -1;
      flag_to_send[2 * i + 1] = -1;
      flag_to_recv[2 * i ] = -1;
      flag_to_recv[2 * i + 1] = -1;
    }

    int arecv_pos = -1;
    MPI_Sendrecv(&flag_to_send[2 * i ], 1, MPI_INT,
                 flag_send_neg_old > -1 ? sb->id_m[i] : MPI_PROC_NULL, 500,
                 &arecv_pos, 1, MPI_INT,
                 flag_recv_pos_old > -1 ? sb->id_p[i] : MPI_PROC_NULL, 500,
                 sb->comm, &statusMPI[0]);
    int arecv_neg = -1;
    MPI_Sendrecv(&flag_to_send[2 * i + 1], 1, MPI_INT,
                 flag_send_pos_old > -1 ? sb->id_p[i] : MPI_PROC_NULL, 600,
                 &arecv_neg, 1, MPI_INT,
                 flag_recv_neg_old > -1 ? sb->id_m[i] : MPI_PROC_NULL, 600,
                 sb->comm, &statusMPI[0]);

    if (!status) {
      flag_to_recv[2 * i] = arecv_neg;
      if (flag_to_recv[2 * i ] == -1) flag_to_send[2 * i ] = -1;

      flag_to_recv[2 * i + 1] = arecv_pos;
      if (flag_to_recv[2 * i + 1] == -1) flag_to_send[2 * i + 1] = -1;
    }
  }
}
#endif

inline int _check_particle_location(const int *binhead, const int bin_size[], const int *bins,
#ifdef OPS_MPI
                                const int *bin_virtual, const int nexpected,
#endif
                                int d_m[], int ix[], double xpoint[], double rad_ins,
                                ops_dat coords, ops_dat env,
#ifdef OPS_MPI
                                double *x_virtual, double *rad_virtual,
#endif
                                int dim) {


  int ix_min = (ix[0] > 0) ? - 1: 0;
  int ix_max = (ix[1] <= bin_size[0] - 1) ? 1 : 0;

  int iy_min = (ix[1] > 0) ? -1 : 0;
  int iy_max = (ix[1] <= bin_size[1] - 1) ? 1 : 0;

#ifdef OPS_3D
  iz_min = (ix[2] > 0) ? -1 : 0;
  iz_max = (ix[2] <= bin_size[1] - 1) ? 1 : 0;

  for (int k = iz_min; k <= iz_max; k++) {
#else
  int  k = 0; {
#endif
    for (int j = iy_min; j <= iy_max; j++) {
      for (int i = ix_min; i <= ix_max; i++) {
        int address = (ix[0] + i - d_m[0]) + (ix[1] + j - d_m[1]) * bin_size[0]
                    + (ix[2] + k - d_m[2]) * bin_size[0] * bin_size[1];
        int ineigh = binhead[address];

        while (ineigh != -1) {
          double d = 0.0;
#ifdef OPS_MPI
          for (int isou = 0; isou < dim; isou++) {
            double xneig = (ineigh < nexpected) ? ((double *) coords->data)[dim * ineigh + isou] :
                           x_virtual[dim * (ineigh - nexpected)];
            d += (xneig - xpoint[isou]) * (xneig - xpoint[isou]);
          }

          double Rp = (ineigh < nexpected) ?
              ((double *) env->data)[ineigh] : rad_virtual[ineigh - nexpected];
#else
          for (int isou = 0; isou < dim; isou++)
            d+= (((double *) coords->data)[dim * ineigh] - xpoint[isou])
              * (((double *) coords->data)[dim * ineigh] - xpoint[isou]);

          double Rp = ((double *) env->data)[ineigh];

#endif
          if (d < 1.01 * (Rp + rad_ins) * (Rp + rad_ins))
            return 1;
#ifdef OPS_MPI
          ineigh = (ineigh >= nexpected) ? bin_virtual[ineigh - nexpected] : bins[ineigh];
#else
          ineigh = bins[ineigh];
#endif
        }
      }
    }
  }

  return 0;

}


inline int _check_particle_location(ops_dat binhead, ops_dat bins, int d_m[], int ix[],double xpoint[],
                                    double rad_ins, int dim, ops_dat coords, ops_dat env) {

  //Get local end and finish
  int ix_min = (ix[0] > 0)  ? -1 : 0;
  int ix_max = (ix[1] <= binhead->size[0] - 1) ? 1: 0;

  int iy_min = (ix[1] > 0) ? - 1 : 0;
  int iy_max = (ix[1] <= binhead->size[1] - 1) ? 1 : 0;

#ifdef OPS_3D
  int iz_min = (ix[2] > 0) ? -1 : 0;
  int iz_max = (ix[2] <= binhead->size[2] - 1) ? 1 : 0;
#endif

#ifdef OPS_3D
  for (int k = iz_min; k <= iz_max; k++)
#else
  int k = 0;
#endif
  {
    for (int j = iy_min; j <= iy_max; j++) {
      for (int i = ix_min; i <= ix_max; i++) {
        int address = (ix[0]  + i - d_m[0]) + (ix[1] + j - d_m[1]) * binhead->size[0]
                    + (ix[2] + k - d_m[2]) * binhead->size[0] * binhead->size[1];

        int ineigh = ((int *)binhead->data)[address];

        while (ineigh != - 1) {
          double d = 0.0;
          for (int isou = 0; isou < dim; isou++)
            d+= (((double *) coords->data)[dim * ineigh + isou] - xpoint[isou])
              * (((double *) coords->data)[dim * ineigh + isou] - xpoint[isou]);

           double Rsq = (rad_ins + ((double *)env->data)[ineigh])
                       * (rad_ins + ((double *) env->data)[ineigh]);

           if (d < 1.01 * Rsq) return 1;

           if (address < 0)   exit(-1);

           ineigh = ((int *)bins->data)[ineigh];

        }
      }
    }
  }


  return 0;
}


inline int check_against_border_candidates(const int ix[],const int dim,
                                           const int bin_size[], const double *xpoint,
                                           const double rad_ins, const double *candidate_parts,
                                           const int ncand_actual) {

  if (    (ix[0] > 1 && ix[0] < bin_size[0] - 1 - 1 - 1 - 1)
       && (ix[1] > 1 && ix[1] < bin_size[1] - 1 - 1 - 1 - 1)
#ifdef OPS_3D
       && (ix[2] > 1 && ix[2] < bin_size[2] - 1 - 1 - 1 -1)
#endif
       )
    return 0;

  for (int ip = 0; ip < ncand_actual; ip++) {
    double d = 0.0;
    for (int isou = 0; isou < dim; isou++)
      d += (xpoint[isou] - candidate_parts[(dim + 1) * ip + isou])
         * (xpoint[isou] - candidate_parts[(dim + 1) * ip + isou]);

    double Rsq = (rad_ins + candidate_parts[(dim + 1) * ip + dim])
               * (rad_ins + candidate_parts[(dim + 1) * ip + dim]);

    if (d < 1.01 * Rsq) return 1;

  }

  return 0;
}

#ifdef OPS_MPI
inline void _push_generated_to_virtual_lists(const double *candidate_parts, int *flag_to_add, const int dim,
                                             const int ncnd_actual,const int ncnd_virtual, int *binhead,
                                             const int size[], const int d_m[], const int d_p[],
                                             ops_point xmin, const double dx[],
                                             int *&bin_virtual, const int *bins, const int n_expected,
                                             double *&xvirtual, double *&rad_virtual, int  &nvirtual,
                                             int &nvirtual_max) {

  int size_loc[OPS_MAX_DIM] = {};
  for (int i = 0; i < dim; i++)
    size_loc[i] = size[i] - d_p[i] + d_m[i]; //TODO: Need to see base
  int ix[OPS_MAX_DIM] = {};
  int nto_add = 0;
  //Loop over candidate particles
  for (int i = 0; i < ncnd_virtual; i++) {
    int ip = ncnd_actual + i;
    //In case that particle within the list. Please add me
    if (flag_to_add[ip] == 0) {
      ix[0] = (int) ops_floor((candidate_parts[(dim + 1) * ip]  - xmin.x)/dx[0]);
      ix[1] = (int) ops_floor((candidate_parts[(dim + 1) * ip + 1]  - xmin.y)/dx[1]);
      ix[2] = (dim == 3) ?
          (int) ops_floor((candidate_parts[(dim + 1) * ip + 2]  - xmin.z)/dx[2]) : 0;

      if ((ix[0] == d_m[0] || ix[0] == size_loc[0]) ||
         (ix[1] == d_m[1] || ix[1] == size_loc[1])
#ifdef OPS_3D
         || (ix[2] == d_m[2] || ix[2] == size_loc[2])
#endif
         )
      {
        nto_add++;
        flag_to_add[ip] = 1;
      }
    }

  }

  if (nvirtual + nto_add > nvirtual_max) {
    nvirtual_max = nvirtual + nto_add + OPS_MAX_PART;
    xvirtual = (double *) ops_realloc(xvirtual, sizeof(double) * dim * nvirtual_max);
    rad_virtual = (double *) ops_realloc(rad_virtual, sizeof(double) * nvirtual_max);
    bin_virtual = (int *) ops_realloc(bin_virtual, sizeof(int) * nvirtual_max);
  }

  for (int i = 0; i < ncnd_virtual; i++) {
    int ip = ncnd_actual + i;
    if (flag_to_add[ip] == 1) {
      for (int isou = 0; isou < dim; isou++)
        xvirtual[nvirtual * dim + isou] = candidate_parts[ip * (dim + 1)+ isou];
      rad_virtual[nvirtual] = candidate_parts[ip * (dim + 1) + dim];

      //push to map
      ix[0] = (int) ops_floor((xvirtual[dim * nvirtual] - xmin.x) / dx[0]);
      ix[1] = (int) ops_floor((xvirtual[dim * nvirtual + 1] - xmin.y)/ dx[1]);
      ix[2] = (dim == 3) ? ops_floor((xvirtual[dim * nvirtual + 2] - xmin.z) / dx[2]) : 0;

      int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                  + (ix[2] - d_m[2]) * size[0] * size[1];

      int ipart = binhead[address];
      binhead[address] = nvirtual;
      bin_virtual[nvirtual] = ipart;

      nvirtual++;
    }
  }
}

inline int particle_in_border(const int ix[],const int dim,const int size[],
                              const int flag_need_send[]) {

  for (int i = 0; i < dim; i++) {

    if (flag_need_send[2 * i] == 1 && ix[i] == 0) {
      return 1;
    }

    if (flag_need_send[2 * i + 1] == 1 && ix[i] == size[i] -3 ) //TODO: Check the positive
      return 1;
  }

  return 0;
}
#endif

/**
 * Function for generating particles randomly in a user defined region. The function sets only
 * the particle positions, envelopes (radius).
 *
 * @param particle      the ops_particle structure for which particles will be inserted in the
 *                      user defined region
 * @param region        user defined region in the form [xmin xmax]x[ymin ymax] x[zmin zmax]
 * @param Nins          maximum number of particles to be added in the given region
 * @param nattemps      maximum attempts per particle before stopping pouring particles
 *                      in the box
 * @param seed          integer for setting the random engine function
 * @param rad_distr     an pointer to an OPSDistribution used in generating particle sizes
 * @param envelope      an ops_dat structure used to define particle sizes
 */

template<template<typename X> class Distribution>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope) {


  //Sanity checks

  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");

  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

  //TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                             " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                             "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");


  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                             "proceed ");


  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);

  }

// Final checks for inserting particles
  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;

  //TODO: Check that this works for the parallel as well in terms of a function
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }

#endif

  //Generate virtual maps
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }


  int ifirst;
#ifdef OPS_MPI
//TODO: Need to generate a second map with virtual particles that we will increase
//      the bins

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Assume that local particles are mapped
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;

#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif

  std::default_random_engine dre(seed);

  //Create an ops_dat for particle data
  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;


//  printf("Prior to generate\n");

  while (n_insert < Ninsert) {
    int a1;
    int ix[OPS_MAX_DIM] = {};
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {


      ops_generate_random_point(part_dist, dre, xpoint);

      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;
    }

    iattempt_tot += iattempt;


    /*TODO: After 50-100 attemps-Try to check virtual elements
      Add them to the existing list & include them as virtual as
      well
    */

#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {



      //PART I: Set exchange flags to send around and recv

      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);

      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);


      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);


      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);




      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot) {
     //TODO: PUSH END OF INSERTION
      break;
    }

//    printf("R %d: Particles inserted %d\n", ops_get_proc(), n_insert);

  }

  /*
  for (int i = 0; i < nlocal; i++)
    printf("R %d: x[%d] = [%f %f] R = %f\n", ops_get_proc(), i,
                                           ((double *)particle->particle_pos_dat->data)[i * dim + 0],
                                           ((double *)particle->particle_pos_dat->data)[i * dim + 1],
                                           ((double *)envelope->data)[i]);
*/

 // printf("Rank %d Particles inserted %d\n", ops_get_proc(), n_insert);

  //ADD END OF INSERTION FOR PARALLEL
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //TODO: Add tags
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  int nexist = particle->no_particles;
  particle->no_particles = nlocal;

  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  particle->no_particles = nlocal;
  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //NOTE: We had hear a reallocation array for maps but particle data structures passd
      // to particle allocation

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }


  //FREE ARRAYS
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);




        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //NOTE: ALlocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}


template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);




        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}


template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);





        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);


          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);






        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);


          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);

        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);


          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);

        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);


          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);

        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);

        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);

        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }



#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation of map data passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18,
         template<typename X19> class Distribution19, typename T19>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18,
                                 OPS_dat_distr<Distribution19, T19> &distr19) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
        insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
          insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocations passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18,
         template<typename X19> class Distribution19, typename T19,
         template<typename X20> class Distribution20, typename T20>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18,
                                 OPS_dat_distr<Distribution19, T19> &distr19,
                                 OPS_dat_distr<Distribution20, T20> &distr20) {

  //Part I: Sanity checks
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");
  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

//TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                           " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                           "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                           "proceed ");

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert;
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;



  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }

  int ifirst;

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
        insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);
        insert_random_dat(distr20.dat->data, distr20.distribution, dre, nlocal-1);


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;

    }

    iattempt_tot += iattempt;

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
          insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);
          insert_random_dat(distr20.dat->data, distr20.distribution, dre, nlocal-1);

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocations passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}

#endif /* OPS_C_INCLUDE_OPS_INSERT_RANDOM_PARTICLES_V2_H_ */
