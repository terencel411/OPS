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
  * @brief OPS mpi run-time support routines for particle structures
  * @author Valantis Tsinginos
  * @details Implements the runtime support routines for the OPS mpi backend
  *          for particle structures
  */

#include <vector>
#include <ops_lib_core.h>
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_mpi_particle_core.h>

#include <ops_particle_mapping_functions.h>
#include <ops_exceptions.h>
#include <cassert>

extern char *halo_buffer_d;
extern char *ops_buffer_send_1;
extern char *ops_buffer_recv_1;
extern char *ops_buffer_send_2;
extern char *ops_buffer_recv_2;
extern int ops_buffer_send_1_size;
extern int ops_buffer_recv_1_size;
extern int ops_buffer_send_2_size;
extern int ops_buffer_recv_2_size;

/*======================================================================================
 * TODO Functions
 *======================================================================================*/

//Functions For halo exchanges



void _ops_particle_update_dependent_halo_groups(OPS_instance *instance,
                                                ops_particle_halo_group halo_grp,
                                                ops_mpi_particle_halo_group *mpi_group) {

  int index = halo_grp->index;
  for (int igrp = 0; igrp < instance->OPS_particle_halo_group_index; igrp++) {
    if (index == igrp) continue;

    ops_particle_halo_group grp_dep = instance->OPS_particle_halo_group_list[igrp];
    if (grp_dep->halo_type == OPS_HALO_GRP_DEFAULT ||
        grp_dep->halo_type == OPS_HALO_GRP_BORDER ||
        grp_dep->halo_type == OPS_HALO_GRP_EXCHANGE)
      continue;

    //Check if the main halo is the same with halo_grp
    if (grp_dep->halo_master->index != halo_grp->index) continue;

    //TIME TO SET STRUCTURES
    ops_mpi_particle_halo_group *mpi_dep =
        &OPS_mpi_particle_halo_group_list[grp_dep->index];

    for (int i = 0; i < mpi_dep->num_neighbors_send; i++)
      mpi_dep->send_bites[i] = 0;

    for (int i = 0; i < mpi_dep->num_neighbors_recv; i++)
      mpi_dep->recv_bites[i] = 0;

    int nhalos = mpi_group->nhalos;
    for (int h = 0; h < nhalos; h++) {
      ops_particle_halo halo = grp_dep->halo_list[h];
      ops_mpi_particle_halo *mpi_halo = mpi_dep->mpi_halos[h];
      int size_halo = halo->nbites;
      for (int isend = 0; isend < mpi_halo->nproc_to; isend++) {
        int iproc = mpi_halo->proclist[isend];
        int iloc;
        for (iloc =0; iloc < mpi_group->num_neighbors_send; iloc++)
          if (mpi_group->neighbors_send[iloc] == iproc) break;

        ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
        mpi_dep->send_bites[iloc] += size_halo * halo_info->nsend;
      }

      for (int irecv = 0; irecv < mpi_halo->nproc_from; irecv++) {
        int iproc = mpi_halo->proclist[irecv + mpi_halo->nproc_to];
        int iloc;
        for (iloc =0; iloc < mpi_group->num_neighbors_send; iloc++)
          if (mpi_group->neighbors_send[iloc] == iproc) break;

        ops_particle_halo_exchange halo_info =
            mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send +iloc) + h];
        mpi_dep->recv_bites[iloc] += size_halo * halo_info->nrecv;
      }
    }

    //Update shifts
    int shift = 0;
    int notot =0;
    for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
      mpi_dep->send_shift[i] = shift;
      shift += mpi_dep->send_bites[i];
      mpi_dep->send_pos_bites[i] = mpi_group->send_pos_bites[i];
      mpi_dep->shift_send_pos[i] = mpi_group->shift_send_pos[i];
      notot += mpi_dep->send_pos_bites[i];
    }

    shift = MAX(shift, notot);
    if (grp_dep->halo_type == OPS_HALO_GRP_FORWARD) {
      if (shift > ops_buffer_send_1_size) {
        ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
        ops_buffer_send_1_size = shift;
      }
    }
    else if (grp_dep->halo_type == OPS_HALO_GRP_BACKWARD) {
      if (shift > ops_buffer_recv_1_size) {
        ops_buffer_recv_1 =
            (char *) OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
        ops_buffer_recv_1_size = shift;
      }

    }

    shift = 0;
    notot = 0;
    for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
      mpi_dep->recv_shift[i] = shift;
      shift += mpi_dep->recv_bites[i];

      mpi_dep->recv_pos_bites[i] = mpi_group->recv_pos_bites[i];
      mpi_dep->shift_recv_pos[i] = mpi_group->shift_recv_pos[i];
      notot += mpi_dep->recv_pos_bites[i];
    }

    shift = MAX(shift, notot);
    if (grp_dep->halo_type == OPS_HALO_GRP_BACKWARD) {
      if (shift > ops_buffer_send_1_size) {
        ops_buffer_send_1 =
            (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
        ops_buffer_send_1_size = shift;
      }
    }
    else if (grp_dep->halo_type ==  OPS_HALO_GRP_FORWARD) {
      if (shift > ops_buffer_recv_1_size) {
        ops_buffer_recv_1 =
            (char *) OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
        ops_buffer_recv_1_size = shift;
      }

    }

  }


}

void _ops_particle_halo_border_transfer_map(OPS_instance *instance,
                                           ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];


  printf("Entering herein for number of halos: %d\n", mpi_group->nhalos);

  if (mpi_group->nhalos == 0) return;
  double c, t1, t2;
  ops_timers_core(&c, &t1);
 // mpi_neigh_size[0] = 0;

  int nhalos = mpi_group->nhalos;

  /* Part I: Allocate & initialize auxiliarry matrices */
  int size = (mpi_group->num_neighbors_send > 0)
      ? mpi_group->num_neighbors_send * mpi_group->nhalos : 1;
  int *nsend =
      (int *) ops_malloc(size * sizeof(int));
  size = (mpi_group->num_neighbors_recv > 0)
      ? mpi_group->num_neighbors_recv * mpi_group->nhalos : 1;
  int *nrecv = (int *)ops_malloc(size * sizeof(int));


  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; i++) {
    nsend[i] = 0;
    mpi_group->halo_info[i]->nsend = 0;
  }

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; i++)
    nrecv[i] = 0;

  size = (mpi_group->num_neighbors_send > 0) ? mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *) ops_malloc(size * sizeof(int));

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    mpi_group->send_bites[i] = 0;
    mpi_group->send_pos_bites[i] = 0;
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_bites[i] = 0;
    mpi_group->recv_pos_bites[i] = 0;
  }

  /* Part II: Get number of sending particles */

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;

    ops_particle_mapping map
    = instance->OPS_particle_halo_list[halo->index]->particle_from->map_list[0];
    ops_dat binhead = map->binhead;
    int *bins = (int *)map->bin->data;
    int *bin_head =(int *)binhead->data;
    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;
    int bites_pos
    = instance->OPS_particle_halo_list[halo->index]->particle_from->particle_pos_dat->elem_size;
    //Find number of elements to send in different halos
    for (int isend = 0; isend < nproc_to; isend++) {

      //Find process and identify storage location
      int iproc = halo->proclist[isend];
      int iloc;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (iproc == mpi_group->neighbors_send[ip]) {iloc = ip; break;}
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];


      //Find number of particles send in the halo to process iproc and stored in
      _ops_particle_number_of_particles_in_range(halo->send_region + 6 * isend, bin_head,
                                                 bins, binhead->size,
                                                 &nsend[nhalos * iloc + h]);


      halo_info->nsend = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist =
            (int *) OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      //TODO: Perform maps into the list
      _ops_particle_mapped_into_region(halo->send_region, bin_head, bins,
                                      binhead->size, halo_info->sendlist);

      //Set size directly
      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
      mpi_group->send_pos_bites[iloc] += bites_pos * halo_info->nsend;

    }
  }

  /* Part III: Sending and receiving of virtual particles per halo */
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos * isend;
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);

  }

  for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
    int ishift = mpi_group->nhalos * irecv;

    MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
              mpi_group->neighbors_recv[irecv],
              1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
  }



  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  /*Part IV: Initialize data structures for receiving particles */
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    ops_particle particle_to
      = instance->OPS_particle_halo_list[halo->index]->particle_to;
    int ifirst = particle_to->no_particles + particle_to->no_virtual;
    int no_new = 0;
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    int nbites_pos = particle_to->particle_pos_dat->elem_size;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];

      //Get storate location
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (iproc == mpi_group->neighbors_recv[ip]) { iloc = ip; break;}
      }

      ops_particle_halo_exchange halo_info =
          mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];
      halo_info->nrecv = nrecv[nhalos * iloc + h];
      halo_info->firstrecv = ifirst;
      ifirst += halo_info->nrecv;
      no_new += halo_info->nrecv;

      mpi_group->recv_bites[iloc] += nbites * halo_info->nrecv; //TODO: Replace with the first
      mpi_group->recv_pos_bites[iloc] += nbites_pos * halo_info->nrecv;
    }
    particle_to->no_virtual += no_new;
  }




  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  /* Part V: Compute shifts & reallocate buffers */
  int shift0 = 0;
  int shift1 = 0;

  for (int  i = 0; i < mpi_group->num_neighbors_send;i++) {
    mpi_group->send_shift[i] = shift0;
    nsend_shift[i] = shift0;
    mpi_group->shift_send_pos[i] = shift1;
    shift1 +=mpi_group->send_pos_bites[i];
    shift0 += mpi_group->send_bites[i];
  }

  int shift = MAX(shift1, shift0);
  //Reallocate send buffers if necessary
  if (shift > ops_buffer_send_1_size) {
    ops_buffer_send_1 =
    (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
  }

  shift = 0;
  shift0 = 0;
  shift1 = 0;
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift0;
    nrecv_shift[i] = shift0;
    mpi_group->shift_recv_pos[i] = shift1;
    shift0 += mpi_group->recv_bites[i];
    shift1 += mpi_group->recv_pos_bites[i];
  }

  //Reallocate recv buffers if necessary
  shift = MAX(shift0, shift1);
  if (shift > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 = (char *) OPS_realloc_fast(ops_buffer_recv_1,
                                                  ops_buffer_recv_1_size, shift);
    ops_buffer_recv_1_size = shift;
  }


  /*Part VI: Pack particle data */
  int ntot_bites = 0;

  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get process
      int iproc = halo_mpi->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];


      int shift = nsend_shift[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites, 1);
      nsend_shift[iloc] += ntot_bites;
    }
  }

  /* Part VI: Send & receive particle data from halos */
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + mpi_group->send_shift[i], mpi_group->send_bites[i],
              MPI_BYTE,
              mpi_group->send_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int shift = mpi_group->recv_shift[i];

    MPI_Irecv(ops_buffer_recv_1 + shift, mpi_group->recv_bites[i],
              MPI_BYTE,
              mpi_group->recv_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  /* Part VII: Reallocate particle data structures if necessary */
  ntot_bites = 0;
  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
  }

  /* Part VIII: Unpack data structures & map received particles*/
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data *halo_data = halo->dat;

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      //Get process and storage id
      int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (mpi_group->neighbors_recv[ip] == iproc) {
          iloc = ip; break;
        }
      }
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];


      int shift = nrecv_shift[iloc];

      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + shift, halo_data, halo->nhalos,
                                        halo_info, halo->dir_to, halo->dir_from, halo->translate,
                                        &ntot_bites, 1);
      nrecv_shift[iloc] += ntot_bites;

      for (int imap = 0; imap < halo->particle_to->particle_map_index; imap++) {
        ops_particle_mapping map = halo->particle_to->map_list[imap];
        _ops_particle_mapping_virtual_from_halo(map, halo->particle_to,
                                                halo_info->firstrecv,
                                                halo_info->nrecv);

      }
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  /* Part IX: Update halo structures of dependent systems */
  _ops_particle_update_dependent_halo_groups(instance, halo_grp, mpi_group);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(nsend);
  ops_free(nrecv);
  ops_free(nsend_shift);
  ops_free(nrecv_shift);
}

void _ops_particle_halo_border_transfer(OPS_instance  *instance,
                                        ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  double c, t1, t2;
  ops_timers_core(&c, &t1);

  int nhalos = mpi_group->nhalos;

  /* Part I: Allocate auxiliarry matrices and initializations */
  int size = (mpi_group->num_neighbors_send > 0) ?
      mpi_group->num_neighbors_send * mpi_group->nhalos : 1;
  int *nsend = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ?
      mpi_group->num_neighbors_recv * mpi_group->nhalos : 1;
  int *nrecv = (int *) ops_malloc(size * sizeof(int));


  printf("R %d mpi_group->num_neighbors_Send = %d nhalos = %d\n", ops_get_proc(),mpi_group->num_neighbors_send);


  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; i++) {
    nsend[i] = 0;
    mpi_group->halo_info[i]->nsend = 0;
  }

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; i++)
    nrecv[i] = 0;

  size = (mpi_group->num_neighbors_send > 0) ? mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *) ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *) ops_malloc(size * sizeof(int));


  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    mpi_group->send_bites[i] = 0;
    mpi_group->send_pos_bites[i] = 0;
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_bites[i] = 0;
    mpi_group->recv_pos_bites[i] = 0;
  }


  /* Part II: Find number of particles to send in each proc */
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;
    int dim = instance->OPS_particle_halo_list[halo->index]->particle_from->block->dims;
    char *xcrds
    = instance->OPS_particle_halo_list[halo->index]->particle_from->particle_pos_dat->data;
    int noParticles = instance->OPS_particle_halo_list[halo->index]->particle_from->no_particles
                    + instance->OPS_particle_halo_list[halo->index]->particle_from->no_virtual;
    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;
    int bites_coords
    = instance->OPS_particle_halo_list[halo->index]->particle_from->particle_pos_dat->elem_size;
    for (int isend = 0; isend < nproc_to; isend++) {

      //Get process and location
      int iproc = halo->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];

      switch(instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<float> *)halo->sendBox[isend],
                                                   dim, (float *) xcrds, noParticles, &nsend[nhalos * iloc + h]);
        break;
      case sizeof(double):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<double> *)halo->sendBox[isend],
                                                   dim, (double *) xcrds, noParticles,
                                                   &nsend[nhalos * iloc + h]);
        break;
      case sizeof(long double):
      _ops_particle_no_particles_in_range_by_box((BoundingBox<long double> *)halo->sendBox[isend],
                                                 dim, (long double *) xcrds, noParticles,
                                                 &nsend[nhalos * iloc + h]);
        break;
      }

      halo_info->nsend = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist = (int *)
            OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      switch(instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
         _ops_particle_mapped_into_region_by_block((BoundingBox<float> *)halo->sendBox[isend], dim,
                                                   (float *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      case sizeof(double):
        _ops_particle_mapped_into_region_by_block((BoundingBox<double> *)halo->sendBox[isend], dim,
                                                 (double *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      case sizeof(long double):
        _ops_particle_mapped_into_region_by_block((BoundingBox<long double> *)halo->sendBox[isend], dim,
                                                 (long double *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      }

      mpi_group->send_bites[iloc]+= bites * halo_info->nsend;
      mpi_group->send_pos_bites[iloc] += bites_coords * halo_info->nsend;
    }

  }

  /* Part III: Sending and receiving the number of virtual particles added
   * by this halo_group
   */
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos * isend;
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);
  }

   //Receiving data for building chuncks
   for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
     int ishift = mpi_group->nhalos * irecv;
     MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
               mpi_group->neighbors_recv[irecv],
               1000+mpi_group->index, OPS_MPI_GLOBAL,
               &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
   }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  //Need to send a big chunk of data at once
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    ops_particle particle_to
      = instance->OPS_particle_halo_list[halo->index]->particle_to;
    int ifirst = particle_to->no_particles + particle_to->no_virtual;
    int no_new = 0;
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    int npos_bites = particle_to->particle_pos_dat->elem_size;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      //Get proc and storage location
      int iproc = halo->proclist[halo->nproc_to + irecv];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (iproc == mpi_group->neighbors_recv[ip]) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info =
          mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];
      halo_info->nrecv = nrecv[nhalos * iloc + h];
      halo_info->firstrecv = ifirst;//TODO: Add it later
      ifirst += halo_info->nrecv;
      no_new += halo_info->nrecv;

      mpi_group->recv_bites[iloc] += nbites * halo_info->nrecv;
      mpi_group->recv_pos_bites[iloc] += npos_bites * halo_info->nrecv;
    }
    particle_to->no_virtual += no_new;
  }


  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);


  /* Part IV: Compute shifts for particle halo exchanges */
  int shift0 = 0;
  int shift1 = 0;
  for (int  i = 0; i < mpi_group->num_neighbors_send;i++) {
    mpi_group->send_shift[i] = shift0;
    nsend_shift[i] = shift0;

    mpi_group->shift_send_pos[i] = shift1;
    shift0 += mpi_group->send_bites[i];
    shift1 += mpi_group->send_pos_bites[i];
  }

  int shift = MAX(shift0, shift1);
  if (shift > ops_buffer_send_1_size) {
    ops_buffer_send_1 =
        (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
  }

  shift = 0;
  shift0 = 0;
  shift1 = 1;
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift0;
    nrecv_shift[i] = shift0;

    mpi_group->shift_send_pos[i] = shift1;
    shift0 +=  mpi_group->recv_bites[i];
    shift1 +=  mpi_group->recv_pos_bites[i];
  }

  shift = MAX(shift0, shift1);
  if (shift > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 =
        (char *) OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
    ops_buffer_recv_1_size = shift;
  }

  /* Part V: Pack particle data */

  int ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get process and loc
      int iproc = halo_mpi->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      int shift = nsend_shift[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites, 1);
      nsend_shift[iloc] += ntot_bites;
    }
  }

  /* Part VI: Send and receive data */
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + mpi_group->send_shift[i], mpi_group->send_bites[i],
              MPI_BYTE,
              mpi_group->send_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    MPI_Irecv(ops_buffer_recv_1 + mpi_group->recv_shift[i], mpi_group->recv_bites[i],
              MPI_BYTE,
              mpi_group->recv_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  /* Part VII: Reallocate of particle data structures */
  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
  }

  /* Part VIII: Unpack particle data from halo_group */
  ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data *halo_data = halo->dat;

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      //Get process and storage point
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (mpi_group->neighbors_recv[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      int shift = nrecv_shift[iloc];
      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + shift, halo_data, halo->nhalos,
                                        halo_info, halo->dir_to, halo->dir_from, halo->translate,
                                        &ntot_bites, 1);
      nrecv_shift[iloc] += ntot_bites;

    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  /* Part IX: Update halo_groups that depend on the current one */
  _ops_particle_update_dependent_halo_groups(instance, halo_grp, mpi_group);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(nrecv_shift);
  ops_free(nsend_shift);
  ops_free(nsend);
  ops_free(nrecv);

}

//TODO:
void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;

  double c, t1, t2;
  ops_timers_core(&c, &t1);

  int nhalos = mpi_group->nhalos;



  int size = (mpi_group->num_neighbors_send > 0) ?
      nhalos * mpi_group->num_neighbors_send : 1;


  int *nsend = (int *) ops_malloc(sizeof(int) * size);
  size = (mpi_group->num_neighbors_recv > 0) ?
      mpi_group->num_neighbors_recv * nhalos : 1;
  int *nrecv = (int *) ops_malloc(sizeof(int) * size);

  size = (mpi_group->num_neighbors_send > 0)? mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *) ops_malloc(sizeof(int) * size);
  size = (mpi_group->num_neighbors_recv > 0)? mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *) ops_malloc(sizeof(int) * size);

  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; i++) {
    nsend[i] = 0;
    mpi_group->halo_info[ i ]->nsend = 0;
  }

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; i++) {
    nrecv[i] = 0;
//    mpi_group->halo_info[input + i]->nrecv = 0;
  }

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    mpi_group->send_bites[i] = 0;

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    mpi_group->recv_bites[i] = 0;

  for (int h = 0;  h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int dim = instance->OPS_particle_halo_list[halo->index]->particle_from->block->dims;

    ops_particle particle = instance->OPS_particle_halo_list[halo->index]->particle_from;
    double *xcrds =(double *)particle->particle_pos_dat->data;
    int noParticles = particle->no_particles;

    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;

    for (int isend = 0; isend < halo->nproc_to; isend++) {
       //Get process and storage location
      int iproc = halo->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      switch(instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<float> *)halo->sendBox[isend],
                                                   dim, (float *) xcrds, noParticles, &nsend[nhalos * iloc + h]);
        break;
      case sizeof(double):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<double> *)halo->sendBox[isend],
                                                   dim, (double *) xcrds, noParticles,
                                                   &nsend[nhalos * iloc + h]);
        break;
      case sizeof(long double):
      _ops_particle_no_particles_in_range_by_box((BoundingBox<long double> *)halo->sendBox[isend],
                                                 dim, (long double *) xcrds, noParticles,
                                                 &nsend[nhalos * iloc + h]);
        break;
      }

      halo_info->nsend  = nsend[nhalos * iloc + h]; ///out of memory

      if (halo_info->nsend > halo_info->nmax) {
         halo_info->sendlist = (int *)
             OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                          sizeof(int) * (halo_info->nsend + 10));
         halo_info->nmax = halo_info->nsend + 10;
      }

      switch(instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
         _ops_particle_mapped_into_region_by_block((BoundingBox<float> *)halo->sendBox[isend], dim,
                                                   (float *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      case sizeof(double):
        _ops_particle_mapped_into_region_by_block((BoundingBox<double> *)halo->sendBox[isend], dim,
                                                 (double *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      case sizeof(long double):
        _ops_particle_mapped_into_region_by_block((BoundingBox<long double> *)halo->sendBox[isend], dim,
                                                 (long double *) xcrds, noParticles,
                                                  halo_info->sendlist);
        break;
      }


      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
    }
  }

  //Communicate sending info
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos * isend;
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);
  }

  for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
    int ishift = mpi_group->nhalos * irecv;
    MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
              mpi_group->neighbors_recv[irecv],
              1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
  }



  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  //Unpack data
  //TODO: Assume that we need to unpack

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];
      int iloc = -1;
      for (int ip = 0; mpi_group->num_neighbors_recv; ip++) {
        if (mpi_group->neighbors_recv[ip] == iproc) {
          iloc = ip; break;
        }
      }
      mpi_group->recv_bites[iloc] += nbites * nrecv[nhalos * iloc + h];

      ops_particle_halo_exchange halo_info
       = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];

      halo_info->nrecv = nrecv[mpi_group->nhalos * iloc + h];

    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  //Compute shifts
  int shift = 0;
  for (int  i = 0; i < mpi_group->num_neighbors_send;i++) {
    mpi_group->send_shift[i] = shift;
    nsend_shift[i] = shift;
    shift += mpi_group->send_bites[i];
  }

   //Allocation of buffers and finding shifts
  if (shift > ops_buffer_send_1_size) {
    ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1,
                                                  ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
   }

  shift = 0;
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift;
    nrecv_shift[i] = shift;
    shift += mpi_group->recv_bites[i];
  }

  if (shift > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 =
        (char *) OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
    ops_buffer_recv_1_size = shift;
  }

  int ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int  isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get proc and location
      int iproc = halo_mpi->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;

        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h]; //TODO: Check it
      shift = nsend_shift[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                     halo_info, &ntot_bites, 1);
      nsend_shift[iloc] += ntot_bites;

    }

  }

   //Sending data via MPI
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int shift = mpi_group->send_shift[i];
    MPI_Isend(ops_buffer_send_1 + shift, mpi_group->send_bites[i], MPI_BYTE,
              mpi_group->send_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int shift = mpi_group->recv_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + shift, mpi_group->recv_bites[i], MPI_BYTE,
              mpi_group->recv_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }



  MPI_Waitall(mpi_group->num_neighbors_recv,
               &mpi_group->requests[mpi_group->num_neighbors_send],
               &mpi_group->statuses[mpi_group->num_neighbors_send]);

  //Finding if reallocations are needed
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc =-1;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++) {
        if (mpi_group->neighbors_recv[iloc] == iproc) break;
      }

      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h]; //IT IS WRONG

      halo_info->firstrecv  = halo->particle_to->no_particles;
      halo->particle_to->no_particles += halo_info->nrecv;
    }
  }

   //TODO: Add processes recv data
 //Reallocate structures before unpack
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle particle = halo->particle_to;

    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }

  }

  //Unpacking exchange data
   for (int h = 0; h < mpi_group->nhalos; h++) {
     ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
     ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
     ops_particle_halo_data *halo_data = halo->dat;

     ops_particle to = halo->particle_to;
     int *mark_del = (int *) to->mark_deletion;
     for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
       int iproc = halo_mpi->proclist[irecv + halo_mpi->nproc_to];
       int iloc = -1;
       for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++) {
         if (mpi_group->neighbors_recv[iloc] == iproc) break;
       }

       ops_particle_halo_exchange halo_info
       = mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];
       shift = nrecv_shift[iloc];
       _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + shift, halo_data, halo->nhalos,
                                         halo_info, halo->dir_to, halo->dir_from, halo->translate,
                                         &ntot_bites, 1);
       nrecv_shift[iloc] += ntot_bites;

       //TODO: Mark for deletion or not

       char *xcrds = to->particle_pos_dat->data;

      switch (instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
        _ops_particle_mark_for_removal((BoundingBox<float> *)to->box_block, (float *)xcrds, mark_del,
                                       to->block->dims, halo_info->firstrecv,
                                       halo_info->firstrecv + halo_info->nrecv);
        break;
      case sizeof(double):
         _ops_particle_mark_for_removal((BoundingBox<double> *)to->box_block, (double *)xcrds, mark_del,
                                        to->block->dims, halo_info->firstrecv,
                                        halo_info->firstrecv + halo_info->nrecv);
        break;
      case sizeof(long double):
         _ops_particle_mark_for_removal((BoundingBox<long double> *)to->box_block, (long double *)xcrds, mark_del,
                                        to->block->dims, halo_info->firstrecv,
                                        halo_info->firstrecv + halo_info->nrecv);
        break;
      }

     }

   }

   MPI_Waitall(mpi_group->num_neighbors_send,
               &mpi_group->requests[0], &mpi_group->statuses[0]);


   ops_timers_core(&c, &t2);
   instance->ops_user_halo_exchanges_time += t2 - t1;

   ops_free(nrecv_shift);
   ops_free(nsend_shift);
   ops_free(nrecv);
   ops_free(nsend);

}


void _ops_particle_halo_exchange_transfer_map(OPS_instance   *instance,
                                              ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  double c, t1, t2;
  ops_timers_core(&c, &t1);

  /* Part I: Allocations  of tmp arrays  */
  int nhalos = mpi_group->nhalos;

  int size = (mpi_group->num_neighbors_send> 0)
      ? nhalos * mpi_group->num_neighbors_send : 1;

  int *nsend = (int *)ops_malloc(size * sizeof(int));

  //nrecv
  size = (mpi_group->num_neighbors_recv > 0)
      ? nhalos * mpi_group->num_neighbors_recv : 1;
  int *nrecv = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_send > 0) ?
      mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ?
      mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *)ops_malloc(size * sizeof(int));

  for (int i= 0; i < mpi_group->num_neighbors_send * nhalos; i++) {
    nsend[i] = 0;
    mpi_group->halo_info[i]->nsend = 0;
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv * nhalos; i++)
    nrecv[i] = 0;

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    mpi_group->send_bites[i] = 0;

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    mpi_group->recv_bites[i] = 0;

  /* Part II: Find number of particles send and recv */
  for (int h = 0;  h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int dim = instance->OPS_particle_halo_list[halo->index]->particle_from->block->dims;

    ops_particle particle = instance->OPS_particle_halo_list[halo->index]->particle_from;
    char *xcrds = particle->particle_pos_dat->data;
    int noParticles = particle->no_particles;
    int *mark_deletion = particle->mark_deletion;
    char *env = (particle->particle_envelope != nullptr) ?
                 particle->particle_envelope->data : nullptr;

    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;
    for (int isend = 0; isend < halo->nproc_to; isend++) {
      //Find proc and storage
      int iproc = halo->proclist[isend];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_send; iloc++)
        if (mpi_group->neighbors_send[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];

      switch(instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<float> *)halo->sendBox[isend],
                                                   dim, (float *) xcrds, noParticles, &nsend[nhalos * iloc + h]);
        break;
      case sizeof(double):
        _ops_particle_no_particles_in_range_by_box((BoundingBox<double> *)halo->sendBox[isend],
                                                   dim, (double *) xcrds, noParticles,
                                                   &nsend[nhalos * iloc + h]);
        break;
      case sizeof(long double):
      _ops_particle_no_particles_in_range_by_box((BoundingBox<long double> *)halo->sendBox[isend],
                                                 dim, (long double *) xcrds, noParticles,
                                                 &nsend[nhalos * iloc + h]);
        break;
      }

      halo_info->nsend  = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist = (int *)
          OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      switch (instance->OPS_particle_halo_list[halo->index]->particle_from->type_box) {
      case sizeof(float):
          _ops_particle_remove_from_region((BoundingBox<float> *) halo->sendBox[isend], (float *)env,
                                           (float *) xcrds, mark_deletion,
                                           particle->no_particles, particle->block->dims,
                                           halo_info->sendlist);
        break;
      case sizeof(double):
        _ops_particle_remove_from_region((BoundingBox<double> *) halo->sendBox[isend], (double *)env,
                                         (double *) xcrds, mark_deletion,
                                         particle->no_particles, particle->block->dims,
                                         halo_info->sendlist);
        break;
      case sizeof(long double):
        _ops_particle_remove_from_region((BoundingBox<long double> *) halo->sendBox[isend],
                                         (long double *)env,
                                         (long double *) xcrds, mark_deletion,
                                         particle->no_particles, particle->block->dims,
                                         halo_info->sendlist);
        break;
      }

      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
    }
  }

  /* Part III: Obtain number of particles to receive */
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos *isend;
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);
  }

  for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
    int ishift = mpi_group->nhalos * irecv;
    MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
              mpi_group->neighbors_recv[irecv],
              1000+mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);




  //Update sending particles
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      mpi_group->recv_bites[iloc] += nbites * nrecv[nhalos * iloc + h];
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      halo_info->nrecv = nrecv[mpi_group->nhalos * iloc +h];
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  //Part IV: Compute shifts and storage and memory allocations
  int shift = 0;
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    mpi_group->send_shift[i] = shift;
    nsend_shift[i] = shift;
    shift += mpi_group->send_bites[i];
  }

  if (shift > ops_buffer_send_1_size) {
    ops_buffer_send_1 = (char *)
        OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
  }

  shift = 0;
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift;
    nrecv_shift[i] = shift;
    shift += mpi_group->recv_bites[i];
  }

  if (shift > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 = (char *)
        OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
    ops_buffer_recv_1_size = shift;
  }

  /* Part V: Data packing */
  int ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int  isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get process and storage loc
      int iproc = halo_mpi->proclist[isend];
      int iloc = -1;
      for (iloc = 0; mpi_group->num_neighbors_send; iloc++)
        if (mpi_group->neighbors_send[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h]; //TODO: Check it
      shift = nsend_shift[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites, 1);
      nsend_shift[iloc] += ntot_bites;
    }
  }

  /* Part VI: Send and receive data */
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int shift = mpi_group->send_shift[i];
    MPI_Isend(ops_buffer_send_1 + shift, mpi_group->send_bites[i], MPI_BYTE,
              mpi_group->send_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int shift = mpi_group->recv_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + shift, mpi_group->recv_bites[i], MPI_BYTE,
              mpi_group->recv_bites[i] > 0 ? iproc : MPI_PROC_NULL,
              1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  /* Part VII: Reallocate particle structures and finalize mapping structures*/
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h]; //IT IS WRONG
      halo_info->firstrecv = halo->particle_to->no_particles;
      halo->particle_to->no_particles += halo_info->nrecv;
    }
  }

  //Reallocate structures before unpack
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle particle = halo->particle_to;
    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }
  }


  /*Part VIII: Update particle lists due to shifted particles */
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data *halo_data = halo->dat;

    ops_particle to = halo->particle_to;
    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++) {
        if (mpi_group->neighbors_recv[iloc] == iproc) break;
      }
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      shift = nrecv_shift[iloc];
      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + shift, halo_data, halo->nhalos,
                                        halo_info, halo->dir_to, halo->dir_from, halo->translate,
                                        &ntot_bites, 1);
      nrecv_shift[iloc] += ntot_bites; //TODO: Need to rmv

      //TODO: Mark for deletion or not
      for (int imap = 0; imap < to->particle_map_index; imap++) {
         ops_particle_mapping map = to->map_list[imap];
         _ops_particle_map_from_exchange(map, to, halo_info->firstrecv,
                                         halo_info->nrecv + halo_info->firstrecv);
     }
    }

  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;


  ops_free(nsend);
  ops_free(nrecv);
  ops_free(nsend_shift);
  ops_free(nrecv_shift);
}




//TODO: NEED SOME WORK ON SIZES AND DATA AFTER THAT
void _ops_particle_halo_forward_map(OPS_instance   *instance,
                                    ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group
    = &OPS_mpi_particle_halo_group_list[halo_grp->index];

  ops_particle_halo_group main_halo_grp
      = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;



  if (mpi_group->nhalos == 0) return;

  int nhalos = mpi_group->nhalos;
  double c, t1, t2;
  ops_timers_core(&c, &t1);
  int size = (mpi_group->num_neighbors_send > 0) ?
     mpi_group->num_neighbors_send : 1;
  int *shift_send = (int *) ops_malloc(size * sizeof(int));

  size =
      (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
  int *shift_recv = (int *) ops_malloc(size * sizeof(int));;

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    shift_send[i] = mpi_group->send_shift[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    shift_recv[i] = mpi_group->recv_shift[i];

  //Packing data to send
  int ntot_bites;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      int iproc = halo_mpi->proclist[isend];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_send; iloc++) {
        if (iproc == mpi_group->neighbors_send[iloc]) break;
      }
      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      int ishift = shift_send[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + ishift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites);
      shift_send[iloc] += ntot_bites;
    }
  }
  //Sending data
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int ishift = mpi_group->send_shift[i];
    MPI_Isend(ops_buffer_send_1 + ishift, mpi_group->send_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int ishift = mpi_group->recv_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + ishift, mpi_group->recv_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    ops_particle_halo halo_main = main_halo_grp->halo_list[h];

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos *(mpi_group->num_neighbors_send + iloc) + h];
      int ishift = shift_recv[iloc];
      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + ishift, halo_data,
                                        halo->nhalos, halo_info, halo_main->dir_to,
                                        halo_main->dir_from,
                                        halo_main->translate, &ntot_bites);
      shift_recv[iloc] += ntot_bites;

      for (int imap = 0; imap < halo->particle_to->particle_map_index; imap++) {
        ops_particle_mapping map = halo->particle_to->map_list[imap];
        _ops_particle_remap_virtual(map, halo->particle_to, halo_info->firstrecv,
                                    halo_info->nrecv);
      }
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(shift_recv);
  ops_free(shift_send);
}

void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  ops_particle_halo_group main_halo_grp
      = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;

  double c, t1, t2;
  ops_timers_core(&c, &t1);

  int nhalos = mpi_group->nhalos;
  int size = (mpi_group->num_neighbors_send > 0) ?
      mpi_group->num_neighbors_send : 1;
   int *shift_send = (int *) ops_malloc(size * sizeof(int));

   size =
       (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
   int *shift_recv = (int *) ops_malloc(size * sizeof(int));;


  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    shift_send[i] = mpi_group->send_shift[i];


  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    shift_recv[i] = mpi_group->recv_shift[i];

  int ntot_bites;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      int iproc = halo_mpi->proclist[isend];
      int iloc;
      for (int iloc = 0; iloc < mpi_group->num_neighbors_send;iloc++)
        if (mpi_group->neighbors_send[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      int ishift = shift_send[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + ishift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites);
      shift_send[iloc] += ntot_bites;
    }
  }

  //Sending data
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int ishift = mpi_group->send_shift[i];
    MPI_Isend(ops_buffer_send_1 + ishift, mpi_group->send_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int ishift = mpi_group->recv_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + ishift, mpi_group->recv_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;
    ops_particle_halo halo_main = main_halo_grp->halo_list[h];

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv;iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      int ishift = shift_recv[iloc];
      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + ishift, halo_data,
                                        halo->nhalos, halo_info, halo_main->dir_to,
                                        halo_main->dir_from,
                                        halo_main->translate, &ntot_bites);
      shift_recv[iloc] += ntot_bites;


    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(shift_send);
  ops_free(shift_recv);
}

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp,
                                         ops_access access) {


  ops_mpi_particle_halo_group *mpi_group  = &OPS_mpi_particle_halo_group_list[halo_grp->index];


  if (mpi_group->nhalos == 0) return;

  ops_particle_halo_group main_halo_grp
      = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;

  double c, t1, t2;
  ops_timers_core(&c, &t1);
  int nhalos = mpi_group->nhalos;

  int size = (mpi_group->num_neighbors_send > 0) ?
     mpi_group->num_neighbors_send : 1;
  int *shift_send = (int *) ops_malloc(size * sizeof(int));

  size =
      (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
  int *shift_recv = (int *) ops_malloc(size * sizeof(int));;

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    shift_send[i] = mpi_group->send_shift[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    shift_recv[i] = mpi_group->recv_shift[i];

  int ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc = halo_mpi->proclist[irecv + halo_mpi->nproc_to];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info
       = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      int ishift = shift_recv[iloc];
      _ops_particle_halo_reverse_copy_tobuf(ops_buffer_send_1 + ishift, halo_data,
                                            halo->nhalos, halo_info, &ntot_bites); //TODO
      shift_recv[iloc] += ntot_bites;

    }
  }

  //Send data
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int ishift = mpi_group->recv_shift[i];
    MPI_Isend(ops_buffer_send_1 + ishift, mpi_group->recv_bites[i], MPI_BYTE,
              iproc, 300 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  //Recv data
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int ishift = mpi_group->send_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + ishift, mpi_group->send_bites[i], MPI_BYTE,
              iproc, 300 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[0]);
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  //Unpack data
  ntot_bites =0;
  for (int h = mpi_group->nhalos; h >= 0; h--) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo halo_main = main_halo_grp->halo_list[h];

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      int iproc  = halo_mpi->proclist[isend];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;
      int ishift = shift_recv[iloc];
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * iloc + h]; //TODO: Check this nunmber
      _ops_particle_halo_reverse_copy_from_buff(ops_buffer_recv_1  + ishift, halo->dat, halo->nhalos,
                                               halo_info, halo_main->dir_from, halo_main->dir_to, &ntot_bites,
                                               access);

      shift_recv[iloc] += ntot_bites;
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(shift_recv);
  ops_free(shift_send);
}

/* Build and send particle pos */
void _ops_particle_halo_border_pos_transfer(OPS_instance *instance,
                                            ops_particle_halo_group halo_grp) {
  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  int nhalos = mpi_group->nhalos;
  double c, t1;
  ops_timers_core(&c, &t1);

  int size = (mpi_group->num_neighbors_send > 0)
      ? mpi_group->num_neighbors_send * mpi_group->nhalos : 1;
  int *nsend =
      (int *) ops_malloc(size * sizeof(int));
  size = (mpi_group->num_neighbors_recv > 0)
      ? mpi_group->num_neighbors_recv * mpi_group->nhalos : 1;
  int *nrecv = (int *)ops_malloc(size * sizeof(int));


  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; nsend++)
    nsend[i] = 0;

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; nsend++)
    nrecv[i] = 0;

  size = (mpi_group->num_neighbors_send > 0) ? mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ? mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *) ops_malloc(size * sizeof(int));

  int *nrecv_pos_shift = (int *) ops_malloc(size * sizeof(int));

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    mpi_group->send_bites[i] = 0;
    mpi_group->send_pos_bites[i] = 0;

  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_bites[i] = 0;
    mpi_group->recv_pos_bites[i] = 0;
  }

  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;

    ops_particle_mapping map
    = instance->OPS_particle_halo_list[halo->index]->particle_from->map_list[0];
    ops_dat binhead = map->binhead;
    int *bins = (int *)map->bin->data;
    int *bin_head =(int *)binhead->data;
    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;

    int bites_pos =
        instance->OPS_particle_halo_list[halo->index]->particle_from->particle_pos_dat->elem_size;
    //Find number of elements to send in different halos
    for (int isend = 0; isend < nproc_to; isend++) {

      //Find process and identify storage location
      int iproc = halo->proclist[isend];
      int iloc;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (iproc == mpi_group->neighbors_send[ip]) {iloc = ip; break;}
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];


      //Find number of particles send in the halo to process iproc and stored in
      _ops_particle_number_of_particles_in_range(halo->send_region + 6 * isend, bin_head,
                                                 bins, binhead->size,
                                                 &nsend[nhalos * iloc + h]);

      halo_info->nsend = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist = (int *)
            OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      //TODO: Perform maps into the list
      _ops_particle_mapped_into_region(halo->send_region, bin_head, bins,
                                      binhead->size, halo_info->sendlist);

      //Set size directly

      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
      mpi_group->send_pos_bites[iloc] += bites_pos * halo_info->nsend;
    }
  }

  //Send the data and wait to be received
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos * isend;
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);

  }

   //Receiving data for building chuncks
  for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
    int ishift = mpi_group->nhalos * irecv;

    MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
              mpi_group->neighbors_recv[irecv],
              1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    ops_particle particle_to
      = instance->OPS_particle_halo_list[halo->index]->particle_to;
    int ifirst = particle_to->no_particles + particle_to->no_virtual;
    int no_new = 0;
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    int pos_bites =
        instance->OPS_particle_halo_list[halo->index]->particle_to->particle_pos_dat->elem_size;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];


      //Get storate location
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (iproc == mpi_group->neighbors_recv[ip]) { iloc = ip; break;}
      }



      ops_particle_halo_exchange halo_info =
          mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];
      halo_info->nrecv = nrecv[nhalos * iloc + h];
      halo_info->firstrecv = ifirst;
      ifirst += halo_info->nrecv;
      no_new += halo_info->nrecv;

      mpi_group->recv_bites[iloc] += nbites * halo_info->nrecv; //TODO: Replace with the first
      mpi_group->recv_pos_bites[iloc] += pos_bites * halo_info->nrecv;
    }
    particle_to->no_virtual += no_new;
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  int shift0  = 0;
  int shift1 = 0;

  for (int  i = 0; i < mpi_group->num_neighbors_send;i++) {
    mpi_group->send_shift[i] = shift0;
    mpi_group->shift_send_pos[i] = shift1;

    nsend_shift[i] = shift1;
    shift0 += mpi_group->send_bites[i];
    shift1 += mpi_group->send_pos_bites[i];
  }

  int shift = (shift0 > shift1) ? shift0 : shift1;
  if (shift > ops_buffer_send_1_size) {
    ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
  }

  shift0 = 0;
  shift1 = 1;
  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift0;
    mpi_group->shift_recv_pos[i] = shift1;

    nsend_shift[i] = shift1;
    shift0 += mpi_group->recv_bites[i];
    shift1 += mpi_group->recv_pos_bites[i];
  }

  shift = (shift0 > shift1) ? shift0 : shift1;
  if (shift > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 =
        (char *) OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
    ops_buffer_recv_1_size = shift;
  }

  //Packing halo-data positions
   int ntot_bites = 0;

  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_dat pos_dat = halo->particle_from->particle_pos_dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
       //Get process
      int iproc = halo_mpi->proclist[isend];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_send; ip++) {
        if (mpi_group->neighbors_send[ip] == iproc) {
          iloc = ip; break;
        }
      }

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];

      int shift = nsend_shift[iloc];
      _ops_particle_halo_dat_to_buf(ops_buffer_send_1 + shift, pos_dat, halo_info,
                                    &ntot_bites); //TODO

      nsend_shift[iloc] += ntot_bites;
    }
  }

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + mpi_group->shift_send_pos[i], mpi_group->send_pos_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    MPI_Irecv(ops_buffer_recv_1 + mpi_group->shift_recv_pos[i], mpi_group->recv_pos_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  ntot_bites = 0;
  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
  }

  //Unpacking positions and mapping
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_dat coords = halo->particle_to->particle_pos_dat;

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      //Get process and storage id
      int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc = -1;
      for (int ip = 0; ip < mpi_group->num_neighbors_recv; ip++) {
        if (mpi_group->neighbors_recv[ip] == iproc) {
          iloc = ip; break;
        }
      }
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (iloc + mpi_group->num_neighbors_send) + h];

      int shift = nrecv_shift[iloc];
      _ops_particle_dat_copy_from_buff(ops_buffer_recv_1 + shift, coords, OPS_PART_POSITION,
                                        halo_info, halo->dir_to, halo->dir_from,
                                        halo->translate,
                                        &ntot_bites); //TODO
      nrecv_shift[iloc] += ntot_bites;
      //NEED TO MAP THOSE ELEMENTS AS WELL
      for (int imap = 0; imap > halo->particle_to->particle_map_index; imap++) {
        ops_particle_mapping map = halo->particle_to->map_list[imap];
        _ops_particle_mapping_virtual_from_halo(map, halo->particle_to,
                                                halo_info->firstrecv,
                                                halo_info->nrecv);

      }
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  _ops_particle_update_dependent_halo_groups(instance, halo_grp, mpi_group);

  ops_free(nsend);
  ops_free(nrecv);
  ops_free(nsend_shift);
  ops_free(nrecv_shift);
}

void _ops_particle_exchange(ops_particle particle) {

  //REMOVE VIRTUAL

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  int nlocal = particle->no_particles;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int *mark_deletion = particle->mark_deletion;
  double  *xpos = (double *)particle->particle_pos_dat->data;

  for (int idir = 0; idir < dim; idir++) {

    size_t nsend_recv_bites[4];
    for (int i = 0; i < 4; i++)
      nsend_recv_bites[i] = 0;

    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (size_t ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 1) {
        //Check if particle would be mapped for allocation
        if (xpos[dim * ipart + idir] < halo_int->region_exch_neg[1]
            && sb->id_m[idir] != MPI_PROC_NULL) {
          nsend_recv_bites[0]++;
          mark_deletion[ipart] = 2;
          continue;
        }

        if (xpos[dim * ipart + idir] > halo_int->region_exch_pos[0]  &&
            sb->id_p[idir] != MPI_PROC_NULL)
        {
          nsend_recv_bites[1]++;
          mark_deletion[ipart] = 3;
        }
      }
    }

    //marking for deletion

      //Allocate if necessary forward_pos structures
    if (nsend_recv_bites[0] > halo_int->nalloc_max_neg && sb->id_m[idir] != MPI_PROC_NULL) {
      halo_int->particle_send_neg = (int *) OPS_realloc_fast((char *) halo_int->particle_send_neg,
                                                             halo_int->nalloc_max_neg,
                                                             nsend_recv_bites[0]);
      halo_int->nalloc_max_neg  = nsend_recv_bites[0];
      if (nsend_recv_bites[0] * sp->bites_in_exchange > ops_buffer_send_1_size) {
        ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                                      nsend_recv_bites[0] * sp->bites_in_exchange);
        ops_buffer_send_1_size = nsend_recv_bites[0] * sp->bites_in_exchange;
      }

        //TODO: Check if we need to allocate structures for sending

    }

    if (nsend_recv_bites[1] > halo_int->nalloc_max_pos && sb->id_p[idir] != MPI_PROC_NULL) {
      halo_int->particle_send_pos = (int *) OPS_realloc_fast((char *) halo_int->particle_send_pos,
                                                             halo_int->nalloc_max_pos,
                                                             nsend_recv_bites[1]);
      halo_int->nalloc_max_pos = nsend_recv_bites[1]; //
      if (nsend_recv_bites[1] * sp->bites_in_exchange > ops_buffer_send_2_size) {
        ops_buffer_send_2 = (char *) OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                                      nsend_recv_bites[1] * sp->bites_in_exchange);
       ops_buffer_send_2_size = nsend_recv_bites[1] * sp->bites_in_exchange;
      }
    }

    int isend_neg = 0;
    int isend_pos = 0;
    //Create packing structures
    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 2) {
        mark_deletion[ipart] = 4;
        halo_int->particle_send_neg[isend_neg] = ipart;
        isend_neg++;
      }
      else if (mark_deletion[ipart] == 3) {
        mark_deletion[ipart] = 4;
        halo_int->particle_send_pos[isend_pos] = ipart;
        isend_pos++;
      }
    }

    //Packing all data
    ops_dat pos_dat = particle->particle_pos_dat;
    int shift_pos = 0;
    int shift_neg = 0;

    _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                    pos_dat,
                                    halo_int->particle_send_neg,
                                    isend_neg);

    _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                    pos_dat,
                                    halo_int->particle_send_pos,
                                    isend_pos);

    shift_neg += nsend_recv_bites[0] * pos_dat->elem_size;
    shift_pos += nsend_recv_bites[1] * pos_dat->elem_size;
    if (particle->particle_envelope != nullptr)  {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      particle->particle_envelope,
                                      halo_int->particle_send_neg,
                                      isend_neg);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_neg,
                                      particle->particle_envelope,
                                      halo_int->particle_send_pos,
                                      isend_pos);

      shift_neg += nsend_recv_bites[0] * particle->particle_envelope->elem_size;
      shift_pos += nsend_recv_bites[1] * particle->particle_envelope->elem_size;
    }

    if (particle->ids != nullptr) {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      particle->ids,
                                      halo_int->particle_send_neg,
                                      isend_neg);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_neg,
                                      particle->ids,
                                      halo_int->particle_send_pos,
                                      isend_pos);

      shift_neg += nsend_recv_bites[0] * particle->ids->elem_size;
      shift_pos += nsend_recv_bites[1] * particle->ids->elem_size;
    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      if (!dat->is_exchangable) continue;

      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      dat,
                                      halo_int->particle_send_neg,
                                      isend_neg);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                      dat,
                                      halo_int->particle_send_pos,
                                      isend_pos);

      shift_neg += nsend_recv_bites[0] * dat->elem_size;
      shift_pos += nsend_recv_bites[1] * dat->elem_size;
    }

    ops_block block = particle->block;
    OPS_instance *instance = OPS_instance::getOPSInstance();

    for (int ihis = 0; ihis < instance->OPS_block_list[block->index].no_history_structures;
         ihis++) {
      //ops_neighbor_history history = ;
      ops_neighbor_history history = instance->OPS_block_list[block->index].histories[ihis];

      if (history->particleI->index != particle->index) continue;


      shift_neg += _ops_particle_intra_hist_to_buff(ops_buffer_send_1 + shift_neg,
                                                    history,
                                                    halo_int->particle_send_neg + shift_neg,
                                                    isend_neg);

     shift_pos += _ops_particle_intra_hist_to_buff(ops_buffer_send_2 + shift_pos,
                                                   history,
                                                   halo_int->particle_send_pos + shift_pos,
                                                   isend_pos);

    }


    //Send and receive element sizes
    MPI_Status status[4];

    MPI_Sendrecv(&nsend_recv_bites[0], 1, MPI_INT, sb->id_m[idir], 100,
                 &nsend_recv_bites[2], 1, MPI_INT, sb->id_p[idir], 100,
                 sb->comm, &status[0]);

    MPI_Sendrecv(&nsend_recv_bites[1], 1, MPI_INT, sb->id_p[idir], 200,
                 &nsend_recv_bites[3], 1, MPI_INT, sb->id_m[idir], 200,
                 sb->comm, &status[0]);

    //Reallocate if necessary the receiving buffs
    if (nsend_recv_bites[2] * sp->bites_in_exchange > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = (char *) OPS_realloc_fast(ops_buffer_recv_1,
                                                    ops_buffer_recv_1_size,
                                                    nsend_recv_bites[2] * sp->bites_in_exchange);
      ops_buffer_recv_1_size = nsend_recv_bites[2] * sp->bites_in_exchange;
    }

    if (nsend_recv_bites[3] * sp->bites_in_exchange > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = (char *) OPS_realloc_fast(ops_buffer_recv_2,
                                                          ops_buffer_recv_2_size,
                                                          nsend_recv_bites[3] * sp->bites_in_exchange);
      ops_buffer_recv_2_size = nsend_recv_bites[3] * sp->bites_in_exchange;

    }

    //Send and receive
    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, nsend_recv_bites[0] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv_bites[0] > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    MPI_Isend(ops_buffer_send_2, nsend_recv_bites[1] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv_bites[1] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    MPI_Irecv(ops_buffer_recv_1, nsend_recv_bites[2] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv_bites[2] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);

    MPI_Irecv(ops_buffer_recv_2, nsend_recv_bites[3] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv_bites[3] > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);

    int nexist = particle->no_particles;
    particle->no_particles += nsend_recv_bites[2] + nsend_recv_bites[3];

    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }

    int shift_recv_neg = 0;
    int shift_recv_pos = 0;


    _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                    particle->particle_pos_dat,
                                    nexist, nsend_recv_bites[3]);

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                    particle->particle_pos_dat,
                                    nexist + nsend_recv_bites[3],
                                    nsend_recv_bites[2]);

    shift_recv_neg += nsend_recv_bites[3] * particle->particle_pos_dat->elem_size;
    shift_recv_pos += nsend_recv_bites[2] * particle->particle_pos_dat->elem_size;

    if (particle->particle_envelope != nullptr) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      particle->particle_envelope, nexist,
                                      nsend_recv_bites[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      particle->particle_envelope,
                                      nexist + nsend_recv_bites[3],
                                      nsend_recv_bites[2]); //TODO
      shift_recv_neg += nsend_recv_bites[3] * particle->particle_envelope->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * particle->particle_envelope->elem_size;
    }

    if (particle->ids != nullptr) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      particle->ids, nexist,
                                      nsend_recv_bites[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      particle->ids,
                                      nexist + nsend_recv_bites[3],
                                      nsend_recv_bites[2]); //TODO
      shift_recv_neg += nsend_recv_bites[3] * particle->ids->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * particle->ids->elem_size;
    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {

      ops_dat dat = particle->particle_dat[idat];
      if (!dat->is_exchangable) continue;
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      dat, nexist,
                                      nsend_recv_bites[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos, dat,
                                      nexist + nsend_recv_bites[3],
                                      nsend_recv_bites[2]);


      shift_recv_neg += nsend_recv_bites[3] * dat->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * dat->elem_size;
    }

    for (int ihis = 0; ihis < instance->OPS_block_list[block->index].no_history_structures;
         ihis++) {
      //ops_neighbor_history history = ;
      ops_neighbor_history history = instance->OPS_block_list[block->index].histories[ihis];
      shift_recv_neg += _ops_particle_intra_buff_to_hist(ops_buffer_recv_2 + shift_recv_neg,
                                                         history, nexist, nsend_recv_bites[3]);

      shift_recv_pos += _ops_particle_intra_buff_to_hist(ops_buffer_recv_1 + shift_recv_pos,
                                                         history, nexist, nsend_recv_bites[2]);
    }

    MPI_Waitall(2, &request[0], &status[0]);

    //TODO: Set and shift for non-local if not-within
    for (size_t ipart = nexist; ipart < particle->no_particles; ipart++) {
      switch(particle->type_box) {
      case sizeof(float):
        particle->mark_deletion[ipart] =
              (! ((BoundingBox<float> *)particle->box_block)->isCoordinateInBoundingBox((float *) particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      break;
      case sizeof(double):
        particle->mark_deletion[ipart] =
         (! ((BoundingBox<double> *)particle->box_block)->isCoordinateInBoundingBox((double *) particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      break;
      case sizeof(long double):
        particle->mark_deletion[ipart] =
                  (! ((BoundingBox<long double> *)particle->box_block)->isCoordinateInBoundingBox((long double *) particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      }

    }
  }

  //Remove actual particles
  _ops_particle_remove_marked_flag(particle, 4);

}


//TODO: We have the same issue herein as well
void _ops_particle_exchange_map_update(ops_particle particle) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  int nlocal = particle->no_particles;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int *mark_deletion = particle->mark_deletion;

  for (int idir = 0; idir < dim; idir++) {
    int nsend_recv[4];

    for (int i = 0; i < 4; i++)
      nsend_recv[i] = 0;

    //TODO: Shift to idp-s: If on virtual elements then shift or beyond
    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 1) {
        switch(particle->type_box) {
        case sizeof(float): {
          if (((float *)particle->particle_pos_dat->data)[ipart * dim + idir] <
              ((float *)halo_int->region_exch_neg)[1] &&
                sb->id_m[idir] != MPI_PROC_NULL) {
            nsend_recv[0]++;
            mark_deletion[ipart] = 2;
            continue;
          }

          if (((float *) particle->particle_pos_dat->data)[ipart * dim + idir] >
              ((float *) halo_int->region_exch_pos)[0] &&
              sb->id_p[idir] != MPI_PROC_NULL) {
                nsend_recv[1]++;
                mark_deletion[ipart] = 3;
          }
          } break;
        case sizeof(double): {
          if (((double *)particle->particle_pos_dat->data)[ipart * dim + idir] <
              ((double *)halo_int->region_exch_neg)[1] &&
              sb->id_m[idir] != MPI_PROC_NULL) {
            nsend_recv[0]++;
            mark_deletion[ipart] = 2;
            continue;
          }

          if (((double *) particle->particle_pos_dat->data)[ipart * dim + idir] >
              ((double *) halo_int->region_exch_pos)[0] &&
              sb->id_p[idir] != MPI_PROC_NULL) {
                nsend_recv[1]++;
                mark_deletion[ipart] = 3;
          }
          } break;
        case sizeof(long double): {
          if (((long double *)particle->particle_pos_dat->data)[ipart * dim + idir] <
              ((long double *)halo_int->region_exch_neg)[1] &&
                sb->id_m[idir] != MPI_PROC_NULL) {
             nsend_recv[0]++;
             mark_deletion[ipart] = 2;
             continue;
          }

          if (((long double *) particle->particle_pos_dat->data)[ipart * dim + idir] >
              ((long double *) halo_int->region_exch_pos)[0] &&
              sb->id_p[idir] != MPI_PROC_NULL) {
                nsend_recv[1]++;
                mark_deletion[ipart] = 3;
          }
          } break;
        }
      }
    }

    if (nsend_recv[0] > halo_int->nalloc_max_neg && sb->id_m[idir] != MPI_PROC_NULL) {
      halo_int->particle_send_neg = (int *) OPS_realloc_fast((char *) halo_int->particle_send_neg,
                                                             halo_int->nalloc_max_neg,
                                                             nsend_recv[0]);
      halo_int->nalloc_max_neg = nsend_recv[0];

      if (nsend_recv[0] * sp->bites_in_exchange > ops_buffer_send_1_size) {
        ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                                      nsend_recv[0] * sp->bites_in_exchange);
        ops_buffer_send_1_size = nsend_recv[0] * sp->bites_in_exchange;
      }
    }

    if (nsend_recv[1] > halo_int->nalloc_max_pos && sb->id_p[idir] != MPI_PROC_NULL) {
      halo_int->particle_send_pos = (int *) OPS_realloc_fast((char *) halo_int->particle_send_pos,
                                                             halo_int->nalloc_max_pos,
                                                             nsend_recv[1]);
      halo_int->nalloc_max_pos = nsend_recv[1]; //
      if (nsend_recv[1] * sp->bites_in_exchange > ops_buffer_send_2_size) {
        ops_buffer_send_2 = (char *) OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                                      nsend_recv[1] * sp->bites_in_exchange);
       ops_buffer_send_2_size = nsend_recv[1] * sp->bites_in_exchange;
      }
    }

    //Creating packing structures
    int isend_neg = 0;
    int isend_pos = 0;



    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 2) {
        mark_deletion[ipart] = 4;
        halo_int->particle_send_neg[isend_neg] = ipart;
        isend_neg++;
      }
      else if (mark_deletion[ipart] == 3) {
        mark_deletion[ipart] = 4;
        halo_int->particle_send_pos[isend_pos] = ipart;
        isend_pos++;
      }
    }


    //Packing data
    ops_dat pos_dat = particle->particle_pos_dat;
    int shift_pos = 0;
    int shift_neg = 0;

    _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                    pos_dat,halo_int->particle_send_neg,
                                    nsend_recv[0]);

    _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                    pos_dat,halo_int->particle_send_pos,
                                    nsend_recv[1]);

    shift_neg += nsend_recv[0] * pos_dat->elem_size;
    shift_pos += nsend_recv[1] * pos_dat->elem_size;

    if (particle->particle_envelope != nullptr) {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      particle->particle_envelope,
                                      halo_int->particle_send_neg,
                                      nsend_recv[0]);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                      particle->particle_envelope,
                                      halo_int->particle_send_pos,
                                      nsend_recv[1]);

      shift_neg += nsend_recv[0] * particle->particle_envelope->elem_size;
      shift_pos += nsend_recv[1] * particle->particle_envelope->elem_size;

    }

    if (particle->ids != nullptr) {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      particle->ids,
                                      halo_int->particle_send_neg,
                                      nsend_recv[0]);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                      particle->ids,
                                      halo_int->particle_send_pos,
                                      nsend_recv[1]);

      shift_neg += nsend_recv[0] * particle->ids->elem_size;
      shift_pos += nsend_recv[1] * particle->ids->elem_size;

    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      dat, halo_int->particle_send_neg,
                                      nsend_recv[0]);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + shift_pos,
                                      dat, halo_int->particle_send_pos,
                                      nsend_recv[1]);

      shift_neg += nsend_recv[0] * dat->elem_size;
      shift_pos += nsend_recv[1] * dat->elem_size;
    }


    //Check what we send
    //Send and receive elements
    MPI_Status status[4];

    MPI_Sendrecv(&nsend_recv[0], 1, MPI_INT, sb->id_m[idir], 100,
                 &nsend_recv[2], 1, MPI_INT, sb->id_p[idir], 100,
                 sb->comm, &status[0]);

    MPI_Sendrecv(&nsend_recv[1], 1, MPI_INT, sb->id_p[idir], 200,
                 &nsend_recv[3], 1, MPI_INT, sb->id_m[idir], 200,
                 sb->comm, &status[0]);

    //Reallocate if necessary
    if (nsend_recv[2] * sp->bites_in_exchange > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = (char *) OPS_realloc_fast(ops_buffer_recv_1,
                                                    ops_buffer_recv_1_size,
                                                    nsend_recv[2] * sp->bites_in_exchange);
      ops_buffer_recv_1_size = nsend_recv[2] * sp->bites_in_exchange;
    }

    if (nsend_recv[3] * sp->bites_in_exchange > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = (char *) OPS_realloc_fast(ops_buffer_recv_2,
                                                          ops_buffer_recv_2_size,
                                                          nsend_recv[3] * sp->bites_in_exchange);
      ops_buffer_recv_2_size = nsend_recv[3] * sp->bites_in_exchange;

    }

    //Send and receive
    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, nsend_recv[0] * sp->bites_in_exchange,
              MPI_BYTE,  nsend_recv[0] > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    MPI_Isend(ops_buffer_send_2, nsend_recv[1] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv[1] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    MPI_Irecv(ops_buffer_recv_1, nsend_recv[2] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv[2] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);

    MPI_Irecv(ops_buffer_recv_2, nsend_recv[3] * sp->bites_in_exchange,
              MPI_BYTE, nsend_recv[3] > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);


    size_t nexist = particle->no_particles;
    particle->no_particles += nsend_recv[2] + nsend_recv[3];

    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }


    //Realloc maps as well //

    //Unpacking data
    int shift_recv_neg = 0;
    int shift_recv_pos = 0;

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                    particle->particle_pos_dat,
                                    nexist,
                                    nsend_recv[3]);

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                    particle->particle_pos_dat,
                                    nexist + nsend_recv[3],
                                    nsend_recv[2]);

    shift_recv_neg += nsend_recv[3] * particle->particle_pos_dat->elem_size;
    shift_recv_pos += nsend_recv[2] * particle->particle_pos_dat->elem_size;

    if (particle->particle_envelope != nullptr) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      particle->particle_envelope, nexist,
                                      nsend_recv[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      particle->particle_envelope,
                                      nexist + nsend_recv[3],
                                      nsend_recv[2]);

      shift_recv_neg += nsend_recv[3] * particle->particle_envelope->elem_size;
      shift_recv_pos += nsend_recv[2] * particle->particle_envelope->elem_size;
    }


    if (particle->ids != nullptr) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      particle->ids, nexist,
                                      nsend_recv[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      particle->ids,
                                      nexist + nsend_recv[3],
                                      nsend_recv[2]);

      shift_recv_neg += nsend_recv[3] * particle->ids->elem_size;
      shift_recv_pos += nsend_recv[2] * particle->ids->elem_size;
    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];


      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg, dat,
                                      nexist, nsend_recv[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos, dat,
                                      nexist + nsend_recv[3], nsend_recv[2]);

      shift_recv_neg += nsend_recv[3] * dat->elem_size;
      shift_recv_pos += nsend_recv[2] * dat->elem_size;
    }

    MPI_Waitall(2, &request[0], &status[0]);

    double *xcrds = (double *) particle->particle_pos_dat->data;
    for (size_t ipart = nexist; ipart < particle->no_particles; ipart++) {
      switch(particle->type_box) {
      case sizeof(float):
        particle->mark_deletion[ipart] =
             (! ((BoundingBox<float> *)particle->box_block)->isCoordinateInBoundingBox((float *)particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      break;
      case sizeof(double):
        particle->mark_deletion[ipart] =
             (! ((BoundingBox<double> *)particle->box_block)->isCoordinateInBoundingBox((double *)particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      break;
      case sizeof(long double):
        particle->mark_deletion[ipart] =
             (! ((BoundingBox<long double> *)particle->box_block)->isCoordinateInBoundingBox((long double *)particle->particle_pos_dat->data + ipart * dim)) ? 1 : 0;
      break;
      }
    }
  }

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    _ops_particle_mapping_virtual_from_halo(map, particle, nlocal,
                                            particle->no_particles - nlocal); //NEEDS CAUTION
  }

  //Remove particles with maps and flag
  _ops_particle_remove_flag_reset_map(particle, 4);


}


void _ops_particle_build_border(ops_particle particle){

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  for (int idir = 0; idir < dim; idir++) {
    int nsed_recv[4];

    int nfirst = particle->no_particles + particle->no_virtual;

    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    int nswaps = halo_int->nswaps;
    size_t ntot_neg = 0;
    size_t ntot_pos = 0;
    int nswap_bites  =0;

    int ishift_neg = 0;
    int ishift_pos = 0;

    int ifirst_send = 0;
    int ilast_send = particle->no_particles + particle->no_virtual;
    for (int iswap = 0; iswap < nswaps; iswap++) {


      sp->particle_halos[idir]->nforward_pos[iswap] = 0;
      sp->particle_halos[idir]->nforward_neg[iswap] = 0;


      _ops_particle_find_intra_box(particle, halo_int, iswap, ifirst_send, ilast_send); //TODO

      ntot_pos += sp->particle_halos[idir]->nforward_pos[iswap];
      ntot_neg += sp->particle_halos[idir]->nforward_neg[iswap];


      if (ntot_neg > halo_int->nalloc_max_neg) {
        halo_int->particle_send_neg = (int *)
            OPS_realloc_fast((char *)halo_int->particle_send_neg, halo_int->nalloc_max_neg * sizeof(int),
                              ntot_neg * sizeof(int));
        halo_int->nalloc_max_neg = ntot_neg;
      }

      if (ntot_pos > halo_int->nalloc_max_pos) {
        halo_int->particle_send_pos = (int *)
            OPS_realloc_fast((char *)halo_int->particle_send_pos, halo_int->nalloc_max_pos * sizeof(int),
                               ntot_pos * sizeof(int));
        halo_int->nalloc_max_pos = ntot_pos;
      }

      _ops_particle_set_intra_border_box(particle, halo_int, iswap, ifirst_send, ilast_send); //TODO


      int elem_size = sp->bites_in_exchange;

      if (sp->particle_halos[idir]->nforward_pos[iswap] * elem_size
           > ops_buffer_send_2_size) {
        ops_buffer_send_2 =  OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                              sp->particle_halos[idir]->nforward_pos[iswap] *
                                              elem_size);
        ops_buffer_send_2_size = sp->particle_halos[idir]->nforward_pos[iswap]
                               * elem_size;
      }

      if (   sp->particle_halos[idir]->nforward_neg[iswap]
          * elem_size > ops_buffer_send_1_size) {
        ops_buffer_send_1 =  OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                              sp->particle_halos[idir]->nforward_neg[iswap]
                                              * elem_size);
        ops_buffer_send_1_size = sp->particle_halos[idir]->nforward_neg[iswap]
                               * elem_size;
      }



      //HERE WE PACK AND SEND ONLY PARTICLE POSITIONS
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1,
                                      particle->particle_pos_dat,
                                      halo_int->particle_send_neg + ishift_neg,
                                      halo_int->nforward_neg[iswap]);
      _ops_particle_intra_dat_to_buff(ops_buffer_send_2,
                                      particle->particle_pos_dat,
                                      halo_int->particle_send_pos + ishift_pos,
                                      halo_int->nforward_pos[iswap]);

      ishift_neg += halo_int->nforward_neg[iswap];
      ishift_pos += halo_int->nforward_pos[iswap];
      //Send and receive data
      MPI_Status status[4];

      int nforward = halo_int->nforward_neg[iswap];
      int nrecv = 0;
      MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_m[idir], 100,
                   &nrecv, 1, MPI_INT, sb->id_p[idir], 100,
                   sb->comm, &status[0]);
      halo_int->nrecv_pos[iswap] = nrecv;

      nforward = halo_int->nforward_pos[iswap];
      nrecv = 0; //Need that for setting the simulation
      MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_p[idir], 200,
                   &nrecv, 1, MPI_INT, sb->id_m[idir], 200,
                   sb->comm, &status[0]);
      halo_int->nrecv_neg[iswap] = nrecv;

      //Allocate data if necessary
      if (elem_size * halo_int->nrecv_pos[iswap] > ops_buffer_recv_1_size) {
        ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                             elem_size * halo_int->nrecv_pos[iswap]);
        ops_buffer_recv_1_size = elem_size * halo_int->nrecv_pos[iswap];
      }

      if (elem_size * halo_int->nrecv_neg[iswap] > ops_buffer_recv_2_size) {
        ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_1_size,
                                             elem_size * halo_int->nrecv_neg[iswap]);
        ops_buffer_recv_2_size =  elem_size * halo_int->nrecv_neg[iswap];
      }

      //Send and receive data
      int size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_neg[iswap];
      MPI_Request request[4];
      MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE, (size_send > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[0]);

      size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_pos[iswap];
      MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE, (size_send > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[1]);

      int size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap];
      MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[2]);

      size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_neg[iswap];
      MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[3]);


      MPI_Waitall(2, &request[2], &status[2]);

      halo_int->irecv_neg[iswap] = ilast_send;
      halo_int->irecv_pos[iswap] = ilast_send + halo_int->nrecv_neg[iswap];



      particle->no_virtual += halo_int->nrecv_pos[iswap] + halo_int->nrecv_neg[iswap];

      if (particle->no_particles + particle->no_virtual > particle->Nmax) {
        ops_particle_realloc_data(particle, particle->no_particles + particle->no_virtual);
      }

      //unpack structures
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2,
                                      particle->particle_pos_dat,
                                      halo_int->irecv_neg[iswap],
                                      halo_int->nrecv_neg[iswap]);


      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1,
                                      particle->particle_pos_dat,
                                      halo_int->irecv_pos[iswap],
                                      halo_int->nrecv_pos[iswap]);

      MPI_Waitall(2, &request[0], &status[0]);



      //Reset the send and receive
       ifirst_send = ilast_send;
       ilast_send = ifirst_send + halo_int->nrecv_neg[iswap] + halo_int->nrecv_pos[iswap];
    }
  }
}

void _ops_particle_border_dats(ops_particle particle, ops_dat *dats,
                               int ndats, ops_neighbor_history *histories,
                               int nhistories) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int elem_size = 0;
  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: ops_dat is not "
                         "a particle ops_dat structure\n");
    elem_size += dats[idat]->elem_size;
  }

  for (int ihist = 0; ihist < nhistories; ihist++) {
    if (histories[ihist]->particleI->index != particle->index)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: History structure not related with "
                                            " the particle\n");
    elem_size += histories[ihist]->n_partnersI->elem_size
               + histories[ihist]->partnersI->elem_size
               + histories[ihist]->data->elem_size;
  }

  for (int idir = 0; idir < dim; idir++) {

 //   printf("Number of swaps R %d dir %d: %d\n", ops_get_proc(), idir,sp->particle_halos[idir]->nswaps);

//    int nsend_recv[4];
    int nfirst = particle->no_particles + particle->no_virtual;

    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    int nswaps = halo_int->nswaps;
    size_t ntot_neg = 0;
    size_t ntot_pos = 0;
    int nswap_bites = 0;

    int ishift_neg = 0;
    int ishift_pos = 0;

    int ifirst_send = 0;
    int ilast_send = particle->no_particles + particle->no_virtual;

    for (int iswap = 0; iswap < nswaps; iswap++) {
      //Part I: Find particles within box */
      sp->particle_halos[idir]->nforward_pos[iswap] = 0;
      sp->particle_halos[idir]->nforward_neg[iswap] = 0;

      //Part I: Find particles within the sending zones & reallocate controlling lists
      _ops_particle_find_intra_box(particle, halo_int, iswap, ifirst_send,
                                   ilast_send);

      ntot_pos += sp->particle_halos[idir]->nforward_pos[iswap];
      ntot_neg += sp->particle_halos[idir]->nforward_neg[iswap];

      if (ntot_neg > halo_int->nalloc_max_neg) {
        halo_int->particle_send_neg = (int *)
            OPS_realloc_fast((char *)halo_int->particle_send_neg, halo_int->nalloc_max_neg * sizeof(int),
                              ntot_neg * sizeof(int));
        halo_int->nalloc_max_neg = ntot_neg;
      }

      if (ntot_pos > halo_int->nalloc_max_pos) { //TODO: Set the nalloc_max_pos to sth large
        halo_int->particle_send_pos = (int *)
            OPS_realloc_fast((char *)halo_int->particle_send_pos, halo_int->nalloc_max_pos * sizeof(int),
                               ntot_pos * sizeof(int));
        halo_int->nalloc_max_pos = ntot_pos;
      }


      //Part II: Build intra-block halo
      _ops_particle_set_intra_border_box(particle, halo_int, iswap, ifirst_send, ilast_send);


      //Part III: Re-allocate structures if necessary
      if (sp->particle_halos[idir]->nforward_pos[iswap] * elem_size
           > ops_buffer_send_2_size) {
        ops_buffer_send_2 =  OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                              sp->particle_halos[idir]->nforward_pos[iswap] *
                                              elem_size);
        ops_buffer_send_2_size = sp->particle_halos[idir]->nforward_pos[iswap]
                               * elem_size;
      }

      if (sp->particle_halos[idir]->nforward_neg[iswap]
          * elem_size > ops_buffer_send_1_size) {
        ops_buffer_send_1 =  OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                              sp->particle_halos[idir]->nforward_neg[iswap]
                                              * elem_size);
        ops_buffer_send_1_size = sp->particle_halos[idir]->nforward_neg[iswap]
                               * elem_size;
      }

      //Part IV: Pack ops-data
      int ishift_pos_buf = 0;
      int ishift_neg_buf = 0;

      for (int idat = 0; idat < ndats; idat++) {
        _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + ishift_neg_buf,
                                        dats[idat], halo_int->particle_send_neg + ishift_neg,
                                        halo_int->nforward_neg[iswap]);
        ishift_neg_buf += dats[idat]->elem_size * halo_int->nforward_neg[iswap];

        _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + ishift_pos_buf,
                                        dats[idat], halo_int->particle_send_pos + ishift_pos,
                                        halo_int->nforward_pos[iswap]);
        ishift_pos_buf += dats[idat]->elem_size * halo_int->nforward_pos[iswap];
      }

      //Part IVa: Pack history-data
      for (int ihis = 0; ihis < nhistories; ihis++) {
        ishift_neg_buf += _ops_particle_intra_hist_to_buff(ops_buffer_send_1 + ishift_neg_buf,
                                                           histories[ihis],
                                                           halo_int->particle_send_neg + ishift_neg,
                                                           halo_int->nforward_neg[iswap]);

        ishift_pos_buf += _ops_particle_intra_hist_to_buff(ops_buffer_send_2 + ishift_pos_buf,
                                                           histories[ihis],
                                                           halo_int->particle_send_pos + ishift_pos,
                                                           halo_int->nforward_pos[iswap]);
      }


      //Update ishifts to get correct data structures
      ishift_neg += halo_int->nforward_neg[iswap];
      ishift_pos += halo_int->nforward_pos[iswap];

      /*Part V: Send and receiving virtual particles in the direction */
      MPI_Status status[4];
      int nforward = halo_int->nforward_neg[iswap];
      int nrecv = 0;
      MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_m[idir], 100,
                   &nrecv, 1, MPI_INT, sb->id_p[idir], 100,
                   sb->comm, &status[0]);
      halo_int->nrecv_pos[iswap] = nrecv;

      nforward = halo_int->nforward_pos[iswap];
      nrecv  = 0;
      MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_p[idir], 200,
                   &nrecv, 1, MPI_INT, sb->id_m[idir], 200,
                   sb->comm, &status[0]);
      halo_int->nrecv_neg[iswap] = nrecv;

      /* Part VI: Data allocation for receiving data */
      if (elem_size * halo_int->nrecv_pos[iswap] > ops_buffer_recv_1_size) {
        ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                             elem_size * halo_int->nrecv_pos[iswap]);
        ops_buffer_recv_1_size = elem_size * halo_int->nrecv_pos[iswap];
      }

      if (elem_size * halo_int->nrecv_neg[iswap] > ops_buffer_recv_2_size) {
        ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_1_size,
                                             elem_size * halo_int->nrecv_neg[iswap]);
        ops_buffer_recv_2_size =  elem_size * halo_int->nrecv_neg[iswap];
      }

      /* Part VII: Sending and receiving data structures */
      int size_send = halo_int->nforward_neg[iswap] * elem_size;
      MPI_Request request[4];

      MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE,
                size_send > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[0]);

      size_send = halo_int->nforward_pos[iswap] * elem_size;
      MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE,
                size_send > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[1]);

      int size_recv = halo_int->nrecv_pos[iswap] * elem_size;
      MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE,
                size_recv > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[2]);

      size_recv = halo_int->nrecv_neg[iswap] * elem_size;
      MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE,
                size_recv > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[3]);

      MPI_Waitall(2, &request[2], &status[2]);

      halo_int->irecv_neg[iswap] = ilast_send;
      halo_int->irecv_pos[iswap] = ilast_send + halo_int->nrecv_neg[iswap];

      particle->no_virtual += halo_int->nrecv_pos[iswap] + halo_int->nrecv_neg[iswap];


      if (particle->no_particles + particle->no_virtual > particle->Nmax) {
        ops_particle_realloc_data(particle, particle->no_particles + particle->no_virtual);
      }

      //Part IX: Unpacking data
      int irecv_shift_neg = 0;
      int irecv_shift_pos = 0;

      for (int idat = 0; idat < ndats; idat++) {
        _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + irecv_shift_neg,
                                        dats[idat],
                                        halo_int->irecv_neg[iswap],
                                        halo_int->nrecv_neg[iswap]);

        irecv_shift_neg += halo_int->nrecv_neg[iswap] * dats[idat]->elem_size;

        _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + irecv_shift_pos,
                                        dats[idat],
                                        halo_int->irecv_pos[iswap],
                                        halo_int->nrecv_pos[iswap]);

        irecv_shift_pos += halo_int->nrecv_pos[iswap] * dats[idat]->elem_size;
      }

      for (int ihis = 0; ihis < nhistories; ihis++) {
        irecv_shift_neg += _ops_particle_intra_buff_to_hist(ops_buffer_recv_2 + irecv_shift_neg,
                                                            histories[ihis],
                                                            halo_int->irecv_neg[iswap],
                                                            halo_int->nrecv_neg[iswap]);

        irecv_shift_pos += _ops_particle_intra_buff_to_hist(ops_buffer_recv_2 + irecv_shift_pos,
                                                            histories[ihis],
                                                            halo_int->irecv_pos[iswap],
                                                            halo_int->nrecv_pos[iswap]); //TODO
      }

      MPI_Waitall(2, &request[0], &status[0]);
      ifirst_send = ilast_send;
      ilast_send = ifirst_send + halo_int->nrecv_neg[iswap] + halo_int->nrecv_pos[iswap];
    }
  }

}


void _ops_particle_border_dats_with_maps(ops_particle particle, ops_dat *dats,
                                         int ndats, ops_neighbor_history *histories,
                                         int nhistories) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  int elem_size = 0;
  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: ops_dat is not "
                         "a particle ops_dat structure\n");
    elem_size += dats[idat]->elem_size;
  }

  for (int ihist = 0; ihist < nhistories; ihist++) {
    if (histories[ihist]->particleI->index != particle->index)
      throw OPSException(OPS_RUNTIME_ERROR, "Error: History structure not related with "
                                            " the particle\n");
    elem_size += histories[ihist]->n_partnersI->elem_size
               + histories[ihist]->partnersI->elem_size
               + histories[ihist]->data->elem_size;
  }


  ops_dat binhead = particle->map_list[0]->binhead;
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    d_m[i] = binhead->d_m[i] + OPS_sub_dat_list[binhead->index]->d_im[i];
    d_p[i] = binhead->d_p[i] + OPS_sub_dat_list[binhead->index]->d_ip[i];
  }

  for (int idir = 0; idir < dim ; idir++) {

    ops_int_particle_halos halo_int = sp->particle_halos[idir];

    if (halo_int->nswaps > 1)
      throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Intra-block halos based on maps "
                                           "require a singe swap");
    int nsend_recv[4];
    int not_neg = 0;
    int not_pos = 0;
    int nswap_bites = 0;
    int ishift_neg = 0;
    int ishift_pos = 0;

    int d_im = OPS_sub_dat_list[binhead->index]->d_im[idir];
    int d_ip = OPS_sub_dat_list[binhead->index]->d_ip[idir];

    //Part I: Find particles for exchange in the given direction
    _ops_particle_find_intra_map(particle, halo_int, binhead,
                                 particle->map_list[0]->bin,
                                 binhead->size);

    //Part II: Allocate exchange structures
    if (halo_int->nalloc_max_neg < halo_int->nforward_neg[0]) {
      halo_int->particle_send_neg = (int *)
          OPS_realloc_fast((char *)halo_int->particle_send_neg,
                           halo_int->nalloc_max_neg * sizeof(int),
                           halo_int->nforward_neg[0] * sizeof(int));
      halo_int->nalloc_max_neg  = halo_int->nforward_neg[0];
    }

    if (halo_int->nalloc_max_pos < halo_int->nforward_pos[0]) {
      halo_int->particle_send_pos = (int *)
          OPS_realloc_fast((char *)halo_int->particle_send_pos,
                           halo_int->nalloc_max_pos * sizeof(int),
                           halo_int->nforward_pos[0] * sizeof(int));
      halo_int->nalloc_max_pos  = halo_int->nforward_pos[0];
    }

    // Part III: Find partcles for exchange
    _ops_particle_set_intra_map(particle, halo_int, binhead,
                                particle->map_list[0]->bin,
                                binhead->size);

    // Part IV: Re-allocate structures if necessary
    if (sp->particle_halos[idir]->nforward_pos[0] * elem_size > ops_buffer_send_2_size) {
      ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                           sp->particle_halos[idir]->nforward_pos[0]
                                           * elem_size);
      ops_buffer_send_2_size = sp->particle_halos[idir]->nforward_pos[0]
                             * elem_size;
    }

    if (sp->particle_halos[idir]->nforward_neg[0] * elem_size > ops_buffer_send_1_size) {
      ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                           sp->particle_halos[idir]->nforward_neg[0]
                                              * elem_size);
      ops_buffer_send_1_size = sp->particle_halos[idir]->nforward_neg[0]
                             * elem_size;

    }

    // Part V: Packing data
    int ishift_pos_buf = 0;
    int ishift_neg_buf = 0;

    for (int idat = 0; idat < ndats; idat++) {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + ishift_neg_buf,
                                      dats[idat], halo_int->particle_send_neg + ishift_neg,
                                      halo_int->nforward_neg[0]);
      ishift_neg_buf += dats[idat]->elem_size * halo_int->nforward_neg[0];

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + ishift_pos_buf,
                                      dats[idat], halo_int->particle_send_pos + ishift_pos,
                                      halo_int->nforward_pos[0]);
      ishift_pos_buf += dats[idat]->elem_size * halo_int->nforward_pos[0];
    }


    //Part IVa: Pack history-data-TODO: Need a modification but this will be after
    for (int ihis = 0; ihis < nhistories; ihis++) {
      ishift_neg_buf += _ops_particle_intra_hist_to_buff(ops_buffer_send_1 + ishift_neg_buf,
                                                         histories[ihis],
                                                         halo_int->particle_send_neg + ishift_neg,
                                                         halo_int->nforward_neg[0]);

      ishift_pos_buf += _ops_particle_intra_hist_to_buff(ops_buffer_send_2 + ishift_pos_buf,
                                                         histories[ihis],
                                                         halo_int->particle_send_pos + ishift_pos,
                                                         halo_int->nforward_pos[0]);
    }

    //Part VI: Exchange particle structures
    MPI_Status status[4];
    int nforward = halo_int->nforward_neg[0];
    int nrecv = 0;

    MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_m[idir], 100,
                 &nrecv, 1, MPI_INT, sb->id_p[idir], 100,
                 sb->comm, &status[0]);
    halo_int->nrecv_pos[0] = (sb->id_p[idir] != MPI_PROC_NULL) ? nrecv : 0;

    nforward = halo_int->nforward_pos[0];
    nrecv = 0;
    MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_p[idir], 200,
                 &nrecv, 1, MPI_INT, sb->id_m[idir], 200,
                 sb->comm, &status[0]);
    halo_int->nrecv_neg[0] = (sb->id_m[idir] != MPI_PROC_NULL) ? nrecv : 0;

    //Part VII: Allocate receiving buffer structures
    if ( elem_size
         * halo_int->nrecv_pos[0] > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                           elem_size * halo_int->nrecv_pos[0]);
      ops_buffer_recv_1_size = elem_size * halo_int->nrecv_pos[0];
    }

    if ( elem_size
          * halo_int->nrecv_neg[0] > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                           elem_size* halo_int->nrecv_neg[0]);
      ops_buffer_recv_2_size = elem_size * halo_int->nrecv_neg[0];
    }

    //Part VII: Send data
    int size_send = elem_size * halo_int->nforward_neg[0];
    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE,
              (size_send > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    size_send = elem_size * halo_int->nforward_pos[0];
    MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE,
              (size_send > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    int size_recv = elem_size * halo_int->nrecv_pos[0];
    MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE,
              (size_recv > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);

    size_recv = elem_size * halo_int->nrecv_neg[0];
    MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE,
              (size_recv > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);


    //Part VIII: Set receiving structures
    halo_int->irecv_neg[0] = particle->no_particles + particle->no_virtual;
    halo_int->irecv_pos[0] = halo_int->irecv_neg[0] + halo_int->nrecv_neg[0];

    particle->no_virtual += halo_int->nrecv_pos[0] + halo_int->nrecv_neg[0];
    if (particle->no_particles + particle->no_virtual > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles + particle->no_virtual);
    }

    int irecv_shift_neg = 0;
    int irecv_shift_pos = 0;
    for (int idat = 0; idat < ndats; idat++) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + irecv_shift_neg,
                                      dats[idat], halo_int->irecv_neg[0],
                                      halo_int->nrecv_neg[0]);
      irecv_shift_neg += halo_int->nrecv_neg[0] * dats[idat]->elem_size;

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + irecv_shift_pos,
                                      dats[idat], halo_int->irecv_pos[0],
                                      halo_int->nrecv_pos[0]);
      irecv_shift_pos += halo_int->nrecv_pos[0] * dats[idat]->elem_size;
    }

    for (int ihis = 0; ihis < nhistories; ihis++) {
      irecv_shift_neg += _ops_particle_intra_buff_to_hist(ops_buffer_recv_2 + irecv_shift_neg,
                                                          histories[ihis],
                                                          halo_int->irecv_neg[0],
                                                          halo_int->nrecv_neg[0]);

      irecv_shift_pos += _ops_particle_intra_buff_to_hist(ops_buffer_recv_2 + irecv_shift_pos,
                                                          histories[ihis],
                                                          halo_int->irecv_pos[0],
                                                          halo_int->nrecv_pos[0]); //TODO
    }


    MPI_Waitall(2, &request[0], &status[0]);

    //Part VII: update maps
    for (int imap = 0; imap < particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      _ops_particle_mapping_virtual_from_halo(map, particle,
                                              halo_int->irecv_neg[0],
                                              halo_int->nrecv_neg[0] + halo_int->nrecv_pos[0]);
    }
  }

}

void _ops_particle_build_border_maps(ops_particle particle) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];
  ops_dat binhead = particle->map_list[0]->binhead;
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    d_m[i] = binhead->d_m[i] + OPS_sub_dat_list[binhead->index]->d_im[i];
    d_p[i] = binhead->d_p[i] + OPS_sub_dat_list[binhead->index]->d_ip[i];
  }


  for (int idir = 0; idir < dim; idir++) {

    ops_int_particle_halos halo_int = sp->particle_halos[idir];


    if (halo_int->nswaps > 1)
      throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Intra-block halos based on maps "
                                           "require a singe swap");
    int nsend_recv[4];
    int not_neg = 0;
    int not_pos = 0;
    int nswap_bites = 0;
    int ishift_neg = 0;
    int ishift_pos = 0;


    int d_im = OPS_sub_dat_list[binhead->index]->d_im[idir];
    int d_ip = OPS_sub_dat_list[binhead->index]->d_ip[idir];


    _ops_particle_find_intra_map(particle, halo_int, binhead,
                                 particle->map_list[0]->bin,
                                 binhead->size);


    if (halo_int->nalloc_max_neg < halo_int->nforward_neg[0]) {
      halo_int->particle_send_neg = (int *)
          OPS_realloc_fast((char *)halo_int->particle_send_neg,
                           halo_int->nalloc_max_neg * sizeof(int),
                           halo_int->nforward_neg[0] * sizeof(int));
      halo_int->nalloc_max_neg  = halo_int->nforward_neg[0];
    }

    if (halo_int->nalloc_max_pos < halo_int->nforward_pos[0]) {
      halo_int->particle_send_pos = (int *)
          OPS_realloc_fast((char *)halo_int->particle_send_pos,
                           halo_int->nalloc_max_pos * sizeof(int),
                           halo_int->nforward_pos[0] * sizeof(int));
      halo_int->nalloc_max_pos  = halo_int->nforward_pos[0];
    }

    _ops_particle_set_intra_map(particle, halo_int, binhead,
                                particle->map_list[0]->bin,
                                binhead->size);


    //Reallocate sending structures

    //Allocate data if necessary
    int elem_size = sp->bites_in_exchange; //At the

    if (sp->particle_halos[idir]->nforward_pos[0] *
        elem_size > ops_buffer_send_2_size) {
      ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                           sp->particle_halos[idir]->nforward_pos[0]
                                           * elem_size);
      ops_buffer_send_2_size = sp->particle_halos[idir]->nforward_pos[0] *
                               elem_size;
    }
    if (sp->particle_halos[idir]->nforward_neg[0] *
         elem_size > ops_buffer_send_1_size) {
      ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                           sp->particle_halos[idir]->nforward_neg[0]
                                              * elem_size);
      ops_buffer_send_1_size = sp->particle_halos[idir]->nforward_neg[0]
                             * elem_size;

    }

    _ops_particle_intra_dat_to_buff(ops_buffer_send_1,
                                    particle->particle_pos_dat,
                                    halo_int->particle_send_neg,
                                    halo_int->nforward_neg[0]);

    _ops_particle_intra_dat_to_buff(ops_buffer_send_2,
                                    particle->particle_pos_dat,
                                    halo_int->particle_send_pos,
                                    halo_int->nforward_pos[0]);


    MPI_Status status[4];
    int nforward = halo_int->nforward_neg[0];
    int nrecv = 0;
    MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_m[idir], 100,
                 &nrecv, 1, MPI_INT, sb->id_p[idir], 100,
                 sb->comm, &status[0]);
    halo_int->nrecv_pos[0] = (sb->id_p[idir] != MPI_PROC_NULL) ? nrecv : 0;


    nforward = halo_int->nforward_pos[0];
    MPI_Sendrecv(&nforward, 1, MPI_INT, sb->id_p[idir], 200,
                 &nrecv, 1, MPI_INT, sb->id_m[idir], 200,
                 sb->comm, &status[0]);
    halo_int->nrecv_neg[0] = (sb->id_m[idir] != MPI_PROC_NULL) ? nrecv : 0;


    if ( elem_size
         * halo_int->nrecv_pos[0] > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                           elem_size * halo_int->nrecv_pos[0]);
      ops_buffer_recv_1_size = elem_size * halo_int->nrecv_pos[0];
    }

    if ( elem_size
          * halo_int->nrecv_neg[0] > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_1_size,
                                           elem_size* halo_int->nrecv_neg[0]);
      ops_buffer_recv_1_size = elem_size * halo_int->nrecv_neg[0];
    }

    //Send and receive data
    int size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_neg[0];
    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE, (size_send > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_pos[0];
    MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE, (size_send > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    int size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[0];
    MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);

    size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_neg[0];
    MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);



    halo_int->irecv_neg[0] = particle->no_particles + particle->no_virtual;
    halo_int->irecv_pos[0] = halo_int->irecv_neg[0] + halo_int->nrecv_neg[0];

    particle->no_virtual += halo_int->nrecv_pos[0] + halo_int->nrecv_neg[0];
    if (particle->no_particles + particle->no_virtual > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles + particle->no_virtual);
    }

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_2,
                                    particle->particle_pos_dat,
                                    halo_int->irecv_neg[0], halo_int->nrecv_neg[0]);

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_1,
                                    particle->particle_pos_dat,
                                    halo_int->irecv_pos[0], halo_int->nrecv_pos[0]);

    MPI_Waitall(2, &request[0], &status[0]);

    //Map particles
    for (int imap = 0; imap < particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      _ops_particle_mapping_virtual_from_halo(map, particle,
                                              halo_int->irecv_neg[0],
                                              halo_int->nrecv_neg[0] + halo_int->nrecv_pos[0]);
    }
  }
}




void _ops_particle_forward_intra_maps(ops_particle particle) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  for (int idir = 0; idir < dim; idir++) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];

    _ops_particle_intra_dat_to_buff(ops_buffer_send_1,
                                    particle->particle_pos_dat,
                                    halo_int->particle_send_neg,
                                    halo_int->nforward_neg[0]);

    _ops_particle_intra_dat_to_buff(ops_buffer_send_2,
                                    particle->particle_pos_dat,
                                    halo_int->particle_send_pos,
                                    halo_int->nforward_pos[0]);

    int size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_neg[0];
    MPI_Request request[4];
    MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE, (size_send > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    size_send = particle->particle_pos_dat->elem_size * halo_int->nforward_pos[0];
    MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE, (size_send > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    int size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[0];
    MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_p[idir] : MPI_PROC_NULL,
                   idir, sb->comm, &request[2]);

    size_recv = particle->particle_pos_dat->elem_size * halo_int->nrecv_neg[0];
    MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE, (size_recv > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Status status[4];
    MPI_Waitall(2, &request[2], &status[2]);

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_2, particle->particle_pos_dat,
                                    halo_int->irecv_neg[0], halo_int->nrecv_neg[0]);

    _ops_particle_intra_buff_to_dat(ops_buffer_recv_1, particle->particle_pos_dat,
                                    halo_int->irecv_pos[0], halo_int->nrecv_neg[0]);

    MPI_Waitall(2, &request[0], &status[0]);

    //Update maps
    for (int imap = 0; imap < particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      _ops_particle_remap_virtual(map, particle, halo_int->irecv_neg[0],
                                  halo_int->nrecv_neg[0] + halo_int->nrecv_pos[0]);
    }


  }
}

void _ops_particle_packer_all(ops_dat dat, ops_particle particle, double range_in[], int  iswap,
                              int idir, int send_recv_offsets[]) {
  //Get particle data
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  ops_int_particle_halos halo_int = sp->particle_halos[idir];

  if (halo_int->nswaps < iswap) return;

  int ishift_n = 0;
  int ishift_p = 0;
  for (int i = 0; i < iswap; i++) {
    ishift_n += halo_int->nforward_neg[iswap];
    ishift_p += halo_int->nforward_pos[iswap];
  }

  _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + send_recv_offsets[0],
                                  dat, halo_int->particle_send_neg + ishift_n,
                                  halo_int->nforward_neg[iswap]);

  _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + send_recv_offsets[0],
                                  dat, halo_int->particle_send_pos + ishift_p,
                                  halo_int->nforward_pos[iswap]);



  send_recv_offsets[0] += halo_int->nforward_neg[iswap] * dat->elem_size;
  send_recv_offsets[1] += halo_int->nforward_pos[iswap] * dat->elem_size;

}

void _ops_particle_forward_dats_with_maps(ops_particle particle, ops_dat *dats,
                                          int ndats) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  int elem_size = 0;
  for (int idat = 0; idat < ndats; idat++) {
    elem_size += dats[idat]->elem_size;
  }

  for (int idir = 0; idir < dim; idir++) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];

    if (halo_int->nswaps > 1)
      throw OPSException(OPS_RUNTIME_ERROR,"Error: multiple swaps are not supported with "
                         "mapping");

    //check if reallocation is needed
    if (elem_size * halo_int->nforward_neg[0] > ops_buffer_send_1_size) {
      ops_buffer_send_1 = OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                           halo_int->nforward_neg[0] * elem_size);
      ops_buffer_send_1_size = halo_int->nforward_neg[0] * elem_size;
    }

    if (elem_size * halo_int->nforward_pos[0] > ops_buffer_send_2_size) {
      ops_buffer_send_2 = OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                           halo_int->nforward_pos[0] * elem_size);
      ops_buffer_send_2_size = halo_int->nforward_pos[0] * elem_size;
    }

    if (elem_size * halo_int->nrecv_pos[0] > ops_buffer_recv_1_size) {
      ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                           halo_int->nrecv_pos[0] * elem_size);
      ops_buffer_recv_1_size = halo_int->nrecv_pos[0] * elem_size;

    }

    if (elem_size * halo_int->nrecv_neg[0] > ops_buffer_recv_2_size) {
      ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                           halo_int->nrecv_neg[0] * elem_size);
      ops_buffer_recv_2_size = halo_int->nrecv_neg[0] * elem_size;
    }

    //Part II: Packing data structures
    int ishift_pos_buf = 0;
    int ishift_neg_buf = 0;

    for (int idat = 0; idat < ndats; idat++) {
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + ishift_neg_buf,
                                      dats[idat], halo_int->particle_send_neg,
                                      halo_int->nforward_neg[0]);
      ishift_neg_buf += dats[idat]->elem_size * halo_int->nforward_neg[0];

      _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + ishift_pos_buf,
                                      dats[idat], halo_int->particle_send_pos,
                                      halo_int->nforward_pos[0]);
      ishift_pos_buf += dats[idat]->elem_size * halo_int->nforward_pos[0];
    }

    //Part III: Send and receive data
    MPI_Request request[4];
    MPI_Status status[4];

    int size_send = halo_int->nforward_neg[0] * elem_size;
    MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE,
              size_send > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[0]);

    size_send = halo_int->nforward_pos[0] * elem_size;
    MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE,
              size_send > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[1]);

    int size_recv = halo_int->nrecv_pos[0] * elem_size;
    MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE,
              size_recv > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
              idir, sb->comm, &request[2]);


    size_recv = halo_int->nrecv_neg[0] * elem_size;
    MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE,
              size_recv > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
              idir + OPS_MAX_DIM, sb->comm, &request[3]);

    MPI_Waitall(2, &request[2], &status[2]);

    //Part IV: Unpack ops_dat structures
    int nrecv_pos_shift = 0;
    int nrecv_neg_shift = 0;


    for (int idat = 0; idat < ndats; idat++) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + nrecv_neg_shift, dats[idat],
                                      halo_int->irecv_neg[0], halo_int->nrecv_neg[0]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + nrecv_pos_shift, dats[idat],
                                      halo_int->irecv_pos[0], halo_int->nrecv_pos[0]);

      nrecv_neg_shift += halo_int->nrecv_neg[0] * dats[idat]->elem_size;
      nrecv_pos_shift += halo_int->nrecv_pos[0] * dats[idat]->elem_size;
    }

    MPI_Waitall(2, &request[0], &status[0]);




    //Part V: Update maps
    for (int imap = 0; imap <  particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      _ops_particle_remap_virtual(map, particle, halo_int->irecv_neg[0],
                                  halo_int->nrecv_neg[0] + halo_int->nrecv_pos[0]);
    }

  }
}

void _ops_particle_forward_dats(ops_particle particle, ops_dat *dats, int ndats) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  int elem_size = 0;
  for (int idat = 0; idat < ndats; idat++) {
    elem_size += dats[idat]->elem_size;
  }

  int nmax_pos = 0;
  int nmax_neg = 0;
  int nmaxr_pos = 0;
  int nmaxr_neg = 0;

  for (int idir = 0; idir < dim; idir++) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int iswap = 0; iswap < halo_int->nswaps; iswap++) {
      nmax_pos = MAX(nmax_pos, halo_int->nforward_pos[iswap]);
      nmax_neg = MAX(nmax_neg, halo_int->nforward_neg[iswap]);
      nmaxr_pos = MAX(nmaxr_pos, halo_int->nrecv_pos[iswap]);
      nmaxr_neg = MAX(nmaxr_neg, halo_int->nrecv_neg[iswap]);
    }
  }



  //Perform if needed allocations
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

  if (ops_buffer_recv_1_size < nmaxr_pos * elem_size) {
    ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                         nmaxr_pos * elem_size);
    ops_buffer_recv_1_size = nmaxr_pos* elem_size;
  }

  if (ops_buffer_recv_2_size < nmaxr_neg * elem_size) {
    ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                         nmaxr_neg * elem_size);
    ops_buffer_recv_2_size = nmaxr_neg* elem_size;
  }


  for (int idir = 0; idir < dim; idir++) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    int nshift_pos = 0;
    int nshift_neg = 0;

    for (int iswap = 0; iswap < halo_int->nswaps; iswap++) {
      //Part I: Pack data structures
      int nshift_dat_p = 0;
      int nshift_dat_n = 0;

      for (int idat  = 0; idat < ndats; idat++) {
        _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + nshift_dat_n,
                                        dats[idat],
                                        halo_int->particle_send_neg + nshift_neg,
                                        halo_int->nforward_neg[iswap]);
        nshift_dat_n += halo_int->nforward_neg[iswap] * dats[idat]->elem_size;

        _ops_particle_intra_dat_to_buff(ops_buffer_send_2 + nshift_dat_p,
                                        dats[idat],
                                        halo_int->particle_send_pos + nshift_pos,
                                        halo_int->nforward_pos[iswap]);
        nshift_dat_p += halo_int->nforward_pos[iswap] * dats[idat]->elem_size;
      }

      nshift_pos += halo_int->nforward_pos[iswap];
      nshift_neg += halo_int->nforward_neg[iswap];

      //Part II: Sending and receiving data
      MPI_Status status[4];
      MPI_Request request[4];

      int size_send = elem_size * halo_int->nforward_neg[iswap];
      MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE,
                size_send > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[0]);

      size_send = elem_size * halo_int->nforward_pos[iswap];
      MPI_Isend(ops_buffer_send_2, size_send, MPI_BYTE,
                size_send > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[1]);

      int size_recv = elem_size * halo_int->nrecv_pos[iswap];
      MPI_Irecv(ops_buffer_recv_1, size_recv, MPI_BYTE,
                size_recv > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[2]);

      size_recv = elem_size * halo_int->nrecv_neg[iswap];
      MPI_Irecv(ops_buffer_recv_2, size_recv, MPI_BYTE,
                size_recv > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[3]);

      MPI_Waitall(2, &request[2], &status[2]);

      //Part III: Unpack data
      int nshift_recv_pos = 0;
      int nshift_recv_neg = 0;
      for (int idat  = 0; idat < ndats; idat++) {
        _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + nshift_recv_neg,
                                        dats[idat],
                                        halo_int->irecv_neg[iswap],
                                        halo_int->nrecv_neg[iswap]);
        nshift_recv_neg += halo_int->nrecv_neg[iswap] * dats[idat]->elem_size;


        _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + nshift_recv_pos,
                                        dats[idat],
                                        halo_int->irecv_pos[iswap],
                                        halo_int->nrecv_pos[iswap]);
        nshift_recv_pos += halo_int->nrecv_pos[iswap] * dats[idat]->elem_size;
      }

      MPI_Waitall(2, &request[0], &status[0]);

    }
  }
}

void _ops_particle_reverse_dats(ops_particle particle, ops_dat *dats, int ndats,
                                ops_access access) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];

  int elem_size = 0;
  for (int idat = 0; idat < ndats; idat++) {
    elem_size += dats[idat]->elem_size;
  }

  int nmax_pos = 0;
  int nmax_neg = 0;
  int nmaxr_pos = 0;
  int nmaxr_neg = 0;

  for (int idir = 0; idir < dim; idir++) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int iswap = 0; iswap < halo_int->nswaps; iswap++) {
      nmaxr_pos = MAX(nmaxr_pos, halo_int->nforward_pos[iswap]);
      nmaxr_neg = MAX(nmaxr_neg, halo_int->nforward_neg[iswap]);
      nmax_pos = MAX(nmax_pos, halo_int->nrecv_pos[iswap]);
      nmax_neg = MAX(nmax_neg, halo_int->nrecv_neg[iswap]);
    }
  }

  //Reallocation of structures if neccessary
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

  if (ops_buffer_recv_1_size < nmaxr_pos * elem_size) {
    ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                                         nmaxr_pos * elem_size);
    ops_buffer_recv_1_size = nmaxr_pos* elem_size;
  }

  if (ops_buffer_recv_2_size < nmaxr_neg * elem_size) {
    ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_2_size,
                                         nmaxr_neg * elem_size);
    ops_buffer_recv_2_size = nmaxr_neg* elem_size;
  }

  for (int idir = dim - 1; idir >= 0; idir--) {
    ops_int_particle_halos halo_int = sp->particle_halos[idir];

    int nrecv_pos = 0;
    int nrecv_neg = 0;
    for (int iswap = 0; iswap < halo_int->nswaps; iswap++) {

      //Pack data to send
      int ishift_pos = 0;
      int ishift_neg = 0;
      for (int idat = 0; idat < ndats; idat++) {
        _ops_particle_intra_pack_rev_dat_to_buff(ops_buffer_send_1 + ishift_neg,
                                                 dats[idat],
                                                 halo_int->irecv_neg[iswap],
                                                 halo_int->nrecv_neg[iswap]);
        ishift_neg += dats[idat]->elem_size * halo_int->nrecv_neg[iswap];

        _ops_particle_intra_pack_rev_dat_to_buff(ops_buffer_send_2 + ishift_pos,
                                                 dats[idat], halo_int->irecv_pos[iswap],
                                                 halo_int->nrecv_pos[iswap]);
        ishift_pos += dats[idat]->elem_size * halo_int->nrecv_pos[iswap];

      }

      //Sending and receiving data structues
      MPI_Status status[4];
      MPI_Request request[4];

      int send_size = halo_int->nrecv_neg[iswap] * elem_size;
      MPI_Isend(ops_buffer_send_1, send_size, MPI_BYTE,
                send_size > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[0]);

      send_size = halo_int->nrecv_pos[iswap] * elem_size;
      MPI_Isend(ops_buffer_send_2, send_size, MPI_BYTE,
                send_size > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[1]);



      int recv_size = halo_int->nforward_pos[iswap] * elem_size;
      MPI_Irecv(ops_buffer_recv_1, recv_size, MPI_BYTE,
                recv_size > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[2]);

      recv_size = halo_int->nforward_neg[iswap] * elem_size;
      MPI_Irecv(ops_buffer_recv_2, recv_size, MPI_BYTE,
                recv_size > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
                idir + OPS_MAX_DIM, sb->comm, &request[3]);

      MPI_Waitall(2, &request[2], &status[2]);

      //TODO: Unpack data

      int nshift_for_n = 0;
      int nshift_for_p = 0;
      for (int idat = 0; idat < ndats; idat++) {
        _ops_particle_unpack_reverse_buff_to_dat(ops_buffer_recv_2 + nshift_for_n,
                                                 dats[idat],
                                                 halo_int->particle_send_neg + nrecv_neg,
                                                 halo_int->nforward_neg[iswap],
                                                 access);
        nshift_for_n += halo_int->nforward_neg[iswap] * dats[idat]->elem_size;

        _ops_particle_unpack_reverse_buff_to_dat(ops_buffer_recv_1 + nshift_for_p,
                                                 dats[idat],
                                                 halo_int->particle_send_pos + nrecv_pos,
                                                 halo_int->nforward_pos[iswap],
                                                 access);
        nshift_for_p += halo_int->nforward_pos[iswap] * dats[idat]->elem_size;

      }

      MPI_Waitall(2, &request[0], &status[0]);


      nrecv_neg += halo_int->nforward_neg[iswap];
      nrecv_pos += halo_int->nforward_pos[iswap];
    }
  }
}

void _ops_particle_unpacker_all(ops_dat dat, ops_particle particle, double range_in[], int iswap,
                                int idir, int send_recv_offsets[]) {

  //Get particle data
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;

  sub_particle sp = sb->sb_particle_list[particle->index];
  ops_int_particle_halos halo_int = sp->particle_halos[idir];

  if (halo_int->nswaps < iswap) return;


  _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + send_recv_offsets[2],
                                  dat, halo_int->irecv_pos[iswap],
                                  halo_int->nrecv_pos[iswap]);

  _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + send_recv_offsets[3],
                                  dat, halo_int->irecv_neg[iswap],
                                  halo_int->nrecv_neg[iswap]);

  send_recv_offsets[2] += halo_int->nrecv_pos[iswap] * dat->elem_size;
  send_recv_offsets[3] += halo_int->nrecv_neg[iswap] * dat->elem_size;

}
