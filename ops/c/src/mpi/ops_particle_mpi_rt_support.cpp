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



//TODO: The given structure assumes that the number of cells is large compared to the
//      halo cells-In case not need re-order & multiple swaps

/* Values of flag
 *  0: No intersection or range box within my region
 *  1: Intersection: Need to find particles for sending
 *  2: My box within region: Send all particles
 */
static int check_intersection(BoundingBox *box, double range_in[], int dim) {
  if (range_in == nullptr) return 2;

  double intersection = 1;
  double ha, hb, dxa, dxb;

  for (int idir = 0; idir < dim; idir++) {
    ha = 0.5 * (box->getMaxCoordDir(idir) + box->getMinCoordDir(idir));
    hb = 0.5 * (range_in[2 * idir + 1] + range_in[2 * idir]);
    dxa = 0.5 * (box->getMaxCoordDir(idir) - box->getMinCoordDir(idir));
    dxb = 0.5 * (range_in[2 * idir + 1] - range_in[2 * idir]);

    double hmin = MIN(ha, hb);
    double hmax = MAX(ha, hb);
    if (hmax + hmin > dxa + dxb) return 0;

    else if (hmax + hmin <= dxa + dxb) { //Check all

      //Special cases
      if (fabs(ha - hb) > fabs(dxb - dxa)) intersection *= 1;
      else if (fabs(ha-hb) < dxa - dxb) intersection *= 2.; //Region within my box special treatment
      else if (fabs(ha-hb) < dxb - dxa) intersection *= 3; //Region within my region
    }
  }

  if (intersection == 0) return 0;

  if (intersection == pow(2, dim)) return 0;

  if (intersection == pow(3, dim)) return 2;

  return 1;

}

/* Values of flag
 *  0: No intersection or range box within my region
 *  1: Intersection: Need to find particles for sending
 */


void ops_particle_pack_dat_forward(ops_dat dat, ops_int_particle_halos halo,
                                   int *send_recv_offsets) {
  if (!dat->is_particle)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Packing particle data structures");

  //Compute sending in the negative direction
  int nsend = halo->nsend_neg * dat->elem_size;
  if (send_recv_offsets[0] + nsend > ops_buffer_send_1_size) {
    ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1,
                                                  ops_buffer_send_1_size,
                                                  send_recv_offsets[0] + 4 * nsend);
    ops_buffer_send_1_size += 4 * nsend;
  }

  if (halo-> nsend_neg > 0) {
    send_recv_offsets[0] += ops_particle_pack_border_data(dat, ops_buffer_send_1,
                                                          halo->particle_send_neg, 0,
                                                          halo->nsend_neg);
  }

  //Compute packing in the positive direction
  nsend = halo->nsend_pos * dat->elem_size;
  if (send_recv_offsets[2] + nsend > ops_buffer_send_2_size) {
    ops_buffer_send_2 = (char *) OPS_realloc_fast(ops_buffer_send_2,
                                                  ops_buffer_send_2_size,
                                                  send_recv_offsets[2] + 4 * nsend);
    ops_buffer_send_2_size += 4 * nsend;

  }

  if (halo->nsend_pos > 0)
    send_recv_offsets[1] += ops_particle_pack_border_data(dat, ops_buffer_send_1,
                                                          halo->particle_send_neg, 0,
                                                          halo->nsend_neg);
}

void ops_particle_unpack_dat_forward(ops_dat dat, ops_int_particle_halos halo,
                                     int *send_recv_offsets) {
  //3: Received from the negative proc
  //1: Received from the positive direction

  if (halo->nrecv_neg[0] > 0)
    ops_particle_unpack_border(dat, ops_buffer_recv_2, halo->irecv_neg[0],
                               halo->nrecv_neg[0]);

  int iforward = halo->nrecv_neg[0] * dat->elem_size;
  if (halo->nrecv_pos[0] > 0)
    ops_particle_unpack_border(dat, ops_buffer_recv_1 + iforward,
                               halo->irecv_pos[0], halo->nrecv_pos[0]);
}

void _ops_particle_border_exchange_build_maps(int idir,ops_particle particle,
                                             ops_int_particle_halos halo,
                                             sub_block_list sb) {

  if (!sb->owned) return;

  int send_recv_offsets[4] = {};

  ops_dat coords = particle->particle_pos_dat;

  ops_particle_pack_dat_forward(coords, halo, send_recv_offsets);

  //Perform exchange
  //Compute receive from positive direction
  send_recv_offsets[1] = halo->nrecv_pos[0] * coords->elem_size;
  send_recv_offsets[3] = halo->nrecv_neg[0] * coords->elem_size;

  if (send_recv_offsets[1] > ops_buffer_recv_1_size) {
    ops_buffer_recv_1 = (char *) OPS_realloc_fast(ops_buffer_recv_1,
                                                  ops_buffer_recv_1_size,
                                                  ops_buffer_recv_1_size +
                                                  4 * send_recv_offsets[1]);
    ops_buffer_recv_1_size += 4 * send_recv_offsets[1];
  }

  if (send_recv_offsets[3] > ops_buffer_recv_2_size) {
    ops_buffer_recv_2 = (char *) OPS_realloc_fast(ops_buffer_recv_2,
                                                  ops_buffer_recv_2_size,
                                                  ops_buffer_recv_2_size +
                                                  4 * send_recv_offsets[3]);
    ops_buffer_recv_2_size += 4 * send_recv_offsets[3];
  }



  MPI_Request request[4];
  MPI_Isend(ops_buffer_send_1, send_recv_offsets[0], MPI_BYTE,
            send_recv_offsets[0] > 0 ? sb->id_m[idir] : MPI_PROC_NULL,
            idir, sb->comm, &request[0]);

  MPI_Isend(ops_buffer_send_2, send_recv_offsets[2], MPI_BYTE,
            send_recv_offsets[2] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
            idir + OPS_MAX_DIM, sb->comm, &request[1]);

  MPI_Irecv(ops_buffer_recv_1, send_recv_offsets[1], MPI_BYTE,
            send_recv_offsets[1] > 0 ? sb->id_p[idir] : MPI_PROC_NULL,
            idir, sb->comm, &request[2]);
  MPI_Irecv(ops_buffer_recv_2, send_recv_offsets[3], MPI_BYTE,
            send_recv_offsets[3] > 0? sb->id_m[idir] : MPI_PROC_NULL,
            idir + OPS_MAX_DIM, sb->comm, &request[3]);

  ops_particle_unpack_dat_forward(coords, halo, send_recv_offsets);

  /* Mapping virtual particles currently received */
  int ifirst = halo->irecv_neg[0];
  int ilast =  halo->irecv_pos[0] + halo->irecv_neg[0];
  ops_particle_update_intra_halo_maps(particle, ifirst, ilast);

}

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

void _ops_particle_halo_border_transfer_vs(OPS_instance *instance,
                                           ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  double c, t1, t2;
  ops_timers_core(&c, &t1);
 // mpi_neigh_size[0] = 0;

  int nhalos = mpi_group->nhalos;
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


  //Need to send a big chunk of data at once

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
      mpi_group->recv_bites[iloc] += nbites_pos * halo_info->nrecv;
    }
    particle_to->no_virtual += no_new;
  }


  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  int shift0 = 0;
  int shift1 = 0;

  for (int  i = 0; i < mpi_group->num_neighbors_send;i++) {
    mpi_group->send_shift[i] = shift0;
    nsend_shift[i] = shift0;
    mpi_group->shift_send_pos[i] = shift1;
    shift1+=mpi_group->send_pos_bites[i];
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


  //Packing halo-data
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
                                    halo_info, &ntot_bites);
      nsend_shift[iloc] += ntot_bites;
    }
  }

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + mpi_group->send_shift[i], mpi_group->send_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    MPI_Irecv(ops_buffer_recv_1 + mpi_group->recv_shift[i], mpi_group->recv_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  //Reallocate main data structures
  ntot_bites = 0;
  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
    for (int imap = 0; imap < to->particle_map_index; imap++) {
      ops_particle_mapping map = to->map_list[imap];
      if (map->Nmax < to->no_particles + to->no_virtual)
        _ops_particle_realloc_map_data(map,   to->no_particles + to->no_virtual);
    }
  }

  //Unpacking forward data
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
                                        &ntot_bites);
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

  int size = (mpi_group->num_neighbors_send > 0) ?
      mpi_group->num_neighbors_send * mpi_group->nhalos : 1;
  int *nsend = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ?
      mpi_group->num_neighbors_recv * mpi_group->nhalos : 1;
  int *nrecv = (int *) ops_malloc(size * sizeof(int));

  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; nsend++)
    nsend[i] = 0;

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; nsend++)
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


  double xminBox[OPS_MAX_DIM], xmaxBox[OPS_MAX_DIM];
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;
    int dim = instance->OPS_particle_halo_list[halo->index]->particle_from->block->dims;
    double *xcrds
    = (double *)instance->OPS_particle_halo_list[halo->index]->particle_from->particle_pos_dat->data;
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
      halo->sendBox[isend]->getLocalMaxMin(xminBox, xmaxBox);

      _ops_particle_number_of_particles_in_range(halo->sendBox[isend], dim, xcrds, noParticles,
                                                 &nsend[nhalos * iloc + h]); //TODO
      halo_info->nsend = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist = (int *)
            OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      _ops_particle_mapped_into_region(halo->sendBox[isend], dim, xcrds,
                                       noParticles, halo_info->sendlist);

      mpi_group->send_bites[iloc]+= bites * halo_info->nsend;
      mpi_group->send_pos_bites[iloc] += bites_coords * halo_info->nsend;
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
      halo_info->firstrecv = ifirst;
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

  //Packing halo-data
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
                                    halo_info, &ntot_bites);
      nsend_shift[iproc] += ntot_bites;
    }
  }

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + mpi_group->send_shift[i], mpi_group->send_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    MPI_Irecv(ops_buffer_recv_1 + mpi_group->recv_shift[i], mpi_group->recv_bites[i],
              MPI_BYTE, iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  //Unpacking halo data
  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
    for (int imap = 0; imap < to->particle_map_index; imap++) {
      ops_particle_mapping map = to->map_list[imap];
      if (map->Nmax < to->no_particles + to->no_virtual)
        _ops_particle_realloc_map_data(map,   to->no_particles + to->no_virtual);
    }
  }

  //Unpacking forward data
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
                                        &ntot_bites);
      nrecv_shift[iloc] += ntot_bites;

    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  _ops_particle_update_dependent_halo_groups(instance, halo_grp, mpi_group);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

  ops_free(nrecv_shift);
  ops_free(nsend_shift);
  ops_free(nsend);
  ops_free(nrecv);

}

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

  for (int i = 0; i <mpi_group->num_neighbors_send * mpi_group->nhalos; nsend++)
    nsend[i] = 0;

  for (int i = 0; i <mpi_group->num_neighbors_recv * mpi_group->nhalos; nsend++)
    nrecv[i] = 0;

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
     int *mark_deletion = particle->mark_deletion;
     double *env = (particle->particle_envelope != nullptr) ?
         (double *)particle->particle_envelope->data : nullptr;

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
       _ops_particle_number_of_particles_in_range(halo->sendBox[isend], dim, xcrds, noParticles,
                                                  &nsend[nhalos * iloc + h]);
       halo_info->nsend  = nsend[nhalos * iloc + h];

       if (halo_info->nsend > halo_info->nmax) {
         halo_info->sendlist = (int *)
             OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                          sizeof(int) * (halo_info->nsend + 10));
         halo_info->nmax = halo_info->nsend + 10;
       }

       _ops_particle_remove_from_region(halo->sendBox[isend], xcrds, env, mark_deletion,
                                        particle->no_particles, particle->block->dims,
                                        halo_info->sendlist);

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
               1000+mpi_group->index, OPS_MPI_GLOBAL,
               &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
   }

   MPI_Waitall(mpi_group->num_neighbors_recv,
               &mpi_group->requests[mpi_group->num_neighbors_send],
               &mpi_group->statuses[mpi_group->num_neighbors_send]);

   //Unpack data
   for (int h = 0; h < mpi_group->nhalos; h++) {
     ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
     int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
     ops_particle particle_to = instance->OPS_particle_halo_list[halo->index]->particle_to;
     int ifirst = particle_to->no_particles;
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
       halo_info->firstrecv = ifirst;
       ifirst += halo_info->nrecv;
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

   //Allocation of buffer
   if (shift > ops_buffer_send_1_size) {
     ops_buffer_send_1 = (char *) OPS_realloc_fast(ops_buffer_send_1,
                                                   ops_buffer_send_1_size, shift);
     ops_buffer_send_1_size = shift;
   }

   shift = 0;
   for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
     mpi_group->recv_shift[i] = shift;
     nrecv_shift[i] = shift;
     shift+= mpi_group->recv_bites[i];
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
                                     halo_info, &ntot_bites);
       nsend_shift[iloc] += ntot_bites;
     }
   }

   //Sending data via MPI
   for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
     int iproc = mpi_group->neighbors_send[i];
     int shift = mpi_group->send_shift[i];
     MPI_Isend(ops_buffer_send_1 + shift, mpi_group->send_bites[i], MPI_BYTE,
               iproc, 1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
   }

   for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
     int iproc = mpi_group->neighbors_recv[i];
     int shift = mpi_group->recv_shift[i];
     MPI_Irecv(ops_buffer_recv_1 + shift, mpi_group->recv_bites[i], MPI_BYTE,
               iproc, 1000 + mpi_group->index, OPS_MPI_GLOBAL,
               &mpi_group->requests[mpi_group->num_neighbors_send + i]);
   }

   MPI_Waitall(mpi_group->num_neighbors_recv,
               &mpi_group->requests[mpi_group->num_neighbors_send],
               &mpi_group->statuses[mpi_group->num_neighbors_send]);

   //Finding if reallocations are needed
   for (int h = 0; h < mpi_group->nhalos; h++) {
     ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
     ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
     int nrecv1 = 0;
     for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
       int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
       int iloc =-1;
       for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++) {
         if (mpi_group->neighbors_recv[iloc] == iproc) break;
       }

       ops_particle_halo_exchange halo_info
        = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h]; //IT IS WRONG
       nrecv1 += halo_info->nrecv;
     }
     halo->particle_to->no_particles += nrecv1;
   }

   //Reallocate structures before unpack
   for (int h = 0; h < mpi_group->nhalos; h++) {
     ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
     ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
     ops_particle particle = halo->particle_to;
     if (particle->no_particles > particle->Nmax) {
       ops_particle_realloc_data(particle, particle->no_particles);
     }

     for (int imap = 0; imap < particle->particle_map_index; imap++) {
       ops_particle_mapping map = particle->map_list[imap];
       if (map->Nmax < particle->no_particles)
         _ops_particle_realloc_map_data(map,   particle->no_particles);
     }
   }

   //Unpacking exchange data

   for (int h = 0; h < mpi_group->nhalos; h++) {
     ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
     ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
     ops_particle_halo_data *halo_data = halo->dat;

     ops_particle to = halo->particle_to;
     int *mark_del = (int *) to->mark_deletion;
     double *xcrds = (double *)to->particle_pos_dat->data;
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
                                         &ntot_bites);
       nrecv_shift[iloc] += ntot_bites;

       //TODO: Mark for deletion or not
      _ops_particle_mark_for_removal(to->box_block, xcrds, mark_del,
                                     to->block->dims, halo_info->firstrecv,
                                     halo_info->firstrecv + halo_info->nsend);


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


  int nhalos = mpi_group->nhalos;

  int size = (mpi_group->num_neighbors_send> 0)
      ? nhalos * mpi_group->num_neighbors_send : 1;

  int *nsend = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0)
      ? nhalos * mpi_group->num_neighbors_recv : 1;
  int *nrecv = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_send > 0) ?
      mpi_group->num_neighbors_send : 1;
  int *nsend_shift = (int *)ops_malloc(size * sizeof(int));

  size = (mpi_group->num_neighbors_recv > 0) ?
      mpi_group->num_neighbors_recv : 1;
  int *nrecv_shift = (int *)ops_malloc(size * sizeof(int));

  for (int i= 0; i < mpi_group->num_neighbors_send * nhalos; i++)
    nsend[i] = 0;

  for (int i = 0; i < mpi_group->num_neighbors_recv * nhalos; i++)
    nrecv[i] = 0;

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
    int *mark_deletion = particle->mark_deletion;
    double *env = (particle->particle_envelope != nullptr) ?
        (double *)particle->particle_envelope->data : nullptr;

    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;
    for (int isend = 0; isend < halo->nproc_to; isend++) {
      //Find proc and storage
      int iproc = halo->proclist[isend];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_send; iloc++)
        if (mpi_group->neighbors_send[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      _ops_particle_number_of_particles_in_range(halo->sendBox[isend], dim, xcrds, noParticles,
                                                 &nsend[nhalos * iloc + h]);
      halo_info->nsend  = nsend[nhalos * iloc + h];

      if (halo_info->nsend > halo_info->nmax) {
        halo_info->sendlist = (int *)
          OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      _ops_particle_remove_from_region(halo->sendBox[isend], xcrds, env, mark_deletion,
                                       particle->no_particles, particle->block->dims,
                                       halo_info->sendlist);

      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
    }
  }

  //Communicate sending info
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

  //Unpack data
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;
    ops_particle particle_to = instance->OPS_particle_halo_list[halo->index]->particle_to;
    int ifirst = particle_to->no_particles;
    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      mpi_group->recv_bites[iloc] += nbites * nrecv[nhalos * iloc + h];
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h];
      halo_info->nrecv = nrecv[mpi_group->nhalos * iloc +h];
      halo_info->firstrecv = ifirst;
      ifirst += halo_info->nrecv;
    }
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  //Compute shifts during packing
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

  //Packing data
  int ntot_bites = 0;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int  isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get process and storage loc
      int iproc = halo_mpi->proclist[isend];
      int iloc;
      for (iloc = 0; mpi_group->num_neighbors_send; iloc++)
        if (mpi_group->neighbors_send[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h]; //TODO: Check it
      shift = nsend_shift[iloc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites);
      nsend_shift[iloc] += ntot_bites;
    }
  }

  //Sending data via MPI
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int shift = mpi_group->send_shift[i];
    MPI_Isend(ops_buffer_send_1 + shift, mpi_group->send_bites[i], MPI_BYTE,
              iproc, 1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int shift = mpi_group->recv_shift[i];
    MPI_Irecv(ops_buffer_recv_1 + shift, mpi_group->recv_bites[i], MPI_BYTE,
              iproc, 1000 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  //Finding if reallocations are needed
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    int nrecv1= 0;
    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++) {
        if (mpi_group->neighbors_recv[iloc] == iproc) break;
      }
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos * (mpi_group->num_neighbors_send + iloc) + h]; //IT IS WRONG
      nrecv1 += halo_info->nrecv;
    }
    halo->particle_to->no_particles += nrecv1; //TODO: Check
  }

  //Reallocate structures before unpack
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle particle = halo->particle_to;
    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }

    for (int imap = 0; imap < particle->particle_map_index; imap++) {
      ops_particle_mapping map = particle->map_list[imap];
      if (map->Nmax < particle->no_particles)
        _ops_particle_realloc_map_data(map,   particle->no_particles);
    }
  }

  //Unpacking exchange data

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
                                        &ntot_bites);
      nrecv_shift[iloc] += ntot_bites;

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

      for (int imap = 0; imap > halo->particle_to->particle_map_index; imap++) {
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
                                         ops_particle_halo_group halo_grp) {


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
  for (int h = 0; h < mpi_group->nhalos; h++) {
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
                                               halo_info, halo_main->dir_from, halo_main->dir_to, &ntot_bites);

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

void _ops_particle_exchange(ops_particle particle) { }

/* Build and send particle pos */
void _ops_particle_halo_border_pos_transfer(OPS_instance *instance,
                                            ops_particle_halo_group halo_grp) {
  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  int nhalos = mpi_group->nhalos;
  double c, t1, t2;
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
  int *nsend_pos_shift = (int *) ops_malloc(size * sizeof(int));


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
    for (int imap = 0; imap < to->particle_map_index; imap++) {
      ops_particle_mapping map = to->map_list[imap];
      if (map->Nmax < to->no_particles + to->no_virtual)
        _ops_particle_realloc_map_data(map,   to->no_particles + to->no_virtual);
    }
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

void _ops_particle_halo_forward_pos_transfer(OPS_instance *instance,
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
    shift_send[i] = mpi_group->shift_send_pos[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    shift_recv[i] = mpi_group->shift_recv_pos[i];

  int ntot_bites;
  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];

    ops_dat pos_dat = halo->particle_from->particle_pos_dat;
    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      int iproc = halo_mpi->proclist[isend];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_send; iloc++) {
        if (iproc == mpi_group->neighbors_send[iloc]) break;
      }
      ops_particle_halo_exchange halo_info = mpi_group->halo_info[nhalos * iloc + h];
      int ishift = shift_send[iloc];
      _ops_particle_halo_dat_to_buf(ops_buffer_send_1 + ishift, pos_dat, halo_info,
                                    &ntot_bites);
      shift_send[iloc] += ntot_bites;
    }
  }

  //Sending data
  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    int ishift = mpi_group->shift_send_pos[i];
    MPI_Isend(ops_buffer_send_1 + ishift, mpi_group->send_pos_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    int ishift = mpi_group->shift_recv_pos[i];
    MPI_Irecv(ops_buffer_recv_1 + ishift, mpi_group->recv_pos_bites[i], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];

    ops_particle_halo halo_main = main_halo_grp->halo_list[h];

    ops_dat coords = halo_main->particle_to->particle_pos_dat;

    for (int irecv = 0; irecv < halo_mpi->nproc_from; irecv++) {
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int iloc;
      for (iloc = 0; iloc < mpi_group->num_neighbors_recv; iloc++)
        if (mpi_group->neighbors_recv[iloc] == iproc) break;

      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[nhalos *(mpi_group->num_neighbors_send + iloc) + h];
      int ishift = shift_recv[iloc];
      _ops_particle_dat_copy_from_buff(ops_buffer_recv_1 + ishift, coords, OPS_PART_POSITION,
                                        halo_info, halo_main->dir_to, halo_main->dir_from,
                                        halo_main->translate,
                                        &ntot_bites);
      shift_recv[iloc] += ntot_bites;

      for (int imap = 0; imap > halo->particle_to->particle_map_index; imap++) {
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

//WORK IN PROGRESS
void ops_particle_halo_exchanges(ops_arg* args, int nargs, double *range,int ndim) {

  int send_recv_offsets[4];

  MPI_Comm comm= MPI_COMM_NULL;

  int nsd = (ndim >= 3) ? 3 : ndim;

  for (int dim = 0; dim < nsd; dim++) {

    int id_m = -1; int id_p = -1;
    int other_dims = 1;

    for (int i = 0; i < 4; i++)
      send_recv_offsets[i] = 0; //TODO: Add additional structures for shifting and sizes

    for (int i = 0; i < nargs; i++) {

      if (args[i].argtype != OPS_ARG_DAT_PARTICLE ||
          args[i].acc == OPS_WRITE || args[i].acc == OPS_MAX ||
          args[i].acc == OPS_MIN || args[i].opt == 0) continue;

      ops_dat dat = args[i].dat;
      int index = dat->block->index;
      int part_index = args[i].part_index;
      int map_index = args[i].map_index;
      ops_particle particle = OPS_instance::getOPSInstance()->OPS_block_list[index].particle[part_index];
      ops_particle_mapping map = particle->map_list[map_index];



//      if (args[i].argtype == OPS_ARG_DAT_PARTICLE  &&
//          args[i].acc ==OPS_READ))
      comm = OPS_sub_block_list[dat->block->index]->comm;

      //Get intersection range

    }
  }


}

void ops_particle_exchange(ops_particle particle) {

  //REMOVE VIRTUAL

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  int nlocal = particle->no_particles;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int *mark_deletion = particle->mark_deletion;
  double  *xpos = (double *)particle->particle_pos_dat->data;

  for (int idir = 0; idir < dim; idir++) {

    int nsend_recv_bites[4];
    for (int i = 0; i < 4; i++)
      nsend_recv_bites[i] = 0;

    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
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
                                    particle->particle_envelope,
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

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      dat,
                                      halo_int->particle_send_neg,
                                      isend_neg);

      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_pos,
                                      dat,
                                      halo_int->particle_send_pos,
                                      isend_pos);

      shift_neg += nsend_recv_bites[0] * dat->elem_size;
      shift_pos += nsend_recv_bites[1] * dat->elem_size;
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

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_2 + shift_recv_neg,
                                      dat, nexist,
                                      nsend_recv_bites[3]);

      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos, dat,
                                      nexist + nsend_recv_bites[3],
                                      nsend_recv_bites[2]);


      shift_recv_neg += nsend_recv_bites[3] * dat->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * dat->elem_size;
    }

    MPI_Waitall(2, &request[0], &status[0]);

    //TODO: Set and shift for non-local if not-within
    double *xcrds = (double *)pos_dat->data;
    for (int ipart = nexist; ipart < particle->no_particles; ipart++) {
      particle->mark_deletion[ipart] =
          (! particle->box_block->isCoordinateInBoundingBox(xcrds + ipart * dim)) ? 1 : 0;
    }
  }

  //Remove actual particles
  ops_particle_remove_marked_flag(particle, 4);

}

void _ops_particle_exchange_map_update(ops_particle particle) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  int nlocal = particle->no_particles;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int *mark_deletion = particle->mark_deletion;
  double  *xpos =(double *) particle->particle_pos_dat->data;

  for (int idir = 0; idir < dim; idir++) {
    int nsend_recv[4];

    for (int i = 0; i < 4; i++)
      nsend_recv[i] = 0;

    //TODO: Shift to idp-s: If on virtual elements then shift or beyond
    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 1) {
        printf("Particle for deletion %d neg[%f %f] x pos[%f %f]\n",ipart,
               halo_int->region_exch_neg[0], halo_int->region_exch_neg[1],
               halo_int->region_exch_pos[0], halo_int->region_exch_pos[1]);

        if (xpos[dim * ipart + idir] < halo_int->region_exch_neg[1] &&
            sb->id_m[idir] != MPI_PROC_NULL) {
          nsend_recv[0]++;
          mark_deletion[ipart] = 2;
          continue;
        }

        if (xpos[dim * ipart + idir] >= halo_int->region_exch_pos[0] &&
            sb->id_p[idir] != MPI_PROC_NULL) {
          nsend_recv[1]++;
          mark_deletion[ipart] = 3;
        }
      }
    }

    printf("Rank %d Number of particle to move in positive or in negative: %d and %d\n",
           ops_get_proc(), nsend_recv[1], nsend_recv[0]);

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

    printf("Rank %d: Packing %d and %d data\n", ops_get_proc(), nsend_recv[0] * sp->bites_in_exchange,
           nsend_recv[1] * sp->bites_in_exchange);

    //Send and receive elements
    MPI_Status status[4];

    MPI_Sendrecv(&nsend_recv[0], 1, MPI_INT, sb->id_m[idir], 100,
                 &nsend_recv[2], 1, MPI_INT, sb->id_p[idir], 100,
                 sb->comm, &status[0]);

    MPI_Sendrecv(&nsend_recv[1], 1, MPI_INT, sb->id_p[idir], 200,
                 &nsend_recv[3], 1, MPI_INT, sb->id_m[idir], 200,
                 sb->comm, &status[0]);


    printf("Rank %d: Receiving %d and %d elements\n",
           ops_get_proc(), nsend_recv[2], nsend_recv[3]);

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


    int nexist = particle->no_particles;
    particle->no_particles += nsend_recv[2] + nsend_recv[3];

    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }

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
    for (int ipart = nexist; ipart < particle->no_particles; ipart++) {
      particle->mark_deletion[ipart] =
          (! particle->box_block->isCoordinateInBoundingBox(xcrds + ipart * dim)) ? 1 : 0;
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


//TODO
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
    int ntot_neg = 0;
    int ntot_pos = 0;
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
      elem_size += sizeof(int); //TODO: Do we need that

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

      MPI_Waitall(0, &request[0], &status[0]);



      //Reset the send and receive
       ifirst_send = ilast_send;
       ilast_send = ifirst_send + halo_int->nrecv_neg[iswap] + halo_int->nrecv_pos[iswap];
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


    printf("Dir = %d: Rank = %d Region in neg =[%d %x]x[%d %d] and in pos [%d %d]x[%d %d]\n", idir, ops_get_proc(),
           halo_int->region_neg[0],halo_int->region_neg[1], halo_int->region_neg[2],
           halo_int->region_neg[3], halo_int->region_pos[0],  halo_int->region_pos[1],
           halo_int->region_pos[2],  halo_int->region_pos[3]);


    printf("Dir = %d: Rank = %d: nparts_send_pos = %d nparts_send_neg = %d\n", idir,
           ops_get_proc(),
           halo_int->nforward_pos[0],
           halo_int->nforward_neg[0]);

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
    int elem_size = sp->bites_in_exchange;
    elem_size += sizeof(int); //Why we add the int?

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

    printf("Dir = %d Rank %d Receiving first element and last in neg dir: %d %d\n", idir, ops_get_proc(), halo_int->irecv_neg[0],
           halo_int->irecv_neg[0] + halo_int->nrecv_neg[0]);
    printf("Dir = %d Rank %d Receiving first element and last in positive direction: %d %d\n", idir, ops_get_proc(), halo_int->irecv_pos[0],
           halo_int->nrecv_pos[0]  + halo_int->irecv_pos[0]);

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

void _ops_particle_packer_inters(ops_dat dat, ops_particle particle, double range_in[],
                                 int iswap, int idir, int send_recv_offsets[]) {

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  sub_particle sp = sb->sb_particle_list[particle->index];

  int nsend_pos = 0;
  int nsend_neg = 0;

  double range_pos[2 * OPS_MAX_DIM], range_neg[2 * OPS_MAX_DIM];
  int dim = particle->block->dims;
  ops_int_particle_halos halo_int = sp->particle_halos[idir];

  if (iswap >= halo_int->nswaps) return;

  range_pos[2 * idir + 1] = halo_int->region_bord_pos[2 * idir + 1];
  range_pos[2 * idir] = halo_int->region_bord_pos[2 * idir];

  range_neg[2 * idir] = halo_int->region_bord_neg[2 * idir];
  range_neg[2 * idir + 1] = halo_int->region_bord_neg[2 * idir + 1];

  for (int i = 0; i < dim; i++) {
    if (i != idir) {
      range_pos[2 * i] = MAX(range_in[2 * i], halo_int->region_bord_pos[2 * i]);
      range_pos[2 * i + 1] = MIN(range_in[2 * i + 1],
                                 halo_int->region_bord_pos[2 * i + 1]);
      range_neg[2 * i + 1] = MIN(range_in[2 * i + 1],
                                 halo_int->region_bord_pos[2 * i + 1]);
      range_neg[2 * i] = MIN(range_in[2 * i], halo_int->region_bord_pos[2 *i]);
    }
  }

  int ishift_n = 0;
  int ishift_p = 0;

  for (int i = 0; i < iswap; i++) {
    ishift_n += halo_int->nforward_neg[i];
    ishift_p += halo_int->nforward_pos[i];
  }

  _ops_particle_pack_intra_reg_dat_to_buff(ops_buffer_send_1 + send_recv_offsets[0],
                                           (double *) particle->particle_pos_dat->data,
                                           dat, dim, range_neg,
                                           halo_int->particle_send_neg + ishift_n,
                                           halo_int->nforward_neg[iswap], &nsend_neg);

  _ops_particle_pack_intra_reg_dat_to_buff(ops_buffer_send_2 + send_recv_offsets[1],
                                           (double *) particle->particle_pos_dat->data,
                                           dat, dim, range_pos,
                                           halo_int->particle_send_pos + ishift_p,
                                           halo_int->nforward_pos[iswap], &nsend_pos);



  send_recv_offsets[0] += nsend_neg * dat->elem_size;
  send_recv_offsets[1] += nsend_pos * dat->elem_size;

  //Find number of receiving elements
  int nrecv_neg = 0;
  int nrecv_pos = 0;

  range_pos[2 * idir] = particle->box_block->getMaxCoordDir(idir);
  range_neg[2 * idir + 1] = particle->box_block->getMinCoordDir(idir);

  _ops_particle_number_of_particles_in_range(range_neg, dim,
                                             (double *)particle->particle_pos_dat->data,
                                             halo_int->irecv_neg[iswap],
                                             halo_int->irecv_neg[iswap] + halo_int->nrecv_neg[iswap],
                                             &nsend_neg);


  _ops_particle_number_of_particles_in_range(range_pos, dim,
                                             (double *)particle->particle_pos_dat->data,
                                             halo_int->irecv_pos[iswap],
                                             halo_int->irecv_pos[iswap] + halo_int->nrecv_pos[iswap],
                                             &nsend_pos);




  send_recv_offsets[3] += nsend_neg * dat->elem_size;
  send_recv_offsets[2] += nsend_pos * dat->elem_size;

}

void _ops_particle_unpacker_inters(ops_dat dat, ops_particle particle, double range_in[],
                                   int iswap, int idir, int send_recv_offsets[]) {

  //Find data
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  sub_particle sp = sb->sb_particle_list[particle->index];

  int nrecv_pos = 0;
  int nrecv_neg = 0;

  double range_pos[2 * OPS_MAX_DIM], range_neg[2 * OPS_MAX_DIM];
  int dim = particle->block->dims;
  ops_int_particle_halos halo_int = sp->particle_halos[idir];

  if (iswap >= halo_int->nswaps) return;

  range_pos[2 * idir + 1] = halo_int->region_bord_pos[2 * idir + 1];
  range_pos[2 * idir] = particle->box_block->getMaxCoordDir(idir);

  range_neg[2 * idir] = halo_int->region_bord_neg[2 * idir];
  range_neg[2 * idir + 1] = particle->box_block->getMinCoordDir(idir);

  for (int i = 0; i < dim; i++) {
    if (i != idir) {
      range_pos[2 * i] = MAX(range_in[2 * i], halo_int->region_bord_pos[2 * i]);
      range_pos[2 * i + 1] = MIN(range_in[2 * i + 1],
                                 halo_int->region_bord_pos[2 * i + 1]);
      range_neg[2 * i + 1] = MIN(range_in[2 * i + 1],
                                 halo_int->region_bord_pos[2 * i + 1]);
      range_neg[2 * i] = MIN(range_in[2 * i], halo_int->region_bord_pos[2 *i]);
    }
  }

  _ops_particle_unpack_intra_reg_buff_to_dat(ops_buffer_recv_2 + send_recv_offsets[3],
                                           (double *) particle->particle_pos_dat->data,
                                           dat, dim, range_neg,
                                           halo_int->irecv_neg[iswap],
                                           halo_int->nrecv_neg[iswap] + halo_int->irecv_neg[iswap],
                                           &nrecv_neg);

  _ops_particle_unpack_intra_reg_buff_to_dat(ops_buffer_recv_1 + send_recv_offsets[2],
                                           (double *) particle->particle_pos_dat->data,
                                           dat, dim, range_pos,
                                           halo_int->irecv_pos[iswap],
                                           halo_int->nrecv_pos[iswap] + halo_int->irecv_pos[iswap],
                                           &nrecv_pos);

  send_recv_offsets[3] += nrecv_neg * dat->elem_size;
  send_recv_offsets[2] += nrecv_pos * dat->elem_size;




}



void ops_particle_halo_exchanges(ops_arg *args, int nargs, double *range_in) {
  int send_recv_offsets[4];
  for (int idim = 0; idim < OPS_MAX_DIM; idim++) {



    //Find number of swaps in the given direction
    int nswaps = 0;
    int flag = 0;
    MPI_Comm comm = MPI_COMM_NULL;

    for (int i = 0; i < nargs; i++) {
      if (args[i].argtype != OPS_ARG_DAT_PARTICLE ||
          (args[i].acc == OPS_WRITE || args[i].acc == OPS_MAX ||
           args[i].acc == OPS_MIN ) || args[i].opt == 0) continue;

      //TODO: Need one more
      ops_dat dat = args[i].dat;
      ops_particle particle =
          OPS_instance::getOPSInstance()->OPS_block_list[dat->block->index].particle[args[i].part_index];
      sub_block *sb = OPS_sub_block_list[dat->block->index];

      nswaps = MAX(nswaps, sb->sb_particle_list[particle->index]->particle_halos[idim]->nswaps);
    }

    for (int iswap = 0; iswap < nswaps; iswap++) {

      for (int i = 0; i < 4; i++)
        send_recv_offsets[i] = 0;

      int id_p = MPI_PROC_NULL;
      int id_m = MPI_PROC_NULL;

      for (int i = 0; i < nargs; i++) {

        //Checks for packing data
        if (args[i].argtype != OPS_ARG_DAT_PARTICLE ||
            (args[i].acc == OPS_WRITE || args[i].acc == OPS_MAX ||
            args[i].acc == OPS_MIN) ||
            args[i].opt == 0)
          continue;

        ops_dat dat = args[i].dat;
        ops_particle particle =
        OPS_instance::getOPSInstance()->OPS_block_list[dat->block->index].particle[args[i].part_index];
        sub_block *sb = OPS_sub_block_list[dat->block->index];

        if (particle->block->dims >= idim)  continue;

        //TODO Another check



        //Perform intersection if needed
        flag = check_intersection(particle->box_block, range_in, idim);


        //Computing also receiving sizes
        if (flag == 0) continue;
        else if (flag == 1)
          _ops_particle_packer_inters(dat, particle, range_in, iswap, idim, send_recv_offsets);
        else if (flag == 2)
          _ops_particle_packer_all(dat, particle, range_in, iswap, idim, send_recv_offsets);

        comm = sb->comm;
        id_m = sb->id_m[idim];
        id_p = sb->id_p[idim];
      }

      if (flag == 0 || comm == MPI_COMM_NULL) continue;

      MPI_Request requests[4];
      MPI_Status  status[4];
      MPI_Isend(ops_buffer_send_1, send_recv_offsets[0], MPI_BYTE,
                send_recv_offsets[0] > 0 ? id_m  : MPI_PROC_NULL, idim, comm,
                &requests[0]);
      MPI_Isend(ops_buffer_send_2, send_recv_offsets[1], MPI_BYTE,
                send_recv_offsets[1] > 0 ? id_p : MPI_PROC_NULL,
                    idim + OPS_MAX_DIM, comm, &requests[1]);

      MPI_Irecv(ops_buffer_recv_1, send_recv_offsets[2], MPI_BYTE,
                send_recv_offsets[2] > 0 ? id_p : MPI_PROC_NULL,
                idim, comm, &requests[2]);

      MPI_Irecv(ops_buffer_recv_2, send_recv_offsets[3], MPI_BYTE,
                send_recv_offsets[1] > 0 ? id_m : MPI_PROC_NULL,
                idim + OPS_MAX_DIM, comm, &requests[3]);

      MPI_Waitall(2, &requests[2], &status[2]);

      //Loop over all particles to unpack

      for (int i = 0; i < 4; i++)
        send_recv_offsets[i] = 0;


      for (int i = 0; i < nargs; i++) {

        if (args[i].argtype != OPS_ARG_DAT_PARTICLE ||
            !(args[i].acc == OPS_READ || args[i].acc == OPS_RW)
            || args[i].opt == 0) continue;

        ops_dat dat = args[i].dat;
        ops_particle particle =
        OPS_instance::getOPSInstance()->OPS_block_list[dat->block->index].particle[args[i].part_index];
        sub_block *sb = OPS_sub_block_list[dat->block->index];

        if (particle->block->dims >= idim)  continue;

        //TODO: Do we need intersection
        flag = check_intersection(particle->box_block, range_in, idim);

        if (flag  == 0) continue;

        if (flag == 1)
          _ops_particle_unpacker_inters(dat, particle, range_in, iswap, idim, send_recv_offsets);
        else if (flag == 2)
          _ops_particle_unpacker_all(dat, particle, range_in, iswap, idim, send_recv_offsets);

      }

      MPI_Waitall(2, &requests[0], &status[0]);
    }

  }
}


void _ops_particle_update_maps_due_to_halos(ops_particle particle,
                                            ops_particle_mapping map) { }



