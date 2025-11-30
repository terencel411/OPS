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
  * @brief OPS mpi support functions for particle management
  * @author
  * @details Setup the fine details of domain decomposition for particle packages within
  *          OPS
  */

#include <math.h>
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_exceptions.h>


/*-------------------------------------------------------------------------------------*
 *  Setting domain partition for particle data involving
 *  1. Particle domain partition
 *  2. Setting details of forward communication for particle exchange data
 */
void ops_particle_setup_partition() {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, " Setting up particle partition must "
                                          "happen after domain partition.");

  // loop over all blocks
  for (int index = 0; index < instance->OPS_block_index; index++) {
    ops_block block= instance->OPS_block_list[index];

    sub_block *sb = OPS_sub_block_list[block->index];

    if (!sb->owned) return;

    /* Get access to particle data */
    sub_particle *spar = sb->sb_particle_list;
    int nparticles = instance->OPS_block_list[index].no_particle_structures;

    for (int ipartlist = 0; ipartlist < nparticles; ipartlist++) {
      ops_particle particle = spar[ipartlist].particle;
      sub_particle sub_part = spar[ipartlist];

      //Build bounding box for particle //
      ops_build_bounding_box(particle);

      //Set communication structures in terms of boxes

      //TODO:1. Find how far we go for data exchange
      //     2. Set number of loops in each  for rcv data
      ops_particle_setup_forward_comm(sub_particle);



    }
  }

}


void ops_decomp_dats(sub_block *sb) {
  ops_block block = sb->block;
  ops_dat_entry *item, *tmp_item;
  for (item = TAILQ_FIRST(&(OPS_instance::getOPSInstance()->OPS_block_list[block->index].datasets));
       item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;

    if (dat->is_particle) continue;

    sub_dat *sd = OPS_sub_dat_list[dat->index]; //TODO: Do we need to add particle dat lists to the
    //structures

    // aggregate size and prod array
    size_t *prod_t = (size_t *)ops_malloc((sb->ndim + 1) * sizeof(size_t));
    size_t *prod = &prod_t[1];
    prod[-1] = 1;
    sd->prod = prod;
    sd->halos = NULL;

    for (int d = 0; d < block->dims; d++) {
      // first store away the details of the dat (i.e global dat details)
      sd->gbl_base[d] = dat->base[d]; // global start base
      sd->gbl_size[d] = dat->size[d]; // global size of data elements in this
                                      // dat (i.e. pure data)
      sd->gbl_d_m[d] =
          dat->d_m[d]; // global dat halo at the beginning (minus size)
      sd->gbl_d_p[d] = dat->d_p[d]; // global dat halo at the end (positive
                                    // size)

      // special treatment if it's an edge dataset in this direction
      if (dat->e_dat && (dat->size[d] == 1)) {
        if (dat->base[d] != 0) {
          OPSException ex(OPS_RUNTIME_ERROR);
          ex << "Error: dataset " << dat->name << " is an edge dataset, but has a non-0 base";
          throw ex;
        }
        prod[d] = prod[d - 1];
        sd->decomp_disp[d] = 0;
        sd->decomp_size[d] = 1;
        sd->d_im[d] = 0; // no intra-block halo
        sd->d_ip[d] = 0;
        continue;
      }

      // global size of dat
      int zerobase_gbl_size =
          dat->size[d] + dat->d_m[d] - dat->d_p[d] + dat->base[d];

      sd->decomp_disp[d] = sb->decomp_disp[d]/dat->stride[d] + (sb->decomp_disp[d]%dat->stride[d] == 0 ? 0:1);
      int next_block = (sb->decomp_disp[d]+sb->decomp_size[d]);
      sd->decomp_size[d] = MAX(0,MIN(next_block/dat->stride[d] + (next_block%dat->stride[d] == 0 ? 0:1) - sd->decomp_disp[d],
                                      zerobase_gbl_size - sd->decomp_disp[d]));
      if (sb->id_m[d] != MPI_PROC_NULL) {
        // if not negative end, then there is no block-level left padding, but
        // intra-block halo padding
        dat->base[d] = 0;
        // TODO: compute this properly, or lazy or something
        sd->d_im[d] = dat->d_m[d]; // intra-block (MPI) halos are set to be
                                   // equal to block halos
        if (OPS_instance::getOPSInstance()->ops_enable_tiling && OPS_instance::getOPSInstance()->ops_tiling_mpidepth>0)
sd->d_im[d] = -OPS_instance::getOPSInstance()->ops_tiling_mpidepth;

        dat->d_m[d] = 0;
      } else {
        sd->decomp_disp[d] +=
            (dat->base[d] + dat->d_m[d]); // move left end to negative for base
                                          // and left block halo
        sd->decomp_size[d] -= (dat->base[d] + dat->d_m[d]); // extend size
        sd->d_im[d] = 0; // no intra-block halo
      }

      if (sb->id_p[d] != MPI_PROC_NULL) {
        // if not positive end
        // TODO: compute this properly, or lazy or something
        sd->d_ip[d] = dat->d_p[d]; // intra-block (MPI) halos are set to be
                                   // equal to block halos

        if (OPS_instance::getOPSInstance()->ops_enable_tiling && OPS_instance::getOPSInstance()->ops_tiling_mpidepth>0)
sd->d_ip[d] = OPS_instance::getOPSInstance()->ops_tiling_mpidepth;

        dat->d_p[d] = 0;

        /*if (d == 0) { // Compute x-dim padding for vectorization
          int temp_size = sd->decomp_size[0] - sd->d_im[0] + sd->d_ip[0];
          int x_pad = (1+((temp_size-1)/32))*32 - temp_size;
          sd->d_ip[0] = x_pad;
        }*/

      } else {
        sd->decomp_size[d] += dat->d_p[d]; // if last in this dimension, extend
                                           // with left block halo size
        sd->d_ip[d] = 0;                   // no intra-block halo

        /*if (d == 0) { // Compute x-dim padding for vectorization
          int x_pad = (1+((sd->decomp_size[0]-1)/32))*32 - sd->decomp_size[0] ;
          sd->decomp_size[0] += x_pad;
          dat->d_p[0] += x_pad;
        }*/
      }
      dat->size[d] = sd->decomp_size[d] - sd->d_im[d] + sd->d_ip[d];
      prod[d] = prod[d - 1] * dat->size[d];
    }

    if (!sb->owned) {
      sd->mpidat = NULL;
      continue;
    }

    // Allocate datasets for ops_dat
    //TODO: Need to include an extra set for particle data structures
    if (dat->data == NULL){
      if (dat->is_hdf5 == 0) {
        //dat->data = (char *)ops_calloc(prod[sb->ndim-1]*dat->elem_size,1);
        dat->data = (char *)ops_malloc(prod[sb->ndim - 1] * dat->elem_size * 1);
        dat->hdf5_file = "none";
        dat->mem =
            prod[sb->ndim - 1] * dat->elem_size; // this includes the halo sizes
      } else {
        dat->data = (char *)ops_calloc(prod[sb->ndim - 1] * dat->elem_size, 1);
        dat->mem =
            prod[sb->ndim - 1] * dat->elem_size; // this includes the halo sizes
        if (ops_read_dat_hdf5_dynamic == NULL) {
          OPSException ex(OPS_RUNTIME_ERROR);
          ex << "Error: using ops_decl_dat_hdf5, but lib_ops_hdf5_mpi.so was not linked";
          throw ex;
        }
        ops_read_dat_hdf5_dynamic(dat);
      }
    } else {
        dat->user_managed = 1;
        dat->is_hdf5 = 0;
        dat->hdf5_file = "none";
    }

    // Compute offset in bytes to the base index
    dat->base_offset = 0; //TODO: First element to access in the list
    size_t cumsize = 1;
    for (int i = 0; i < block->dims; i++) {
      dat->base_offset += (OPS_instance::getOPSInstance()->OPS_soa ? dat->type_size : dat->elem_size)
                          * cumsize *
                          (-dat->base[i] - dat->d_m[i] - sd->d_im[i]);
      cumsize *= dat->size[i];
    }

    ops_cpHostToDevice(dat->block->instance, (void **)&(dat->data_d), (void **)&(dat->data),
                       prod[sb->ndim - 1] * dat->elem_size);

    // TODO: halo exchanges should not include the block halo part for
    // partitions that are on the edge of a block

    /// MPI data types are no longer used as the manual data types were found to
    /// be more optimal
    // sd->mpidat = (MPI_Datatype *) ops_malloc(sizeof(MPI_Datatype)*sb->ndim *
    // MAX_DEPTH);
    // MPI_Datatype new_type_p; //create generic type for MPI comms
    // MPI_Type_contiguous(dat->elem_size, MPI_CHAR, &new_type_p);
    // MPI_Type_commit(&new_type_p);

    sd->halos =
        (ops_int_halo *)ops_calloc(MAX_DEPTH * sb->ndim , sizeof(ops_int_halo));

    for (int n = 0; n < sb->ndim; n++) {
      for (int d = 0; d < MAX_DEPTH; d++) {
        /// MPI data types are no longer used as the manual data types were
        /// found to be more optimal
        // MPI_Type_vector(prod[sb->ndim - 1]/prod[n], d*prod[n-1],
        //                prod[n], new_type_p, &(sd->mpidat[MAX_DEPTH*n+d]));
        // MPI_Type_commit(&(sd->mpidat[MAX_DEPTH*n+d]));

        // populate the struct duplicating information in MPI_Datatypes for
        // (strided) halo access
        sd->halos[MAX_DEPTH * n + d].count = prod[sb->ndim - 1] / prod[n];
        sd->halos[MAX_DEPTH * n + d].blocklength =
            d * prod[n - 1] * dat->type_size;
        sd->halos[MAX_DEPTH * n + d].stride = prod[n] * dat->type_size;

        // printf("Datatype: %d %d %d\n", prod[sb->ndim - 1]/prod[n], prod[n-1],
        // prod[n]);
        // printf("dat->name %s, Datatype %d %d
        // %d\n",dat->name,sd->halos[MAX_DEPTH*n+d].count,
        // sd->halos[MAX_DEPTH*n+d].blocklength,
        // sd->halos[MAX_DEPTH*n+d].stride);
      }
    }
  }
}

void ops_particle_halo_set_exchage_send(ops_mpi_particle_halo &mpi_halo,
                                        double *recv_box_regions,
                                        int dim) {

  ops_particle_halo halo = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo.index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block];
  if (!sb->owned) {mpi_halo.nproc_to = 0; return; }

  int nsend_max = mpi_halo.nproc_to;
  int nsend = 0;


  BoundingBox *sendBox = halo->particle_from->box_block;
  if (!sendBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting particle halos must occur after "
                                          "all block boxes are set\n");
  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  sendBox->getLocalMaxMin(xmin, xmax);
  double xrecv_min[OPS_MAX_DIM], xrecv_max[OPS_MAX_DIM];
  double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];



  for (int i = 0; i < nsend_max; i++) {
    //Set box to identify region
    for (int isou = 0; isou < dim; isou++) {
      int irecv_dir = halo->dir_to[isou];
      int isend_dir = halo->dir_from[isou];
      xrecv_min[isou] = recv_box_regions[2 * dim * i + 2 * isou]
                      - halo->translate[isend_dir];
      xrecv_max[isou] = recv_box_regions[2 * dim * i + 2 * isou + 1]
                      - halo->translate[isend_dir];
    }

    int intersect = ops_check_box_intersection(dim, xmin, xmax,
                                               xrecv_min, xrecv_max); //TODO: Assume ok for the moment

    //We have a possible interaction with 8 surrounding elements for the moment
    if (intersect == 1) {


      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {

          for (int j = 0; j < dim; j++) {
            xsend_min[j] = xmin[j];
            xsend_max[j] = xmax[j];
          }

          xsend_min[is] = (iswap == 0) ? xmin[is] - 0.5 * BIG : xmax[i];
          xsend_max[is] = (iswap == 0) ? xmin[is] : xmax + 0.5 * BIG;


          int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                              xrecv_min, xrecv_max);
          if (a1 == 1) {
            for (int j = 0; j < dim; j++) {
              if (is != j) {
                xsend_min[j] = (fabs(xsend_min[j] - xrecv_min[j]) < 1.e-9) ?
                    -BIG : MAX(xrecv_min[j], xrecv_min[j]);
                xsend_max[j] = (fabs(xsend_max[j] - xrecv_max[j]) < 1.e-9) ?
                    BIG : MIN(xsend_max[j], xrecv_max[j]);
              }
            }

            //Setting boxes etc
            mpi_halo.proclist[nsend] = mpi_halo.proclist_complete[i];
            mpi_halo.sendBox[nsend] = new BoundingBox(dim);
            mpi_halo.sendBox[nsend]->setBoundingBoxLocalBound(xsend_min, xsend_max);
            nsend++;
            goto endline;
          }
        }
      }
    }

    endline:
    int a1 = 0;


  }

  mpi_halo.nproc_to = nsend;

}

void     ops_particle_halo_set_recv_block(ops_mpi_particle_halo &mpi_halo,
                                          double *send_box_regions, int dim) {

  ops_particle_halo halo
    = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo.index];
  sub_block *sb = OPS_sub_block_list[halo->particle_to->block->index];
  if (!sb->owned) {mpi_halo.nproc_from = 0; return;}

  int nrecv_max = mpi_halo.nproc_from;
  int nrecv = 0;


}

void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {

  double *recv_box_regions = NULL;
  double *send_box_regions = NULL;
  int size_recv_max = 0;
  int size_send_max = 0;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (!OPS_sub_block_list[halo->particle_to->block->index]->owned &&
        !OPS_sub_block_list[halo->particle_from->block->index]->owned) continue;

    ops_mpi_particle_halo  mpi_halo = OPS_mpi_particle_halo_list[halo->index];


    int dim = halo->particle_to->block->dims;


    int size_recv = 2 * sizeof(double) * dim * mpi_halo.nproc_to;

    if (size_recv> size_recv_max) {
      recv_box_regions = (double *)ops_realloc(recv_box_regions, size_recv);
      size_recv_max = size_recv;
    }

    int size_send = 2 * sizeof(double) * dim * OPS_mpi_particle_halo_list[halo->index].nproc_from;
    if (size_send > size_send_max) {
      send_box_regions = (double *)ops_realloc(send_box_regions, size_send);
      size_send_max = size_send;
    }


    ops_get_send_recv_box(recv_box_regions, send_box_regions, OPS_mpi_particle_halo_list[halo->index], dim);

    //TODO: Add elements for checking

    ops_particle_halo_set_exchage_send(OPS_mpi_particle_halo_list[halo->index],
                                       recv_box_regions, dim);

    ops_particle_halo_set_exchange_recv(OPS_mpi_particle_halo_list[halo->index],
                                        send_box_regions, dim);

    //Can we??????
  }


  //Create the halo group
  ops_mpi_particle_halo_group *mpi_group = &OPS_mpi_particle_halo_group_list[halo_grp->index];
  int owned = 0;
  for (int j = 0; j < halo_grp->nhalos; j++) {
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_from > 0  ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_to > 0)
      owned++;
  }

  mpi_group->group = halo_grp;
  mpi_group->nhalos = owned;
  mpi_group->index = halo_grp->index;

  mpi_group->mpi_halos
    = (ops_mpi_particle_halo **) ops_malloc(owned * sizeof(ops_mpi_particle_halo *));

  for (int i = 0; i < mpi_group->nhalos; i++)
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_from > 0 ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_to > 0) {
      mpi_group->mpi_halos[i] = &OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index];
    }


  mpi_group->num_neighbors_send = 0;
  mpi_group->num_neighbors_recv = 0;
  mpi_group->nhalo_info = 0;

  int *neighbor_send = (int *)ops_malloc(ops_comm_global_size * sizeof(int));
  int *neighbor_recv = (int *)ops_malloc(ops_comm_global_size * sizeof(int));

  for (int i = 0; i < ops_comm_global_size; i++) {
    neighbor_send[i] = 0;
    neighbor_recv[i] = 0;
  }

  for (int i = 0; i < mpi_group->nhalos; i++) {
    for (int k = 0; k < mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_send[iproc]++;
    }

    for (int k = mpi_group->mpi_halos[i]->nproc_to;
             k < mpi_group->mpi_halos[i]->nproc_from
               + mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_recv[iproc]++;
    }
  }

  for (int i = 0; i < ops_comm_global_size; i++) {
    mpi_group->num_neighbors_send += (neighbor_send[i] > 0) ? 1 : 0;
    mpi_group->num_neighbors_send += (neighbor_recv[i] > 0) ? 1 : 0;
  }



  int ntot = 0;
  for (int i = 0; i < mpi_group->nhalos;i++) {
    ntot += mpi_group->mpi_halos[i]->nproc_to + mpi_group->mpi_halos[i]->nproc_from;
  }

  mpi_group->halo_info = (ops_particle_halo_exchange *)ops_malloc(sizeof(ops_particle_halo_exchange) * ntot);
  mpi_group->nhalo_info = ntot;
  for (int igroup = 0; igroup < ntot; igroup++)
    mpi_group->halo_info[i] = (ops_particle_halo_exchange) ops_malloc(sizeof(OPS_particle_halo_exchange_info_core));

    //TODO: We do not need the process, we will need to simply copy them
  mpi_group->neighbors_send =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
  mpi_group->neighbors_recv =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);

  mpi_group->requests = (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                   sizeof(MPI_Request));
  mpi_group->statuses = (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                  sizeof(MPI_Status));
  int k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j) {
    if (neighbor_send[j] > 0)
      mpi_group->neighbors_send[k] = j;

  }


  k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j)
    if (neighbor_recv[j] > 0)
      mpi_group->neighbors_recv[k] = j;








  //TODO: Set up the halo_group exchange structures

  ops_free(recv_box_regions);
  ops_free(send_box_regions);

  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}

void ops_particle_halo_set_exchage_send(ops_mpi_particle_halo *mpi_halo,
                                        double *recv_box_regions,
                                        int dim) {

  ops_particle_halo halo = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo.index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block];
  if (!sb->owned) {mpi_halo->nproc_to = 0; return; }

  int nsend_max = mpi_halo->nproc_to_max;
  int nsend = 0;


  BoundingBox *sendBox = halo->particle_from->box_block;
  if (!sendBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting particle halos must occur after "
                                          "all boundary boxes are set\n");
  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  sendBox->getLocalMaxMin(xmin, xmax);
  double xrecv_min[OPS_MAX_DIM], xrecv_max[OPS_MAX_DIM];
  double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];



  for (int i = 0; i < nsend_max; i++) {
    //Set box to identify region
    for (int isou = 0; isou < dim; isou++) {
      int irecv_dir = halo->dir_to[isou];
      int isend_dir = halo->dir_from[isou];
      xrecv_min[isend_dir] = recv_box_regions[2 * dim * irecv_dir + 2 * isou]
                      - halo->translate[isend_dir];
      xrecv_max[isend_dir] = recv_box_regions[2 * dim * irecv_dir + 2 * isou + 1]
                      - halo->translate[isend_dir];
    } //tODO: Check

    int intersect = ops_check_box_intersection(dim, xmin, xmax,
                                               xrecv_min, xrecv_max); //TODO: Assume ok for the moment

    //We have a possible interaction with 8 surrounding elements for the moment
    if (intersect == 1) {


      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {

          for (int j = 0; j < dim; j++) {
            xsend_min[j] = xmin[j];
            xsend_max[j] = xmax[j];
          }

          xsend_min[is] = (iswap == 0) ? xmin[is] - 0.5 * BIG : xmax[i];
          xsend_max[is] = (iswap == 0) ? xmin[is] : xmax + 0.5 * BIG;


          int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                              xrecv_min, xrecv_max);
          if (a1 == 1) {
            for (int j = 0; j < dim; j++) {
              if (is != j) {
                xsend_min[j] = (fabs(xsend_min[j] - xrecv_min[j]) < 1.e-9) ?
                    -BIG : MAX(xrecv_min[j], xrecv_min[j]);
                xsend_max[j] = (fabs(xsend_max[j] - xrecv_max[j]) < 1.e-9) ?
                    BIG : MIN(xsend_max[j], xrecv_max[j]);
              }
            }

            //Setting boxes etc
            mpi_halo->proclist[nsend] = mpi_halo->proclist_complete[i];
            mpi_halo->sendBox[nsend] = new BoundingBox(dim);
            mpi_halo->sendBox[nsend]->setBoundingBoxLocalBound(xsend_min, xsend_max);
            nsend++;
            goto endline;
          }
        }
      }
    }

    endline:
    int a1 = 0;


  }

  mpi_halo->nproc_to = nsend;

}

void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp) { }

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp) { }

void _ops_particle_halo_border_transfer_vs(OPS_instance *instance,
                                           ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];

  if (mpi_group->nhalos == 0) return;
  double c, t1, t2;
  ops_timers_core(&c, &t1);
 // mpi_neigh_size[0] = 0;

  int nhalos = mpi_group->nhalos;
  int nsend[ops_comm_global_size * mpi_group->nhalos];
  for (int i = 0; i < ops_comm_global_size; i++)
    nsend[i] = 0;

  int nrecv[ops_comm_global_size * mpi_group->nhalos];

  int nsend_size[ops_comm_global_size];
  int nrecv_size[ops_comm_global_size];
  int nsend_shift[ops_comm_global_size];
  int nrecv_shift[ops_comm_global_size];

  for (int i = 0; i < ops_comm_global_size; i++) {
    nsend_size[i] = 0;
    nrecv_size[i] = 0;
  }

  int ntot = 0;
  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;

    ops_particle_mapping map
    = instance->OPS_particle_halo_list[halo->index]->particle_from->map_list[0];
    ops_dat binhead = map->binhead;
    int *bins = (int *)map->bin->data;
    int *bin_head =(int *)binhead->data;
    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;

    //Find number of elements to send in different halos
    for (int isend = 0; isend < nproc_to; isend++) {
      ops_particle_halo_exchange halo_info = mpi_group->halo_info[ntot + isend];
      int iproc = halo->proclist[isend];

      _ops_particle_number_of_particles_in_range(halo->send_region, bin_head, bins,
                                                 binhead->size, &nsend[nhalos * iproc + h]);

      halo_info->nsend = nsend[nhalos * iproc + h];

      if (halo_info->nsend > halo_info->nmax) {
        OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      //TODO: Perform maps into the list
      _ops_particle_mapped_into_region(halo->send_region, bin_head, bins,
                                      binhead->size, halo_info->sendlist);

      nsend_size[iproc] += bites * halo_info->nsend;
    }

    ntot += halo->nproc_to + halo->nproc_from;
  }



  //Send the data and wait to be received
  for (int isend = 0; isend < mpi_group->num_neighbors_send; isend++) {
    int ishift = mpi_group->nhalos * mpi_group->neighbors_send[isend];
    MPI_Isend(nsend + ishift, mpi_group->nhalos, MPI_INT, mpi_group->neighbors_send[isend],
              1000 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[isend]);

  }

   //Receiving data for building chuncks
  for (int irecv = 0; irecv < mpi_group->num_neighbors_recv; irecv++) {
    int ishift = mpi_group->nhalos * mpi_group->neighbors_recv[irecv];
    MPI_Irecv(nrecv + ishift, mpi_group->nhalos, MPI_INT,
              mpi_group->neighbors_recv[irecv],
              1000+mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + irecv]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  //Need to send a big chunk of data at once
  ntot = 0;

  for (int h = 0; h < mpi_group->nhalos; h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    ops_particle particle_to
      = instance->OPS_particle_halo_list[halo->index]->particle_to;
    int ifirst = particle_to->no_particles + particle_to->no_virtual;
    int no_new = 0;
    int nbites = instance->OPS_particle_halo_list[halo->index]->nbites;

    for (int irecv = 0; irecv < halo->nproc_from; irecv++) {
      int iproc = halo->proclist[halo->nproc_to + irecv];
      ops_particle_halo_exchange halo_info =
          mpi_group->halo_info[ntot + halo->nproc_to + irecv];
      halo_info->nrecv = nrecv[nhalos * iproc + h];
      halo_info->firstrecv = ifirst;
      ifirst += halo_info->nrecv;
      no_new += halo_info->nrecv;

      nrecv_size[iproc] += nbites * halo_info->nrecv;
    }
    particle_to->no_virtual += no_new;
    ntot += halo->nproc_from + halo->nproc_to;
  }


  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  int shift = 0;
  int shift0 = 0;
  for (int i = 0; i < ops_comm_global_size; i++) {
    nsend_shift[i] = shift;
    shift += nsend_size[i];
    nrecv_shift[i] = shift0;
    shift0 += nrecv_size[i];
  }


  for (int  i = 0; i < mpi_group->neighbors_send;i++) {
    int iproc = mpi_group->neighbors_send[i];
    mpi_group->send_bites[i] = nsend_size[iproc];
    mpi_group->send_shift[i] = nsend_shift[iproc];
  }


  for (int i = 0; i < mpi_group->neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    mpi_group->recv_bites[i]= nrecv_size[iproc];
    mpi_group->recv_shift[i] = nrecv_shift[iproc];
  }


  //Reallocate buffers if necessary
  int nmax_send = 0;
  int nmax_recv = 0;

  int shift_send[ops_comm_global_size + 1];
  shift_send[0] = 0;
  int shift_recv[ops_comm_global_size +1];
  shift_recv[0] = 0;
  for (int i = 0; i < ops_comm_global_size; i++) {
    nmax_send += nsend_size[i];
    nmax_recv += nrecv_size[i];
    shift_send[i + 1] = nsend_size[i];
    shift_recv[i + 1] = nrecv_size[i];
  }

  if (nmax_send > ops_buffer_send_1_size) {
    OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, nmax_send);
    ops_buffer_send_1_size = nmax_send;
  }

  if (nmax_recv > ops_buffer_recv_1_size) {
    OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, nmax_recv);
    ops_buffer_recv_1_size = nmax_recv;
  }

  //Packing halo-data
  ntot = 0;
  int ntot_bites = 0;

  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo_mpi = mpi_group->mpi_halos[h];
    ops_particle_halo halo = instance->OPS_particle_halo_list[halo_mpi->index];
    ops_particle_halo_data  *halo_data = halo->dat;

    for (int isend = 0; isend < halo_mpi->nproc_to; isend++) {
      //Get data
      ops_particle_halo_exchange halo_info = mpi_group->halo_info[ntot + h];
      int iproc = halo_mpi->proclist[isend];
      int shift = shift_send[iproc];
      _ops_particle_halo_copy_tobuf(ops_buffer_send_1 + shift, halo_data, halo->nhalos,
                                    halo_info, &ntot_bites);
      shift_send[iproc] += ntot_bites;
    }
    ntot += halo_mpi->nproc_to + halo_mpi->nproc_from; //TODO

  }

  for (int i = 0; i < mpi_group->num_neighbors_send; i++) {
    int iproc = mpi_group->neighbors_send[i];
    MPI_Isend(ops_buffer_send_1 + shift_send[iproc], nsend_size[iproc], MPI_BYTE,
             iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL, &mpi_group->requests[i]);
  }

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++) {
    int iproc = mpi_group->neighbors_recv[i];
    MPI_Irecv(ops_buffer_recv_1 + shift_recv[iproc], nrecv_size[iproc], MPI_BYTE,
              iproc, 100 + mpi_group->index, OPS_MPI_GLOBAL,
              &mpi_group->requests[mpi_group->num_neighbors_send + i]);
  }

  MPI_Waitall(mpi_group->num_neighbors_recv,
              &mpi_group->requests[mpi_group->num_neighbors_send],
              &mpi_group->statuses[mpi_group->num_neighbors_send]);


  //Unpacking halo data
  ntot = 0;
  int ntot_bites = 0;


  for (int h = 0; h < halo_grp->nhalos; h++) {
    ops_particle to = halo_grp->halo_list[h]->particle_to;
    if (to->no_virtual + to->no_particles > to->Nmax)
      ops_particle_realloc_data(to, to->no_virtual + to->no_particles);
    for (int imap = 0; imap < to->map_list[imap]; imap++) {
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
      ops_particle_halo_exchange halo_info
      = mpi_group->halo_info[ntot + halo_mpi->nproc_to + h];
      int iproc  = halo_mpi->proclist[halo_mpi->nproc_to + irecv];
      int shift = shift_recv[iproc];
      _ops_particle_halo_copy_from_buff(ops_buffer_recv_1 + shift, halo_data, halo->nhalos,
                                        halo_info, halo->dir_to, halo->dir_from, halo->translate,
                                        &ntot_bites);
      shift_recv[iproc] += ntot_bites;
      //NEED TO MAP THOSE ELEMENTS AS WELL
      for (int imap = 0; imap > halo->particle_to->particle_map_index; imap++) {
        ops_particle_mapping map = halo->particle_to->map_list[imap];
        _ops_particle_mapping_virtual_from_halo(map, halo->particle_to,
                                                halo_info->firstrecv,
                                                halo_info->nrecv);

      }
    }
    ntot += halo_mpi->nproc_to + halo_mpi->nproc_from;
  }

  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0], &mpi_group->statuses[0]);

  ops_timers_core(&c, &t2);
  instance->ops_user_halo_exchanges_time += t2 - t1;

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

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    mpi_group->send_bites[i] = 0;

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    mpi_group->recv_bites[i] = 0;

  for (int h = 0; h < mpi_group->nhalos;h++) {
    ops_mpi_particle_halo *halo = mpi_group->mpi_halos[h];
    int nproc_to = halo->nproc_to;

    ops_particle_mapping map
    = instance->OPS_particle_halo_list[halo->index]->particle_from->map_list[0];
    ops_dat binhead = map->binhead;
    int *bins = (int *)map->bin->data;
    int *bin_head =(int *)binhead->data;
    int bites = instance->OPS_particle_halo_list[halo->index]->nbites;

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
        OPS_realloc_fast((char *)halo_info->sendlist, sizeof(int) * halo_info->nmax,
                         sizeof(int) * (halo_info->nsend + 10));
        halo_info->nmax = halo_info->nsend + 10;
      }

      //TODO: Perform maps into the list
      _ops_particle_mapped_into_region(halo->send_region, bin_head, bins,
                                      binhead->size, halo_info->sendlist);

      //Set size directly

      mpi_group->send_bites[iloc] += bites * halo_info->nsend;
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
    }
    particle_to->no_virtual += no_new;
  }


  MPI_Waitall(mpi_group->num_neighbors_send,
              &mpi_group->requests[0],
              &mpi_group->statuses[0]);

  int shift = 0;
  for (int  i = 0; i < mpi_group->neighbors_send;i++) {
    mpi_group->send_shift[i] = shift;
    nsend_shift[i] = shift;
    shift += mpi_group->send_bites[i];
  }

  //Reallocate send buffers if necessary
  if (shift > ops_buffer_send_1_size) {
    OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size, shift);
    ops_buffer_send_1_size = shift;
  }

  shift = 0;
  for (int i = 0; i < mpi_group->neighbors_recv; i++) {
    mpi_group->recv_shift[i] = shift;
    nrecv_shift[i] = shift;
    shift += mpi_group->recv_bites;
  }

  //Reallocate recv buffers if necessary
  if (shift > ops_buffer_recv_1_size) {
    OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size, shift);
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
    for (int imap = 0; imap < to->map_list[imap]; imap++) {
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

void _ops_particle_halo_forward_map(OPS_instance   *instance,
                                    ops_particle_halo_group halo_grp) {

  ops_mpi_particle_halo_group *mpi_group
    = OPS_mpi_halo_group_list[halo_grp->index];

  ops_particle_halo_group main_halo_grp
      = (halo_grp->halo_type == OPS_HALO_GRP_DEFAULT) ? halo_grp :halo_grp->halo_master;



  if (mpi_group->nhalos == 0) return;

  int nhalos = mpi_group->nhalos;
  double c, t1, t2;
  ops_timers_core(&c, &t1);
  int size = (mpi_group->num_neighbors_send > 0) ?
     mpi_group->neighbors_send : 1;
  int *shift_send = (int *) ops_malloc(size * sizeof(int));

  size =
      (mpi_group->num_neighbors_recv > 0) ? mpi_group->neighbors_recv > 0 : 1;
  int *shift_recv = (int *) ops_malloc(size * sizeof(int));;

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    shift_send[i] = mpi_group->send_shift[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    shift_recv = mpi_group->recv_shift[i];

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

void ops_particle_exchange(ops_particle particle) {

  //REMOVE VIRTUAL

  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  int nlocal = particle->no_particles;
  sub_particle sp = sb->sb_particle_list[particle->index];

  int *mark_deletion = particle->mark_deletion;
  double  *xpos = particle->particle_pos_dat->data;

  for (int idir = 0; idir < dim; idir++) {

    int nsend_recv_bites[4];
    for (int i = 0; i < 4; i++)
      nsend_recv_bites[i] = 0;

    ops_int_particle_halos halo_int = sp->particle_halos[idir];
    for (int ipart = 0; ipart < particle->no_particles; ipart++) {
      if (mark_deletion[ipart] == 1) {
        //Check if particle would be mapped for allocation
        if (xpos[dim * ipart + idir] < halo_int->particle_send_neg[1]) {
          nsend_recv_bites[0]++;
          mark_deletion[ipart] = 2;
          continue;
        }

        if (xpos[dim * ipart + idir] > halo_int->particle_send_pos[0]) {
          nsend_recv_bites[1]++;
          mark_deletion[ipart] = 3;
        }
      }
    }

      //Allocate if necessary forward_pos structures
    if (nsend_recv_bites[0] > halo_int->nalloc_max_neg && sb->id_m != MPI_PROC_NULL) {
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

    if (nsend_recv_bites[1] > halo_int->nalloc_max_pos && sb->id_p != MPI_PROC_NULL) {
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
        sp->bites_in_exchange[isend_neg] = ipart;
        isend_neg++;
      }
      else if (mark_deletion[ipart] == 3) {
        mark_deletion[ipart] = 4;
        sp->bites_in_exchange[isend_pos] = ipart;
        isend_pos++;
      }
    }

    //Packing all data
    ops_dat pos_dat = particle->particle_pos_dat;
    int shift_pos = 0;
    int shift_neg = 0;

    _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                    ops_buffer_send_2 + shift_pos,
                                    pos_dat,halo_int,
                                    nsend_recv_bites[0],
                                    nsend_recv_bites[1]); //TODO


    shift_neg += nsend_recv_bites[0] * pos_dat->elem_size;
    shift_pos += nsend_recv_bites[1] * pos_dat->elem_size;
    if (particle->particle_envelope != nullptr)  {
     _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                     ops_buffer_send_2 + shift_pos,
                                     particle->particle_envelope,
                                     halo_int,
                                     nsend_recv_bites[0],
                                     nsend_recv_bites[1]);

      shift_neg += nsend_recv_bites[0] * particle->particle_envelope->elem_size;
      shift_pos += nsend_recv_bites[1] * particle->particle_envelope->elem_size;
    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1 + shift_neg,
                                      ops_buffer_send_2 + shift_pos,
                                      dat, halo_int,
                                      nsend_recv_bites[0],
                                      nsend_recv_bites[1]); //TODO
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
              MPI_BYTE, sb->id_m[idir], idir, sb->comm, &request[0]);

    MPI_Isend(ops_buffer_send_2, nsend_recv_bites[1] * sp->bites_in_exchange,
              MPI_BYTE, sb->id_p[idir], idir + OPS_MAX_DIM, sb->comm,
              &request[1]);

    MPI_Irecv(ops_buffer_recv_1, nsend_recv_bites[2] * sp->bites_in_exchange,
              MPI_BYTE, sb->id_p[idir], idir, sb->comm, &request[2]);

    MPI_Irecv(ops_buffer_recv_2, nsend_recv_bites[3] * sp->bites_in_exchange,
              MPI_BYTE, sb->id_m[idir], idir + OPS_MAX_DIM, sb->comm,
              &request[2]);

    MPI_Status status[4];
    MPI_Waitall(2, &request[2], &status[2]);

    int nexist = particle->no_particles;
    particle->no_particles += nsend_recv_bites[2] + nsend_recv_bites[3];

    if (particle->no_particles > particle->Nmax) {
      ops_particle_realloc_data(particle, particle->no_particles);
    }

    int shift_recv_neg = 0;
    int shift_recv_pos = 0;


    _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                    ops_buffer_recv_2 + shift_recv_neg,
                                    particle->particle_pos_dat, nexist,
                                    nsend_recv_bites[2],
                                   nsend_recv_bites[3]); //TODO
    shift_recv_neg += nsend_recv_bites[3] * particle->particle_pos_dat->elem_size;
    shift_recv_pos += nsend_recv_bites[2] * particle->particle_pos_dat->elem_size;

    if (particle->particle_envelope != nullptr) {
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      ops_buffer_recv_2 + shift_recv_neg,
                                      particle->particle_envelope, nexist,
                                      nsend_recv_bites[2],
                                      nsend_recv_bites[3]); //TODO
      shift_recv_neg += nsend_recv_bites[3] * particle->particle_envelope->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * particle->particle_envelope->elem_size;
    }

    for (int idat = 0; idat < particle->particle_dat_index; idat++) {
      ops_dat dat = particle->particle_dat[idat];
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1 + shift_recv_pos,
                                      ops_buffer_recv_2 + shift_recv_neg,
                                      dat, nexist, nsend_recv_bites[2],
                                      nsend_recv_bites[3]);

      shift_recv_neg += nsend_recv_bites[3] * dat->elem_size;
      shift_recv_pos += nsend_recv_bites[2] * dat->elem_size;
    }

    MPI_Waitall(2, &request[0], &status[0]);

    //TODO: Set and shift for non-local if not-within
    double *xcrds = (double *);
    for (int ipart = nexist; ipart < particle->no_particles; ipart++) {
      particle->mark_deletion[ipart] =
          (! particle->box_block->isCoordinateInBoundingBox(xcrds + ipart * dim)) ? 1 : 0;
    }
  }

  //Remove actual particles
  ops_particle_remove_marked_flag(particle, 4);

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

      if (sp->particle_halos[idir]->nforward_pos[iswap] * sp->bites_in_exchange > ops_buffer_send_2_size) {
        ops_buffer_send_2 =  OPS_realloc_fast(ops_buffer_send_2, ops_buffer_send_2_size,
                                              sp->particle_halos[idir]->nforward_pos[iswap] * sp->bites_in_exchange);
        ops_buffer_send_2_size = sp->particle_halos[idir]->nforward_pos[iswap] * sp->bites_in_exchange;
      }

      if (sp->particle_halos[idir]->nforward_neg[iswap] * sp->bites_in_exchange > ops_buffer_send_1_size) {
        ops_buffer_send_1 =  OPS_realloc_fast(ops_buffer_send_1, ops_buffer_send_1_size,
                                              sp->particle_halos[idir]->nforward_neg[iswap] * sp->bites_in_exchange);
        ops_buffer_send_1_size = sp->particle_halos[idir]->nforward_neg[iswap] * sp->bites_in_exchange;
      }



      //HERE WE PACK AND SEND ONLY PARTICLE POSITIONS
      _ops_particle_intra_dat_to_buff(ops_buffer_send_1, ops_buffer_send_2,
                                      particle->particle_pos_dat,
                                      halo_int->particle_send_neg + ishift_neg,
                                      halo_int->particle_send_pos + ishift_pos,
                                      halo_int->nforward_neg[iswap],
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


      if (particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap] > ops_buffer_recv_1_size) {
        ops_buffer_recv_1 = OPS_realloc_fast(ops_buffer_recv_1, ops_buffer_recv_1_size,
                         particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap]);
        ops_buffer_recv_1_size = particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap];
      }

      if (particle->particle_pos_dat->elem_size * halo_int->nrecv_neg[iswap] > ops_buffer_recv_2_size) {
        ops_buffer_recv_2 = OPS_realloc_fast(ops_buffer_recv_2, ops_buffer_recv_1_size,
                         particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap]);
        ops_buffer_recv_1_size = particle->particle_pos_dat->elem_size * halo_int->nrecv_pos[iswap];
      }

      //Send and receive data
      int size_send = particle->particle_pos_dat->elem_size * halo_int->nsend_neg[iswap];
      MPI_Request request[4];
      MPI_Isend(ops_buffer_send_1, size_send, MPI_BYTE, (size_send > 0) ? sb->id_m[idir] : MPI_PROC_NULL,
                idir, sb->comm, &request[0]);

      size_send = particle->particle_pos_dat->elem_size * halo_int->nsend_pos[iswap];
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
      _ops_particle_intra_buff_to_dat(ops_buffer_recv_1,
                                      ops_buffer_recv_2,
                                      particle->particle_envelope,
                                      particle->no_particles + particle->no_virtual,
                                      halo_int->nrecv_pos[iswap], halo_int->nrecv_neg[iswap]); //TODO


      //Reset the send and receive
       ifirst_send = ilast_send;
       ilast_send = ifirst_send + halo_int->nrecv_neg[iswap] + halo_int->nrecv_pos[iswap];
    }
  }
}


void _ops_particle_halo_exchange_intersect(ops_particle particle,ops_dat dat,
                                           double range_in[], int idim,int dims) {

  double range_pos[2 * OPS_MAX_DIM];
  double range_neg[2 * OPS_MAX_DIM];

  //Get sub-particle //
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  int dim = particle->block->dims;
  sub_particle sp = sb->sb_particle_list[particle->index];

  ops_int_particle_halos halo = sp->particle_halos[idim];

  int nswaps = halo->nswaps;

  //Compute intersection region
  range_pos[2 * idim] = halo->region_pos[2 * idim];
  range_pos[2 * idim + 1] = halo->region_pos[2 * idim + 1];
  for (int i = 0; i < dims; i++) {
    if (i != idim) {
      range_pos[2 * i] = MIN(halo->region_pos[2 * i], range_in[2 * i]);
      range_pos[2 * i + 1] = MIN(halo->region_pos[2 * i + 1], range_in[2 * i + 1]);
    }
  }

  range_neg[2 * idim] = halo->region_neg[2 * idim];
  range_neg[2 * idim + 1] = halo->region_neg[2 * idim + 1];
  for (int i = 0; i < dims; i++) {
    if ( i != dim) {
      range_neg[2 * i] = MIN(halo->region_neg[2 * i], range_in[2 * i]);
      range_neg[2 * i + 1] = MIN(halo->region_neg[2 * i + 1], range_in[2 * i + 1]);
    }
  }

  int ishift_neg = 0;
  int ishift_pos = 0;
  for (int iswap = 0; iswap < nswaps; iswap++) {

    //Find particle in range and realloc structure to send if necessary
    int nparts_neg = 0;
    int nparts_pos = 0;

    //TODO: Assumed of fixed size equal to maximum & reallocate

    _ops_particle_pack_within_reg((double *)particle->particle_pos_dat->data,
                                  halo->particle_send_neg + ishift_neg,
                                  halo->particle_send_pos + ishift_pos,
                                  ops_buffer_send_1, ops_buffer_send_2,
                                  halo->nsend_neg[iswap], halo->nsend_pos[iswap],
                                  &nparts_neg, &nparts_pos); //TODO

    ishift_neg += halo->nsend_neg[iswap];
    ishift_pos += halo->nsend_pos[iswap];

    int nrecv_neg = 0;
    int nrecv_pos = 0;

    MPI_Status status[4];
    MPI_Request requests[4];
    MPI_Sendrecv(&nparts_neg, 1, MPI_INT, sb->id_m[idim], 100,
                 &nrecv_pos, 1, MPI_INT, sb->id_p[idim], 100,
                 sb->comm, &status[0]);

    int size = nparts_neg * (sizeof(int) + dat->elem_size);
    MPI_Isend(ops_buffer_send_1, size, MPI_BYTE, size > 0 ? sb->id_m[idim] : MPI_PROC_NULL,
              200, sb->comm, &requests[0]);
    size = nparts_neg * (sizeof(int) + dat->elem_size);
    MPI_Isend(ops_buffer_send_2, size, MPI_BYTE, size > 0 ? sb->id_p[idim] : MPI_PROC_NULL,
              200 + OPS_MAX_DIM, sb->comm, &requests[1]);

    size = nrecv_pos * (sizeof(int) + dat->elem_size);
    MPI_Irecv(ops_buffer_recv_1, size, MPI_BYTE, size > 0 ? sb->id_p[idim] : MPI_PROC_NULL,
              200, sb->comm, &requests[3]);

    size = nrecv_neg * (sizeof(int) + dat->elem_size);
    MPI_Irecv(ops_buffer_recv_2, size, MPI_BYTE, size > 0? sb->id_m[idim] : MPI_PROC_NULL,
              200 + OPS_MAX_DIM, sb->comm, &requests[2]);

    MPI_Waitall(2, &requests[2], &status[2]);

    //Unpack data
    _ops_particle_unpack_within_reg(ops_buffer_recv_2, ops_buffer_recv_1,
                                    halo->irecv_neg[iswap], halo->irecv_pos[iswap],
                                    halo->nrecv_neg[iswap], halo->nrecv_pos[iswap],
                                    nrecv_neg, nrecv_pos); //TODO

    MPI_Waitall(2, &requests[0], &status[0]);

  }
}

void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp) {

  double *recv_box_regions = nullptr;
  double *send_box_regions = nullptr;
  int size_recv_max = 0;
  int size_send_max = 0;

  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];

    if (!OPS_sub_block_list[halo->particle_to->block->index]->owned &&
        !OPS_sub_block_list[halo->particle_from->block->index]->owned) continue;

    ops_mpi_particle_halo  *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];


    int dim = halo->particle_to->block->dims;


    int size_recv = 2 * sizeof(double) * dim * mpi_halo->nproc_to_max;

    if (size_recv_max == 0) {
      recv_box_regions = (double *) ops_malloc( size_recv);
      size_recv_max = size_recv;
    }
    else if (size_recv> size_recv_max) {
      recv_box_regions = (double *)ops_realloc(recv_box_regions, size_recv);
      size_recv_max = size_recv;
    }

    int size_send = 2 * sizeof(double) * dim
                      * mpi_halo->nproc_from_max;


    if (size_send_max == 0 && size_send > 0) {
      send_box_regions = (double *) ops_malloc( size_send);
      size_send_max = size_send;
    }
    else if (size_send > size_send_max) {
      send_box_regions = (double *)ops_realloc(send_box_regions, size_send);
      size_send_max = size_send;
    }


    //TODO: Add elements for checking
    ops_get_send_recv_box(recv_box_regions, send_box_regions, mpi_halo, dim);


    _ops_particle_halo_set_exchange_send(mpi_halo,
                                       recv_box_regions, dim);


    ops_particle_halo_set_exchange_recv(mpi_halo,
                                        send_box_regions, dim);

    ops_free(mpi_halo->proclist_complete);
  }

  //Create the halo group
  ops_mpi_particle_halo_group *mpi_group = &OPS_mpi_particle_halo_group_list[halo_grp->index];
  int owned = 0;
  for (int j = 0; j < halo_grp->nhalos; j++) {
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_from > 0  ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[j]->index].nproc_to > 0)
      owned++;
  }

  mpi_group->group = halo_grp;
  mpi_group->nhalos = owned;
  mpi_group->index = halo_grp->index;

  if (owned > 0)
    mpi_group->mpi_halos
      = (ops_mpi_particle_halo **) ops_malloc(owned * sizeof(ops_mpi_particle_halo *));

  for (int i = 0; i < mpi_group->nhalos; i++)
    if (OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_from > 0 ||
        OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index].nproc_to > 0) {
      mpi_group->mpi_halos[i] = &OPS_mpi_particle_halo_list[halo_grp->halo_list[i]->index];
    }


  mpi_group->num_neighbors_send = 0;
  mpi_group->num_neighbors_recv = 0;
  mpi_group->nhalo_info = 0;


  int *neighbor_send = (int *)ops_malloc(ops_comm_global_size * sizeof(int));
  int *neighbor_recv = (int *)ops_malloc(ops_comm_global_size * sizeof(int));

  for (int i = 0; i < ops_comm_global_size; i++) {
    neighbor_send[i] = 0;
    neighbor_recv[i] = 0;
  }

  for (int i = 0; i < mpi_group->nhalos; i++) {
    for (int k = 0; k < mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_send[iproc]++;
    }

    for (int k = mpi_group->mpi_halos[i]->nproc_to;
             k < mpi_group->mpi_halos[i]->nproc_from
               + mpi_group->mpi_halos[i]->nproc_to; k++) {
      int iproc = mpi_group->mpi_halos[i]->proclist[k];
      neighbor_recv[iproc]++;
    }
  }

  for (int i = 0; i < ops_comm_global_size; i++) {
    mpi_group->num_neighbors_send += (neighbor_send[i] > 0) ? 1 : 0;
    mpi_group->num_neighbors_recv += (neighbor_recv[i] > 0) ? 1 : 0;
  }

  int ntot = (mpi_group->num_neighbors_send + mpi_group->num_neighbors_recv)
           * mpi_group->nhalos;

  if (ntot > 0)
    mpi_group->halo_info = (ops_particle_halo_exchange *)ops_malloc(sizeof(ops_particle_halo_exchange) * ntot);

  mpi_group->nhalo_info = ntot;


  for (int igroup = 0; igroup < ntot; igroup++)
    mpi_group->halo_info[igroup] =
        (ops_particle_halo_exchange) ops_malloc(sizeof(OPS_particle_halo_exchange_info_core));

  if (mpi_group->num_neighbors_send > 0) {
    mpi_group->neighbors_send =
       (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_shift
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_sizes
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_bites
     = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  }
  if (mpi_group->num_neighbors_recv > 0) {
    mpi_group->neighbors_recv =
       (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_bites
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
    mpi_group->recv_shift
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
    mpi_group->recv_sizes
     = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  }

  if (mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send > 0) {
    mpi_group->requests
       = (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                    sizeof(MPI_Request));
    mpi_group->statuses
       = (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                   sizeof(MPI_Status));
  }

  int k = 0;
  for (int j = 0; j < ops_comm_global_size; ++j) {
    if (neighbor_send[j] > 0) {
      mpi_group->neighbors_send[k] = j;
      k++;
    }
  }

  k = 0;
  for (int j = 0; j < ops_comm_global_size; j++)
    if (neighbor_recv[j] > 0) {
      mpi_group->neighbors_recv[k] = j;
      k++;
    }

  /* Allocationn of arrays for sending & receiving data */


  ops_free(recv_box_regions);
  ops_free(send_box_regions);
  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}
