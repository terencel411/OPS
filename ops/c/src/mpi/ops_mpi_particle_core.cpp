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
  * @brief OPS MPI functions for particle functionalities
  * @author  C. Tsinginos
  * @details Implements runtime support functions applicable to MPI backend for
  *          particle functionalities
  */

#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <math.h>
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_mpi_particle_core.h>

#include <string>
#include <assert.h>
#include <array>
#include <limits>


void BoundingBox::partitionBoundingBox(ops_block block, ops_dat map_bin, double *dx) {

  if (owned) return;

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: BoundingBox decomposition "
                                          "failed. Box cannot be decomposed before "
                                          "the domain partition");

  if (!OPS_sub_block_list[block->index]->owned) return;

  if (this->volume < DBL_EPSILON) {

    sub_block *sb = OPS_sub_block_list[block->index];


    double xmin[OPS_MAX_DIM];
    double xmax[OPS_MAX_DIM] = {};
    double x_min_glob[OPS_MAX_DIM], x_max_glob[OPS_MAX_DIM];
    double xmax_loc[OPS_MAX_DIM];


    _ops_construct_local_box_from_dat(this->coords, nullptr,
                                      block->dims, xmin, xmax);



    MPI_Status status;
    for (int isou = 0; isou < dim; isou++) {
      xmax_loc[isou] = xmax[isou];
      MPI_Sendrecv(&xmin[isou], 1, MPI_DOUBLE, sb->id_m[isou], 10 + isou,
                   &xmax_loc[isou], 1, MPI_DOUBLE, sb->id_p[isou],
                   10+isou, sb->comm, &status);
    }

    this->setBoundingBoxLocalBound(xmin, xmax_loc);

    //Setting global box: TODO: Change it
    MPI_Allreduce(xmin, x_min_glob, block->dims, MPI_DOUBLE, MPI_MIN,
                  sb->comm);
    MPI_Allreduce(xmax, x_max_glob, block->dims, MPI_DOUBLE, MPI_MAX,
                  sb->comm);

    this->setBoundingBoxGlobalBound(x_min_glob, x_max_glob);
    this->owned = true;

  }
  else {
    if (map_bin == nullptr)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: BoundingBox cannot be defined unless"
                                            "one of the following is not specified: "
                                            "i. Grid structure"
                                            "ii. Mapping structure");

    //Get points
    ops_point xGl_max = this->getGlobalMax();
    ops_point xGl_min = this->getGlobalMin();
    int dim = block->dims;
    double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
    sub_dat *sdat = OPS_sub_dat_list[map_bin->index];
    int size[OPS_MAX_DIM];

    //Uniform grid
    for (int i = 0; i < dim; i++) {
      int ifirst = sdat->decomp_disp[i] - map_bin->base[i] -map_bin->d_m[i]; //TODO Check

      int isize = sdat->decomp_size[i]  + map_bin->base[i] + map_bin->d_m[i] - map_bin->d_p[i];

      int ilast = ifirst + isize;

      xmin[i] = this->getGlobalMin(i) + static_cast<double>(ifirst) * dx[i];
      xmax[i] = this->getGlobalMin(i) + static_cast<double>(ilast) * dx[i];
    }

    this->setBoundingBoxLocalBound(xmin, xmax);

    this->owned = true;

  }

  sub_block *sb = OPS_sub_block_list[block->index];
  if (sb->owned && OPS_instance::getOPSInstance()->OPS_diags>2) {
    printf("Rank %d: Local box [%f %f]x[%f %f]x[%f %f]\n", ops_get_proc(),
           boundingBox[0].x, boundingBox[1].x, boundingBox[0].y, boundingBox[1].y,
           boundingBox[0].z, boundingBox[1].z);
  }

}


static void _compute_mapping_region(BoundingBox *box, int *sending_reg, int dim,
                                    ops_particle particle, ops_particle_mapping map)
{
  int size[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    size[i] = (map != nullptr) ? map->binhead->size[i] : 0;
  }

  ops_point xmin, xmax, xmin_send, xmax_send;
  double dx[OPS_MAX_DIM];

  for (int i = 0; i < box->getDim(); i++) dx[i] = map->dx[i];

  xmin =particle->box_block->getLocalMin();
  xmax = particle->box_block->getLocalMax();

  xmin_send = box->getLocalMin();
  xmax_send = box->getLocalMax();

  ops_dat binhead = map->binhead;
  ops_dat grid = map->grid;

  int d_p[OPS_MAX_DIM], d_m[OPS_MAX_DIM];
  for (int isou = 0; isou < dim; isou++) {
    d_m[isou] = binhead->d_m[isou] + OPS_sub_dat_list[binhead->index]->d_im[isou];
    d_p[isou] = binhead->d_p[isou] + OPS_sub_dat_list[binhead->index]->d_ip[isou];
  }

  if (box->getOwnership() && OPS_instance::getOPSInstance()->OPS_diags > 2) {
    printf("dx = [%f %f]\n",dx[0], dx[1]);
    printf("Block region [%f %f]x[%f %f] sending reg [%f %f]x[%f %f] dx = [%f %f]\n", xmin.x, xmax.x,
         xmin.y, xmax.y, xmin_send.x, xmax_send.x, xmin_send.y, xmax_send.y, dx[0], dx[1]);
  }

  xmin.x += static_cast<double>(d_m[0]) * dx[0];
  xmax.x += static_cast<double>(d_p[0]) * dx[0];

  xmin.y += static_cast<double>(d_m[1]) * dx[1];
  xmax.y += static_cast<double>(d_p[1]) * dx[1];

  if (dim == 3) {
    xmin.z += static_cast<double>(d_m[2]) * dx[2];
    xmax.z += static_cast<double>(d_p[2]) * dx[2];
  }

  double inv_dx = 1./ dx[0];

  sending_reg[0] = (xmin_send.x < xmin.x) ? 0 : (int) ops_floor((xmin_send.x - xmin.x) / dx[0]);
  sending_reg[1] = (xmax_send.x > xmax.x) ? size[0] : (int) ops_ceil((xmax_send.x - xmin.x) / dx[0]);

  sending_reg[2] = (xmin_send.y < xmin.y) ? 0 : (int ) ops_floor((xmin_send.y - xmin.y) / dx[1]);
  sending_reg[3] =  (xmax_send.y > xmax.y) ? size[1] : (int ) ops_ceil((xmax_send.y - xmin.y) / dx[1]);

  sending_reg[4] = 0;
  sending_reg[5] = 1;
  if (dim == 3) {
    sending_reg[4] = (xmin_send.z < xmin.z) ? 0 : (int ) ops_floor((xmin_send.z - xmin.z) / dx[2]);
    sending_reg[5] = (xmax_send.z > xmax.z) ? size[2] : (int ) ops_ceil((xmax_send.z - xmin.z) / dx[2]);

  }

}


static int _check_box_intersection(int dim, double xmin[], double xmax[],
                                   double xmin2[], double xmax2[]) {

  int a1{1};

  int i{0};

  double R1 = 0.0;
  double R2 = 0.0;
  double h = 0.0;

  for (i = 0; i < dim; i++) {
    double hA = 0.5 * (xmin[i] + xmax[i]);
    double hB = 0.5 * (xmin2[i] + xmax2[i]);

    double rA = 0.5 * fabs(xmax[i] - xmin[i]);
    double rB = 0.5 * fabs(xmax2[i] - xmin2[i]);

    if (fabs(hB-hA) > rA + rB) a1 = 0;

    h += (hB - hA) * (hB - hA);
    R1 += rA * rA;
    R2 += rB * rB;
  }

  if (a1 == 1) {
    if (h >= R1 + R2) a1 = 0;
  }

  return a1;

}


void _ops_mapping_def_core(ops_particle particle, ops_dat grid, ops_stencil stencil,
                           ops_with_virtual &include_virtual, int size[],
                           int base[], int d_m[], int d_p[]) {

  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Empty particle structure");

  if (grid == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Empty grid structure. Mapping structures "
                                             "require a user-defined grid (ops_dat) structure");

  ops_block block_particle = particle->block;
  ops_block block_grid = grid->block;
  if (strcmp(block_particle->name, block_grid->name) != 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "ERROR: Inconsistent block assignment: Particle and grid "
                                             "(ops_dat) structure are assigned to different ops_block "
                                             "structures");

  include_virtual = OPS_WITH_VIRTUAL; //Becomes default for modelling particle structures

  int dim = block_particle->dims;
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i] + grid->d_m[i] - grid->d_p[i] -1;

    base[i] = grid->base[i];
    d_m[i] = MIN(grid->d_m[i], -1);
    d_p[i] = MAX(grid->d_p[i], 1);
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

//TODO: Need to pass them to template structures use size to handle them
void _ops_mapping_set_structures(ops_particle particle, double skin[], int  d_m[],
                                 int d_p[], int d_mb[], int d_pb[], int size[],
                                 double dx_map[],  ops_with_virtual &include_virtual) {

  if (particle->box_block->getBlockVolume() < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Block of non-positive volume\n");

//Sanity checks for skin
  for (int i = 0; i < particle->block->dims; i++)
    if (skin[i] < DBL_EPSILON)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Bin cells have non-positive volume ");


  //Finding particle per direction
  for (int i = 0; i < particle->block->dims; i++) {
    double length
    = particle->box_block->getGlobalMax(i) - particle->box_block->getGlobalMin(i);
    int icells = floor(length / skin[i]);
    size[i] = (icells > 0) ? icells : 1;
    dx_map[i] = (length) / static_cast<double>(size[i]);

    d_mb[i] = (d_m != nullptr) ? MIN(d_m[i], -1) : -1;
    d_pb[i] = (d_p != nullptr) ? MAX(d_m[i], 1) : 1;
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) {
    size[i] = 1;
    dx_map[i] = 0.0;
    d_mb[i] = d_pb[i] = 0;
  }

  include_virtual = OPS_WITH_VIRTUAL; //Default for MPI
}

void ops_build_bounding_box(ops_particle particle) {

  //TODO: Predifine communication map if necessary-
  //For the moment is the first entry map
  particle->box_block->partitionBoundingBox(particle->block,
                                            particle->map_list[0]->binhead,
                                            particle->map_list[0]->dx);
}

void   ops_particle_update_intra_halo_maps(ops_particle particle, int ifirst,
                                           int ilast) {

  sub_block *sb =  OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  for (int imap = 0; imap < particle->particle_map_index; imap++) {
    ops_particle_mapping map = particle->map_list[imap];
    _ops_particle_update_map_int_halos(map, particle, ifirst, ilast);
  }
}

//TODO: Check if we need it
bool ops_particle_global_rebuild(bool flag) {
  return flag;
}


/*=====================================================================================*
 *              HALO DEFINITIONS FOR PARALLEL COMMS                                                                       *
 *=====================================================================================*/

void ops_get_send_recv_box(double *recv_box_regions, double *send_box_regions,
                           const ops_mpi_particle_halo *mpi_halo, int dim) {

  ops_particle_halo halo
  = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];
  sub_block *sb_from = OPS_sub_block_list[halo->particle_from->block->index];
  sub_block *sb_to = OPS_sub_block_list[halo->particle_to->block->index];

  double send_local[2 * OPS_MAX_DIM] = { };
  double recv_local[2 * OPS_MAX_DIM] = { };



  MPI_Request request_send[mpi_halo->nproc_from_max + mpi_halo->nproc_to_max];
  MPI_Request request_recv[mpi_halo->nproc_from_max + mpi_halo->nproc_to_max];

  if (sb_from->owned) {
    if (!halo->particle_from->box_block->getOwnership())
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Local Boxes need to be defined prior to "
                         " halo setup (sending block)");

    for (int i = 0; i < dim; i++) {
      send_local[2 * i] = halo->particle_from->box_block->getMinCoordDir(i);
      send_local[2 * i + 1] = halo->particle_to->box_block->getMaxCoordDir(i);
    }


    //Send data
    for (int i = 0; i < mpi_halo->nproc_to_max; i++) {
      int idp = mpi_halo->proclist_complete[i];
      MPI_Isend(send_local, 2 * dim, MPI_DOUBLE, idp, 200,
                OPS_MPI_GLOBAL, &request_send[i]);
    }

  }

  if (sb_to->owned) {
    if (!halo->particle_to->box_block->getOwnership())
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications"
                                            " require the partition of bounding box");

    for (int i = 0; i < dim; i++) {
      recv_local[2 * i] = halo->particle_to->box_block->getMinCoordDir(i);
      recv_local[2 * i + 1] = halo->particle_to->box_block->getMaxCoordDir(i);
    }

    for (int i = 0; i < mpi_halo->nproc_from_max; i++) {
      int idp = mpi_halo->proclist_complete[mpi_halo->nproc_to_max + i];

      MPI_Isend(recv_local, 2 * dim, MPI_DOUBLE, idp, 100, OPS_MPI_GLOBAL,
                &request_send[i +  mpi_halo->nproc_to_max]);
    }
  }

  //Perform receive and unpack
  double buff[2 * OPS_MAX_DIM];
  if (sb_from->owned) {
    for (int i = mpi_halo->nproc_to_max - 1; i >= 0; i--) {
      int idp = mpi_halo->proclist_complete[i];
      MPI_Irecv(buff, 2 * dim, MPI_DOUBLE, idp, 100, OPS_MPI_GLOBAL,
                &request_recv[i]);
      MPI_Status status;
      MPI_Wait(&request_recv[i], &status);
      for (int j = 0; j < dim; j++)  {
        recv_box_regions[2 * dim * i + 2 * j] = buff[2 * j];
        recv_box_regions[2 * dim * i + 2 * j + 1] = buff[2 * j + 1];
      }

    }

  }

  if (sb_to->owned) {
    for (int  i = mpi_halo->nproc_from_max - 1; i >= 0; i--) {
      int idp = mpi_halo->proclist_complete[mpi_halo->nproc_to_max + i];
      MPI_Irecv(buff, 2 * dim, MPI_DOUBLE, idp, 200, OPS_MPI_GLOBAL,
                &request_recv[i + mpi_halo->nproc_to]);

      MPI_Status status;
      MPI_Wait(&request_recv[i + mpi_halo->nproc_to], &status);

      for (int j = 0; j < dim; j++) {
        send_box_regions[2 * dim * i + 2 * j] = buff[2 * j];
        send_box_regions[2 * dim * i + 2 * j + 1] = buff[2 * j + 1];

      }

    }

  }

  MPI_Status status_recv[mpi_halo->nproc_to_max + mpi_halo->nproc_from_max];
  if (sb_from->owned) {
    MPI_Waitall(mpi_halo->nproc_to_max, &request_send[0], &status_recv[0]);
  }
  if (sb_to->owned) {
    MPI_Waitall(mpi_halo->nproc_from_max, &request_send[mpi_halo->nproc_to_max],
                &status_recv[mpi_halo->nproc_to_max]);
  }

}

void _ops_particle_halo_set_exchange_send(ops_mpi_particle_halo *mpi_halo,
                                          double *recv_box_regions,
                                          int dim) {

  ops_particle_halo halo =
      OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block->index];
  if (!sb->owned) {mpi_halo->nproc_to = 0; return; }

  int nsend_max = mpi_halo->nproc_to_max;
  int nsend = 0;


  BoundingBox *sendBox = halo->particle_from->box_block;
  if (!sendBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting up particle communications "
                                          " requires the partition of BoundingBoxes");
  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  sendBox->getLocalMaxMin(xmin, xmax);
  double xrecv_min[OPS_MAX_DIM], xrecv_max[OPS_MAX_DIM];
  double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];

  int nmax = mpi_halo->nproc_from_max + mpi_halo->nproc_to_max;
  int *proclist = (int *) ops_malloc(nmax * sizeof(int));
  BoundingBox **boxes =
      (BoundingBox **)ops_malloc(mpi_halo->nproc_to_max *sizeof(BoundingBox *));


  for (int i = 0; i < nsend_max; i++) {
    //Set box to identify region
    for (int isou = 0; isou < dim; isou++) {

      int irecv_dir = halo->dir_to[isou];
      int isend_dir = halo->dir_from[isou];
      xrecv_min[isend_dir] = recv_box_regions[2 * dim * i + 2 * irecv_dir]
                      - halo->translate[isend_dir];
      xrecv_max[isend_dir] = recv_box_regions[2 * dim * i + 2 * irecv_dir + 1]
                      - halo->translate[isend_dir];
    } //tODO: Check


    int intersect = ops_check_box_intersections2(dim, xmin, xmax,
                                               xrecv_min, xrecv_max);

    if (intersect == 1) {
      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {
          for (int j = 0; j < dim; j++) {
            xsend_min[j] = xmin[j];
            xsend_max[j] = xmax[j];
          }

          xsend_min[is] = (iswap == 0) ? xmin[is] - 0.5 * BIG : xmax[is];
          xsend_max[is] = (iswap == 0) ? xmin[is] : xmax[is] + 0.5 * BIG; //0.5 * fabs(xmax[is]);



          int a1 = ops_check_box_intersections2(dim, xsend_min, xsend_max,
                                              xrecv_min, xrecv_max);

          if (a1 == 1) {

            for (int j = 0; j < dim; j++) {
              if (j != is) {
                xsend_min[j] = (fabs(xsend_min[j] - xrecv_min[j]) < 1.e-9) ?
                  -BIG : MAX(xrecv_min[j], xrecv_min[j]);
                xsend_max[j] = (fabs(xsend_max[j] - xrecv_max[j]) < 1.e-9) ?
                  BIG : MIN(xsend_max[j], xrecv_max[j]);
              }
            }

            proclist[nsend] = mpi_halo->proclist_complete[i];

            boxes[nsend] = new BoundingBox(dim);
            boxes[nsend]->setBoundingBoxLocalBound(xsend_min, xsend_max);
            nsend++;
              //goto endline;
            }
        }
      }
    }

    endline:
    int a1 = 0;


  }

  mpi_halo->nproc_to = nsend;

  //Allocation of arrays

  if (mpi_halo->nproc_to > 0) {

    mpi_halo->proclist = (int *) ops_malloc(mpi_halo->nproc_to * sizeof(int));
    mpi_halo->sendBox = (BoundingBox **) ops_malloc(mpi_halo->nproc_to * sizeof(BoundingBox *));
    mpi_halo->send_region = (int *) ops_malloc(mpi_halo->nproc_to * 6  * sizeof(int));

     for (int i = 0; i < mpi_halo->nproc_to; i++) {
      mpi_halo->proclist[i] = proclist[i];
      mpi_halo->sendBox[i] = boxes[i];
    }
  }




  ops_free(boxes);
  ops_free(proclist);

  if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
    printf("Halo %d: Number of processes to send from rank %d\n",
           mpi_halo->index, mpi_halo->nproc_to, ops_get_proc());
    if (mpi_halo->nproc_to > 0) {
      printf("Rank %d: Processes to send ", ops_get_proc());
      for (int i = 0; i < mpi_halo->nproc_to; i++)
        printf("%d Region [%f %f]x[%f %f] ", mpi_halo->proclist[i],
               mpi_halo->sendBox[i]->getMinCoordDir(0),
               mpi_halo->sendBox[i]->getMaxCoordDir(0),
               mpi_halo->sendBox[i]->getMinCoordDir(1),
               mpi_halo->sendBox[i]->getMaxCoordDir(1));
        printf("\n");
    }
  }

}

void ops_particle_halo_set_border_send(ops_mpi_particle_halo *mpi_halo,
                                       double *recv_box_regions, int dim) {

  ops_particle_halo halo = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];

  sub_block *sb = OPS_sub_block_list[halo->particle_from->block->index];
  if (!sb->owned) {mpi_halo->nproc_to = 0; return; }

  int nsend_max = mpi_halo->nproc_to_max;
  int nsend = 0;

  int nmax = mpi_halo->nproc_to_max;
  int *proclist = (int *) ops_malloc(nmax * sizeof(int));
  BoundingBox **boxes =
      (BoundingBox **)ops_malloc(mpi_halo->nproc_to_max *sizeof(BoundingBox *));
  int *sending_reg = (int *) ops_malloc(mpi_halo->nproc_to_max * 6 * sizeof(int));

  BoundingBox *sendBox = halo->particle_from->box_block;
  if (!sendBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: For setting inter-block halos "
                                          "require the partition of Bounding Boxes\n");

  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  sendBox->getLocalMaxMin(xmin, xmax);
  double xrecv_min[OPS_MAX_DIM], xrecv_max[OPS_MAX_DIM];
  double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];

  for (int i = 0; i < nsend_max; i++) {
    for (int isou = 0; isou < dim; isou++) {
      int isend_dir = halo->dir_from[isou];
      int irecv_dir = halo->dir_to[isou];

      xrecv_min[isend_dir] = recv_box_regions[2 * dim * i + 2 *  irecv_dir]
                           - halo->translate[isend_dir];
      xrecv_max[isend_dir] = recv_box_regions[2 * dim * i + 2* irecv_dir + 1]
                           - halo->translate[isend_dir];
    }

    int intersect = ops_check_box_intersections2(dim, xmin, xmax, xrecv_min, xrecv_max);

    if (intersect) {
      for (int is = 0; is < dim; is++) {
        for (int iswap = 0; iswap < 2; iswap++) {
          for (int j = 0; j < dim; j++) {
            xsend_min[j] = xmin[j];
            xsend_max[j] = xmax[j];
          }

          //Get binhead to receive the proper region
          double dx[OPS_MAX_DIM];


          xsend_min[is] = (iswap == 0) ? xmin[is] - 0.5 * BIG :
                              xmax[is] - halo->dx[is];
          xsend_max[is] = (iswap == 0) ? xmin[is] + halo->dx[is] : xmax[is] + 0.5 * BIG;

          int a1 = ops_check_box_intersection2(dim, xsend_min, xsend_max,
                                              xrecv_min, xrecv_max);

          if (a1 == 1) {
            for (int j = 0; j < dim; j++) {
              if (is != j) {
                xsend_min[j] = (fabs(xsend_min[j] - xrecv_min[j]) < 1.e-9) ?
                                -BIG : MAX(xsend_min[j], xrecv_min[j]);
                xsend_max[j] = (fabs(xsend_max[j] - xrecv_max[j]) < 1.e-9) ?
                                 BIG : MIN(xsend_max[j], xrecv_max[j]);
              }
            }

            proclist[nsend] = mpi_halo->proclist_complete[i];
            boxes[nsend] = new BoundingBox(dim);
            boxes[nsend]->setBoundingBoxLocalBound(xsend_min, xsend_max);


            _compute_mapping_region(boxes[nsend], sending_reg + 6 * nsend, dim,
                                    halo->particle_to, halo->particle_to->map_list[0]);


            if (OPS_instance::getOPSInstance()->OPS_diags > 2) {
               printf("Halo %d Rank %d sending region [%d %d]x[%d %d]x[%d %d] "
                      "with box [%f %f]x[%f %f]\n", mpi_halo->index, ops_get_proc(),
                      sending_reg[6 * nsend], sending_reg[6 * nsend + 1],
                      sending_reg[6 * nsend + 2], sending_reg[6 * nsend + 3],
                      sending_reg[6 * nsend + 4], sending_reg[6 * nsend + 5],
                      xsend_min[0], xsend_max[0], xsend_min[1], xsend_max[1]);
            }
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

  if (mpi_halo->nproc_to > 0) {
    mpi_halo->proclist = (int *) ops_malloc(mpi_halo->nproc_to * sizeof(int));
    mpi_halo->sendBox = (BoundingBox **) ops_malloc(mpi_halo->nproc_to * sizeof(BoundingBox *));
    mpi_halo->send_region = (int *)ops_malloc(mpi_halo->nproc_to * 6 * sizeof(int));

    for (int  i = 0; i < mpi_halo->nproc_to; i++) {
      mpi_halo->proclist[i] = proclist[i];
      mpi_halo->sendBox[i] = boxes[i];
      for (int j = 0; j < 6; j++)
        mpi_halo->send_region[6 * i + j] = sending_reg[6 * i + j];
    }
  }
  ops_free(proclist);
  ops_free(boxes);
  ops_free(sending_reg);
}

void  ops_particle_halo_set_exchange_recv(ops_mpi_particle_halo *mpi_halo,
                                          double *send_box_regions, int dim) {

  ops_particle_halo halo
    = OPS_instance::getOPSInstance()->OPS_particle_halo_list[mpi_halo->index];
  sub_block *sb = OPS_sub_block_list[halo->particle_to->block->index];
  if (!sb->owned) {mpi_halo->nproc_from = 0; return;}

  int nrecv_max = mpi_halo->nproc_from_max;
  int nrecv = 0;

  BoundingBox *recvBox = halo->particle_to->box_block;
  if (!recvBox->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Setting interblock particle halos "
                                          " require the partition of block boxes\n");

  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  recvBox->getLocalMaxMin(xmin, xmax);
  double xsend_min[OPS_MAX_DIM], xsend_max[OPS_MAX_DIM];

  int *proc_recv = (int *)ops_malloc(nrecv_max * sizeof(int));
  for (int i = 0; i < nrecv_max; i++) {
    for (int isou = 0; isou < dim; isou++) {
      int isend_dir = halo->dir_to[isou];
      int irecv_dir = halo->dir_from[isou];
      xsend_min[irecv_dir] = send_box_regions[2 * dim * i + 2 * isend_dir]
                           + halo->translate[irecv_dir];
      xsend_max[irecv_dir] = send_box_regions[2 * dim * i + 2 * isend_dir + 1]
                           + halo->translate[irecv_dir];
    }

    int intersect = ops_check_box_intersections2(dim, xmin, xmax, xsend_min, xsend_max);

    if (intersect == 1) {
      proc_recv[nrecv] =   mpi_halo->proclist_complete[i + mpi_halo->nproc_to_max];
      nrecv++;
    }


  }

  mpi_halo->nproc_from = nrecv;
  int ntot = mpi_halo->nproc_from + mpi_halo->nproc_to;
  if (mpi_halo->nproc_from + mpi_halo->nproc_to > mpi_halo->nproc_to)
    mpi_halo->proclist = (int *) ops_realloc((char *)mpi_halo->proclist,
                                             sizeof(int) * ntot);

  for (int i = 0; i < mpi_halo->nproc_from; i++) {
    mpi_halo->proclist[i + mpi_halo->nproc_to] = proc_recv[i];
  }

  ops_free(proc_recv);
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

    if (size_recv_max == 0 && size_recv > 0) {
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
  for (int ihalo = 0; ihalo < mpi_group->nhalo_info; ihalo++) {
    mpi_group->halo_info[ihalo]->nmax = 10;
    mpi_group->halo_info[ihalo]->sendlist = (int *)ops_malloc(sizeof(int) * 10);
  }

  ops_free(recv_box_regions);
  ops_free(send_box_regions);
  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}

void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp) {

  double *recv_box_regions = NULL;
  double *send_box_regions = NULL;
  int size_recv_max = 0;
  int size_send_max = 0;

  /*Part I: Find processes to which this process will send */
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo  = halo_grp->halo_list[ihalo];

    if (!OPS_sub_block_list[halo->particle_to->block->index]->owned &&
        !OPS_sub_block_list[halo->particle_to->block->index]->owned) continue;


    int dim = halo->particle_to->block->dims;
    ops_mpi_particle_halo  *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];

    int size_recv = 2 * sizeof(double) * dim * mpi_halo->nproc_to_max;

    if (size_recv_max ==  0 && size_recv > 0) {
      recv_box_regions = (double *) ops_malloc(size_recv);
      size_recv_max = size_recv;
    }
    else if (size_recv > size_recv_max) {
      recv_box_regions = (double *) ops_realloc(recv_box_regions, size_recv);
      size_recv_max = size_recv;
    }

//    printf("Number of max procs recv %d and send %d\n", mpi_halo->nproc_from_max, mpi_halo->nproc_to_max);

    int size_send = 2 * sizeof(double) * dim * mpi_halo->nproc_from_max;
    if (size_send_max == 0 && size_send  > 0) {
//      printf("Should enter here\n");
      send_box_regions = (double *) ops_malloc(size_send);
      size_send_max = size_send;
    }
    else if (size_send > size_send_max) {
      send_box_regions = (double *) ops_realloc(send_box_regions, size_send);
      size_send_max = size_send;
    }

    //Find send and receive regions
    ops_get_send_recv_box(recv_box_regions, send_box_regions, mpi_halo, dim);

    ops_particle_halo_set_border_send(mpi_halo, recv_box_regions, dim); //TODO

    ops_particle_halo_set_exchange_recv(mpi_halo, send_box_regions, dim);

    ops_free(mpi_halo->proclist_complete);

  }

  /* Part II: Create mpi halo groups for particle structures */
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

  /*Part III: Start populating particle halo group structures */
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
    mpi_group->halo_info
       = (ops_particle_halo_exchange *)ops_malloc(sizeof(ops_particle_halo_exchange) * ntot);

  mpi_group->nhalo_info = ntot;


  for (int igroup = 0; igroup < ntot; igroup++)
    mpi_group->halo_info[igroup]
       = (ops_particle_halo_exchange) ops_malloc(sizeof(OPS_particle_halo_exchange_info_core));


    //TODO: We do not need the process, we will need to simply copy them
  if (mpi_group->num_neighbors_send > 0) {
    mpi_group->neighbors_send =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_shift =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_send);
    mpi_group->send_sizes =
      (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->send_pos_bites =
        (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
    mpi_group->shift_send_pos =
        (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  }

  if (mpi_group->num_neighbors_recv > 0) {
    mpi_group->neighbors_recv =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_shift =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_sizes =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->recv_pos_bites =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
    mpi_group->shift_recv_pos =
      (int *) ops_malloc(sizeof(int) * mpi_group->num_neighbors_recv);
  }


  if (mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send) {
    mpi_group->requests =
        (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                   sizeof(MPI_Request));
  mpi_group->statuses =
      (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
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
  for (int j = 0; j < ops_comm_global_size; ++j)
    if (neighbor_recv[j] > 0) {
      mpi_group->neighbors_recv[k] = j;
      k++;
    }

  /* Allocation of arrays for sending & receiving data */
  for (int ihalo = 0; ihalo < mpi_group->nhalo_info; ihalo++) {
    mpi_group->halo_info[ihalo]->nmax = 10;
    mpi_group->halo_info[ihalo]->sendlist = (int *)ops_malloc(sizeof(int) * 10); //TODO: Need to add only recv
  }

  ops_free(recv_box_regions);
  ops_free(send_box_regions);
  ops_free(neighbor_send);
  ops_free(neighbor_recv);
}

void _ops_particle_setup_for_rev_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp) {

  ops_particle_halo_group halo_main = halo_grp->halo_master;

  if (halo_main == nullptr)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR: Main halo for forward/backward halo types is"
                       "not defined");
  if (halo_main->halo_type == OPS_HALO_GRP_BORDER)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR: The type of the main halo must be of border type");

  ops_mpi_particle_halo_group *mpi_main_grp =
      &OPS_mpi_particle_halo_group_list[halo_main->index];

  ops_mpi_particle_halo_group *mpi_group =
      &OPS_mpi_particle_halo_group_list[halo_grp->index];


  if (halo_main->nhalos != halo_grp->nhalos)
    throw OPSException(OPS_RUNTIME_CONFIGURATION_ERROR,
                       "ERROR:  Inconsistent halo configuration: Reverse and main particle "
                       "halos have different counts");


  //Allocate halo structures and initialize
  for (int i = 0; i < mpi_group->nhalos; i++) {
    ops_mpi_particle_halo *mpi_halo = mpi_group->mpi_halos[i];
    ops_mpi_particle_halo *mpi_halo_main = mpi_main_grp->mpi_halos[i];

    mpi_halo->nproc_from = mpi_halo_main->nproc_from;
    mpi_halo->nproc_to = mpi_halo_main->nproc_to;

    //Allocate structures
    mpi_halo->particle_halo = instance->OPS_particle_halo_list[mpi_halo->index];

    if (mpi_halo->nproc_from + mpi_halo->nproc_to > 0) {
      mpi_halo->proclist = (int *) ops_malloc((mpi_halo->nproc_to + mpi_halo->nproc_from)
                                               * sizeof(int));
    }

    for (int i = 0; i < mpi_halo->nproc_to + mpi_halo->nproc_from; i++) {
      mpi_halo->proclist[i] = mpi_halo_main->proclist[i];
    }

    ops_free(mpi_halo->proclist_complete);
  }

  //Allocate mpi_group structures
  mpi_group->num_neighbors_send = mpi_main_grp->num_neighbors_send;
  mpi_group->num_neighbors_recv = mpi_main_grp->num_neighbors_recv;

  //Allocate structures

  mpi_group->group = halo_grp;
  mpi_group->nhalos = mpi_main_grp->nhalos;
  mpi_group->nhalo_info = mpi_main_grp->nhalo_info;

  mpi_group->mpi_halos
   = (ops_mpi_particle_halo **) ops_malloc(mpi_group->nhalos
                                           * sizeof(ops_mpi_particle_halo *));

  mpi_group->neighbors_send
   = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->neighbors_recv
   = (int *) ops_malloc(mpi_group->num_neighbors_recv
                        * sizeof(int));

  mpi_group->halo_info =
      (ops_particle_halo_exchange *) ops_malloc(sizeof(ops_particle_halo_exchange)
                                                * mpi_group->nhalo_info);

  for (int i = 0; i < mpi_group->nhalo_info; i++)
    mpi_group->halo_info[i] = mpi_main_grp->halo_info[i]; //TODO: Check

  mpi_group->requests = (MPI_Request *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                   sizeof(MPI_Request));
  mpi_group->statuses = (MPI_Status *) ops_calloc(mpi_group->num_neighbors_recv + mpi_group->num_neighbors_send,
                                                  sizeof(MPI_Status));

  for (int i = 0; i < mpi_group->num_neighbors_send; i++)
    mpi_group->neighbors_send[i] = mpi_main_grp->neighbors_send[i];

  for (int i = 0; i < mpi_group->num_neighbors_recv; i++)
    mpi_group->neighbors_recv[i] = mpi_main_grp->neighbors_recv[i];

  mpi_group->send_shift = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->send_sizes = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->send_bites = (int *) ops_malloc(mpi_group->num_neighbors_send * sizeof(int));
  mpi_group->recv_bites = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  mpi_group->recv_shift = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));
  mpi_group->recv_sizes = (int *) ops_malloc(mpi_group->num_neighbors_recv * sizeof(int));

}

void ops_particle_setup_intrablock_comms(ops_particle particle) {

  sub_block_list  sb = OPS_sub_block_list[particle->block->index];

  if (!sb->owned) return;

  if (!particle->box_block->getOwnership())
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Box block is not partitioned\n");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: At least one ops_particle_mapping structure "
                       "must be defined\n");

  sub_particle sp = sb->sb_particle_list[particle->index];

  int dim = particle->block->dims;

  //Get bounding box
  BoundingBox *box = particle->box_block;
  double epsilon = 1.e-12;

  for (int idim = 0; idim < particle->block->dims; idim++) {

    sp->particle_halos[idim]->region_exch_neg[0] = -BIG;
    sp->particle_halos[idim]->region_exch_neg[1] = box->getMinCoordDir(idim);

    sp->particle_halos[idim]->region_exch_pos[0] = box->getMaxCoordDir(idim);

    sp->particle_halos[idim]->region_exch_pos[1] = BIG;


    //TODO: Need to set up comms in x-direction and y-direction for forward
//    printf("Rank %d: Dir %d: Exchange neg. region [%f %f] and in positive [%f %f]\n", ops_get_proc(), idim, sp->particle_halos[idim]->region_exch_neg[0],
//           sp->particle_halos[idim]->region_exch_neg[1], sp->particle_halos[idim]->region_exch_pos[0],
//           sp->particle_halos[idim]->region_exch_pos[1]);

    for (int isou = 0; isou < particle->block->dims; isou++)
      if (isou != idim)  {
        sp->particle_halos[idim]->region_bord_neg[2 * isou] = -BIG;
        sp->particle_halos[idim]->region_bord_neg[2 * isou + 1] = BIG;
        sp->particle_halos[idim]->region_bord_pos[2 * isou] = -BIG;
        sp->particle_halos[idim]->region_bord_pos[2 * isou + 1] = BIG;
      }

    double xmin = particle->box_block->getMinCoordDir(idim);
    double xmax = particle->box_block->getMaxCoordDir(idim);

    //TODO: Fix the mappings for getting dx
    ops_dat binhead = particle->map_list[0]->binhead;
    int d_m = binhead->d_m[idim] + OPS_sub_dat_list[binhead->index]->d_im[idim];
    int d_p = binhead->d_p[idim] + OPS_sub_dat_list[binhead->index]->d_ip[idim];

    //TODO: Fix the size
    double dx = (xmax - xmin) / static_cast<double>(binhead->size[idim] - d_p + d_m);

    //TODO: If not part of a mapping we need to redefine the dx and get it directly
    sp->particle_halos[idim]->region_bord_neg[2 * idim] = -BIG;
    sp->particle_halos[idim]->region_bord_neg[2 * idim + 1]
       = (sb->id_m[idim] != MPI_PROC_NULL) ? xmin - dx * OPS_sub_dat_list[binhead->index]->d_im[idim] + epsilon : -BIG;
    sp->particle_halos[idim]->region_bord_pos[2 * idim + 1] = BIG;
    sp->particle_halos[idim]->region_bord_pos[2 * idim] = (sb->id_p[idim] != MPI_PROC_NULL) ?
        xmax - dx * OPS_sub_dat_list[binhead->index]->d_ip[idim] - epsilon : BIG;

    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Rank %d Region[%d] =[%f %f]x[%f %f] region_neg = [%f %f]x[%f %f]\n",
             ops_get_proc(), idim, sp->particle_halos[idim]->region_bord_pos[0],
             sp->particle_halos[idim]->region_bord_pos[1],
             sp->particle_halos[idim]->region_bord_pos[2],
             sp->particle_halos[idim]->region_bord_pos[3],
             sp->particle_halos[idim]->region_bord_neg[0],
             sp->particle_halos[idim]->region_bord_neg[1],
             sp->particle_halos[idim]->region_bord_neg[2],
             sp->particle_halos[idim]->region_bord_neg[3]);

    //Set up region in bin cells
    if (dim < 3) {
      sp->particle_halos[idim]->region_neg[4] = 0;
      sp->particle_halos[idim]->region_neg[5] = 1;

      sp->particle_halos[idim]->region_pos[4] = 0;
      sp->particle_halos[idim]->region_pos[5] = 1;
    }

    for (int isou = 0; isou < dim; isou++) {
      if (isou != idim) {
        sp->particle_halos[idim]->region_neg[2 * isou] = 0;
        sp->particle_halos[idim]->region_neg[2 * isou + 1] = binhead->size[isou];
        sp->particle_halos[idim]->region_pos[2 * isou] = 0;
        sp->particle_halos[idim]->region_pos[2 * isou + 1] = binhead->size[isou];
      }
    }

    //Set-up
    sp->particle_halos[idim]->region_neg[2 * idim] = 0;
    sp->particle_halos[idim]->region_neg[2 * idim + 1] = (sb->id_m[idim] != MPI_PROC_NULL) ?
       -2 * OPS_sub_dat_list[binhead->index]->d_im[idim] : 0;
    sp->particle_halos[idim]->region_pos[2 * idim] = (sb->id_p[idim] != MPI_PROC_NULL) ?
       binhead->size[idim] -2 * OPS_sub_dat_list[binhead->index]->d_ip[idim] : binhead->size[idim];
    sp->particle_halos[idim]->region_pos[2 * idim + 1] =  binhead->size[idim];

    if (OPS_instance::getOPSInstance()->OPS_diags > 2)
      printf("Rank %d Forward regions in %d: Neg (%d) [%d %d]x[%d %d]x[%d %d] "
             "and pos (%d): [%d %d]x[%d %d]x[%d %d]\n",
             ops_get_proc(), idim, sb->id_m[idim], sp->particle_halos[idim]->region_neg[0],
             sp->particle_halos[idim]->region_neg[1], sp->particle_halos[idim]->region_neg[2],
             sp->particle_halos[idim]->region_neg[3], sp->particle_halos[idim]->region_neg[4],
             sp->particle_halos[idim]->region_neg[5],
             sb->id_p[idim], sp->particle_halos[idim]->region_pos[0],
             sp->particle_halos[idim]->region_pos[1], sp->particle_halos[idim]->region_pos[2],
             sp->particle_halos[idim]->region_pos[3], sp->particle_halos[idim]->region_pos[4],
             sp->particle_halos[idim]->region_pos[5]);
  }
}

ops_dat ops_decl_particle_dat_char(ops_particle particle, int dim, int *dataset_size,
                                   int *base, int *d_m, int *d_p, int *stride,
                                   char *data, int type_size, char const *type,
                                   char const *name, bool assign, bool exchanging) {

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_decl_particle_dat_char "
                                                         "called after ops_partition");

  ops_dat dat = ops_decl_dat_temp_core(particle->block, dim, dataset_size, base,
                                       d_m, d_p, stride, data, type_size, type, name);

  dat->user_managed = 0;
  dat->is_hdf5 = 0;

  //Setting up the sd one-Need for
  // create list to hold sub-grid decomposition geometries for each mpi process
    OPS_sub_dat_list = (sub_dat_list *)ops_realloc(
        OPS_sub_dat_list, OPS_instance::getOPSInstance()->OPS_dat_index * sizeof(sub_dat_list));

    sub_dat_list sd = (sub_dat_list)ops_calloc(1, sizeof(sub_dat));
    sd->dat = dat;
    sd->dirty_dir_send =
        (int *)ops_malloc(sizeof(int) * 2 * particle->block->dims * MAX_DEPTH);
    for (int i = 0; i < 2 * particle->block->dims * MAX_DEPTH; i++)
      sd->dirty_dir_send[i] = 1;
    sd->dirty_dir_recv =
        (int *)ops_malloc(sizeof(int) * 2 * particle->block->dims * MAX_DEPTH);
    for (int i = 0; i < 2 * particle->block->dims * MAX_DEPTH; i++)
      sd->dirty_dir_recv[i] = 1;
    for (int i = 0; i < OPS_MAX_DIM; i++) {
      sd->d_ip[i] = 0;
      sd->d_im[i] = 0;
    }
    OPS_sub_dat_list[dat->index] = sd;

    //Set now the required variables
    dat->is_particle = true;
    dat->is_exchangable = exchanging;

    if (assign) {
      particle->particle_dat_index++;
      ops_particle_realloc_list(particle);
      particle->particle_dat[particle->particle_dat_index - 1] = dat;
    }

    return dat;
}




void ops_particle_print_data_to_txtfile(ops_particle particle, const char *file_name) {

  if (!OPS_sub_block_list[particle->block->index]->owned) return;

  ops_get_data(particle->particle_pos_dat);
  if (particle->particle_envelope != nullptr)
    ops_get_data(particle->particle_envelope);

  for (int idat = 0; idat < particle->particle_dat_index; idat++) {
    ops_get_data(particle->particle_dat[idat]);
  }

  ops_particle_print_mpi_data_to_txt_file_core(particle, file_name);
}

void ops_particle_print_dats_to_txtfile(ops_particle particle, ops_dat *dats, int ndats,
                                        const char *file_name) {

  if (!OPS_sub_block_list[particle->block->index]->owned) return;

  for (int idat = 0; idat < ndats; idat++) {
    if (!dats[idat]->is_particle)
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ops_dat is not linked to particle"
                                            " data\n");
    ops_get_data(dats[idat]);
  }

  ops_particle_print_mpi_dats_to_txt_file_core(particle, dats, ndats, file_name);

}
