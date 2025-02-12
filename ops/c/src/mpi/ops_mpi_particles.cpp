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
  * @brief OPS mpi core routines for particle treament
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the core mpi decl routines for the OPS particle implementation
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>
#include <string>


/*-----------------------------------------------------------------------------------------------------------*/
/* Finds the box of each process based on an ops_dat structure
 */
void _ops_construct_local_box_from_dat(ops_dat coords, double grid_size, int dim, double* xmin, double *xmax) {
  /* Get minimum lower bound */
  int imin[OPS_MAX_DIM], imax[OPS_MAX_DIM];
  int size[OPS_MAX_DIM];

  sub_block_list sb = OPS_sub_block_list[coords->block->index];
  if (!sb->owned) return;

  for (int i = 0; i < dim; i++) {
    imin[i] = -coords->d_m[i] - OPS_sub_dat_list[coords->index]->d_im[i];
    imax[i] = -coords->d_m[i] - OPS_sub_dat_list[coords->index]->d_im[i]
            - (  coords->d_p[i] + OPS_sub_dat_list[coords->index]->d_ip[i]
               - coords->d_m[i] + OPS_sub_dat_list[coords->index]->d_im[i])
            + OPS_sub_dat_list[coords->index]->decomp_size[i] - 1;//TODO: Check
    size[i] = OPS_sub_dat_list[coords->index]->decomp_size[i];
  }

  double *data = (double *)coords->data; //Assume no CUDA

  /* Get xmin per direction */
  OPS_instance  *instance = coords->block->instance;
  if (dim == 2) {
    /*Get xmin per direction */
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
  }

  /* Uniform assumption for the moment */
  for (int i = 0; i < dim ; i++) {
    xmin[i] -= 0.5 * grid_size;
    xmax[i] += 0.5 *grid_size;
  }
}

/*----------------------------------------------------------------------------------------------------------------------*/
/* Computes the local box from global box                                                                               */
/*----------------------------------------------------------------------------------------------------------------------*/
bool  ops_get_bounding_box_local_to_global(ops_block block,double* xmin,double *xmax,double* xglb_min, double* xglb_max)
{
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned)
    return false;

  MPI_Allreduce(xmin, xglb_min, 3, MPI_DOUBLE, MPI_MIN, sb->comm);
  MPI_Allreduce(xmax, xglb_max, 3, MPI_DOUBLE, MPI_MAX, sb->comm);

  return true;
}


bool ops_bounding_box_global_to_local(const ops_block block, int dim, std::array<ops_point, 2>& globalBoundingBox,
                                      std::array<ops_point, 2>& boundingBox) {

  sb_block_list sb = OPS_sub_block_list[block->index];
  if (! ops_partitioned()) {
    boundingBox[0].x = globalBoundingBox[0].x;
    boundingBox[0].y = globalBoundingBox[0].y;

    boundingBox[1].x = globalBoundingBox[1].x;
    boundingBox[1].y = globalBoundingBox[1].y;

    if (dim == 3) {
      boundingBox[0].z = globalBoundingBox[0].z;
      boundingBox[1].z = globalBoundingBox[1].z;
    }
  }
  else {
    /* Box defined after partition */
    sub_block_list sb = OPS_sub_block_list[block->index];
    if (!sb->owned) {return false;}

    int no_x{sb->pdims[0]};
    int no_y{sb->pdims[1]};
    int no_z{sb->pdims[2]};

    int loc_x{sb->coords[0]};
    int loc_y{sb->coords[1]};
    int loc_z{sb->coords[1]};

    Real size_x{globalBoundingBox[1].x - globalBoundingBox[0].x};
    Real size_y{globalBoundingBox[1].y - globalBoundingBox[0].y};

    Real size_z;
    if (dim == 3)
      size_z = globalBoundingBox[1].z - globalBoundingBox[0].z;

    Real dx{size_x / static_cast<double>(no_x)};
    Real dy{size_y / static_cast<double>(no_y)};
    Real dz{ dim==3 ? size_z / static_cast<double>(no_z) : 0.0};

    /* Get domain data */
    boundingBox[0].x = gloBoundingBox[0].x
                     + dx * static_cast<double>(loc_x);
    boundingBox[1].x = globalBoundingBox[0].x
                     + dx * static_cast<double>(loc_x + 1);

    boundingBox[0].y = globalBoundingBox[0].y
                     + dy * static_cast<double>(loc_y);
    boundingBox[1].y = globalBoundingBox[0].y
                     + dy * static_cast<double>(loc_y + 1);
    if (dim == 3) {
      boundingBox[0].z = globalBoundingBox[0].z
                       + dx * static_cast<double>(loc_z);
      boundingBox[1].z = globalBoundingBox[0].z
                       + dx * static_cast<double>(loc_z + 1);
    }
  }
  return true;
}

/*-------------------------------------------------------------------------*
 *     Particle mapping functions                                          *
 *-------------------------------------------------------------------------*/

int _ops_particle_moved_outside(ops_particle particle) {


  size_t nParticles = particle->no_particles;
  int dim = particle->block->dims;
  BoundingBox *box = particle->box_block;
  double *xlocal = (double *) particle->particle_pos_dat[0]->data;
  for (size_t i = 0; i < nParticles; i++) {
    ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                    (dim == 3) ? xlocal[dim * i + 2] : 0.0};
     bool decide = box->isCoordinateInBoundingBox(point);
     if (!decide) return 1;
  }

  return 0;
}

int _ops_particle_mapping_decide(ops_particle_mapping map,
                                 ops_particle particle, bool enforce)
{
  if (enforce)
    return 1;

  /* 1. Particle number changes */
  int local_flag = 0;

  local_flag = _ops_particle_moved_outside(particle);

  /* 2. Particle movement-list rebuild */
  double *x = (particle->particle_pos_dat[0]->data != NULL) ?
      (double *)particle->particle_pos_dat[0]->data : NULL;
  double *x_old = (map->pos_old->data != NULL) ?
      (double *)map->pos_old->data : NULL;

  if (local_flag == 0) {
    double *rad = (map->Rp->data != NULL) ? (double *) map->Rp->data : NULL;
    double *rad_old =
        (map->particle_changes
            == ops_particle_mapping_core::ops_shape_change::change) ?
        (double *) map->Rp_old->data : nullptr;

    int dim = particle->block->dims;
    double dx[dim];
    for (size_t i = 0; i <map->nParticles; i++) {
      double dx_sq{0.0}, dr{0.0};
      for (int isou = 0; isou < dim; isou++) {
        dx[isou] = x[isou + dim * i] - x_old[isou + dim * i];
        dx_sq += dx[isou] * dx[isou];


        /* Changing size */
        if (map->particle_changes== ops_particle_mapping_core::ops_shape_change::change)
          dr = rad[i] - rad_old[i];
      }

      /* Case I: Particle of fixed envelope */
      if (map->particle_changes == ops_particle_mapping_core::ops_shape_change::constant) {
        if (dx_sq > map->skin * map->skin) {
          local_flag = 1;
          break;
        }
      }
      else {
        double dr_sq = dr * dr;
        if (dr_sq > map->skin * map->skin || dx_sq > map->skin * map->skin) {
          local_flag = 1;
          break;
        }
      }
    }
  }

  //TODO: Broadcast and return
  int flag{0};
  sub_block_list sb = OPS_sub_block_list[particle->block->index];
  MPI_Allreduce(&local_flag, &flag, 1, MPI_INT, MPI_MAX, sb->comm);

  return flag;

}
