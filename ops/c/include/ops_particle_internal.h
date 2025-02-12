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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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
 * @brief OPS internal types and function declarations for particle data
 * @author Valantis Tsinginos
 * @details This header file contains types and function declarations needed by
 *          ops_particle_lib_core.h
 */


#ifndef __OPS_PARTICLE_INTERNAL_H
#define __OPS_PARTICLE_INTERNAL_H

#define OPS_HALO_GRP_EXCHANGE 0
#define OPS_HALO_GRP_BORDER   1
#define OPS_HALO_GRP_FORWARD  2
#define OPS_HALO_GRP BACKWARD 3
#define OPS_HALO_GRP_DEFAULT  4


#define OPS_PART_ORIENT_OFF 0
#define OPS_PART_ORIENT_ON  1
#define OPS_PART_POSITION   2

#define OPS_PART_LOOP_ALL   0
#define OPS_PART_LOOP_LOCAL 1

/*-------------------------------------------------------------------------------------*/
/* Auxiliary BoundingBox Functions                                                     */
/*                                                                                     */

void _ops_construct_local_box_from_dat(ops_dat coords, double grid_Size, int dim,
                                      double* xmin, double* xmax);

/*-------------------------------------------------------------------------------------*/
/* Particle auxiliary functions                                                        */
/*-------------------------------------------------------------------------------------*/

template<typename T>
void  _ops_particle_add_elem(int dim, T *x_local, ops_dat pos, int loc) {
  int size_dat = 1;
  for (int i = 0; i < pos->block->dims; i++)
    size_dat *= pos->size[i];

  if (size_dat < loc)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_dat size smaller than location");

  T* data = (T *)pos->data;

  for (int i = 0; i < data; i++)
    data[dim * loc + i] = x_local[i];
}

void _ops_particle_pack_halo_data(char *buf, ops_particle_halo_data* halo_data, int ndata,
                                  int ipart);

void _ops_particle_unpack_halo_data(char *buff, ops_particle_halo_data* halo_data,
                                    int ndata, int iloc);

/*-------------------------------------------------------------------------------------*/
/* Particle halo and halo group auxiliary functions                                    */
/*------------------------------------------------------------------------------------ */

ops_particle_halo_data _ops_particle_decl_halo_data_core(OPS_instance *instance,
                                                         ops_dat from,
                                                         ops_dat to, int *dir_from,
                                                         int *dir_to, double *translate,
                                                         ops_part_orient orient_flag);

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, ops_particle_halo_data halos[],
                                          int nhalos, double *critical_length);

ops_particle_halo_group _ops_particle_decl_halo_group(OPS_instance *instance,
                                                      ops_particle_halo particle_halos[],
                                                      int nhalos,
                                                      ops_part_halo_grp_type halo_type,
                                                      ops_part_loop_type loop_type,
                                                      ops_particle_halo_group master);

void  _ops_particle_set_exchange_zone(OPS_instance *instance,
                                      ops_particle_halo halo);

void _ops_particle_halo_border_transfer(OPS_instance  *instance,
                                        ops_particle_halo_group halo_grp);

void _ops_particle_setup_exchange_comm(OPS_instance *instance,
                                       ops_particle_halo_group halo_grp);

void _ops_particle_setup_border_comm(OPS_instance *instance,
                                     ops_particle_halo_group halo_grp);

void _ops_particle_setup_default_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp);

void _ops_particle_setup_for_rev_comm(OPS_instance *instance,
                                      ops_particle_halo_group halo_grp);

void _ops_particle_set_exchange_border_zone(OPS_instance *instance,
                                                 ops_particle_halo halo);



void _ops_particle_halo_forward_transfer(OPS_instance  *instance,
                                         ops_particle_halo_group halo_grp);

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp);

void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp);

/*--------------------------------------------------------------------------------------*/
/* Mapping auxiliary functions                                                          */
/*--------------------------------------------------------------------------------------*/

void _ops_get_max_min(double &minv, double &maxv, const double* dat, const size_t size);

void  _ops_compute_bin_size(const ops_point xmin, const ops_point xmax, const int dim,
                            const double dx, int &Nx, double &dx_x, int &Ny,
                            double &dx_y, int &Nz, double &dx_z);

int _ops_coord_to_bin(const int dim, const ops_point xmin,const  ops_point xmax,
                      const double *dx, const int *Ngrid, const double *xp);

int _ops_coord_to_bin_dir(const int dir, const int dim, const ops_point xmin,
                          const double *dx, const int *Ngrid, const double *xp);

void _ops_uniform_build_map(const int init, const int dim, const size_t Np,
                            const int *Ngrid, const double* xp, const ops_point xmin,
                            const ops_point xmax, const double *dx, int *binhead,
                            int *bin);

void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin);

void  _ops_compute_uniform_dx(ops_dat grid, const int dims,double *dx);

void _ops_get_grid_size_per_node(const int dims, const ops_dat grid,
                                 const ops_point xmin, const ops_point xmax,
                                 double *grid_dx, double *grid_shape);


int _ops_check_particle_nunif_grid_inters(const int dim,const double *xGrid,
                                          const double *dxGrid, const double *xp);


void _ops_particle_to_non_uniform_grid_intersection(const int init, const int dim,
                                                    const ops_point xmin,
                                                    const int *binhead_grids, const int *Npoints,
                                                    const int *bin_grid, const double* xGrid,
                                                    const double *grid_dx, const int Ngrid,
                                                    const int *binhead_particles, const int *Ng_parts,
                                                    const int *bin_parts, const double *xp, const int Np,
                                                    const double *dx_grid, const double *dx_p_grid,
                                                    int *binhead, const int *size, int *bin);

int _ops_particle_moved_outside(ops_particle particle);
#endif /* __OPS_PARTICLE_INTERNAL_H */
