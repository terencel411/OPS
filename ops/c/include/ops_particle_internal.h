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

enum ops_particle_iterate_type {
  OPS_PARTICLE_ITERATE_LOCAL = 0,
  OPS_PARTICLE_ITERATE_ALL = 1,
  OPS_PARTICLE_ITERATE_RANDOM = 2 //TODO: Replace with different name not convenient
};

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

  for (int i = 0; i < dim; i++)
    data[dim * loc + i] = x_local[i];
}


void _ops_particle_halo_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                   int nhalos, ops_particle_halo_exchange  halo_info,
                                   int *ntot_bites, int flag = 0);

void _ops_particle_halo_dat_to_buf(char *buff, ops_dat dat,
                                   ops_particle_halo_exchange halo_info,
                                   int *ntot_bites);

void _ops_particle_halo_reverse_copy_tobuf(char *buff, ops_particle_halo_data *halo_data,
                                           int nhalos, ops_particle_halo_exchange halo_info,
                                           int *ntot_bites);

void _ops_particle_halo_reverse_copy_from_buff(char *buff, ops_particle_halo_data *halo_data, int nhalos,
                                                ops_particle_halo_exchange info, int dir_from[],
                                                int dir_to[], int *ntot_bities, ops_access access);


void _ops_particle_halo_copy_from_buff(char *buff, ops_particle_halo_data *halo_data,
                                       int nhalos, ops_particle_halo_exchange halo_info,
                                       int dir_to[], int dir_from[], char* translate,
                                       int *ntot_bites, int flag = 0);

void _ops_particle_copy_mapping_data_to(ops_particle_mapping map, int to, int from);

void _ops_particle_swap_mapping_data(ops_particle_mapping map, int from, int to);

void _ops_particle_mapping_virtual_from_halo(ops_particle_mapping map,ops_particle particle,
                                             int ifirst, int n_to_map);

/*--------------------------------------------------------------------------------------*/
/* Allocate temporary array for particle halos
 */

void _ops_particle_setup_tmp_array(OPS_instance *instance);


void _ops_particle_allocate_tmp_array(int size_elem);

void _ops_particle_free_tmp_array();


/*-------------------------------------------------------------------------------------*/
/* Particle halo and halo group auxiliary functions                                    */
/*------------------------------------------------------------------------------------ */

ops_particle_halo_data _ops_particle_decl_halo_data_core(OPS_instance *instance,
                                                         ops_dat from,
                                                         ops_dat to,
                                                         ops_part_orient orient_flag);

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, ops_particle_halo_data halos[],
                                          int nhalos, double *critical_length,
                                          int dir_from, int dir_to,
                                          double *translate);

ops_particle_halo _ops_particle_decl_halo(OPS_instance *instance, ops_particle from,
                                          ops_particle to, int nhalos,
                                          ops_particle_halo_data halos[],
                                          char *sending_region,
                                          int *dir_from, int *dir_to,
                                          char *translate, int type_size);

ops_particle_halo_group _ops_particle_decl_halo_group(OPS_instance *instance,
                                                      ops_particle_halo particle_halos[],
                                                      int nhalos,
                                                      ops_part_halo_grp_type halo_type,
                                                      ops_with_virtual with_virtual,
                                                      ops_particle_halo_group master);

ops_particle_halo _ops_free_particle_halo(ops_particle_halo halo);
ops_particle_halo_group _ops_free_particle_halo_group(ops_particle_halo_group halo_grp);

void  _ops_particle_set_exchange_zone(OPS_instance *instance,
                                      ops_particle_halo halo);

void _ops_particle_halo_border_transfer(OPS_instance  *instance,
                                        ops_particle_halo_group halo_grp);

void _ops_particle_halo_border_transfer_map(OPS_instance *intance,
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


void  _ops_particle_halo_exchange_transfer_map(OPS_instance   *instance,
                                               ops_particle_halo_group halo_grp);

void _ops_particle_halo_forward_map(OPS_instance   *instance,
                                    ops_particle_halo_group halo_grp);

void _ops_particle_halo_reverse_transfer(OPS_instance *instance,
                                         ops_particle_halo_group halo_grp,
                                         ops_access access);

void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp);

/*--------------------------------------------------------------------------------------*
 * Mapping functions
 *--------------------------------------------------------------------------------------*/

void _ops_mapping_def_core(ops_particle particle, ops_dat grid, ops_stencil stencil,
                          ops_with_virtual &include_virtual, int size[],
                          int base[], int d_m[], int d_p[]);

void _ops_mapping_set_structures(ops_particle particle, char *skin, int  d_m[],
                                 int d_p[], int d_mb[], int d_pb[], int size[],
                                 char *dx_map,  ops_with_virtual &include_virtual);

ops_particle_mapping _ops_decl_mapping_core(ops_particle particle, char *skin,
                                           int size[], int d_m[],
                                           int d_p[], int base[],
                                           ops_with_virtual with_virtual,
                                           ops_grid_type grid_type,
                                           int Ng);

ops_particle_mapping  _ops_decl_mapping_core(ops_particle particle, ops_dat grid,
                                             int size[], int d_m[], int d_p[],
                                             int base[], int stride[],
                                             ops_stencil stencil,
                                             ops_with_virtual include_virtual,
                                             ops_grid_type grid_type,
                                             double skin, int Ng);

void _ops_particle_map_validation(ops_particle_mapping map);

void _ops_partition_flat_wall(ops_particle particle);

void  _ops_particle_init_map(ops_particle_mapping map);

void  _ops_particle_build_local_uniform(ops_particle_mapping map, ops_particle particle);

int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle);

int _ops_particle_decide_build_only_local_uniform(ops_particle_mapping map,
                                                  ops_particle particle);

void _ops_particle_update_local_uniform(ops_particle_mapping map, ops_particle particle);

void _ops_particle_setup_map(ops_particle particle, ops_particle_mapping map);

void _ops_particle_map_from_exchange(ops_particle_mapping map, ops_particle particle,
                                     int ifirst, int ilast);

void _ops_particle_setup_map_virtual(ops_particle particle, ops_particle_mapping map);

void  _ops_build_particle_to_grid(const size_t nParticles, const ops_dat bin,
                                  const ops_dat binhead, ops_dat parts_to_grid);

void _ops_particle_remap_virtual(ops_particle_mapping map, ops_particle particle,
                                 int istart, int ilast);

/*------------------------------------------------------------------------------------*
 * Communication functions
 *------------------------------------------------------------------------------------*/

void _ops_particle_exchange(ops_particle particle);

void _ops_particle_build_border(ops_particle particle);

void _ops_particle_build_border_maps(ops_particle particle);

void _ops_particle_forward_intra_maps(ops_particle particle);

void _ops_particle_exchange_map_update(ops_particle particle);


void _ops_particle_border_dats(ops_particle particle, ops_dat *dats,
                               int ndats, ops_neighbor_history *histories = nullptr,
                               int nhistories = 0);

void _ops_particle_border_dats_with_maps(ops_particle particle, ops_dat *dats, int ndats,
                                         ops_neighbor_history *histories = nullptr,
                                         int nhistories = 0);

void _ops_particle_forward_dats_with_maps(ops_particle particle, ops_dat *dats,
                                          int ndats);

void _ops_particle_reverse_dats(ops_particle particle, ops_dat *dats, int ndats,
                                ops_access access);

void _ops_particle_forward_dats(ops_particle particle, ops_dat *dats,
                                int ndats);


/*--------------------------------------------------------------------------------------*/
/* Mapping auxiliary functions                                                          */
/*--------------------------------------------------------------------------------------*/


bool _ops_particle_moved_to_exchange_zone(int bin_old[], int bin_new[], int rmv_limits[],
                                          int dim);

int _ops_check_particle_nunif_grid_inters(const int dim,const double *xGrid,
                                          const double *dxGrid, const double *xp);


int _ops_particle_moved_outside(ops_particle particle);

void ops_particle_print_data_to_txtfile_core(ops_particle particle, const char *file_name);

void ops_particle_print_dats_to_txtfile_core(ops_particle particle, ops_dat *dats, int ndats,
                                            const char *file_name_in);

void _ops_particle_append_dat_point(ops_dat dat, char *&buff, size_t &len,
                                    size_t &cap, const int ip);

void _ops_particle_remove_marked(ops_particle particle);

void _ops_particle_remove_marked_flag(ops_particle particle, int flag);

void _ops_particle_remove_marked_with_maps(ops_particle particle);

void _ops_particle_reset_marked(ops_particle particle);



#endif /* __OPS_PARTICLE_INTERNAL_H */
