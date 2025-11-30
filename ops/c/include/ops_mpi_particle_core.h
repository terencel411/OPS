#ifndef  __OPS_MPI_PARTICLE_CORE_H_
#define  __OPS_MPI_PARTICLE_CORE_H_

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
  * @brief core header file for the ops-particle MPI backend
  * @author Valantis
  * @details Header file for OPS MPI backend for particle data
  */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <mpi.h>
#include <ops_lib_core.h>


/*------------------------------------------------------------------------
 *          SUPPORT FUNTION CHANGE WITH ARCHITECTURE
 *------------------------------------------------------------------------*/

void _ops_particle_number_of_particles_in_range(const int *region, const int *binhead,
                                                const int *bins, const int *size,
                                                int *nsend);

void _ops_particle_number_of_particles_in_range(BoundingBox *box,int dim,double  *xcrds,
                                                int noParticles, int *nsend);

void _ops_particle_number_of_particles_in_range(double *region, int dim,
                                                double *xcrds,
                                                int ifirst, int ilast, int *nwithin);

void _ops_particle_mapped_into_region(const int *region, const int *binhead,
                                      const int *bins, const int *size,
                                      int *sendlist);

void _ops_particle_mapped_into_region(BoundingBox *box, int dim, double  *xcrds,
                                      int noParticles, int *sendlist);


void ops_particle_border_build_maps(int idir,ops_particle particle,
                                    ops_int_particle_halos halo,
                                    sub_block_list sb);

void  _ops_particle_build_map_to_dir(int idir,ops_particle particle,
                                    ops_int_particle_halos halo,
                                    const ops_dat binhead, const ops_dat bin,
                                    const sub_block_list sb);

void _ops_particle_update_map_int_halos(ops_particle_mapping map,
                                        ops_particle particle,
                                       int ifirst, int ilast);

void _ops_particle_remove_from_region(BoundingBox *box, const double *env,
                                      const double *xcrds, int *mark_deletion,
                                      size_t noParticles,
                                      const int dim, int *sendlist);

void _ops_particle_mark_for_removal(BoundingBox *box, double *xcrds, int *mark_del,
                                    int dim, int first, int last);

void _ops_particle_dat_copy_from_buff(char *buff, ops_dat dat, ops_part_orient orient,
                                      ops_particle_halo_exchange halo_info,
                                      int dir_to[], int dir_from[], double translate[],
                                      int *ntot_bites);



void _ops_particle_intra_dat_to_buff(char *buff,  ops_dat dat,
                                     int *sendlist, int nsend);

void   _ops_particle_pack_intra_reg_dat_to_buff(char *buff, double *xcrds, ops_dat dat,
                                                int dim, const double *range,
                                                const int *send_list, const int nelems,
                                                int *nsend);

void _ops_particle_intra_buff_to_dat(char  *buff, ops_dat dat,
                                     size_t nexist, int nrecv);

void _ops_particle_unpack_within_reg(char *buff_recv_neg, char *buff_recv_pos, ops_dat dat,
                                     const int ifirst_neg, const int ifirst_pos,
                                     const int nrecv_neg_tot, const int nrecv_pos_tot,
                                     const int nrecv_neg, const int nrecv_pos);

void _ops_particle_unpack_intra_reg_buff_to_dat(char *buff, double *xcrds, ops_dat dat,
                                                const int dim, const double *range,
                                                const int ifirst, const int ilast,
                                                int *nrecv);



void _ops_particle_remove_flag_reset_map(ops_particle particle, int flag);

void _ops_particle_find_intra_box(ops_particle particle, ops_int_particle_halos halo,
                                  int iswap, int ifirst, int ilast);

void _ops_particle_find_intra_map(ops_particle particle, ops_int_particle_halos halo,
                                  ops_dat binhead, ops_dat bins, int *size);

void _ops_particle_set_intra_border_box(ops_particle particle, ops_int_particle_halos halo,
                                        int iswap, int ifirst, int ilast);

void _ops_particle_set_intra_map(ops_particle particle, ops_int_particle_halos halo,
                                 ops_dat binhead, ops_dat bins, int size[]);




/*------------------------------------------------------------------------
 *   FUNCTION DO NOT CHANGE WITH SYSTEM ARCHITECTURE
 *------------------------------------------------------------------------*/

void   ops_particle_update_intra_halo_maps(ops_particle particle, int ifirst,
                                           int ilast);

void  _ops_particle_border_exchange_build_maps(int idir,ops_particle particle,
                                               ops_int_particle_halos halo,
                                               sub_block_list sb);

void _ops_particle_update_dependent_halo_groups(OPS_instance *instance,
                                                ops_particle_halo_group halo_grp,
                                                ops_mpi_particle_halo_group *mpi_group);

void _ops_particle_packer_inters(ops_dat dat, ops_particle particle, double range_in[],
                                 int iswap, int idir, int send_recv_offsets[]);

void _ops_particle_unpacker_inters(ops_dat dat, ops_particle particle, double range_in[],
                                   int iswap, int idir, int send_recv_offsets[]);



#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /* __OPS_MPI_PARTICLE_CORE_H_ */
