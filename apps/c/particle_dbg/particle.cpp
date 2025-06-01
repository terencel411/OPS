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

/** @brief 3D heat diffusion PDE solved using ADI
 *  @author Endre Lazlo, converted to OPS by Gihan Mudalige
 *  @details PDE is solved with the ADI (Alternating Direction Implicit) method
 *  uses the Scalar tridiagonal solver for CPU and GPU written by Endre. Lazslo
**/

// standard headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <random>

// OPS header file
#define OPS_3D
#include "ops_seq_v2.h"
#include "ops_particle_seq.h"
#include "ops_particle_insert_del.h"
#include "ops_grid_part_seq.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#ifdef OPS_MPI
#include <mpi.h>
#endif
//#include "print_kernel.h"

#include "generate_kernel.h"
int main(int argc, char *argv[]) {
  // Set defaults options
  int base[3]={0, 0, 0};

  /**--- Initialisation----**/

  // OPS initialisation
  ops_init(argc, argv, 2);

  /* Insert block */
  ops_block block3D = ops_decl_block(3, "block3D");

  /* Generating box bound */
  double Lx = 10.0;
  ops_point xlow{0.0, 0.0, 0.0};
  ops_point xmax{Lx, Lx, Lx};

  BoundingBox *boundBlock = ops_create_bounding_box(3);

  /* Insert particle structure in the block */
  ops_particle particle = ops_decl_particle(block3D, boundBlock);
  double *temp = NULL;
  ops_dat particle_crd = ops_decl_particle_dat(particle, 3, base, temp, "double", "crd_x");

  ops_dat particle_act_crd = ops_decl_particle_pos_dat(particle, 3, base,
                                                       temp, "double","coords");

  ops_dat particle_act_shape = ops_decl_particle_envelope(particle, base, temp, "double",
                                                          "envelope");

  printf("Size of double %d\n", sizeof(double));

  int size[3] = {100, 100, 100};
  int d_m[3] ={ 0, 0, 0};
  int d_p[3] ={0, 0, 0};

  int d_mf[3] = {-1, -1, -1};
  int d_pf[3] = { 1, 1, 1};

  ops_dat field = ops_decl_dat(block3D, 3, size, base, d_mf, d_pf, temp, "double", "test_field");


  ops_dat coords_data  = ops_decl_dat(block3D, 3, size, base, d_m, d_p, temp, "double", "coords_grid");

  /* declare stencil */
  int loca_dat[3]={0, 0, 0};
  ops_stencil local = ops_decl_stencil(3, 1, loca_dat, "local_Stencil");

  ops_particle_mapping map = ops_decl_mapping(particle, coords_data, particle_act_shape,
                                              OPS_NO_VIRTUAL, OPS_CONST_SHAPE,
                                              OPS_UNIFORM_GRID, 1, 0.001); //Overload mapping function with \pm grid_points
  //Add also a dx

  /* Declare halos */
  int dir_from[3] = {0, 1, 2};
  double translate[3] = {Lx, 0., 0.};

  //TODO: INFORM ONLY FOR TYPE
  ops_particle_halo_data xcrd_halo =  ops_particle_decl_data_halo(particle_act_crd, particle_act_crd,
                                                                  dir_from,
                                                                  dir_from, translate,
                                                                  OPS_PART_ORIENT_ON);

  ops_particle_halo_data env_halo = ops_particle_decl_data_halo(particle_act_shape, particle_act_shape,
                                                                 dir_from, dir_from, translate,
                                                                 OPS_PART_ORIENT_OFF);

  /* Define particle halo */
  ops_particle_halo_data halo_data_grp[2] = {xcrd_halo,  env_halo};
  double crit_length[3] = {0.0, 0.0, 0.0}; //Example-need to dive


  ops_particle_halo halo_part = ops_particle_decl_halo(particle, particle,
                                                       halo_data_grp, 2,
                                                       crit_length, dir_from,
                                                       dir_from, translate);

  //TODO: Merge together with the other one or just keep the translation to the y.
  double dx1 = Lx / static_cast<double>(size[0] - 1);
  double dx_halo[3] = {dx1, 0, 0};
  ops_particle_halo halo_part_exchange = ops_particle_decl_halo(particle, particle,
                                                                halo_data_grp, 2,
                                                                dx_halo, dir_from,
                                                                dir_from, translate);


  translate[0] = -Lx;
  /* Define for the opposite direction halos */
  ops_particle_halo halo_part_opp = ops_particle_decl_halo(particle, particle,
                                                           halo_data_grp, 2,
                                                           crit_length, dir_from,
                                                           dir_from, translate);
  ops_particle_halo halo_part_grp[2] = {halo_part, halo_part_opp};

  ops_particle_halo halo_part_excnh_opp = ops_particle_decl_halo(particle, particle,
                                                                 halo_data_grp, 2,
                                                                 dx_halo, dir_from,
                                                                 dir_from, translate);

  ops_particle_halo halo_part_exch_grp[2] = {halo_part_exchange, halo_part_excnh_opp};


  ops_particle_halo_group halo_group = ops_particle_decl_halo_group(halo_part_grp, 2,
                                                                    OPS_HALO_GRP_EXCHANGE,
                                                                    OPS_PART_LOOP_LOCAL,
                                                                    nullptr);

  ops_particle_halo_group halo_group_border = ops_particle_decl_halo_group(halo_part_exch_grp, 2,
                                                                           OPS_HALO_GRP_BORDER,
                                                                           OPS_PART_LOOP_LOCAL,
                                                                           nullptr);

  ops_partition("particle_partition");

  /* Init domain */
  double dx[1];
  dx[0]= (xmax.x - xlow.x) / static_cast<double>(size[0]-1);

  ops_printf("dx = %f", dx[0]);
  int range[6]{0, size[0], 0, size[1], 0, size[2]};
  ops_par_loop(KerGenerateKernel, "KerGenerateKernel", block3D, 3, range,
               ops_arg_dat(coords_data, 3, local, "double", OPS_WRITE),
               ops_arg_dat(field, 3, local, "double", OPS_WRITE),
               ops_arg_gbl(dx, 1, "double", OPS_READ),
               ops_arg_idx());

  /* Verify partition */
  //TODO: Insert Bounding Box
  ops_set_bounding_box_from_dat(boundBlock, coords_data, 0.0, 3);

  /* Setting up halos */
  ops_particle_set_halo_group(halo_group);
  ops_particle_set_halo_group(halo_group_border);

  double xCrds[3 * 20];
  double rshape[20];
  double trial[3 * 20];
  std::uniform_real_distribution<double> xf(0., 10);
  std::default_random_engine re;

  for (int i = 0; i < 20; i++) {
    xCrds[3 * i] = xf(re);
    xCrds[3 * i + 1] = xf(re);
    xCrds[3 * i + 2] = xf(re);
//    printf("Candidate Particle %d: [%f %f %f]\n", i, xCrds[3 * i],xCrds[3 * i + 1], xCrds[3 * i + 2]);
    rshape[i] = 0.02 * xf(re);
    trial[3 * i] = - xf(re);
    trial[3 * i + 1] = xf(re) * xf(re);
    trial[3 * i + 2] = 0.0;//xf(re) - 1;

  }

  ops_particle_insert(KerInsertData, "KerInsertData", KerDecide, particle, 0, 3, 20, xCrds, rshape,
                      ops_arg_gbl_particle(xCrds, 3, "double", OPS_READ),
                      ops_arg_dat_particle(particle_act_shape, 1, "double", OPS_WRITE),
                      ops_arg_gbl_particle(rshape, 1, "double", OPS_READ),
                      ops_arg_dat_particle(particle_crd, 3, "double", OPS_WRITE),
                      ops_arg_gbl_particle(trial, 3, "double", OPS_READ));

  //TODO: Vrf mapping without removing and inserting

  double *xcoords = (double *) particle_act_crd->data;
  double *rad =(double *)particle_act_shape->data;
  for (int i = 0; i < 20; i++) {
//    printf("Particle %d [%f %f %f] Rp = %f\n", i, xcoords[3 * i], xcoords[3 * i + 1],
//           xcoords[3 * i + 2], rad[i]);
  }
  ops_particle_update_map_lists(particle);


  ops_particle_build_maps(particle);
  //TODO-1: Add also the opposite where particle is mapped in the cell

  double dt{0.001};
  double rangeLoop[6] = {0, 10., 0. ,10., 0., 10.};

  ops_particle_par_loop(KerComputeVel, "KerComputeVel", particle,
                        map, 3, rangeLoop,
                        ops_arg_gbl(&dt, 1, "double", OPS_READ),
                        ops_arg_dat(field, 3, local, "double", OPS_READ),
                        ops_arg_dat_particle(particle_crd, 3, "double", OPS_WRITE));

  int iter_range[] = {0, size[0], 0, size[1], 0, size[2]};

  double *part_dta  =(double *)particle_crd->data;
  for (int i = 0; i < particle->no_particles; i++)
    printf("%d [%f %f %f]\n", i, part_dta[3 * i], part_dta[3 * i + 1], part_dta[3 * i + 2]);

  ops_par_loop(KernelGrid, "KernelGrid", block3D, map, local, 3, iter_range,
               ops_arg_dat(field, 3, local, "double", OPS_WRITE),
               ops_arg_dat_particle(particle_crd, 3, "double", OPS_WRITE));

  //TODO-2: Add the quadrature scheme

  double *us = (double *) particle_crd->data;

  //for (int i = 0; i < 20; i++)
 //   printf("Particle %d: u =[%f %f %f]\n", i, us[3 *i], us[3 * i + 1], us[3 * i + 2]);
  ops_exit();
  return 0;
}
