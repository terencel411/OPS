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

/** @Test application for particle-wall interactions
  * @author Valantis Tsinginos
  */


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>

#define OPS_3D

#include <ops_seq_v2.h>
#include <ops_particle_grid_seq.h>
#include <ops_grid_part_seq_v2.h>

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif


#include "grid_kernels.h"
#include "grid_part_loop.h"

typedef double Real;
//typedef float Real;

double et_act = 0.0;
double et_rem = 0.0;
double et_for = 0.0;
double et_bord = 0.0;
double et_tot = 0.0;

Real OMEGA = 1.0;
Real rho0 =1.0;
Real deltaUX = 1.e-6;

//TODO: Standarized it

Real _intersection_volume(Real *region, Real *my_region, Real *region_ins,
                          int dim) {

  Real volume = 1.0;
  for (int i = 0; i < dim; i++) {
    region_ins[2 * i] = MAX(region[2 * i], my_region[2 * i]);
    region_ins[2 * i + 1] = MIN(region[2 * i + 1], my_region[2 * i + 1]);
    int intersect = (region_ins[2 * i + 1] - region_ins[2 * i]) > 0;
    volume *= (intersect) ? region_ins[2 * i + 1] - region_ins[2 * i] : 0.0;
  }

  return volume;
}

int _insert_particles_in_range(int ntotal, ops_particle particle, Real *region,
                               int dim) {



  Real volume = ((BoundingBox<Real> *)particle->box_block)->getBlockVolume();

  Real reg_volume = 1;
  for (int i = 0; i < dim; i++)
    reg_volume *= (region[2 * i + 1] - region[2 * i]);



  //Get region to insert
  Real reg_to_insert[2 * OPS_MAX_DIM];

  if (((BoundingBox<Real> *) particle->box_block)->getBlockVolume() < 1.e-12)
    return 0;


  Real region_block[2 * OPS_MAX_DIM];
  Real region_ins[2 * OPS_MAX_DIM];

  for (int i = 0; i < dim; i++) {
    region_block[2 * i ] = ((BoundingBox<Real> *)particle->box_block)->getMinCoordDir(i);
    region_block[2 * i + 1] = ((BoundingBox<Real> *)particle->box_block)->getMaxCoordDir(i);
  }

  Real volume_ins = _intersection_volume(region, region_block, region_ins, dim);

  int Np = (int) ceil(ntotal * volume_ins / reg_volume);


  printf("R %d: Inserting  %d particles in loc volume = %12.9e (volume = %12.9e)\n", ops_get_proc(), Np,
         volume_ins, reg_volume);

  if (Np == 0) return 0;

  printf("R %d: Inserting region [%f %f]x[%f %f]x[%f %f]\n", ops_get_proc(), region_ins[0], region_ins[1],
         region_ins[2], region_ins[3], region_ins[4], region_ins[5]);


  //Check for particle allocation
  if (Np + particle->no_particles > particle->Nmax)
    ops_particle_realloc_data(particle, Np + particle->no_particles);

  Real *xpos = (Real *) particle->particle_pos_dat->data;
  int nexist = particle->no_particles;

  int nins = 0;
  int ntrial = 0;
  while (nins < Np) {

    Real xmin = region_ins[0] + (region_ins[1] - region_ins[0]) * ((Real) rand() / RAND_MAX);
    Real ymin = region_ins[2] + (region_ins[3] - region_ins[2]) * ((Real) rand() / RAND_MAX);
    Real zmin = region_ins[4] + (region_ins[5] - region_ins[4]) * ((Real) rand() / RAND_MAX);

    ntrial++;
    if (ops_get_proc() == 1) {
      printf("ntrial = %d  xmin\n", ntrial);
    }
    if ((xmin >= region_ins[0] && xmin < region_ins[1]) && (ymin >= region_ins[2] && ymin < region_ins[3])
        && (zmin >= region_ins[4] && zmin < region_ins[5])){
      xpos[3 * (nins + nexist)] = xmin;
      xpos[3 * (nins + nexist) + 1] = ymin;
      xpos[3 * (nins + nexist) + 2] = zmin;
      nins++;
    }
  }

  //Random particle insertion
  particle->no_particles += nins;


  return Np;

}

void setup_maps(ops_particle particle, ops_dat *border, int nsize,
                ops_neighbor_history *histories, int nhistories) {




  ops_particle_setup_maps_with_dats(particle, border, nsize,
                                    histories, nhistories);

  printf("R %d: PAssed maps\n", ops_get_proc());
  ops_particle_halo_transfer_group(OPS_HALO_GRP_BORDER, true);


}


int build_maps(ops_particle particle, ops_dat *dat_border, ops_dat *dat_forward,
                int nborder, int nforward, ops_neighbor_history *histories,
                int nhistories, int iter) {
  double et1, et0;
  double ct1, ct0;

  ops_timers(&ct0, &et0);
  double et_in = et0;

  int decide =  ops_particle_update_map_lists_actual_hybrid(particle);//ops_particle_update_map_lists(particle);

  //Part I: Add halos for exchange
  ops_particle_halo_transfer_group_map(OPS_HALO_GRP_EXCHANGE, decide);


  //History transfer


  /* Part II: Remove particles marked for deletion */
  ops_timers(&ct0, &et0);
  ops_particle_remove_delete_maps(particle, decide);
  ops_timers(&ct1, &et1);
  et_rem += et1 - et0;


  if (decide) {

    //Halo exchanges-before border map update


    ops_timers(&ct0, &et0);
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder,
                                              histories, nhistories);

    ops_particle_halo_transfer_group_map(OPS_HALO_GRP_BORDER, decide);

    et_bord += et1 - et0;
  } else {
    ops_timers(&ct0, &et0);
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);

    ops_particle_halo_transfer_group_map(OPS_HALO_GRP_FORWARD, decide);


    ops_timers(&ct1, &et1);
    et_for += et1 - et0;
  }

  ops_particle_reset_flags(particle, decide);

  ops_timers(&ct1, &et1);
  et_tot += et1 - et_in;

  return decide;
}

void output_particle_dat(ops_particle particle, ops_dat *grp, int ngroup,
                         int timestep) {

  std::string filename = "particle_timestep_" + std::to_string(timestep) + ".txt";
  ops_particle_print_dats_to_txtfile(particle, grp, ngroup, filename.c_str());

}

int main(int argc, char **argv) {
  ops_init(argc, argv, 1);

  ops_init(argc, argv, 1);

  const int Nx = 101;
  const int Ny = 101;
  const int Nz = 101;

  const Real wi[] ={4. / 9., 1./ 9., 1./ 9., 1./ 9. , 1. /9.,
      1./36., 1./36., 1./ 36., 1./36.};
  const Real cs = sqrt(3);

  const int cx[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  const int cy[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

  Real tau = 0.0001;

  Real energy;

  Real ct0,et0,ct1,et1; //timer variables

  int *temp_int = NULL;
  Real *temp_dbl = NULL;

  //====================================
  // Declare & define key data structures
  //====================================




  ops_block lb_block = ops_decl_block(3, "lattice-boltzmann_grid");

  int size[] {101, 101, 101};
  int base[] = {0, 0, 0};
  int d_m[] = {-1, -1, -1};
  int d_p[] = { 1, 1, 1};
  Real *data_ptr = nullptr;

  ops_dat grid = ops_decl_dat(lb_block, 3, size, base, d_m, d_p,
                              data_ptr, "double", "grid_points");

  Real dx1[] = {0., 0., 0.};
  BoundingBox<Real> *box_block = ops_create_bounding_box(lb_block, grid, 3,
                                                        dx1, OPS_BOX_CONFORMING,
                                                        OPS_BOX_TO_BLOCK);



  //===================================================================
  // Declaire particle data structures
  //==================================================================

  Real dx_box[] = {0., 0., 0.};
  Real region[] = {-0.212, -.2, -.2, 0.598, .6, .6}; //[x_min, y_min z_min]x[x_max, y_max _z_max];

  BoundingBox<Real> *box = ops_create_bounding_box(lb_block,
                                                   3, region, OPS_BOX_NONCONFORMING, 0);

  ops_particle particle = ops_decl_particle(lb_block, "Markers", box);

  //Assign variables for testing
  ops_dat particle_pos = ops_decl_particle_pos_dat(particle, 3, base, temp_dbl,
                                                   "double", "part_coords");

  ops_dat particle_vel = ops_decl_particle_dat(particle, 3, base,
                                               temp_dbl, "double", "part_vels");


  ops_dat particle_ids = ops_decl_particle_id_dat(particle, base, (int *) nullptr, "int", "particle_ids");
  //Generate map for the grid




  Real skin[] = {0.01, 0.01, 0.01};
  int stride[] = {1, 1, 1};

  int stencil_points[] = {-1, 0, 0, 0, 0, 0, 1, 0, 0};
  ops_stencil stencil = ops_decl_stencil(3, 3, stencil_points, "testing_stencil");

  // 3x3x3 neighbourhood
  int stencil_inter[] = {-1,-1,-1,  0,-1,-1,  1,-1,-1,  -1, 0,-1,  0, 0,-1,  1, 0,-1,  -1, 1,-1,  0, 1,-1,  1, 1,-1,
                         -1,-1, 0,  0,-1, 0,  1,-1, 0,  -1, 0, 0,  0, 0, 0,  1, 0, 0,  -1, 1, 0,  0, 1, 0,  1, 1, 0,
                         -1,-1, 1,  0,-1, 1,  1,-1, 1,  -1, 0, 1,  0, 0, 1,  1, 0, 1,  -1, 1, 1,  0, 1, 1,  1, 1, 1};
  ops_stencil stencil_int = ops_decl_stencil(3, 27, stencil_inter, "iter_stencil");

  int stencil_loc[] = {0, 0, 0};
  ops_stencil sten_loc = ops_decl_stencil(3, 1, stencil_loc, "local_stencil");

  Real dx_m[] = {0.01, 0.01, 0.01}; //Defines the grid size

  ops_particle_mapping map = ops_decl_mapping(particle, grid, stencil,
                                              OPS_WITH_VIRTUAL, d_m, d_p,
                                              OPS_NON_UNI_STAG, dx_m);

  //Data structures for border and forward communications-In your case the same
  ops_dat dat_borders[] = {particle_pos, particle_vel};
  ops_dat dat_forward[] = {particle_pos, particle_vel};

//  printf("Dat %s: d_m = [%d %d] d_p = [%d %d]\n", map->grid_head->name, map->grid_head->d_m[0],
//         map->grid_head->d_m[1], map->grid_head->d_p[0], map->grid_head->d_p[1]);

//  ops_dat parts_dat_outputs[] = {particle_pos, particle_vel};


  ops_partition("");

  int range[] = {0, size[0], 0, size[1], 0, size[2]};
  double dx[] = {1. / (size[0] - 1), 1./(size[1] - 1), 1./(size[2] - 1)};

  /*ops_par_loop(KernelInitGrid, "KernelInitGrid", lb_block, 3, range,
               ops_arg_dat(grid, 3, sten_loc, "double", OPS_WRITE),
               ops_arg_gbl(dx, 3, "double", OPS_READ),
               ops_arg_idx());
  */

  //Update particle data only independent of fields
  // ops_particle_par_loop(KerInitPartField, "KernelInitTorque", particle,
  //                       (ops_particle_group) nullptr, 2, OPS_PARTICLE_ITERATE_LOCAL,
  //                       (Real *) nullptr, map,
  //                       ops_arg_dat_particle(om_p, 1, "double", particle, map, OPS_WRITE),
  //                       ops_arg_gbl(&nfield, 1, "int", OPS_READ));

  //Check for p


  ops_par_loop(KernelInitNonUnf,"KernelInitNonUnf", lb_block, 3, range,
               ops_arg_dat(grid, 3, sten_loc, "double", OPS_WRITE),
               ops_arg_gbl(dx, 3, "double", OPS_READ),
               ops_arg_idx());

  /*
  int depths[4 * OPS_MAX_DIM];
  for (int i = 0; i < 2; i++) {
    for (int  j = 0; j < 4; j++)
    depths[4 * i + j] = 1;
  }

  for (int i = 2; i < OPS_MAX_DIM; i++)
    for (int j = 0; j < 4; j++)
      depths[4 * i + j] = 0;
  ops_halo_exchanges_datlist(&grid, 1, depths);

*/
    // debug prints
  // int sizey = grid->size[1] + (grid->d_m[1] + OPS_sub_dat_list[grid->index]->d_im[1])
  //           - (grid->d_p[1] + OPS_sub_dat_list[grid->index]->d_ip[1]);

  // int sizex = grid->size[0] + (grid->d_m[0] + OPS_sub_dat_list[grid->index]->d_im[0])
  //           - (grid->d_p[0] + OPS_sub_dat_list[grid->index]->d_ip[0]);



  // printf("R %d size = [%d %d]\n", ops_get_proc(), sizex, sizey);
/*
  for (int j = 0; j < sizey; j++)  {
    int d_my = OPS_sub_dat_list[grid->index]->d_im[1] + grid->d_m[1];
    for (int i = 0; i < sizex; i++) {
      int d_mx = OPS_sub_dat_list[grid->index]->d_im[0] + grid->d_m[0];
      int address = 2 * (i - d_mx) + 2 * (j - d_my) * grid->size[0];
      printf("R %d [%d %d] xf = [%12.9e %12.9e]\n", ops_get_proc(), ((double *) grid->data)[address],
             i, j, ((double *) grid->data)[address + 1]);
    }

  }
*/
  //=======================================
  //  Setup particle simulation
  //=======================================





  printf("Map grid name: %s and type %d \n", map->grid->name, map->map_type);


  printf("Halo exchanged\n");
  ops_particle_setup_partition();

  printf("Partition is set for map type = %d\n", map->map_type);
  //Insert particle in

  Real insertion_reg[] = {0, 0.56, 0, 0.56, 0, 0.56};
  int n_ins =  _insert_particles_in_range(6, particle, insertion_reg, 3);

  int prod = map->grid->size[0] * map->grid->size[1] * map->grid->size[2];

  /*
  for (int i = 0; i < prod; i++)
    printf(" Grid_to_map[%d] (%d %d) = %d dx = [%12.9e %12.9e] xmin = [%12.9e %12.9e]\n", i, map->grid_head->size[0], map->grid_head->size[1],
           ((int *) map->grid_to_map->data)[i], ((double *) map->dx)[0],
           ((double *) map->dx)[1], ((double *) map->xmin)[0], ((double *) map->xmin)[1]);
*/
  printf("Gridhead = %s binhead = %s\n", map->grid_head->name, map->binhead->name);
  printf("Kernel initiated\n");
  for (int i = 0; i < particle->no_particles; i++)
    printf("R %d  X[%d] = [%12.9e %12.9e %12.9e] in rage [%f %f]x[%f %f]x[%f %f]\n", ops_get_proc(),
           i, ((Real *) particle->particle_pos_dat->data)[3 * i] ,
           ((Real *) particle->particle_pos_dat->data)[3 * i +1],
           ((Real *) particle->particle_pos_dat->data)[3 * i +2],
           box_block->getLocalMin().x, box_block->getLocalMax().x, box_block->getLocalMin().y,
           box_block->getLocalMax().y, box_block->getLocalMin().z, box_block->getLocalMax().z);


  setup_maps(particle, dat_borders, 2,
             nullptr, 0);
  //add particles
  printf("Maps are build\n");


  Real gravity[3] = {0, -9.81, 0};
  Real dt = 0.001;


  for (int i = 0; i < particle->no_particles;i++) {
    int address = ((int *) map->parts_to_grid->data)[i];

    printf("%d x = [%12.9e %12.9e %12.9e] in  %d and grid_point = %d\n", i, ((double *) particle->particle_pos_dat->data)[3 * i],
           ((double *) particle->particle_pos_dat->data)[3 * i + 1],
           ((double *) particle->particle_pos_dat->data)[3 * i + 2], ((int *) map->parts_to_grid->data)[i],
           ((int *) map->grid_head->data)[address]);
  }



  ops_par_grid_part_loop(KerGridPartInter,"KerGridPartInter", particle, nullptr,
               map, stencil_int, 3, range,
               ops_arg_dat(grid, 3, sten_loc, "double", OPS_READ),
               ops_arg_dat_particle(particle_pos, 3, "double", particle, map, OPS_READ));


  printf("Grids are generated and interactions found\n");
  ops_par_particle_grid_loop(KerPartGridInter,"KerPartGridInter", particle,
                             nullptr, map, 3, OPS_PARTICLE_ITERATE_ALL, (Real*) nullptr,
                             stencil_int,
                             ops_arg_dat_particle(particle_pos, 3, "double", particle, map,
                                                  OPS_READ),
                             ops_arg_dat(grid, 3, sten_loc, "double", OPS_READ));


  // let every rank finish printing before shutdown; ops_exit() segfaults in ops_mpi_exit
  fflush(nullptr);
#ifdef OPS_MPI
  MPI_Barrier(OPS_MPI_GLOBAL);
  MPI_Finalize();
#endif
  return 0;

  //ops_exit();


}
