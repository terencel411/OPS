/*
 * testing_update_parts.cpp
 *
 *  Created on: Jun 24, 2026
 *      Author: valantis
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>

#define OPS_2D

#include <ops_seq_v2.h>

#include <ops_particle_seq.h>
#include <ops_particle_grid_seq.h>
#include <ops_grid_part_seq_v2.h>

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "lattice_kernels.h"
#include "particle_kernels.h"
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

void insert_particles(int ix1, int iy1, Real box[], ops_particle particle,
                      ops_dat up, ops_dat tag, ops_dat Fd, ops_dat mass) {

  Real dx, dy;
  Real xmin[2], xmax[2];

  int ix = (ix1 <= 0) ? 1 : ix1;
  int iy = (iy1 <= 0) ? 1 : iy1;

  ops_block block = particle->block;
  int ix_c, iy_c;
#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];

  ix_c = ix / sb->pdims[0];
  iy_c = iy / sb->pdims[1];
#else
  ix_c = ix;
  iy_c = iy;
#endif

  Real dx_min[2], dx_max[2];
  dx_min[0] = box[0] - ((BoundingBox<Real> *) particle->box_block)->getGlobalMin().x;
  dx_min[1] = box[2] - ((BoundingBox<Real> *) particle->box_block)->getGlobalMin().y;

  dx_max[0] = ((BoundingBox<Real> *)particle->box_block)->getGlobalMax().x - box[1];
  dx_max[1] = ((BoundingBox<Real> *)particle->box_block)->getGlobalMax().y - box[3];

  xmin[0] = ((BoundingBox<Real> *) particle->box_block)->getLocalMin().x + 0.075 * dx_min[0];
  xmin[1] = ((BoundingBox<Real> *) particle->box_block)->getLocalMin().y + 0.075 * dx_min[1];

  xmax[0] = ((BoundingBox<Real> *) particle->box_block)->getLocalMax().x - 0.075 * dx_max[0];
  xmax[1] = ((BoundingBox<Real> *) particle->box_block)->getLocalMax().y - 0.075 * dx_max[1];

  //Setting limits based on needs
  xmin[0] = MAX(xmin[0], box[0]);
  xmin[1] = MAX(xmin[1], box[2]);

  xmax[0] = MIN(xmax[0], box[1]);
  xmax[1] = MIN(xmax[1], box[3]);

  dx = (xmax[0] - xmin[0]) / static_cast<Real>(ix_c + 1);
  dy = (xmax[1] - xmin[1]) / static_cast<Real>(iy_c + 1);

  int nparts = ix_c * iy_c;
  if (nparts > particle->Nmax)
    ops_particle_realloc_data(particle, nparts);

  int ipart = 0;
  Real *xpos = (Real *)particle->particle_pos_dat->data;
  Real *velp = (Real *)up->data;
  Real *massp = (Real *) mass->data;
  Real *Fdp =(Real *)Fd->data;
  int  *ids = (int *)tag->data;

  for (int i = 0; i < ix_c; i++) {
    Real x1 = xmin[0] + dx * static_cast<Real>(i + 1);
    for (int j = 0; j < iy_c; j++) {
      Real y1 = xmin[1] + dy * static_cast<Real>(j + 1);
      xpos[2 * ipart] = x1;
      xpos[2 * ipart + 1] = y1;

      velp[2 * ipart] = 0.0;
      velp[2 * ipart + 1] = 0.0;

      ids[ipart] = ipart + 1;
      Fdp[2 * ipart] = 0.0;
      Fdp[2 * ipart + 1] = 0.0;

      massp[ipart] = 0.002;
      ipart++;
    }
  }

  particle->no_particles = ipart;

#ifdef OPS_MPI
  int nelems[ops_num_procs()];
  MPI_Allgather(&ipart, 1, MPI_INT, nelems, 1, MPI_INT, sb->comm);
  int ntot = 0;
  int myrank;
  MPI_Comm_rank(sb->comm, &myrank);
  for (int i = 0; i < myrank; i++)
    ntot += nelems[i];


  for (int ip = 0; ip < particle->no_particles; ip++) {
     ids[ip] += ntot;
  }
#else
  for (int i = 0; i < particle->no_particles; i++)
    ids[i]  += 1;
#endif

 // for (int i = 0; i < nparts; i++)
 //   printf("Particle %d  [%12.9e %12.9e]\n", ids[i], xpos[2 * i], xpos[2 * i +1]);
}

void setup_maps(ops_particle particle, ops_dat *border, int nsize) {




  ops_particle_setup_maps_with_dats(particle, border, nsize);

  ops_particle_halo_transfer_group(OPS_HALO_GRP_BORDER, true);


}

//TODO: Standarized it
void build_maps(ops_particle particle, ops_dat *dat_border, ops_dat *dat_forward,
                int nborder, int nforward, int iter) {
  double et1, et0;
  double ct1, ct0;

  ops_timers(&ct0, &et0);
  double et_in = et0;

  int decide =  ops_particle_update_map_lists_actual_hybrid(particle);//ops_particle_update_map_lists(particle);




  //Part I: Add halos for exchange
  ops_particle_halo_transfer_group_map(OPS_HALO_GRP_EXCHANGE, decide);



  /* Part II: Remove particles marked for deletion */
  ops_timers(&ct0, &et0);
  ops_particle_remove_delete_maps(particle, decide);
  ops_timers(&ct1, &et1);
  et_rem += et1 - et0;


  if (decide) {

    //Halo exchanges-before border map update


    ops_timers(&ct0, &et0);
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);

    ops_particle_halo_transfer_group_map(OPS_HALO_GRP_BORDER, decide);

    ops_timers(&ct1, &et1);
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

}

void output_grid_dat(ops_dat *group, int ngroup, int timestep) {

  //Assume rho, u, v
  std::string rho_out = "rho_timestep_" + std::to_string(timestep);
  std::string u_out = "u_timestep_" + std::to_string(timestep);
  std::string v_out = "v_timestep_" + std::to_string(timestep);
  std::string xgrid_out = "grid_time_step";

  ops_print_dat_to_txtfile(group[0], rho_out.c_str());
  ops_print_dat_to_txtfile(group[1], u_out.c_str());
  ops_print_dat_to_txtfile(group[2], v_out.c_str());

  if (timestep == 0)
    ops_print_dat_to_txtfile(group[3], xgrid_out.c_str());

//  std::string particle_data = "particle_data_" + std::to_string(timestep) + ".txt";
//  ops_particle_print_data_to_txtfile(particle, particle_data.c_str());

}

void output_particle_dat(ops_particle particle, ops_dat *grp, int ngroup,
                         int timestep) {

  std::string filename = "particle_timestep_" + std::to_string(timestep) + ".txt";
  ops_particle_print_dats_to_txtfile(particle, grp, ngroup, filename.c_str());

}

void  implement_bc(ops_block block, ops_dat f, ops_stencil S2D_00, int size[]) {

  double uw[] = {0.01, 0.};
  double zeros[] = {0., 0.};

  //Bottom wall


  //Right wall
  int range_right[] = {size[0]-1, size[0], 1, size[1] - 1};
  ops_par_loop(Ker_Diffuse_Refl_Norm_xpos, "Ker_Diffuse_Refl_Norm_ypos", block, 2,
               range_right,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(zeros, 2, "double", OPS_READ));

  //Left wall
  int range_left[] = {0, 1,  1, size[1] - 1};
  ops_par_loop(Ker_Diffuse_Refl_Norm_xneg, "KerDiffuse_Refl_Norm_yneg", block, 2,
               range_left,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(zeros, 2, "double", OPS_READ));

  //Top wall moving
  int range_top[] = {1, size[0]-1, size[1]-1, size[1]};
  ops_par_loop(Ker_Diffuse_Refl_Norm_ypos, "KerDiffuse_moving", block, 2,
               range_top,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(uw, 2, "double", OPS_READ));

  int range_bot[] = {1, size[0]-1, 0, 1};
  ops_par_loop(Ker_Diffuse_Refl_Norm_yneg,"Ker_Diffuse_Refl_Norm_xneg", block, 2,
               range_bot,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(zeros, 2, "double", OPS_READ));

  //Bottom-Left corner
  int range_BL_C[] = {0, 1, 0, 1};
  ops_par_loop(Ker_Diffuse_Refl_Norm_xneg_yneg, "Ker_Diffuse_Refl_Norm_xneg_yneg",
               block, 2, range_BL_C,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(zeros, 2, "double", OPS_READ));

  //Bottom-Right corner
  int range_BR_C[] = {size[0] - 1, size[0], 0, 1};
  ops_par_loop(Ker_Diffuse_Refl_Norm_xpos_yneg, "Ker_Diffuse_Refl_Norm_xpos_yneg",
               block, 2, range_BR_C,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(zeros, 2, "double", OPS_READ));

  int range_TR_C[] = {size[0] - 1, size[0], size[1] - 1, size[1]};
  ops_par_loop(Ker_Diffuse_Refl_xpos_ypos,"Ker_Diffuse_Refl_xpos_ypos",
               block, 2, range_TR_C,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(uw, 2, "double", OPS_READ));

  int range_TL_C[] = {0, 1, size[1] - 1, size[1]};
  ops_par_loop(Ker_Diffuse_Refl_Norm_xneg_ypos, "Ker_Diffuse_Refl_xneg_ypos",
               block, 2, range_TL_C,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(uw, 2, "double", OPS_READ));

}

int main(int argc, char **argv) {
  ops_init(argc, argv, 1);

  ops_init(argc, argv, 1);

  const int Nx = 101;
  const int Ny = 101;

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

  ops_block lb_block = ops_decl_block(2, "lattice-boltzmann_grid");

  const int NX = 101;
  const int NY = 101;

  int size[] = {NX, NY};
  int base[] = {0,0};     // this is in C indexing - start from 0

  int d_m[]  = {-1,-1};     // max boundary depths for the dat in the negative direction
  int d_p[]  = { 1, 1};     // max boundary depths for the dat in the possitive direction
   // Single dim dats
  ops_dat rho   = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl,
                               "double", "rho");
  ops_dat u_x    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p,
                                temp_dbl, "double", "ux");
  ops_dat u_y    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p,
                                temp_dbl, "double", "uy");

  int size_c[] = {10, 10};
  int stride_c[] = {10, 10};
  ops_dat test = ops_decl_dat(lb_block, 1, size_c, base, d_m, d_p, stride_c,
                              temp_dbl, "double", "testing");

  ops_dat f = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl,
                           "double", "f");
  ops_dat f_copy = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl,
                                "double", "f");

  int d_mz[] = {-1, -1};
  int d_pz[] = { 1,  1};
  ops_dat x_grid = ops_decl_dat(lb_block, 2, size, base, d_m, d_p, temp_dbl,
                                "double", "xgrid");

  // Declare stencils
  int s2d_00[] = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
  int s2d_9pt[] = {-1,-1, -1,0, -1,1, 0,-1, 0,0, 0,1, 1,-1, 1,0, 1,1};
  ops_stencil S2D_9pt = ops_decl_stencil(2,9,s2d_9pt,"9pt");

  int s2d_4pt[] = {0, 0, 1, 0, 1, 1, 0, 1};
  ops_stencil S2D_4pt = ops_decl_stencil(2, 4, s2d_4pt, "4pt");

  int s2d_4pt_opp[] =  {0, 0, -1, 0, 0, -1, -1, -1};
  ops_stencil S2D_4pt_opp = ops_decl_stencil(2, 4, s2d_4pt_opp, "4pt_opp");

  ops_dat group_out[] = {rho, u_x, u_y, x_grid}; //TODO: Need HDF5: Particle can be outputed randomly

  Real dx = 1.0/(size[0] - 1.0);
  Real dt = dx / cs;
  OMEGA = dt / (tau + 0.5 * dt);

  //===================================================================
  // Declaire particle data structures
  //==================================================================

  Real dx_box[] = {0., 0.};
  BoundingBox<Real> *box = ops_create_bounding_box(lb_block,
                                                   x_grid, 2, dx_box);

  ops_particle particle = ops_decl_particle(lb_block, "Markers", box);

  //Assign variables for testing
  ops_dat particle_pos = ops_decl_particle_pos_dat(particle, 2, base, temp_dbl,
                                                   "double", "part_coords");

  ops_dat particle_vel = ops_decl_particle_dat(particle, 2, base,
                                               temp_dbl, "double", "part_vels");

  ops_dat particle_ids = ops_decl_particle_dat(particle, 1, base, temp_int,
                                               "int", "part_ids");

  printf("%s has dim = %d\n", particle_ids->name, particle_ids->dim);

  ops_dat particle_drag = ops_decl_particle_dat(particle, 2, base,
                                                temp_dbl, "double", "part_fd");

  ops_dat particle_mass = ops_decl_particle_dat(particle, 1, base,
                                                temp_dbl, "double", "part_mass");

  ops_particle_mapping map = ops_decl_mapping(particle, x_grid, S2D_9pt,
                                              OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);

  ops_dat dat_borders[] = {particle_pos, particle_vel, particle_ids, particle_mass};
  ops_dat dat_forward[] = {particle_pos, particle_vel};


  ops_dat parts_dat_outputs[] = {particle_ids, particle_pos, particle_vel};

  ops_decl_const("OMEGA",1,"double",&OMEGA);
  ops_decl_const("rho0",1,"double",&rho0);

  /* Declaire halos */
  ops_particle_halo_data halo_coords = ops_particle_decl_data_halo(particle_pos, particle_pos,
                                                                   OPS_PART_ORIENT_ON);

  ops_particle_halo_data halo_ids = ops_particle_decl_data_halo(particle_ids, particle_ids,
                                                                OPS_PART_ORIENT_OFF);

  ops_particle_halo_data halo_vels = ops_particle_decl_data_halo(particle_vel, particle_vel,
                                                                 OPS_PART_ORIENT_OFF);

  //Halo exchange decleration
  Real dx_x[] = {0.01, 0};
  Real translate_x[] = {1., 0};
  int dir[] = {0, 1};

  ops_particle_halo_data grp_data[] = {halo_coords, halo_ids, halo_vels};


  ops_particle_halo exchange_x1 = ops_particle_decl_halo(particle, particle, grp_data, 3, dx_x,
                                                         dir, dir, translate_x);
  Real translate_xopp[] = {-1, .0};

  ops_particle_halo exchange_x2 = ops_particle_decl_halo(particle, particle, grp_data, 3, dx_x,
                                                         dir, dir, translate_xopp);

  ops_particle_halo border_x1 = ops_particle_decl_halo(particle, particle, grp_data, 3, dx_x,
                                                       dir, dir, translate_x);

  ops_particle_halo border_x2 = ops_particle_decl_halo(particle, particle, grp_data, 3, dx_x,
                                                       dir, dir, translate_xopp);



  //Create halo group for exchange
  ops_particle_halo exchange_group_x[] = {exchange_x1, exchange_x2};
  ops_particle_halo_group group_exch_x = ops_particle_decl_halo_group(exchange_group_x, 2,
                                                                      OPS_HALO_GRP_EXCHANGE,
                                                                      OPS_WITH_VIRTUAL);

  ops_particle_halo border_group_x[] = {border_x1, border_x2};
  ops_particle_halo_group group_bord_x = ops_particle_decl_halo_group(border_group_x, 2,
                                                                      OPS_HALO_GRP_DEFAULT,
                                                                      OPS_WITH_VIRTUAL);


  printf("Prior partition\n");
  printf("Rank %d: Binhead size = [%d %d]\n", ops_get_proc(), map->binhead->size[0],
         map->binhead->size[1]);
  ops_partition("");

  int full_range[] = {-1,NX+1, -1,NY+1};
  int interior_range[] = {0,NX, 0,NY};

  ops_par_loop(KerInitGrid, "KerInitGrid", lb_block, 2, interior_range,
               ops_arg_dat(x_grid, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_gbl(&dx, 1, "double", OPS_READ),
               ops_arg_idx());

  //=======================================
  //  Setup particle simulation
  //=======================================

  ops_particle_setup_partition();

  ops_particle_set_halo_group(group_exch_x);
  ops_particle_set_halo_group(group_bord_x);


  printf("Number of halos = %d\n", group_exch_x->nhalos);
  for (int ihalos = 0; ihalos < group_exch_x->nhalos; ihalos++) {
    ops_particle_halo halo  = group_exch_x->halo_list[ihalos];
    ops_mpi_particle_halo *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];
    printf("R %d Halos = %d nproc to = %d\n", ops_get_proc(), ihalos, mpi_halo->nproc_to);
    for (int i = 0; i < mpi_halo->nproc_to; i++) {
      printf("Halo %d: Rank %d send to %d with box [%f %f %f %f]\n",// or [%d %d]x[%d %d]\n",
             ihalos, ops_get_proc(), mpi_halo->proclist[i],
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMinCoordDir(0),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMaxCoordDir(0),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMinCoordDir(1),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMaxCoordDir(1));
          //   mpi_halo->send_region[0], mpi_halo->send_region[1],
          //   mpi_halo->send_region[2], mpi_halo->send_region[3]);
    }
  }

  printf("Number of halos = %d\n", group_bord_x->nhalos);
  for (int ihalos = 0; ihalos < group_bord_x->nhalos; ihalos++) {
    ops_particle_halo halo  = group_bord_x->halo_list[ihalos];
    ops_mpi_particle_halo *mpi_halo = &OPS_mpi_particle_halo_list[halo->index];
    printf("R %d Halos = %d nproc to = %d\n", ops_get_proc(), ihalos, mpi_halo->nproc_to);
    for (int i = 0; i < mpi_halo->nproc_to; i++) {
      printf("Halo %d: Rank %d send to %d with box [%f %f %f %f]\n or [%d %d]x[%d %d]\n",
             ihalos, ops_get_proc(), mpi_halo->proclist[i],
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMinCoordDir(0),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMaxCoordDir(0),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMinCoordDir(1),
             ((BoundingBox<Real> **) mpi_halo->sendBox)[i]->getMaxCoordDir(1), //);
             mpi_halo->send_region[0], mpi_halo->send_region[1],
             mpi_halo->send_region[2], mpi_halo->send_region[3]);
    }
  }

  /*============================================================
   * Initial & first output of grid sim (LBM data)
   *============================================================*/

  ops_par_loop(KerInitMacros, "KerInitMacros", lb_block, 2, full_range,
               ops_arg_dat(rho, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(u_x, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(u_y, 1, S2D_00, "double", OPS_WRITE));


  ops_par_loop(KerInitF,"KerInitF", lb_block, 2, full_range,
               ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(rho, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(u_x, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(u_y, 1, S2D_00, "double", OPS_READ),
               ops_arg_gbl(cx, 9, "int", OPS_READ),
               ops_arg_gbl(cy, 9, "int", OPS_READ),
               ops_arg_gbl(wi, 9, "double", OPS_READ),
               ops_arg_gbl(&cs, 1, "double", OPS_READ));

  ops_par_loop(KerComputeMacros, "KerComputeMacros", lb_block, 2,
               interior_range,
               ops_arg_dat(rho, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(u_x, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(u_y, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(f, 9, S2D_00, "double", OPS_READ),
               ops_arg_gbl(cx, 9, "int", OPS_READ),
               ops_arg_gbl(cy, 9, "int", OPS_READ),
               ops_arg_gbl(&cs, 1, "double", OPS_READ));

  output_grid_dat(group_out, 4, 0); //TODO


  /*============================================================
   *   Insert part to simulation and setting up maps
   *============================================================*/
  int igrid[] = {4, 4};
  Real box_insert[] = {0.025, 0.975, 0.025, 0.975};

  insert_particles(igrid[0], igrid[1], box_insert, particle, particle_vel, particle_ids,
                   particle_drag, particle_mass);

  setup_maps(particle, dat_borders, 3); //TODO
  printf("Maps are build\n");

  if (ops_get_proc() == 0) {
   ((Real *) particle_pos->data)[2 * 3] = -0.001;
  }
  build_maps(particle, dat_borders, dat_forward, 3, 2, 0);


  for (int i = 0; i < particle->no_particles + particle->no_virtual;i++) {
    printf("R %d: id[%d] = %d xp = [%12.9e %12.9e]\n", ops_get_proc(), i, ((int *) particle_ids->data)[i],
           ((Real *) particle_pos->data)[2 * i], ((Real *) particle_pos->data)[2 * i + 1]);
  }
  exit(-1);

  ops_exit();


}
