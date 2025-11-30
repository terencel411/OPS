/* STEP 5 - Code generation with OPS translator for different targets
*/
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>

double OMEGA = 1.0;
double rho0 = 1.0;
double deltaUX = 10e-6;

// Including main OPS header file, and setting 2D
#define OPS_2D
#include <ops_seq_v2.h>
#include <ops_particle_seq.h>
#include "ops_particle_insert_del.h"

// Including applicaiton-specific "elemental kernels"
#include "lattice_kernels.h"
#include "particle_kernels.h"


static inline int mod(int v, int m) {
    int val = v%m;
    if (val<0) val = m+val;
    return val;
}

void output_grid_dat(ops_dat *group, int ngroup, ops_particle particle,
                     int timestep) {

  //Assume rho, u, v
  std::string rho_out = "rho_timestep_" + std::to_string(timestep) + ".txt";
  std::string u_out = "u_timestep_" + std::to_string(timestep) + ".txt";
  std::string v_out = "v_timestep_" + std::to_string(timestep) + ".txt";

  ops_print_dat_to_txtfile(group[0], rho_out.c_str());
  ops_print_dat_to_txtfile(group[1], u_out.c_str());
  ops_print_dat_to_txtfile(group[2], v_out.c_str());

  std::string particle_data = "particle_data_" + std::to_string(timestep) + ".txt";
  ops_particle_print_data_to_txtfile(particle, particle_data.c_str());

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

void  init_particles(int npart, ops_particle particle, ops_dat particle_pos,
                     ops_dat particle_vel, ops_particle_mapping map, double dx) {

  //Assume the following
  int ix_x = sqrt(npart);
  int iy_y = npart / ix_x;

  int npart_ins = ix_x * iy_y;

  double xpos[2 * npart_ins];
  double uins[2 * npart_ins];
  double Rp[npart_ins];

  for (int i = 0; i < npart_ins; i++)
    Rp[i] = 0.001;

  int ipart = 0;
  for (int i = 0; i < ix_x; i++) {
    for (int j = 0; j < iy_y; j++) {
      xpos[2 * ipart] = dx * i + 0.2;
      xpos[2 * ipart + 1] = 0.8 - 10 * dx * j;
      ipart++;
    }
  }

  printf("ix_x  = %d iy_x = %d\n", ix_x, iy_y);
  printf("particles to insert %d\n", npart_ins);
  printf("x = [%f %f]\n", xpos[0], xpos[1]);
  printf("x = [%f %f]\n", xpos[2], xpos[3]);

  ops_particle_insert(KerInsertData,"KerInsertData", KerDecide, particle, 0, 2,
                      npart_ins, xpos, nullptr,
                      ops_arg_dat_particle(particle_vel, 2, "double", particle, map, OPS_WRITE));


}

void insert_particles(int ix, int iy, double box[], ops_particle particle,
                      ops_dat particle_vel, ops_particle_mapping map) {

  int ix1 = (ix <= 0) ? 1 : ix;
  int iy1 = (iy <= 0) ? 1 : iy;
  double dx = (box[1] - box[0]) / static_cast<double>(ix1 + 1);
  double dy = (box[3] - box[2]) / static_cast<double>(iy1 + 1);

  int nparts = ix1 *iy1;

  double xpos[2 * nparts];
  double Rp[nparts];

  int ipart = 0;
  for (int i = 0; i < ix1; i++) {
    double x1 = box[0] + dx * static_cast<double>(i+1);
    for (int j = 0; j < iy1; j++) {
      double y1 = box[2] + dy * static_cast<double>(j + 1);
      xpos[2 * ipart] = x1;
      xpos[2 * ipart + 1] = y1;
      Rp[ipart] = 0.001;
      ipart++;
    }
  }

  ops_particle_insert(KerInsertData, "KerInsertData", KerDecide, particle, 0, 2,
                      nparts, xpos, nullptr,
                      ops_arg_dat_particle(particle_vel, 2, "double", particle, map, OPS_WRITE));

  for (int i = 0; i < nparts; i++)
    printf("x[%d] = [%f %f]\n", i, xpos[2 * i], xpos[2 * i +1]);

 // ops_particle_reset_marked(particle);


}

void setup_maps(ops_particle particle) {

  ops_particle_build_maps(particle, true);

}


void build_maps(ops_particle particle, int timestep) {

  int decide =  ops_particle_update_map_lists_actual_hybrid(particle);//ops_particle_update_map_lists(particle);


  if (decide) {
    printf("List is rebuild at %d ", timestep);

    ops_particle_remove_delete_maps(particle, 1);
    ops_particle_reset_flags(particle, true);
    printf("Remaining particles : %d\n", particle->no_particles);
  }


//  int *bin2grid = (int *)particle->map_list[0]->parts_to_grid->data;
//  for (int i = 0; i < particle->no_particles; i++)
//    printf("Particle %d: mapped to %d\n", i, bin2grid[i]);
}


int main(int argc, char ** argv) {
    // Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
    ops_init(argc, argv, 1);

    const int NX = 101;
    const int NY = 101;

    const double wi[] ={4. / 9., 1./ 9., 1./ 9., 1./ 9. , 1. /9.,
                       1./36., 1./36., 1./ 36., 1./36.};
    const double cs = sqrt(3);

    const int cx[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int cy[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    double tau = 0.0001;

    
    double energy;

    double ct0,et0,ct1,et1; //timer variables

    int *temp_int = NULL;
    double *temp_dbl = NULL;

    //====================================
    // Declare & define key data structures
    //====================================

    // The 2D block
    ops_block lb_block = ops_decl_block(2, "lattice-boltzmann_grid");

    int size[] = {NX, NY};
    int base[] = {0,0};     // this is in C indexing - start from 0

    int d_m[]  = {-1,-1};     // max boundary depths for the dat in the negative direction
    int d_p[]  = { 1, 1};     // max boundary depths for the dat in the possitive direction
    // Single dim dats
    ops_dat rho   = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "rho");
    ops_dat u_x    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "ux");
    ops_dat u_y    = ops_decl_dat(lb_block, 1, size, base, d_m, d_p, temp_dbl, "double", "uy");

    ops_dat f = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl, "double", "f");
    ops_dat f_copy = ops_decl_dat(lb_block, 9, size, base, d_m, d_p, temp_dbl, "double", "f");

    int d_mz[] = {0, 0, 0};
    int d_pz[] = {0, 0, 0};
    ops_dat x_grid = ops_decl_dat(lb_block, 2, size, base, d_mz, d_pz, temp_dbl, "double", "xgrid");

    // Declare stencils
    int s2d_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
    int s2d_9pt[] = {-1,-1, -1,0, -1,1, 0,-1, 0,0, 0,1, 1,-1, 1,0, 1,1};
    ops_stencil S2D_9pt = ops_decl_stencil(2,9,s2d_9pt,"9pt");

    int s2d_4pt[] = {0, 0, 1, 0, 1, 1, 0, 1};
    ops_stencil S2D_4pt = ops_decl_stencil(2, 4, s2d_4pt, "4pt");

    ops_dat group_out[] = {rho, u_x, u_y};

    double dx = 1.0/(size[0] - 1.0);
    double dt = dx / cs;
    OMEGA = dt / (tau + 0.5 * dt);

    //===================================================================
    // Declaire particle data structures
    //==================================================================

    //Decleration of a bounding box
    double dx_box[] = {0., 0.};
    BoundingBox *bounding_box = ops_create_bounding_box(lb_block, x_grid, 2, dx_box);

    //Declaire particle structure
    ops_particle particle = ops_decl_particle(lb_block, "markers", bounding_box);

    //Declaire particle ops_dat structures
    ops_dat particle_pos = ops_decl_particle_pos_dat(particle, 2, base, temp_dbl, "double", "part_coords");

    ops_dat particle_vel = ops_decl_particle_dat(particle, 2, base, temp_dbl, "double", "part_vels");

    ops_particle_mapping  map = ops_decl_mapping(particle, x_grid, nullptr,
                                                 S2D_00, OPS_WITH_VIRTUAL, OPS_CONST_SHAPE,
                                                 OPS_UNIFORM_STAG, dx, 1);




    printf("OMEGA = %12.9e dx = %12.9e dt = %12.9e\n", OMEGA, dx, dt);
    // Declare and define global constants
    ops_decl_const("OMEGA",1,"double",&OMEGA);
    ops_decl_const("rho0",1,"double",&rho0);

    ops_partition("");


       // Range
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

    //==========================================================
    // Initialize & first output of grid based part (LBM data)
    //==========================================================
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




    //=================================================================
    //Insert particles into a grid and setup maps
    //=================================================================
    int igrid[] = {1, 1};
    double box_insert[] = {0.05, 0.95, 0.05, 0.95};
    insert_particles(igrid[0], igrid[1], box_insert, particle,
                      particle_vel, map);

    //Build map
    setup_maps(particle);

    output_grid_dat(group_out, 3, particle, 0);


//    for (int i = 0; i < particle->no_particles; i++)
//      printf("Particle %d: mark_flag = %d\n", i, particle->mark_deletion[i]);

    //Start timer
    ops_timers(&ct0, &et0);
    // Main time loop
    int Nsteps = 1000000;
    int nprint = 2000;
    for (int i = 0; i < Nsteps; i++) {

      //Perform collisions
      ops_par_loop(KerCollision, "KerComputeCollision", lb_block,
                   2, interior_range,
                   ops_arg_dat(f_copy, 9, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(f, 9, S2D_00, "double", OPS_READ),
                   ops_arg_dat(rho, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(u_x, 1, S2D_00, "double", OPS_READ),
                   ops_arg_dat(u_y, 1, S2D_00, "double", OPS_READ),
                   ops_arg_gbl(&OMEGA, 1, "double", OPS_READ),
                   ops_arg_gbl(cx, 9, "int", OPS_READ),
                   ops_arg_gbl(cy, 9, "int", OPS_READ),
                   ops_arg_gbl(wi, 9, "double", OPS_READ),
                   ops_arg_gbl(&cs, 1, "double", OPS_READ));

      //Perform Streaming
      ops_par_loop(KerStream, "KerStream", lb_block, 2,
                   interior_range,
                   ops_arg_dat(f, 9, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(f_copy, 9, S2D_9pt, "double", OPS_READ),
                   ops_arg_gbl(cx, 9, "int", OPS_READ),
                   ops_arg_gbl(cy, 9, "int", OPS_READ));

      //BCs
      implement_bc(lb_block, f, S2D_00, size);

      ops_par_loop(KerComputeMacros, "KerComputeMacros", lb_block, 2,
                   interior_range,
                   ops_arg_dat(rho, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(u_x, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(u_y, 1, S2D_00, "double", OPS_WRITE),
                   ops_arg_dat(f, 9, S2D_00, "double", OPS_READ),
                   ops_arg_gbl(cx, 9, "int", OPS_READ),
                   ops_arg_gbl(cy, 9, "int", OPS_READ),
                   ops_arg_gbl(&cs, 1, "double", OPS_READ));



      //Performing particle operations (Map particles)
      double range_parts[] = {0., 1., 0., 1.};
      ops_particle_par_loop(KerUpdateParticleVelocity, "KerUpdateParticleVelocity",
                            particle, 2, OPS_PARTICLE_ITERATE_LOCAL, range_parts,
                            map,
                            ops_arg_dat_particle(particle_vel, 2, "double", particle, map, OPS_WRITE),
                            ops_arg_dat_particle(particle_pos, 2, "double", particle, map, OPS_READ),
                            ops_arg_dat(u_x, 1, S2D_4pt, "double", OPS_READ),
                            ops_arg_dat(u_y, 1, S2D_4pt, "double", OPS_READ),
                            ops_arg_dat(x_grid, 2, S2D_4pt, "double", OPS_READ),
                            ops_arg_gbl(&dx, 1, "double", OPS_READ));

      ops_particle_par_loop(KerUpdatePosition, "KerUpdatePosition", particle,
                            2, OPS_PARTICLE_ITERATE_LOCAL, range_parts, map,
                            ops_arg_dat_particle(particle_pos, 2, "double", particle, map, OPS_WRITE),
                            ops_arg_dat_particle(particle_vel, 2, "double", particle, map, OPS_READ),
                            ops_arg_gbl(&dt, 1, "double", OPS_READ));


      int timestep = i + 1;

      build_maps(particle, i);

      if (timestep % nprint == 0)
        output_grid_dat(group_out, 3, particle, timestep);


    }

    if ((Nsteps ) % nprint != 0)
      output_grid_dat(group_out, 3, particle, Nsteps + 1);



    //End timer
    ops_timers(&ct1, &et1);
    ops_printf("\nTotal Wall time %lf seconds\n",et1-et0);

    /*if (true) {
        std::ofstream myfile;
        myfile.open ("output_velocity.txt");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << SOLID[j*NX+i] << " " << ux[j*NX+i] << " " << uy[j*NX+i] << std::endl;
            }
        }
        myfile.close();
    }*/

    // Finalising the OPS library
    ops_exit();
    return 0;
    
}// End of main function

