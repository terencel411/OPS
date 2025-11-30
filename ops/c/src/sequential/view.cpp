/*
 * view.cpp
 *
 *  Created on: Aug 26, 2025
 *      Author: valantis
 */


//TODO: We need to add herein what it builds (local or all)
int _ops_particle_decide_build_local_uniform(ops_particle_mapping map,
                                             ops_particle particle) {

  //TODO: Do we need this to be false by default???
  map->decide = false;



  int stag = 1; //Staggering is by def 1 for now

 // printf("Entering to check: %d\n");

  /* Checking if mapping declaired particles < total */
  int changed{0};
  int dim = particle->block->dims;
  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;
  int *part_to_grid = (int *)map->parts_to_grid->data;
  int *mark_del = particle->mark_deletion;


  /*Accessing local particles only */
  /* 1. Check for deletion */
  /* 2. Check if need to move */

  int *d_p = map->binhead->d_p;
  int *d_m = map->binhead->d_m;
  int *size = map->binhead->size;

  size_t nnodes = 1;
  int ilocal[OPS_MAX_DIM], ilocal_new[OPS_MAX_DIM];
  int grid_nodes[OPS_MAX_DIM];
  int zeros[OPS_MAX_DIM];


 /* Set particles for mapping */
  int nmapping
    = (map->mapping_type != OPS_WITH_VIRTUAL) ? particle->no_particles :
                          particle->no_particles + particle->no_virtual;


  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();

  double dx[particle->block->dims];
  /* Get grid size the structure */
  _ops_compute_uniform_dx(map->grid, particle->block->dims, dx);
  //Upate xmin and xmax due to special conditions

  xmin.x += static_cast<double>(map->binhead->d_m[0]) * dx[0];
  xmax.x += static_cast<double>(map->binhead->d_p[0]) * dx[0];

  xmin.y += static_cast<double>(map->binhead->d_m[1]) * dx[1];
  xmax.y += static_cast<double>(map->binhead->d_p[1]) * dx[1];

  if (particle->block->dims == 3) {
   xmin.z += static_cast<double>(map->binhead->d_m[2]) * dx[2];
   xmax.z += static_cast<double>(map->binhead->d_p[2]) * dx[2];
  }

  double *xpos = (double *)particle->particle_pos_dat->data;
  double *xold = (double *)map->pos_old->data; //Assumed here in that
  //Nparticles in list is build properly

  int nlimits = 2 * dim;
  int rmv_limits[2 * OPS_MAX_DIM];

  for (int i =  0; i < dim; i++) {
    rmv_limits[2 *i] = 0;
    rmv_limits[2 * i + 1] = size[i] + d_m[i] - d_p[i] - 1;
  }

//  printf("rmv_limits = [%d %d]x[%d %d]\n", rmv_limits[0], rmv_limits[1], rmv_limits[2], rmv_limits[3]);

/*
  for (int i = 0; i < particle->no_particles + particle->no_virtual; i++) {
    if (i < particle->no_particles)
      printf("Actual particle  %d: x = [ %f %f ] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
             bins[i], part_to_grid[i]);
    else printf("Virtual %d: x = [ %f %f ] xold =[%f %f] bins = %d and bin = %d\n", i, xpos[2 * i], xpos[2 * i + 1],
                 xold[2 * i], xold[2 * i + 1],bins[i], part_to_grid[i]);
  }
*/

  int nactual = particle->no_particles;

  //TODO: For GPU: Split into two parts
  for (int i = 0; i < particle->no_particles; i++) {
    int flag =  local_decide_rebuild(xpos + i * dim, xold + i * dim,  map->skin,
                                     dim);

    if (flag) {

      int address = part_to_grid[i];
      int iPart = binhead[address];

      get_local_point(address, map->binhead->size, map->binhead->d_m, dim,
                      ilocal);

      _remove_particle_from_bins(address, i, binhead, bins);

      //Actual particle perform two checks

        //TODO-1: Check for possible removal
      int del_flag =  _ops_particle_check_for_deletion(i, ilocal, dim, rmv_limits,
                                                       xpos + i * dim, particle->box_block);

      if (del_flag) {
        part_to_grid[i] = -1;
        map->decide = true;//TODO: Think if it is needed
        nactual--;
        particle->mark_deletion[i] = 1;
        continue;
      }

      //Start-mapping
      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size,xpos + dim * i);
      if (address < 0) map->decide = true;

      bins[i] = binhead[address];
      binhead[address] = i;

      part_to_grid[i] = address;

      get_local_point(address, map->binhead->size, map->binhead->d_m,
                       dim, ilocal_new);

      get_coord_point((double *)map->grid->data, map->grid->size, map->grid->d_m,
                       ilocal_new, dim, xold + dim * i, map->skin, 1);

      //
      bool flag_build  = _ops_particle_moved_to_exchange_zone(ilocal, ilocal_new, rmv_limits, dim);

 //     if (flag_build) printf("Particle %d to exchange zone\n");

      if (!map->decide) map->decide = flag_build;

    }
  }

  nactual = particle->no_particles;

  int ifirst = particle->no_particles;
  int nvirtual_act = particle->no_virtual;

//  printf("Exit actual mapping: ");

  //printf("Building virtual maps\n");


   //TODO: We need a new version which keeps th

  for (int i = ifirst; i < nmapping; i++) {

    int flag = local_decide_rebuild(xpos + i * dim, xold + i * dim, map->skin, dim);

    if (flag) { //To-Rebuild for virtual particle

      int address = part_to_grid[i];
      _remove_particle_from_bins(address, i, binhead, bins);
      part_to_grid[i] = -1;


      address = _ops_coord_to_bin(dim, xmin, xmax, dx, map->binhead->size, xpos + dim * i);

      if (address < 0) {
  //      printf("Particle %d non-positive address (%d)\n", i, address);
        map->decide = true; continue;}

      //Virtual still projected'


      get_local_point(address, map->binhead->size, map->binhead->d_m,
                      dim, ilocal);

      int flag_in = _virtual_within(ilocal, rmv_limits, xpos + dim * i, particle->box_block, dim);

      //Need to remove



      if (flag_in) { //Virtual become actual

        nactual++; //Increase local.
        nvirtual_act--;
        //swap->particle data

        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


        _ops_particle_swap_data(particle->particle_pos_dat->data, i, nactual - 1,
                                particle->particle_pos_dat->elem_size);

        //compute new positions and swap: TODO

        if (particle->particle_envelope != nullptr) {
          _ops_particle_swap_data(particle->particle_envelope->data, i, nactual - 1,
                                  particle->particle_envelope->elem_size);
        }

        for (int idat = 0; idat < particle->particle_dat_index; idat++) {
          ops_dat dat = particle->particle_dat[idat];
          _ops_particle_swap_data(dat->data, i, nactual - 1, dat->elem_size);
        }

        particle->mark_deletion[i] = 0;
        _ops_particle_swap_data((char *)particle->mark_deletion, i, nactual - 1,
                                sizeof(int));

        //swap->map with first virtual

        //set flag map to true

        _ops_particle_swap_mapping_data(map, i, nactual - 1);
        _ops_particle_swap_data(map->pos_old->data, i, nactual -1, \
                                map->pos_old->elem_size);
        //TODO: Need to swap positions as well then rebuild
        map->decide = true;

      }

      if (!map->decide)  {
        part_to_grid[i] = address;
        bins[i] = binhead[address];
        binhead[address] = i;


        get_coord_point_virtual_point((double *) map->grid->data, map->grid->size,
                                      map->grid->d_m, ilocal, dim, xold + dim * i,
                                      map->skin, 1);
      }

    }


  }
  //Perform action for virtual particles



  //TODO: Add this info only for the mapping list
  if (nactual > particle->no_particles)
    particle->no_particles = nactual;

  particle->no_virtual = nvirtual_act;

  return (int) map->decide;
}

void _ops_particle_halo_exchange_transfer(OPS_instance *instance,
                                          ops_particle_halo_group halo_grp) {


  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;

    BoundingBox *box = halo->sendBox;

    if (box == nullptr) continue;
    info->nsend = 0;
    int no_particles = particle_from->no_particles;
    for (int i = 0; i < no_particles; i++) {

      if (particle_from->mark_deletion[i] == 1) {

        if (particle_from->particle_envelope != nullptr)
          if (((double *)particle_from->particle_envelope->data)[i] < 0) continue;

        ops_point point{xlocal[dim * i], xlocal[dim * i + 1],
                        (dim == 3) ? xlocal[dim * i + 2] : 0.0};
        bool decide = box->isCoordinateInBoundingBox(point);


        if (decide) {
          //mark particle for deletion now or shift it later
          info->nsend++;
          if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
            instance->ops_halo_buffer_size += (10  + info->nsend) * halo->nbites;
            instance->ops_halo_buffer = (char *)ops_realloc(instance->ops_halo_buffer , instance->ops_halo_buffer_size
                                           * sizeof(char));
          }

          particle_from->mark_deletion[i] = 2; //particle is exchanged
         _ops_particle_pack_halo_data(instance->ops_halo_buffer  + halo->nbites * (info->nsend - 1),
                                      halo->dat, halo->nhalos, i);
        }
      }
    }

    /* Unpack data */
    ops_particle particle_to = halo->particle_to;
    int nrecv = info->nrecv = info->nsend;
    int ifirst = particle_to->no_particles;

    particle_to->no_particles += nrecv;
    if (particle_to->no_particles > particle_to->Nmax)
      ops_particle_realloc_data(particle_to, particle_to->no_particles); //TODO: Add default number


    double *particle_crds = (double *)particle_to->particle_pos_dat->data;

    for (int i = 0; i < nrecv; i++) {
      int ielem = ifirst + i;

      _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + halo->nbites * i, halo->dat,
                                     halo->nhalos, ielem, halo->dir_to, halo->dir_from, halo->translate);

      //If particle within mark as actual
      ops_point xpoint{particle_crds[dim * ielem], particle_crds[dim * ielem + 1], (dim == 3) ? particle_crds[dim * ielem + 2] : 0.0};
      bool a1 =  particle_to->box_block->isCoordinateInBoundingBox(xpoint);
      particle_to->mark_deletion[ielem] = (a1) ? 0 : 1;

     //printf("Particle for deletion %d: %d\n", ielem, particle_to->mark_deletion[i]);
    }

    //Update the list initialization for inserted points

    for (int imap = 0; imap < particle_to->particle_map_index; imap++) {
      ops_particle_mapping map = particle_to->map_list[imap];
      if (map->Nmax < particle_to->no_particles) {
        _ops_particle_realloc_map_data(map, particle_to->no_particles);
      }

      //Set memories for bin and ibin2 from previous owned to currently owned


 //     int ifirst = sizeof(int) * particle_to->no_particles - nrecv;
 //     int size = nrecv * sizeof(int);

    //    memset(map->bin->data + ifirst, -1, size);
    //    memset(map->parts_to_grid->data + ifirst, -1, size);
      int *bins = (int *)map->bin->data;
      int *part2grid = (int *)map->parts_to_grid->data;
      for (int i = ifirst; i < particle_to->no_particles; i++) {
        bins[i] = -1;
        part2grid[i] = -1;
      }



//      for (int iPart = 0; iPart < particle_to->no_particles; iPart++)/
 //printf("")

    }

 //   int *part2grid = (int *)particle_to->map_list[0]->parts_to_grid->data;
//  for (int i = 0; i < particle_to->no_particles; i++)
//      printf("Part2Grid[%d] = %d\n", i, part2grid[i]);

//    printf("nparticles after exchange %d\n", particle_to->no_particles);

  }

}



