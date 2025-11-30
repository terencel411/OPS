/*
 * halo.cpp
 *
 *  Created on: Jul 25, 2025
 *      Author: valantis
 */


void _ops_particle_halo_border_transfer(OPS_instance *instance,
                                        ops_particle_halo_group halo_grp) {

  //Construct sending
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo  halo= halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange  info =  halo_grp->halo_info[ihalo];

    ops_particle particle_from = halo->particle_from;
    int dim = particle_from->block->dims;

    int imin = 0;
    int imax = (halo_grp->with_virtual != OPS_NO_VIRTUAL) ?
        particle_from->no_particles + particle_from->no_virtual :
                                      particle_from->no_particles;

    BoundingBox *box = halo->sendBox;
    info->nsend = 0;

    double *xlocal = (double *)particle_from->particle_pos_dat->data;
    for (int i = imin; i < imax; i++) {
      ops_point point{xlocal[dim * i], xlocal[dim * i + 1], (dim == 3) ?
                      xlocal[dim * i + 2] : 0.0};


      bool decide = box->isCoordinateInBoundingBox(point);
      if (decide) {
        info->nsend++;
        if (info->nsend > info->nmax) {
          info->nmax = info->nsend + OPS_MAX_PART;
          info->sendlist = (int *) ops_realloc(info->sendlist,
                                               info->nmax * sizeof(int));
        }

        info->sendlist[info->nsend - 1] = i;

        /* Reallocatr buffer as well */
        /* Ensure that buffer is large enough to receive data */
        if (info->nsend * halo->nbites > instance->ops_halo_buffer_size) {
          instance->ops_halo_buffer_size += (OPS_MAX_PART + info->nsend) * halo->nbites;
          instance->ops_halo_buffer  = (char *)ops_realloc(instance->ops_halo_buffer,
                                                           instance->ops_halo_buffer_size
                                                           * sizeof(char));
        }

      }

    }

  }

  //Upgrading recv information
  for (int ihalo = 0; ihalo < halo_grp->nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];


    ops_particle particle_to = halo->particle_to;
    info->nrecv = info->nsend;

    //Compute location of first;
    info->firstrecv = particle_to->no_particles + particle_to->no_virtual;
    particle_to->no_virtual += info->nrecv;
    //Realloc buffer lists -Only for MPI
  }

  //Perform pack exchange
  for (int ihalo = 0; ihalo < halo_grp-> nhalos; ihalo++) {
    ops_particle_halo halo = halo_grp->halo_list[ihalo];
    ops_particle_halo_exchange info = halo_grp->halo_info[ihalo];

    for (int isend = 0; isend < info->nsend; isend++) {
      int i = info->sendlist[isend];
      _ops_particle_pack_halo_data(instance->ops_halo_buffer  + isend * halo->nbites,
                                   halo->dat, halo->nhalos, i);
    }

    ops_particle particle_to = halo->particle_to;
    if (particle_to->no_particles + particle_to->no_virtual > particle_to->Nmax) {
      ops_particle_realloc_data(particle_to, particle_to->no_particles + particle_to->no_virtual);
    }


    //npack data
    for (int irecv = 0; irecv < info->nrecv; irecv++) {
        int ifirst = info->firstrecv;
        /* Realocation of particle data */
        int iloc = ifirst + irecv;
        _ops_particle_unpack_halo_data(instance->ops_halo_buffer  + irecv * halo->nbites,
                                       halo->dat, halo->nhalos, iloc, halo->dir_to,
                                       halo->dir_from, halo->translate); //TODO: check

    }


  }

}


