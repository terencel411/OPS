/*
 * sss.h
 *
 *  Created on: Sep 8, 2026
 *      Author: valantis
 */

#ifndef OPS_C_INCLUDE_SSS_H_
#define OPS_C_INCLUDE_SSS_H_

template<template<typename X> class Distribution, typename T1, typename T2>
void ops_insert_random_particles(ops_particle particle, T1 *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, T2> *rad_distr,
                                 ops_dat envelope) {


  //Sanity checks

  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function must be called after "
                                             " the ops_particle structure is defined\n");

  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

  //TODO: Add checks for sanity
  if (!ops_partitioned())
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: The function requires partition of"
                                             " the simulation domain");

  if (particle->particle_map_index == 0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: At least an ops_particle_mapping "
                                             "structure needs to be defined\n");

  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");


  if (envelope == nullptr)
    envelope = particle->particle_envelope;

  if (envelope == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty particle envelope. Please define one to "
                                             "proceed ");


  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  T1 dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  T1 region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  T1 vol = 1;
  T1 glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = ((BoundingBox<T1> *) particle->box_block)->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = ((BoundingBox<T1> *) particle->box_block)->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > std::numeric_limits<T1>::epsilon()) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < std::numeric_limits<T1>::epsilon())
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > std::numeric_limits<T1>::epsilon()) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  T1 *weights = (T1 *) ops_malloc(sizeof(T1) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  if (sizeof(T1) == sizeof(float))
    MPI_Allgather(&sf, 1, MPI_FLOAT, weights, 1, MPI_FLOAT, sb->comm);
  else
    MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);


  T1 sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < std::numeric_limits<T1>::epsilon())
    return;


  ops_weight_particle_partition(weights, Nins, comm_size, nparticles_ins);
  int rank;
  MPI_Comm_rank(sb->comm, &rank);

  Ninsert = nparticles_ins[rank];



  ops_free(weights);
  ops_free(nparticles_ins);

  //Get and receives elements in the x and y direction.
  int flag_need_send[2 * OPS_MAX_DIM];

  int flag_need_recv[2 * OPS_MAX_DIM];
  MPI_Status status[2];
  int flag_send = (Ninsert > 0) ? 1 : -1;
  for (int isou = 0; isou < dim; isou++) {

    flag_need_send[2 * isou] = (Ninsert > 0 && sb->id_m[isou] != MPI_PROC_NULL) ? 1 : -1;
    flag_need_send[2 * isou + 1] = (Ninsert > 0 && sb->id_p[isou] != MPI_PROC_NULL) ? 1 : -1;



    if (region_insert[2 * isou] > ((BoundingBox<T1> *) particle->box_block)->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 700,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 700,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < ((BoundingBox<T1>) particle->box_block)->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 800,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 800,
                 sb->comm, &status[1]);

  }

// Final checks for inserting particles
  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;

  //TODO: Check that this works for the parallel as well in terms of a function
  BoundingBox<T1> *insertBox =  ops_find_intersection_region((BoundingBox<T1> *)particle->box_block,
                                                            region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Generate virtual maps
  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], size[OPS_MAX_DIM];

#ifdef OPS_MPI
  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i] < 0) ? - 1: 0;
    d_p[i] = (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i] > 0) ? 1: 0;

    size[i] = map->binhead->size[i] + (OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i])
            - (OPS_sub_dat_list[map->binhead->index]->d_ip[i] + map->binhead->d_p[i]);
    if (d_m[i] < 0)
      size[i] += 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }


  int nbins = particle->no_particles + Ninsert + particle->no_virtual;

#else

  for (int i = 0; i < particle->block->dims; i++) {
    d_m[i] = ( map->binhead->d_m[i] < 0) ? -1 : 0;
    d_p[i] = (map->binhead->d_p[i] > 0) ? 1 : 0;
    size[i] = map->binhead->size[i] - map->binhead->d_p[i] + map->binhead->d_m[i];
    if (d_m[i] < 0)
      size[i] -= 1;
    if (d_p[i] > 0)
      size[i] += 1;
  }

  int nbins = particle->no_particles + Ninsert; //TODO: Shift to unsigned
#endif

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0; d_p[i] = 0;  size[i] = 1;
  }

  int nexpected = particle->no_particles + Ninsert;

  //Allocate structures
  int prod = 1;
  for (int i = 0; i < dim; i++) prod *= size[i];
  int *binhead = (int *) ops_malloc(sizeof(int) * prod);

  for (int i = 0; i < prod; i++)
    binhead[i] = -1;

  int *bins = (int *) ops_malloc(sizeof(int) * nbins);
  for (int i = 0; i < particle->no_particles; i++)
    bins[i] = -1;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected + particle->no_particles);

  //Get xmin and xmax for mapping procedures as dx as well
  ops_point<T1> xmin = ((BoundingBox<T1> *) particle->box_block)->getLocalMin();
  ops_point<T1> xmax = ((BoundingBox<T1> *) particle->box_block)->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  T1 *xpos = (T1 *)particle->particle_pos_dat->data;
  for (int i = 0; i < particle->no_particles; i++) {
    ix[0] = (int) ops_floor((xpos[dim * i] - xmin.x) / dx[0]);
    ix[1] = (int) ops_floor((xpos[dim * i + 1] - xmin.y) / dx[1]);
    ix[2] = (dim == 3) ?
        (int) ops_floor((xpos[dim * i + 2] - xmin.z) /dx[2]) : 0;

    int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                + (ix[2] - d_m[2]) * size[0] * size[1];

    bins[i] = binhead[address];
    binhead[address] =i;
  }


  int ifirst;
#ifdef OPS_MPI
//TODO: Need to generate a second map with virtual particles that we will increase
//      the bins

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (char *) ops_malloc(sizeof(T1) * nvirtual_max * dim);
  rad_virtual = (char *) ops_malloc(sizeof(T2) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local<T1,T2>(sb, particle, envelope, binhead, bins, size, flag_need_send,
                                flag_need_recv, nvirtual,
                                nvirtual_max, nexpected, xmin, dx);

  candidate_pos = (char *) ops_malloc(sizeof(T1) * ncand_max * dim);
  candidate_rad = (char *) ops_malloc(sizeof(T2) * ncand_max);
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Assume that local particles are mapped
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;

#ifdef OPS_MPI
  int flg_vol_brd = 0;
  T1 vol_bord = 0.0;
  for (int isou = 0; isou < dim; isou++) {
        vol_bord += _compute_volume_normal(isou, dim, flag_need_send + 2 * isou,
                                           region_insert, dx);
  }


  int nlocal_insert_max = (Ninsert > 0) ? (int) ops_floor(( 1. - vol_bord / vol) * Ninsert) : 0;
  int nattempt_loc_max = (Ninsert > 0) ? (int) ops_floor((1. - vol_bord / vol) * nattempts) : 0;

  //No need to insert
  int nborder_required =  (nlocal_insert_max < Ninsert) ? MIN(Ninsert - nlocal_insert_max, 50) : 100 * Ninsert;

//#ifdef OPS_MPI
  int my_rank;
  MPI_Comm_rank(sb->comm, &my_rank);
  seed += my_rank;

  int ninsert_with_cands = 0;
#endif

  std::default_random_engine dre(seed);

  //Create an ops_dat for particle data
  OPSDistribution<std::uniform_real_distribution, T1> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, T1>(particle->block->dims, region_insert);

  int iattempt_tot = 0;
  int iattempt = 0;
  T1 xpoint[OPS_MAX_DIM];
  T2 rad_ins;
  int n_insert = 0;


//  printf("Prior to generate\n");

  while (n_insert < Ninsert) {
    int a1;
    int ix[OPS_MAX_DIM] = {};
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {


      ops_generate_random_point(part_dist, dre, xpoint);

      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, (T2 *)envelope->data,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        (T1 *)candidate_pos,
                                        (T2 *) candidate_rad, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_pos = (char *) ops_realloc(candidate_pos, sizeof(T1) * dim * (ncand_actual + OPS_MAX_PART));
          candidate_rad = (char *) ops_realloc(candidate_pos, sizeof(T2) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(flag_to_add, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }

        for (int isou = 0; isou < dim; isou++)
          ((T1 *)candidate_pos)[dim  * (ncand_actual - 1) + isou] = xpoint[isou];

        ((T2 *)candidate_rad)[(ncand_actual - 1)] = rad_ins;
        flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((T1 *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((T2 *) envelope->data)[nlocal - 1] = rad_ins;


        //Add point to bin
        int address = (ix[0] - d_m[0]) + (ix[1] - d_m[1]) * size[0]
                    + (ix[2] - d_m[2]) *size[1] * size[0];

        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;

        //Set xold as well //


        break;
      }
      else if (a1 == 1) {iattempt++; break;}

      iattempt++;
    }

    iattempt_tot += iattempt;


    /*TODO: After 50-100 attemps-Try to check virtual elements
      Add them to the existing list & include them as virtual as
      well
    */

#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {



      //PART I: Set exchange flags to send around and recv

      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);

      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts<T1,T2>(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);


      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);


      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(  (T1 *) candidate_pos, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, (T1 *) xvirtual, (T2 *) rad_virtual, nvirtual,
                                       nvirtual_max);




      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((T1 *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                ((T1 *) candidate_pos)[dim  * ip + isou];
          ((T2 *) envelope->data)[nlocal - 1] = ((T2 *) candidate_rad)[ip];

          if (n_insert == Ninsert) break;
        }
      }

//      printf("R %d: I insert %d cands (prior %d tot %d)\n", ops_get_proc(), n_insert - nisert_localised,
//             nisert_localised, n_insert);
 //     exit(-1);

      //Part IV: Reset the lists
      ncand_actual = 0;
      ncand_virtual = 0;

      ninsert_with_cands = n_insert;
    }
#endif

    if (iattempt_tot >=  nattempts_tot) {
     //TODO: PUSH END OF INSERTION
      break;
    }

//    printf("R %d: Particles inserted %d\n", ops_get_proc(), n_insert);

  }

  /*
  for (int i = 0; i < nlocal; i++)
    printf("R %d: x[%d] = [%f %f] R = %f\n", ops_get_proc(), i,
                                           ((double *)particle->particle_pos_dat->data)[i * dim + 0],
                                           ((double *)particle->particle_pos_dat->data)[i * dim + 1],
                                           ((double *)envelope->data)[i]);
*/

 // printf("Rank %d Particles inserted %d\n", ops_get_proc(), n_insert);

  //ADD END OF INSERTION FOR PARALLEL
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //TODO: Add tags
  if (particle->ids != nullptr) {

    if (strcmp(particle->ids->type, "int") == 0 ||
        strcmp(particle->ids->type, "int(4)") == 0 ||
        strcmp(particle->ids->type, "integer") == 0 ||
        strcmp(particle->ids->type, "integer(4)") == 0 ||
        strcmp(particle->ids->type, "integer(kind=4)") == 0)
#ifdef OPS_MPI
      update_particleIds<int>((int *) particle->ids->data, ifirst,
                              n_insert,sb);
#else
      update_particleIds<int>((int *) particle->ids->data, ifirst,
                              n_insert);
#endif
    else if (strcmp(particle->ids->type, "int64") == 0 ||
             strcmp(particle->ids->type, "uint64_t") == 0 ||
             strcmp(particle->ids->type, "uint64") == 0 ||
             strcmp(particle->ids->type, "int64_t") == 0)

#ifdef OPS_MPI
      update_particleIds<int64_t>((int64_t *) particle->ids->data,
                                  ifirst, n_insert, sb);
#else
    update_particleIds<int64_t>((int64_t *) particle->ids->data,
                                ifirst, n_insert);
#endif
    else {
      if (particle->ids->type_size == sizeof(int))
#ifdef OPS_MPI
        update_particleIds<int>((int *) particle->ids->data, ifirst,
                                n_insert,sb);
#else
       update_particleIds<int>((int *) particle->ids->data, ifirst,
                                  n_insert);
#endif
      else if (particle->ids->type_size == sizeof(int64_t))
#ifdef OPS_MPI
        update_particleIds<int64_t>((int64_t *) particle->ids->data,
                                    ifirst, n_insert, sb);
#else
        update_particleIds<int64_t>((int64_t *) particle->ids->data, ifirst,
                                   n_insert);
#endif

    }

  }

  int nexist = particle->no_particles;
  particle->no_particles = nlocal;

  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  particle->no_particles = nlocal;
  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //NOTE: We had hear a reallocation array for maps but particle data structures passd
      // to particle allocation

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }


  //FREE ARRAYS
#ifdef OPS_MPI
  ops_free(candidate_pos);
  ops_free(candidate_rad);
  ops_free(flag_to_add);


  for (int i = 0; i < 2 * dim; i++) {
    ops_free(forward_candids[i]);
  }


  ops_free(xvirtual);
  ops_free(rad_virtual);
  ops_free(bin_virtual);
#endif

  ops_free(binhead);
  ops_free(bins);

}



#endif /* OPS_C_INCLUDE_SSS_H_ */
