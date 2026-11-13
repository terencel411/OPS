/*
 * rest_functions.h
 *
 *  Created on: Sep 8, 2026
 *      Author: valantis
 */

#ifndef OPS_C_INCLUDE_REST_FUNCTIONS_H_
#define OPS_C_INCLUDE_REST_FUNCTIONS_H_


template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);

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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);


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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);

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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);

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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);

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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }



#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocation of map data passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18,
         template<typename X19> class Distribution19, typename T19>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18,
                                 OPS_dat_distr<Distribution19, T19> &distr19) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }

#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
        insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
          insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocations passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8,
         template<typename X9> class Distribution9, typename T9,
         template<typename X10> class Distribution10, typename T10,
         template<typename X11> class Distribution11, typename T11,
         template<typename X12> class Distribution12, typename T12,
         template<typename X13> class Distribution13, typename T13,
         template<typename X14> class Distribution14, typename T14,
         template<typename X15> class Distribution15, typename T15,
         template<typename X16> class Distribution16, typename T16,
         template<typename X17> class Distribution17, typename T17,
         template<typename X18> class Distribution18, typename T18,
         template<typename X19> class Distribution19, typename T19,
         template<typename X20> class Distribution20, typename T20>

void ops_insert_random_particles(ops_particle particle, double *region, int Nins,
                                 int nattempts, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 ops_dat envelope,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10,
                                 OPS_dat_distr<Distribution11, T11> &distr11,
                                 OPS_dat_distr<Distribution12, T12> &distr12,
                                 OPS_dat_distr<Distribution13, T13> &distr13,
                                 OPS_dat_distr<Distribution14, T14> &distr14,
                                 OPS_dat_distr<Distribution15, T15> &distr15,
                                 OPS_dat_distr<Distribution16, T16> &distr16,
                                 OPS_dat_distr<Distribution17, T17> &distr17,
                                 OPS_dat_distr<Distribution18, T18> &distr18,
                                 OPS_dat_distr<Distribution19, T19> &distr19,
                                 OPS_dat_distr<Distribution20, T20> &distr20) {

  //Part I: Sanity checks
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

  //Part II: Find number of particles inserted in each process
  int Ninsert;
  int dim = particle->block->dims;
  ops_particle_mapping map = particle->map_list[0];
  double dx[OPS_MAX_DIM];
  ops_particle_map_get_dx(map, dx);

  double region_insert[OPS_MAX_DIM * 2];

#ifdef OPS_MPI
//TODO
  sub_block *sb = OPS_sub_block_list[particle->block->index];
  if (!sb->owned) return;

  double vol = 1;
  double glb_vol = 1;
  for (int isou = 0; isou < dim; isou++) {
    region_insert[2 * isou] = particle->box_block->getMinCoordDir(isou);
    region_insert[2 * isou + 1] = particle->box_block->getMaxCoordDir(isou);

    if (region[2 * isou] > region_insert[2 * isou])
      region_insert[2 * isou] = region[2 * isou];

    if (region[2 * isou + 1] < region_insert[2 * isou + 1])
      region_insert[2 * isou + 1] = region[2 * isou + 1];

//    double dxb = (region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
//                  region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0;

    vol *= ((region_insert[2 * isou + 1] - region_insert[2 * isou] > DBL_EPSILON) ?
             region_insert[2 * isou + 1] - region_insert[2 * isou] : 0.0);
    glb_vol *= region[2 * isou + 1] - region[2 * isou];

  }

  if (glb_vol < DBL_EPSILON)
    throw OPSException(OPS_RUNTIME_ERROR,"ERROR: Non-positive volume for particle insertion\n");

  vol = (vol > DBL_EPSILON) ? vol : 0.0;
  double sf = vol / glb_vol;


  int comm_size;
  MPI_Comm_size(sb->comm, &comm_size);

  double *weights = (double *) ops_malloc(sizeof(double) * comm_size);

  int *nparticles_ins = (int *) ops_malloc(sizeof(int) * comm_size);

  MPI_Allgather(&sf, 1, MPI_DOUBLE, weights, 1, MPI_DOUBLE, sb->comm);

  double sum = 0.0;
  for (int i = 0; i < comm_size; i++)
    sum+= weights[i];

  if (sum < DBL_EPSILON)
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



    if (region_insert[2 * isou] > particle->box_block->getMinCoordDir(isou) + dx[isou])
      flag_need_send[2 * isou] = -1;

    flag_need_recv[2 * isou] = -1;
    flag_need_recv[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou], 1, MPI_INT, sb->id_m[isou], 100,
                 &flag_need_recv[2 * isou +1], 1, MPI_INT, sb->id_p[isou], 100,
                 sb->comm, &status[0]);

    if (region_insert[2 * isou + 1] < particle->box_block->getMaxCoordDir(isou) - dx[isou])
      flag_need_send[2 * isou + 1] = -1;

    MPI_Sendrecv(&flag_need_send[2 * isou + 1], 1, MPI_INT, sb->id_p[isou], 200,
                 &flag_need_recv[2 * isou ], 1, MPI_INT, sb->id_m[isou], 200,
                 sb->comm, &status[1]);
  }

  for (int isou = 0; isou < 2 * dim; isou++)
    if (flag_need_send[isou] == -1 || flag_need_recv[isou] == -1) {
      flag_need_send[isou] = -1;
      flag_need_recv[isou] = -1;
    }


#else
  int inters = 2;
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  if (inters == 2) { delete insertBox; return;};
  Ninsert =  Nins;
  for (int i = 0; i < particle->block->dims; i++)  {
    region_insert[2 * i] = insertBox->getMinCoordDir(i);
    region_insert[2 * i + 1] = insertBox->getMaxCoordDir(i);
  }
#endif

  //Part III: Create maps for particle insertion
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

  int nbins = particle->no_particles + Ninsert;
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
  ops_point xmin = particle->box_block->getLocalMin();
  ops_point xmax = particle->box_block->getLocalMax();


  //Map existing particles to map
  int ix[OPS_MAX_DIM] = {};
  double *xpos = (double *)particle->particle_pos_dat->data;
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

  //Part IIIa: Insert virtual particles (for intra-block comms) and
  //           generate structures for candidate-boundary (interior) particles
  //TODO: Add halos
#ifdef OPS_MPI

  int nvirtual_max = particle->no_virtual + OPS_MAX_PART;

  xvirtual = (double *) ops_malloc(sizeof(double) * nvirtual_max * dim);
  rad_virtual = (double *) ops_malloc(sizeof(double) * nvirtual_max);

  int nvirtual = 0;
  ifirst = 0;

  bin_virtual = (int *) ops_malloc(sizeof(int) * nvirtual_max);

  shift_virtual_to_local(sb, particle, envelope, binhead, bins, size, flag_need_send,
                         flag_need_recv, nvirtual,
                         nvirtual_max, nexpected, xmin, dx);

  candidate_parts = (double *) ops_malloc(sizeof(double) * ncand_max * (dim + 1));
  flag_to_add = (int *) ops_malloc(sizeof(int) * ncand_max);

  ncand_actual = 0;
  ncand_virtual = 0;


  //Allocate candidate exchange

  for (int i = 0; i < 2 * dim;i++) {
    nforward_max[i] = 10;
    forward_candids[i] = (int *) ops_malloc(sizeof(int) * nforward_max[i]);
  }

#endif

  //Part IV: Reallocate particle structures based on estimated number
  int nattempts_tot = nattempts * Ninsert;

  if (nexpected > particle->Nmax)
    ops_particle_realloc_data( particle, nexpected);

  int nlocal = particle->no_particles;
  ifirst = nlocal;


  //Part V: Get limits for checking creation in border cells & get seed for random engines
#ifdef OPS_MPI
  int flg_vol_brd = 0;
  double vol_bord = 0.0;
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


  /* Part VI: Generate random engine and OPS-distribution for particle positions */
  std::default_random_engine dre(seed);

  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(particle->block->dims, region_insert);

  /* Part VII: Particle generation */
  int iattempt_tot = 0;
  int iattempt = 0;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  int n_insert = 0;

  while (n_insert < Ninsert) {

    int a1;
    int ix[OPS_MAX_DIM] = {};

    //Generate radius
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;

    while (iattempt < nattempts) {
      //Generate a random point
      ops_generate_random_point(part_dist, dre, xpoint);
      ix[0] = (int ) ops_floor((xpoint[0] - xmin.x)/dx[0]);
      ix[1] = (int ) ops_floor((xpoint[1] - xmin.y) / dx[1]);
      ix[2] = (particle->block->dims == 3) ?
          (int ) ops_floor((xpoint[2] - xmin.z) / dx[2]) : 0;

      a1 = _check_particle_location(binhead, size, bins,
#ifdef OPS_MPI
                                    bin_virtual, nexpected,
#endif
                                    d_m, ix, xpoint, rad_ins, particle->particle_pos_dat, envelope,
#ifdef OPS_MPI
                                    xvirtual, rad_virtual,
#endif
                                    dim);

      //Additional checks agaist border particles
#ifdef OPS_MPI

      //Check against candidates
      if (a1 == 0) {
        check_against_border_candidates(ix, dim, size, xpoint, rad_ins,
                                        candidate_parts, ncand_actual);
      }

      if (a1 == 0 && particle_in_border(ix, dim, size, flag_need_send)) {

        ncand_actual++;
        if (ncand_actual > ncand_max) {
          candidate_parts = (double *) ops_realloc(candidate_parts, sizeof(double) * (dim  + 1) * (ncand_actual + OPS_MAX_PART));
          flag_to_add = (int *) ops_realloc(candidate_parts, sizeof(int) * (ncand_actual + OPS_MAX_PART));
          ncand_max = ncand_actual + OPS_MAX_PART;
        }
          for (int isou = 0; isou < dim; isou++)
            candidate_parts[(dim + 1) * (ncand_actual - 1) + isou] = xpoint[isou];
          candidate_parts[(dim + 1) * (ncand_actual - 1) + dim] = rad_ins;
          flag_to_add[ncand_actual - 1] = 0; //TO-BE Inserted if ok

        a1 = 1;
      }
#endif

      //Particle insertion
      if (a1 == 0)  {
        n_insert++;
        nlocal++;
        iattempt++;

        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *) envelope->data)[nlocal - 1] = rad_ins;
        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
        insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
        insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
        insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
        insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
        insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
        insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
        insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
        insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
        insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
        insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);
        insert_random_dat(distr20.dat->data, distr20.distribution, dre, nlocal-1);


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

    //Check if border particles must be inserted in the list
#ifdef OPS_MPI
    //TODO: We need to set also the max attempts per point-which are reset
    if (nborder_required == ncand_actual ||
        n_insert - ninsert_with_cands == nlocal_insert_max) {
      //PART I: Set exchange flags to send around and recv
      _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                             flag_need_send, flag_need_recv);
      //PART II: Exchange particles and check in each direction
      _exchange_candidate_parts(sb, particle->particle_pos_dat, envelope, dim, flag_need_send,
                                flag_need_recv, xmin, xmax, dx, binhead, size, d_m, d_p, bins);
      //PART III: Perform reverse operation (Can become default)
      _reverse_operations(sb, dim, flag_to_add);

      //PART III: Push virtual to actual
      _push_generated_to_virtual_lists(candidate_parts, flag_to_add, dim, ncand_actual, ncand_virtual,
                                       binhead, size, d_m, d_p, xmin, dx,
                                       bin_virtual, bins, nexpected, xvirtual, rad_virtual, nvirtual,
                                       nvirtual_max);

      //PART IV: Insert particles to the list & generate elements
      for (int ip = 0; ip < ncand_actual; ip++) {
        //INSERT PARTICLE
        if (flag_to_add[ip] == 0) {
          n_insert++;
          nlocal++;

          for (int isou = 0; isou < dim; isou++)
            ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou] =
                candidate_parts[(dim + 1) * ip + isou];
          ((double *) envelope->data)[nlocal - 1] = candidate_parts[(dim + 1) * ip + dim];

          //Update additional lists
          insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
          insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
          insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
          insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
          insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
          insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
          insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
          insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
          insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);
          insert_random_dat(distr10.dat->data, distr10.distribution, dre, nlocal-1);
          insert_random_dat(distr11.dat->data, distr11.distribution, dre, nlocal-1);
          insert_random_dat(distr12.dat->data, distr12.distribution, dre, nlocal-1);
          insert_random_dat(distr13.dat->data, distr13.distribution, dre, nlocal-1);
          insert_random_dat(distr14.dat->data, distr14.distribution, dre, nlocal-1);
          insert_random_dat(distr15.dat->data, distr15.distribution, dre, nlocal-1);
          insert_random_dat(distr16.dat->data, distr16.distribution, dre, nlocal-1);
          insert_random_dat(distr17.dat->data, distr17.distribution, dre, nlocal-1);
          insert_random_dat(distr18.dat->data, distr18.distribution, dre, nlocal-1);
          insert_random_dat(distr19.dat->data, distr19.distribution, dre, nlocal-1);
          insert_random_dat(distr20.dat->data, distr20.distribution, dre, nlocal-1);

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

    if (iattempt_tot >=  nattempts_tot)  break;

  }

  //Part VIII: Finalize insertion
#ifdef OPS_MPI
  _update_exchange_flags(sb, dim, iattempt_tot, nattempts_tot, n_insert, Ninsert,
                         flag_need_send, flag_need_recv);

#endif

  //Part IX: Update particle tags if necessary
  if (particle->ids != nullptr) {

    int max_tag = 0;
    int *tags = (int *)particle->ids->data;
    for (int i = 0; i < ifirst; i++)
      max_tag = MAX(max_tag, tags[i]);

#ifdef OPS_MPI

    //Get number of processes in the system
    int nranks;
    MPI_Comm_size(sb->comm, &nranks);
    MPI_Comm_size(sb->comm, &my_rank);
    int *tag_ranks = (int *) ops_malloc(sizeof(int) * nranks);
    int *recv_elems = (int *) ops_malloc(sizeof(int) * nranks);

  //Gather
    MPI_Allgather(&max_tag, 1, MPI_INT, tag_ranks, 1, MPI_INT, sb->comm);
    MPI_Allgather(&n_insert, 1, MPI_INT, recv_elems, 1, MPI_INT, sb->comm);

    max_tag = 0;
    int nrecv_max = 0;
    for (int i = 0; i < nranks; i++) {
      max_tag = MAX(tag_ranks[i], max_tag);
      nrecv_max += (my_rank < i ? n_insert : 0);
    }
//#else
    max_tag += nrecv_max;

    ops_free(tag_ranks);
    ops_free(recv_elems);
#endif
    //Find max

    for (int  i = 0; i < n_insert; i++) {
      tags[i + ifirst] = max_tag + (i+1);
    }
  }

  //Part X: Finalize particle insertion and map update
  int nexist = particle->no_particles;
  particle->no_particles = nlocal;
  for (int i = nexist; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;

  if (n_insert > 0) {
    for (int i = 0; i < particle->particle_map_index; i++) {

      //Allocations passed to particles

      //map the remaining particles
      particle->map_list[i]->nParticles = nlocal;
      _ops_particle_map_from_exchange(particle->map_list[i], particle, ifirst,  nlocal);
      particle->map_list[i]->decide = true;
    }
  }

  //Part XI: Free structures
#ifdef OPS_MPI
  ops_free(candidate_parts);
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

#endif /* OPS_C_INCLUDE_OPS_INSERT_RANDOM_PARTICLES_V2_H_ */



#endif /* OPS_C_INCLUDE_REST_FUNCTIONS_H_ */
