template<template<typename X> class Distribution>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr)
{
  if (particle == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: This function must be called after defining an"
                       " ops_particle structure");

  if (particle->box_block == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Bounding box is not defined\n");

  //TODO: Find actual interesection  region
  if (rad_distr->limits[0] <= 0 || rad_distr->limits[1]<=0)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Non-positive radii");

  if (nmult <= 0.0)
    nmult = 10;

  double Dx = 2. * rad_distr->limits[1] * static_cast<double>(nmult);
  int dim = particle->block->dims;

  double xmin[OPS_MAX_DIM], xmax[OPS_MAX_DIM];
  for (int i = 0; i < particle->block->dims; i++) {
    xmin[i] = region[2 * i];
    xmax[i] = region[2 * i + 1];
  }

  //TODO: Create Bounding Box for intersection
  int inters{2};
  BoundingBox *insertBox =  ops_find_intersection_region(particle->box_block, region, inters);
  printf("inters = %d\n", inters);
  printf("Box = [%f %f %f] x[%f %f %f]\n", insertBox->getMinCoordDir(0), insertBox->getMinCoordDir(1),
         insertBox->getMinCoordDir(2), insertBox->getMaxCoordDir(0), insertBox->getMaxCoordDir(1),
         insertBox->getMaxCoordDir(2));
  if (inters == 2) { delete insertBox; return;};


  int ncells[OPS_MAX_DIM], d_p[OPS_MAX_DIM], d_m[OPS_MAX_DIM];
  long int size = 1;
  for (int i = 0; i < particle->block->dims; i++) {
    ncells[i] = static_cast<int>(ceil(region[2 * i + 1] - region[2 * i]) / Dx);
    d_p[i] = 1; d_m[i] = -1;
    size = size *  (ncells[i] + 2);
  }

  for (int i = particle->block->dims; i < OPS_MAX_DIM; i++) {
    ncells[i] = 1;
    d_p[i] = 0;
    d_m[i] = 0;
  }

//  for (int i = 0; i < OPS_MAX_DIM; i++)
//    printf("%d: size = %d d_m = %d d_p = %d\n", i, ncells[i], d_m[i], d_p[i]);

//  printf("cells = [%d %d %d]\n", ncells[0], ncells[1], ncells[2]);
//  printf("Dim = %d\n dx = %f", dim, Dx);
//  printf("Nlocal = %d\n", particle->no_particles);

  int *binhead = (int *) ops_malloc(sizeof(int) * size);
  //TODO: Add also virtual elems to ensure continuity of search
  int *bins = (int *) ops_malloc(sizeof(int) * (particle->no_particles + Nins));
  int nlocal = particle->no_particles;

  for (int i = 0; i < size; i++)
    binhead[i] = -1;

  for (int i = 0; i < nlocal; i++)
    bins[i] = -1;


  double *xcrd = (double *)particle->particle_pos_dat->data;

  for (int iPar = 0; iPar < nlocal; iPar++) {
    if (insertBox->isCoordinateInBoundingBox(xcrd + dim * iPar)) {
      int ix[OPS_MAX_DIM];
      for (int isou = 0; isou < dim; isou++) {
        ix[isou] = floor(xcrd[iPar * dim + isou] - insertBox->getMinCoordDir(isou)) / Dx;
      }

      ix[2] = (dim == 2) ? ix[2] : 0;
      int address = get_address(dim, ix, d_m, d_p, ncells);
      bins[iPar] = binhead[address];
      binhead[address] = iPar;
    }
  }

//  printf("Ready to insert particles\n");
  //Create a random engine
  std::default_random_engine dre(seed);

  //Create an ops_dat for particle data
  OPSDistribution<std::uniform_real_distribution, double> *part_dist
  = ops_declaire_distribution<std::uniform_real_distribution, double>(dim, region);

  int n_insert = 0;
  int nattempt;
  double xpoint[OPS_MAX_DIM];
  double rad_ins;
  while (n_insert < Nins) {
    ops_generate_random_point(rad_distr, dre, &rad_ins);
    int iattempt = 0;
    while (iattempt < nattemps) {
      ops_generate_random_point(part_dist, dre, xpoint);


      //Get bin and surrounding
      int ix[OPS_MAX_DIM];
      for (int isou = 0; isou < dim ; isou++)
        ix[isou] = static_cast<int>(floor((xpoint[isou] - insertBox->getMinCoordDir(isou)) / Dx));

//      printf("Particle %d Attempt %d: Particle to insert [ %f %f %f ] R = %f (%d %d %d)\n",
//             nlocal, iattempt, xpoint[0], xpoint[1], xpoint[2],
//             rad_ins, ix[0], ix[1], ix[2]);
      ix[2] = (dim == 3) ? ix[2] : 0;
      int a1 = 0;
#ifdef OPS_3D
      for (int k = -1; k <= 1; k++)
#elif defined(OPS_2D)
      int  k = 0;
#endif
      {
        for (int j = -1; j <= 1; j++) {
          for (int i = -1; i <= 1; i++) {
            int address = (ix[0] + i - d_m[0]) + (ix[1] + j - d_m[1]) * (ncells[0] + d_p[0] - d_m[0])
                        + (ix[2] + k - d_m[2]) * (ncells[0] + d_p[0] - d_m[0]) * (ncells[1] + d_p[1] - d_m[1]);
f
            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);

//              printf("Sanity check for %d ([%d %d %d] address = %d) with iattemp %d for %d ", ineigh, ix[0]+i, ix[1]+j, ix[2]+k,
//                      address, iattempt, n_insert);
//              printf("d = %12.9e R = %12.9e\n",d, Rsq);

              if (d < 1.1 * Rsq) {
//                printf("Particle [%f %f %f] R = %f not inserted\n", xpoint[0], xpoint[1], xpoint[2], rad_ins);
                a1 = 1; goto endline;
              }


              ineigh = bins[ineigh];
            }
          }
        }
      }

      endline:
      if (a1 == 1) iattempt++;
      else if (a1 == 0) {
        n_insert++;
        nlocal++;
        if (nlocal > particle->Nmax)
          ops_particle_realloc_data( particle, nlocal);


        //Insert particle positions
        for (int isou = 0; isou < dim; isou++)
          ((double *)particle->particle_pos_dat->data)[dim * (nlocal - 1) + isou]
                                                       = xpoint[isou];

        ((double *)particle->particle_envelope->data)[nlocal - 1] = rad_ins;

        //insert point to list
        int ibin[OPS_MAX_DIM];
        for (int isou = 0; isou < dim ; isou++)
          ibin[isou] = static_cast<int>(floor((xpoint[isou] - insertBox->getMinCoordDir(isou)) / Dx));

        ibin[2] = (dim == 3) ? ibin[2] : 0;
        int address = (ibin[0] - d_m[0]) + (ibin[1] - d_m[1]) * (ncells[0] + d_p[0] - d_m[0])
                    + (ibin[2] - d_m[2]) * (ncells[0] + d_p[0] - d_m[0]) * (ncells[1] + d_p[1] - d_m[1]);


        bins[nlocal - 1] = binhead[address];
        binhead[address] = nlocal - 1;
        break;
      }

    }

    if (iattempt == nattemps) {
//      printf("Less particles generated within box\n");
      break;
    }
  }

  particle->no_particles = nlocal;

  //Force list rebuild
  if (n_insert > 0)
   for (int i = 0; i < particle->particle_map_index; i++) {
     particle->map_list[i]->decide = true; //enforce build

   }

  ops_free(bins);
  ops_free(binhead);
}
