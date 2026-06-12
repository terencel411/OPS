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
  * @brief Functionalities for inserting random particles into simulations
  * @author Valantis Tsinginos
  * @details Inserting random particles to simulations for different
  *          backends
  */

#ifndef __OPS_INSERT_RANDOM_PARTICLES_H
#define __OPS_INSERT_RANDOM_PARTICLES_H

#include "ops_lib_core.h"
#include "ops_distributions.h"
#include <limits>
inline int get_address(int dim, int ix[], int d_m[],int d_p[], int size[]) {

  int address = 0;
  for (int isou = 0; isou < dim; isou++) {
    int total = 1;
    for (int jsou = 0; jsou < isou; jsou++)
      total *= (size[jsou] + d_p[jsou] - d_m[isou]);
    address += (ix[isou] - d_m[isou]) * total;

  }

  return address;
}

inline int  get_max_local_size(int array[], int n) {

  int maxn = 0;

  for (int i = 0; i < n; i++) {
    maxn = MAX(maxn, array[i]);
  }

  return maxn;
}

/**
 * Definition an ops_dat distribution structures for populating
 * a given ops_dat structure
 *
 */

template <template<typename Elem> class Distribution, typename T>
struct OPS_dat_distr {
  OPSDistribution<Distribution, T> *distribution;  /**< OPSDistribution structure for populating
                                                        ops_dat structure */
  ops_dat dat;                                     /**< ops_dat that will be initialized based on
                                                        the OPSDistribution structure */
};

template <template<typename> class Distribution, typename T>
OPS_dat_distr<Distribution, T> ops_dat_distr(ops_dat dat, T* params, T* limit) {

  if (dat == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT,"Empty ops_dat structure");

  if (sizeof(T) != dat->type_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "The size of T differs from the size of the elements "
                                             "of the ops_dat structure\n");
  OPSDistribution<Distribution, T> *distrib =
      ops_declaire_distribution<Distribution, T>(dat->dim, params, limit);

  OPS_dat_distr<Distribution, T> dat_distr;
  dat_distr.distribution = distrib;
  dat_distr.dat = dat;

  return dat_distr;
}

template <template<typename Elem> class Distribution, typename T>
OPS_dat_distr<Distribution, T> ops_dat_distr(ops_dat dat,
                                             OPSDistribution<Distribution, T> *distr) {

  if (dat == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT,"Error: Empty ops_dat structure");

  if (distr == NULL)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Empty OPSDistribution structure");

  OPS_dat_distr<Distribution, T> dat_distr;

  if (dat->dim != distr->dims)
    throw OPSException(OPS_INVALID_ARGUMENT, "Error: Incompatible sizes of the distribution "
                                              " and ops_dat structures.");

  dat_distr.distribution = distr;
  dat_distr.dat = dat;
  return dat_distr;

}

template <template<typename Elem> class Distribution, typename T>
void insert_random_dat(char *data, OPSDistribution<Distribution, T>  *dist,
                       std::default_random_engine &dre, int i) {

  T* array = (T *)data;
  int dim = dist->dims;

  T local_point[dim];
  ops_generate_random_point(dist, dre, local_point);

  for (int isou = 0; isou < dim; isou++)
    array[isou + i * dim] = local_point[isou];
}


/**
 * An insert function for 0 additional arguments. The function simply inserts particles in
 * a user defined region
 *
 * @param particle      the ops_particle structure for which particles are inserted
 * @param region        use defined region in the form [xmin xmax]x[ymin ymax] x[zmin zmax]
 * @param Nins          maximum number of particle to be inserted in the domain
 * @param nmult         parameter for defining the actual bin size for particle insertion
 * @param nattemps      maximum attemps made before stopping pouring particles in the box
 * @param seed          integer for setting the functions engine to a given state
 * @param rad_distr     an OPSDistribution object for generating given particle distribution
 */
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

    //TODO: For the MPI add a second to check the Nins against the border elements (Add halos as well)

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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
      break;
    }
  }

  particle->no_particles = nlocal;

  //Force list rebuild
  if (n_insert > 0)
   for (int i = 0; i < particle->particle_map_index; i++) {
     particle->map_list[i]->decide = true; //enforce build
   }

  for (int i = 0; i < particle->no_particles; i++)
    particle->mark_deletion[i] = 0;


  ops_free(bins);
  ops_free(binhead);
}

/**
 * An insert function for 0 additional arguments. The function simply inserts particles in
 * a user defined region
 *
 * @param particle      the ops_particle structure for which particles are inserted
 * @param region        use defined region in the form [xmin xmax]x[ymin ymax] x[zmin zmax]
 * @param Nins          maximum number of particle to be inserted in the domain
 * @param nmult         parameter for defining the actual bin size for particle insertion
 * @param nattemps      maximum attemps made before stopping pouring particles in the box
 * @param seed          integer for setting the functions engine to a given state
 * @param rad_distr     an OPSDistribution object for generating given particle distribution
 * @param distr1        an OPS_dat_distr structure for initializing a particle dat structures
 */
template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);


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


template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);


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

template<template<typename X> class Distribution,
         template<typename X1> class Distribution1, typename T1,
         template<typename X2> class Distribution2, typename T2,
         template<typename X3> class Distribution3, typename T3,
         template<typename X4> class Distribution4, typename T4,
         template<typename X5> class Distribution5, typename T5,
         template<typename X6> class Distribution6, typename T6,
         template<typename X7> class Distribution7, typename T7,
         template<typename X8> class Distribution8, typename T8>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);


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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

        insert_random_dat(distr1.dat->data, distr1.distribution, dre, nlocal-1);
        insert_random_dat(distr2.dat->data, distr2.distribution, dre, nlocal-1);
        insert_random_dat(distr3.dat->data, distr3.distribution, dre, nlocal-1);
        insert_random_dat(distr4.dat->data, distr4.distribution, dre, nlocal-1);
        insert_random_dat(distr5.dat->data, distr5.distribution, dre, nlocal-1);
        insert_random_dat(distr6.dat->data, distr6.distribution, dre, nlocal-1);
        insert_random_dat(distr7.dat->data, distr7.distribution, dre, nlocal-1);
        insert_random_dat(distr8.dat->data, distr8.distribution, dre, nlocal-1);
        insert_random_dat(distr9.dat->data, distr9.distribution, dre, nlocal-1);


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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
                                 OPS_dat_distr<Distribution1, T1> &distr1,
                                 OPS_dat_distr<Distribution2, T2> &distr2,
                                 OPS_dat_distr<Distribution3, T3> &distr3,
                                 OPS_dat_distr<Distribution4, T4> &distr4,
                                 OPS_dat_distr<Distribution5, T5> &distr5,
                                 OPS_dat_distr<Distribution6, T6> &distr6,
                                 OPS_dat_distr<Distribution7, T7> &distr7,
                                 OPS_dat_distr<Distribution8, T8> &distr8,
                                 OPS_dat_distr<Distribution9, T9> &distr9,
                                 OPS_dat_distr<Distribution10, T10> &distr10)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution11, T11> &distr11)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution12, T12> &distr12)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution13, T13> &distr13)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution14, T14> &distr14)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution15, T15> &distr15)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution16, T16> &distr16)
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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution17, T17> &distr17)

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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution18, T18> &distr18)

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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution19, T19> &distr19)

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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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
         template<typename X19> class Distribution20, typename T20>
void ops_insert_random_particles(ops_particle particle, double *region, int Nins, int nmult,
                                 int nattemps, int seed,
                                 OPSDistribution<Distribution, double> *rad_distr,
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
                                 OPS_dat_distr<Distribution20, T20> &distr20)


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

            int ineigh = binhead[address];
            while (ineigh != - 1) {
              double d = 0.0; // (xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]) *
                         //(xpoint[0] - ((double *)particle->particle_pos_dat->data)[dim * ineigh]);
              for (int isou = 0; isou < dim; isou++)
                d += (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou])
                   * (xpoint[isou] - ((double *)particle->particle_pos_dat->data)[dim * ineigh + isou]);

              double Rsq = (rad_ins + ((double *)particle->particle_envelope->data)[ineigh])
                         * (rad_ins + ((double *)particle->particle_envelope->data)[ineigh]);


              if (d < 1.1 * Rsq)
                a1 = 1; goto endline;



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

#endif /* OPS_C_INCLUDE_OPS_INSERT_RANDOM_PARTICLES_H_ */
