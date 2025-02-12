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
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
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
  * @brief OPS core library function declarations for particle treatment
  * @author Gihan Mudalige
  * @details function declarations header file for the particle library
  * utilized by all OPS backends
  */

#ifndef __OPS_PARTICLES_LIB_CORE_H
#define __OPS_PARTICLES_LIB_CORE_H

#include <string>
#include <vector>
#include <cstring>
#ifdef __unix__
#include <string.h>
#include "queue.h" //contains double linked list implementation
#include <strings.h>
#endif

#include <stdint.h>
#include <complex>
#include <random>
#include<array>

#define OPS_HALO_GRP_EXCHANGE 0
#define OPS_HALO_GRP_BORDER   1
#define OPS_HALO_GRP_FORWARD  2
#define OPS_HALO_GRP_BACKWARD 3
#define OPS_HALO_GRP_DEFAULT  4

#define OPS_PART_ORIENT_OFF 0
#define OPS_PART_ORIENT_ON  1
#define OPS_PART_POSITION   2

#define OPS_PART_LOOP_ALL   0
#define OPS_PART_LOOP_LOCAL 1
#define OPS_PART_LOOP_VIRTUAL 2

#include <ops_exceptions.h>

typedef int ops_part_orient;
typedef int ops_part_halo_grp_type;
typedef int ops_part_loop_type;


class BoundingBox;
class ops_particle_mapping_core;

//Structure that defines
struct ops_point {
    ops_point(double _x, double _y, double _z) {
        x = _x;
        y = _y;
        z = _z;
    };
    ops_point() { };

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};
class BoundingBox {
  public:
   BoundingBox(const ops_block block, int dim, ops_point minCrd,
               ops_point maxCrd);
   BoundingBox(const ops_dat coords, const double grid_size, int dim);
   BoundingBox(int dim);
   BoundingBox(int dim, double *xmin, double *xmax);
   ~BoundingBox();

   const ops_point& getLocalMin() const;
   const ops_point& getLocalMax() const;
   const ops_point& getGlobalMax() const;
   const ops_point& getGlobalMin() const;
   void getLocalMaxMin(double *xmin, double* xmax);
   bool isCoordinateInBoundingBox(const ops_point& point);
   bool isCoordinateInGlobalBoundingBox(const ops_point& point);
   void setBoundingBoxLocalBound(const ops_point &xlow, const ops_point &xmax);
   void setBoundingBoxLocalBound(const double* xlow, const double* xmax);
   void setBoundingBoxGlobalBound(const ops_point &xlow, const ops_point &xmax);
   void setBoundingBoxGlobalBound(double* xlow, double* xmax);

   void setOwnership(bool flag) {owned = flag;}
   bool getOwnership() {return owned;};
//   void setBlock(const ops_block Block, int dim = 3);
//   void setBoundingBox(ops_point xmin, ops_point xmax); //TODO
//   void setBoundingBox(ops_dat dat); //TODO
   inline int getDim() const { return dim;}
#ifdef OPS_MPI
   void generateLocalBoundingBox(/*TODO: */);
#endif
  private:
   void generateGlobalBoundingBox(int count);
   int dim = 0;
   bool owned = true;

   std::array<ops_point,2> boundingBox; /* bounding box owned by this rank */
   std::array<ops_point,2> globalBoundingBox; /* Global dimensions of box-bound */
};

/** Particle decleration lists */
//TODO: Consider smoving the remaining functionalities into a different map
class ops_particle_core {
public:
   //TODO: Is size_t the right size for developing this list
  size_t no_particles; /**<number of particles owned by the block */
  size_t no_virtual;  /* Number of virtual particles assigned to the block */
  size_t nRemoved; /* Number of particles removed from the block */
  size_t global_particles; /* Global number of particles */
  ops_block block;
  std::vector<ops_dat> particle_data;
  std::vector<ops_dat> mapping_to_grid; /**<Particle mapping to multiple grids **/

  ops_dat particle_envelope; /* Envelope of particle shape */

  std::vector<ops_dat> particle_pos_dat;

  std::vector<int> local_to_remove; /* Particles marked for removal */
  std::vector<int> local_to_exchange; /* Particles mapped for forward and reverse communications */

  std::vector<int> local_to_remove_halo; /* Particles marked for halo creation/exchange */
  std::vector<int> local_to_forward_halo; /* Particles marked for forward and reverse commns */

  int *mark_deletion;  /* Marking particles for deletion */

  size_t Nmax{100};
  size_t n_halo;  /* Virtual particles coming from halo communications */
  size_t n_halo_sb{0}; /* Virtual particles in block due to intra-block communications */

  BoundingBox *box_block; //TODO-C

  std::vector<ops_particle_mapping_core*> mapping_list;

  //TODO: Add a constructor

  //ADD also functionalities for creating a block
  size_t& noLocalParticles() { return no_particles; }
  void    setLocalParticles(size_t nparticles) { no_particles = nparticles; }

};

typedef ops_particle_core *ops_particle;

/*************************************************************************************/
/*   Particle halo data structure definitions                                        */
/*************************************************************************************/

/*-----------------------------------------------------------------------------------*/
/*  Particle ops_dat structures  inserted into halo                                  */
/*-----------------------------------------------------------------------------------*/

struct ops_particle_halo_data_core {
  ops_dat from;  /* Particle data set from which the halo reads */
  ops_dat to;    /* Particle data set to which the halo sends */
  ops_part_orient  orient;
  int from_dir[OPS_MAX_DIM]; /* Copy from orientation send to orientation */
  int to_dir[OPS_MAX_DIM];   /* Send to orientation */
  double translate[OPS_MAX_DIM]; /* Data translation to shift into local coordinate
                                    system of receiving box (Only for particle
                                    positional data */
};

typedef ops_particle_halo_data_core *ops_particle_halo_data;

struct OPS_particle_halo_exchange_info_core {
  int nsend;  // number of particles send by this halo
  int nrecv; // number of particles received by this halo


  /* Forward and backward communication data */
  int *sendlist;
  int nmax; //Maximum number of particles in send-list
  int firstrecv;

};

typedef OPS_particle_halo_exchange_info_core *ops_particle_halo_exchange;

/* Particle halo groups. Based on needs of particle (point) simulations
 * Four different halo groups need to be defined
 */
struct ops_particle_halo_core {
  int nhalos;  /* Number of halos */
  size_t nPoints;  /*Number of points (particles) to be exchanged in this halo group */
  ops_particle particle_from; /* Particle structure from which data send */
  ops_particle particle_to;   /* Particle structure to which data send */
  ops_particle_halo_data *dat;   /* Halo data list */
  OPS_instance *instance;

  BoundingBox* sendBox;
  double dx[OPS_MAX_DIM];
  int index;

  int nbites; //number of bites per particle (point) to be send

  //TODO: Add in case of reverse-mapping
};

typedef ops_particle_halo_core *ops_particle_halo;

struct ops_particle_halo_group_core {
  int nhalos; /* Nunber of particle halos */
  ops_part_halo_grp_type halo_type; /* Forward Exchange Backward type */

  ops_particle_halo_group_core *halo_master;
  ops_particle_halo *halo_list; /* Halos defined in this region */
  ops_particle_halo_exchange  *halo_info; /* Halo info for data exchange */
  ops_part_loop_type loop_type;
};

typedef ops_particle_halo_group_core *ops_particle_halo_group;

class ops_particle_mapping_core {
  public:

    enum ops_mapping_type{ actual_only = 0, with_virtual = 1, error_type = -1};
    enum ops_grid_type{ uniform_grid =0, non_uniform = 1};
    enum ops_shape_change{constant =0, change = 1};

    double skin;      /* Distance prior to list rebuild */

    ops_dat pos_old;  /* Particle positions when list is build */
    ops_dat Rp_old;        /* Particle envelope when the list is build */
    ops_dat Rp;       /* Particle envelope */
    ops_dat grid;     /* Mapping to grid */
    ops_dat binhead;       /* Data structure containing the first element in a grid cell*/
    ops_dat bin;           /* Particle joining list ptr to next particle in grid cell */
    size_t nParticles;
    ops_dat mapping_grid; /* Pointer to ops_dat structure for mapping particles */

    int     Ngrids;   /* Number of grid for building the particle-grid list */

    ops_mapping_type mapping_type; /* Mapping actual or actual-virtual particles */

    ops_grid_type grid_type; /* Grid type uniform and non-uniform grid */

    ops_shape_change particle_changes; /* Mapping function for changing size */
    bool decide{true};            /* Building or not mapping list */

    //TODO: Consider mapping on non-uniform spacing
};

typedef ops_particle_mapping_core *ops_particle_mapping;

/*--------------------------------------------------------------------------------*/
/* Define a local box based on ops_dat structures */
/*--------------------------------------------------------------------------------*/

bool  ops_get_bounding_box_local_to_global(ops_block block, double* xmin, double *xmax,
                                           double* xglb_min, double* xglb_max);

bool ops_bounding_box_global_to_local(const ops_block block, int dim, std::array<ops_point, 2>& globalBoundingBox,
                                      std::array<ops_point, 2>& boundingBox);

BoundingBox* ops_create_bounding_box(int dim);

BoundingBox* ops_create_bounding_box(const ops_block block, int dim,
                                    ops_point &point_low, ops_point &point_max);

void ops_set_bounding_box_local(BoundingBox* box, ops_block block, const ops_point &xlow,
                                const ops_point &xmax);

void ops_set_bounding_box_local(BoundingBox* box, ops_block block,
                                double* xlow, double* xmax);

void ops_set_bounding_box_global(BoundingBox* box, ops_block block, const ops_point &xlow,
                                 const ops_point &xmax);

void ops_set_bounding_box_global(BoundingBox* box, ops_block block, double*  xlow,
                                 double* xmax);

/* Set a bounding box from an ops dat structure */
void ops_set_bounding_box_from_dat(BoundingBox* box, const ops_dat coords, double dx, int dim);

BoundingBox* ops_find_send_box_projection(int idef,int dim,double *xbox1_low,
                                          double * xbox1_max, double *xbox2_min,
                                          double *xbox2_max);


/*-------------------------------------------------------------------------------------
 * Auxiliary functions for particle handling into various functions
 *------------------------------------------------------------------------------------*/

ops_arg ops_arg_dat_particle(ops_dat dat, int dim, char const *type, ops_access acc); //TODO: Add a stencil to the list

/*-------------------------------------------------------------------------------------*
 * Passes a temporary particle structure into particle loop-Used for assigning data
 * into particle structures
 *-------------------------------------------------------------------------------------*/

template<typename T>
ops_arg ops_arg_gbl_particle(T *data, int dim, char const *type, ops_access acc) {
  (void)type;
  ops_arg temp = ops_arg_gbl_char((char *)data, dim, sizeof(T), acc);
  (&temp)->argtype = OPS_ARG_DAT_PARTICLE;
  return temp;
}

/*--------------------------------------------------------------------------------------*
 *     Function definitions
 *--------------------------------------------------------------------------------------*/

OPS_FTN_INTEROP
ops_particle ops_decl_particle(ops_block  block, BoundingBox *box = nullptr);

template <class T>
ops_dat ops_decl_particle_dat(ops_particle particle, int data_size, int *base,
                              T* data, char const *type, char const* name) {

   if (particle == nullptr) {
      ops_printf("Empty ops_particle structure. Please defined particle related ops_dat after"
                 "setting particle list\n");
      exit(-1);
   }

   int block_size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], stride[OPS_MAX_DIM];

   d_m[0] = d_p[0] = 0;
   stride[0] = 1;
   block_size[0] = particle->Nmax; //TODO

   for (int i = 1; i < particle->block->dims; i++) {
     d_m[i] = 0;
     d_p[i] = 0;
     block_size[i] = 1;
     stride[i] = 1;
   }

   ops_dat particle_dat = ops_decl_dat_char(particle->block, data_size, block_size,
                                            base, d_m, d_p, stride, (char *)data,
                                            sizeof(T), type, name);
   particle_dat->is_particle = true;

   /* Add the new ops structure to list */
   particle->particle_data.push_back(particle_dat);
   return particle_dat;
}

/*------------------------------------------------------------------------------------*/
/* Setting ops_dat structure for particle envelope                                    */
/*------------------------------------------------------------------------------------*/

template<class T>
ops_dat ops_decl_particle_envelope(ops_particle particle, int *base, double *data,
                                   char const *type, char const *name) {
  if (particle == nullptr) {
    ops_printf("Empty ops_particle structure. Please define ops_particle_dat structure "
               "first.\n");
    exit(-1);
  }

  int block_size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], stride[OPS_MAX_DIM];

  d_m[0] = d_p[0] = 0;
  stride[0] = 1;
  block_size[0] = particle->Nmax; //TODO

  for (int i = 1; i < particle->block->dims; i++) {
    d_m[i] = 0;
    d_p[i] = 0;
    block_size[i] = 1;
    stride[i] = 1;
  }

  ops_dat particle_dat = ops_decl_dat_char(particle->block, 1, block_size, base,
                                           d_m, d_p, stride, (char *)data, sizeof(T),
                                           type, name);
  particle_dat->is_particle = true;

  return particle_dat;
}
template <class T>
ops_dat ops_decl_particle_pos_dat(ops_particle particle, int data_size, int* base,
                                       T* data, char const *type, char const* name)
{
  if (particle == nullptr) {
     ops_printf("Empty ops_particle structure. Please define particle related ops_dat after"
                " setting the particle structure.\n");
     exit(-1);
  }

  int dim = particle->block->dims;

  if (dim < 2 && dim > 3) {
    ops_printf("Particle ops_dat are defined only for two and three- dimensional"
               "structures\n");
    exit(-1);
  }

  /* Verify that positional data are not defined */
  if (particle->particle_pos_dat.size() ==1) {
    if (particle->particle_pos_dat.at(0)->dim == dim) {
       ops_printf("Particle position ops structure is already defined\n");
       exit(-1);

    }
  }
  else if (particle->particle_pos_dat.size() == dim) {
     ops_printf("Particle position data structures are already defined\n");
     exit(-1);
  }
  else {
     int int_dim = 0;
     for (auto x : particle->particle_pos_dat)
        int_dim += x->dim;

     if (int_dim >= dim) {
       ops_printf("Particle positions are defined. Please check your settings via "
                  " non uniform structures\n");
       exit(-1);
     }
  }

  int block_size[OPS_MAX_DIM], d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM], stride[OPS_MAX_DIM];

  d_m[0] = d_p[0] = 0;
  stride[0] = 1;
  block_size[0] = particle->Nmax; //TODO

  for (int i = 1; i < particle->block->dims; i++) {
    d_m[i] = 0;
    d_p[i] = 0;
    block_size[i] = 1;
    stride[i] = 1;
  }

  ops_dat particle_dat = ops_decl_dat_char(particle->block, data_size,block_size, base,
                                           d_m, d_p, stride, (char *)data, sizeof(T),
                                           type, name);
  particle_dat->is_particle = true;

  /* Add the new ops structure to list */
  particle->particle_pos_dat.push_back(particle_dat);
  return particle_dat;
}

/* Rellaocation data for particle ops_dat structures */
void ops_particle_realloc_data(ops_particle particle);

void ops_particles_insert(ops_particle particle, int Np);

template<typename T>
void ops_particles_insert_data(ops_particle particle, ops_dat data, T* insert, int Np) {

  T* array = ops_get_particle_data<T>(data);
  int dim = data->dim;
  int Nbase = (particle->no_particles - Np) * dim;
  /*Adding elements to the  array */
  for (int i = 0; i < Np; i++) {
    for (int idim = 0; idim < dim; idim++) {
      int loc = idim + dim * i;
      array[loc + Nbase] = insert[loc];
    }
  }
}



/*-------------------------------------------------------------------------------------*
 *  Particle halo and particle halo definitions
 *-------------------------------------------------------------------------------------*/

ops_particle_halo_data ops_particle_decl_data_halo(ops_dat from, ops_dat to,
                                                   int *dir_from,
                                                   int dir_to, double *translate,
                                                   ops_part_orient orient_flag);


ops_particle_halo ops_particle_decl_halo(ops_particle from, ops_particle to,
                                         ops_particle_halo_data particle_halos[],
                                         int  nhalos, double *crit_length = NULL);

ops_particle_halo_group ops_particle_decl_halo_group(ops_particle_halo particle_halos[],
                                                     int nhalos,
                                                     ops_part_halo_grp_type halo_type,
                                                     ops_part_loop_type loop_type,
                                                     ops_particle_halo_group master = nullptr);


void ops_particle_set_halo_group(ops_particle_halo_group halo_grp); //TODO:

/* Perform halo transfer for specific type as groups */
void ops_particle_halo_transfer_group(ops_part_halo_grp_type exchange_type);

void ops_particle_halo_transfer(ops_particle_halo_group halo_grp);
/*--------------------------------------------------------------------------------------*/
/* Neighbor build function declerations
 *--------------------------------------------------------------------------------------*/

ops_particle_mapping  ops_decl_mapping(ops_particle particle, ops_dat grid, ops_dat Rp,
                                       int uniform_grid, int include_virtual,
                                       int particle_changes, int grid_type,
                                       double epsilon,
                                       double crit_length, int Ng = 1);

int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce);

void ops_particle_list_build(ops_particle particle, ops_particle_mapping map,
                             bool enforce = false);

void  _ops_particle_build_local_uniform(ops_particle_mapping map, ops_particle particle);

void _ops_particle_build_local_non_uniform(ops_particle_mapping map, ops_particle particle);


/*----------------------------------------------------------------------------------------*/
/* Auxiliarry functions to be moved
 * ---------------------------------------------------------------------------------------*/

void _ops_particle_swap_data(char *data, int i, int j, int elems);

int _ops_particle_owned_dat(ops_particle particle, ops_dat dat); //TODO: Move

#include <ops_particle_internal.h>

#endif /* OPS_C_INCLUDE_OPS_PARTICLES_LIB_CORE_H_ */
