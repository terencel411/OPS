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
  * @author V. Tsinginos
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

#define OPS_UNIFORM_STAG 0
#define OPS_UNIFORM_COLL 1
#define OPS_NON_UNI_STAG 2
#define OPS_NON_UNI_COLL 3

#define OPS_CONST_SHAPE 0
#define OPS_EVOLV_SHAPE 1

#define OPS_WITH_VIRTUAL 0
#define OPS_NO_VIRTUAL   1

#define OPS_MAX_PART 1000


#define OPS_HISTORY_NEWTON_ON 0
#define OPS_HISTORY_NEWTON_OFF 1

#define OPS_HISTORY_UPDATE_UNIQUE 0
#define OPS_HISTORY_UPDATE_ONE_WAY 1
#define OPS_HISTORY_UPDATE_BOTH_WAYS 2

#define OPS_EXCHANGE_PARTICLE_DAT 0
#define OPS_EXCHANGE_HISTORY 1



#include <ops_exceptions.h>
//#include <ops_internal2.h>

typedef int ops_part_orient;
typedef int ops_part_halo_grp_type;
typedef int ops_part_loop_type;

/* Mapping definitions */
typedef int ops_grid_type;
typedef int ops_shape_evolve;


typedef int ops_with_virtual;


class BoundingBox;
class ops_particle_mapping_core;
class ops_neighbor_history_core;

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



/**
 * Structure for handling the simulation domain
 */
class BoundingBox {
  public:
   BoundingBox(const ops_block block, int dim, ops_point minCrd,
               ops_point maxCrd);
   BoundingBox(const ops_dat coords, const double *grid_size, int dim);
   BoundingBox(int dim);
   BoundingBox(int dim, double *xmin, double *xmax);
   ~BoundingBox();

   const ops_point getLocalMin() const;
   const ops_point getLocalMax() const;
   const ops_point getGlobalMax() const;
   const ops_point getGlobalMin() const;
   double getDx(int idir) { if (idir < dim) return dx[idir];
                            return 0.0;}
   void getLocalMaxMin(double *xmin, double* xmax);
   double getGlobalMax(int idir);
   double getGlobalMin(int idir);
   bool isCoordinateInBoundingBox(ops_point& point);
   bool isCoordinateInBoundingBox(const double *point);
   bool isCoordinateInGlobalBoundingBox(const ops_point& point);
   void setBoundingBoxLocalBound(const ops_point &xlow, const ops_point &xmax);
   void setBoundingBoxLocalBound(const double* xlow, const double* xmax);
   void setBoundingBoxGlobalBound(const ops_point &xlow, const ops_point &xmax);
   void setBoundingBoxGlobalBound(double* xlow, double* xmax);
   void setBoundingBoxLocalBound(const double *boxregion);

   void partitionBoundingBox(ops_block block, ops_dat map_bin = nullptr, double *dx = nullptr);
   double getBlockVolume() { return volume;}
   void setOwnership(bool flag) {owned = flag;}
   bool getOwnership() {return owned;};
   inline int getDim() const { return dim;}
   double getMinCoordDir(int dir);
   double getMaxCoordDir(int dir);
#ifdef OPS_MPI
   void generateLocalBoundingBox(/*TODO: */);
#endif
  private:
   void generateGlobalBoundingBox(int count);
   int dim = 0; /**< Size of the spatial space */
   bool owned = true; /**< Ownership of the BoundaryBlock on this rank */

   double volume;

   std::array<ops_point,2> boundingBox; /**< Part of the simulation box owned by the given rank */
   std::array<ops_point,2> globalBoundingBox; /**< The simulation box of the given block  */
   double dx[OPS_MAX_DIM]; /**< Expansion of the local simulation box (Linked to staggered grids) */
   ops_dat coords; /** < Pointer to an ops_dat structure  */
};

/** Particle decleration lists */
class ops_particle_core {
public:
   //TODO: Is size_t the right size for developing this list
  size_t no_particles; /**<number of particles owned by the block */
  size_t no_virtual;  /**< Number of virtual particles owned by the block */
  size_t global_particles; /* Global number of particles */

  size_t nRemoved; /* Number of particles removed from the block */

  ops_block block;

  ops_dat *particle_dat; /**< ops_dat structures related to the given particle */
  int particle_dat_index; /**< Number of particle ops_dat structures */
  int particle_dat_max; /**< Maximum number of particle_ops_dat structures */
  ops_dat particle_envelope; /**< Envelope of particle shape */

  ops_dat particle_pos_dat;
  ops_dat ids;

  int *mark_deletion;  /**< Marking particles for deletion */

  size_t Nmax{OPS_MAX_PART}; /**< Maximum number of particles */
  size_t n_halo;  /**< Depracated */
  size_t n_halo_sb{0}; /**< Depracated ?? */

  BoundingBox *box_block; /**< Pointer to an BoundingBox structure */

  ops_particle_mapping_core **map_list; /**< List of mapping structures  */
  int particle_map_index{0};  /**< Number of mapping used for this type of particle data*/
  int particle_map_max; /**< Maximum number of mapping structures allocated for the given particle */

  int index; /**< index of particle structure within global particle list */

  //ADD also functionalities for creating a block
  size_t& noLocalParticles() { return no_particles; }
  void    setLocalParticles(size_t nparticles) { no_particles = nparticles; }

  ops_neighbor_history_core **histories;  /* List of contact histories linked to particle */
  int nhistories{0};  /* Number of histories linked to the given particle type */
  int nhistories_max{5};
  char* name; /**< Name of the particle structure */

};

typedef ops_particle_core *ops_particle;


class ops_neighbor_history_core {
public:
  ops_particle particleI; /**< Pointer to the first type of particles that uses this
                               neighbor history and controls the loop */
  ops_particle particleJ; /**< Pointer to the second type of particles that uses this
                               neighbor history */

  ops_dat n_partnersI; /*< An ops_dat structure that stores the number of partners for
                           particles of type I */
  ops_dat n_partnersJ; /*< An ops_dat structure that stores the number of partners for
                           particle of type J-It can point to the same structure with I
                           or to a nullptr */

  ops_dat flagI;     /*< An ops_dat structure for reseting contacts owned by particle I*/
  ops_dat flagJ;      /*< An ops_dat structure for reseting contacts owned by particle J*/

  ops_dat partnersI; /*< An ops_dat structure that stores the current partners of particle
                         of type I */

  ops_dat partnersJ; /*< An ops_dat structure that stores the currents partners of particle
                         of type J. It can point to the same structure with I or to a nullptr
                      */

  ops_dat indexI;  /*< An ops_dat structure that stores the location of the data index */
  ops_dat indexJ;  /*< An ops_dat structure that stores the location of the data index for
                       particleI */

  ops_dat indexing; /*< An ops_dat structure that returns the index of the history of the pair
                        (i, j) in local and global terms in the form of a key */

  ops_dat flag; /*< An ops_dat structure that returns the current state of the history */

  ops_dat data;   /*< An array containing the contact history  */

  ops_dat new_neighbors; /*< An array containing temporarily the pair (i,j) for
                            creating new contacts */

  char* name; //* <Name of the particle structure */

  int history_type{0}; /*< Type of contact history */

  int num_neighsI; //* <Maximum number of contacts of particles of type I
  int num_neighsJ; //* <Maximum number of contacts of particles of type J

  int update_type{0}; /*< Flag for defining the update of neighbor history during list
                          update */

  int nmax_cont;
  int nconts;
  int nmax_new;

  int update_list{100}; /*< Number of neighbor list before merging and sorting list */
  int frequency{0}; /*< Current iterations before involing list sortage*/
  int index;

  bool flag_update;      /*< Flag for rebuilding neighbor history */

};

typedef ops_neighbor_history_core *ops_neighbor_history;


/*************************************************************************************/
/*   Particle halo data structure definitions                                        */
/*************************************************************************************/

/*-----------------------------------------------------------------------------------*/
/*  Particle ops_dat structures  inserted into halo                                  */
/*-----------------------------------------------------------------------------------*/


/*  Structure containing particle based ops_dat structures */
struct ops_particle_halo_data_core {
  ops_dat from;  /**< Particle data set from which the halo reads */
  ops_dat to;    /**< Particle data set to which the halo sends */
  ops_part_orient  orient; /**< Flag for translating and rotating and orienting data */
  int halo_type;   /**< Flag for defining a particle related ops_dat structure  or
                        a history structure */
  ops_neighbor_history history_from; /** An ops_neighbor_history for
                                         which data will be send */
  ops_neighbor_history history_to; /** An ops_neighbor history structure which will
                                       receive data */
};

typedef ops_particle_halo_data_core *ops_particle_halo_data;


/* Structure for handling exchange of particles by a given ops_particle_halo structure */
struct OPS_particle_halo_exchange_info_core {
  int nsend;  /**< number of particles send by this halo */
  int nrecv; /**< number of particles received by this halo */

  /* Forward and backward communication data */
  int *sendlist; /**< List of particles that send data in forward communications */
  int nmax; /**< Number of particles allocated in the send-list */
  int firstrecv; /**< Location of first received point in ops_dat structures */

};

typedef OPS_particle_halo_exchange_info_core *ops_particle_halo_exchange;

/* Particle halo groups-*/
struct ops_particle_halo_core {
  int nhalos;  /**< Number of ops_particle_halo_datas linked to this halo exchange */
  size_t nPoints;  /**< Number of points (particles) to be exchanged in this halo group */
  ops_particle particle_from; /**< Particle structure from which data  are send */
  ops_particle particle_to;   /**< Particle structure which receives particles */
  ops_particle_halo_data *dat;   /**< List of ops_particle_halo_data linked to the given halo*/
  OPS_instance *instance; /**< Pointer to an OPS_instance structure  */

  BoundingBox* sendBox; /**< BoundingBox for setting the region from which data are shifted to adjacent node */
  double dx[OPS_MAX_DIM]; /**< Array for expanding and shrinking the simulation box of the sending block */
  int dir_from[OPS_MAX_DIM]; /**< Direction from which data are send (Valid for vectors */
  int dir_to[OPS_MAX_DIM]; /**< Direction to which data are received */
  int index; /**< Index of the given structure */
  double translate[OPS_MAX_DIM]; /**<Vector that translates the sending block with respect to the receiving block */
  int nbites; /**<number of bites per particle (point) to be send */
  int isend[2 * OPS_MAX_DIM]; /** Containes upper to lower bound for exchange via bins */
};

typedef ops_particle_halo_core *ops_particle_halo;

/**< Halo groups for performing different types of data exchange */
struct ops_particle_halo_group_core {
  int nhalos; /**< Number of particle halos */
  ops_part_halo_grp_type halo_type; /**< Forward Exchange Backward type */
  ops_with_virtual  with_virtual; /**< Take into account virtual particles or not */
  ops_particle_halo_group_core *halo_master; /**< ops_particle_halo_group which maps
                                                   are used to exchange data (Forward/backward) */
  ops_particle_halo *halo_list; /**< Halos defined for this exchange block */
  ops_particle_halo_exchange  *halo_info; /**< Arrays for exchanging particles between blocks */
  int index;

};

typedef ops_particle_halo_group_core *ops_particle_halo_group;

class ops_particle_mapping_core {
  public:


    double skin;      /**< Length used in deciding if particle map will be rebuild */

    ops_dat pos_old;  /**<  Particle positions when list is build (Translates to bin center */
    ops_dat grid;     /**< Pointer to an ops_dat structure used to generate mappings */
    ops_dat binhead;  /**< Data structure containing the first element (particle) in a grid cell*/
    ops_dat bin;      /**< Particle joining list ptr to next particle in grid cell */
    size_t nParticles; /**< Number of particles mapped at the previous mapping stage
                            May include virtual particles as well*/
    size_t Nmax; /**< Size of binhead */
    ops_dat mapping_grid; /**< Pointer to ops_dat structure for mapping particles */
    ops_dat parts_to_grid; /**< Particles to grid */

    int     Ngrids;   /**< Number of grid for building the particle-grid list */

    /* Flags for building mapping structures */
    ops_with_virtual mapping_type; /**< Mapping actual or actual-virtual particles */
    ops_grid_type grid_type; /**< Grid type uniform and non-uniform grid */
    ops_shape_evolve particle_changes; /**< Mapping function for changing size */
    ops_stencil  mapping_stencil; /**< Stencil for grid extension in case of virtual particles */

    bool decide{true};            /**< Flag for building or not the mapping list */
    int index;                    /**< Location in the list of a given particle */

    ops_particle particle;        /**< Pointer an ops_particle_core_structure */

    double dx[OPS_MAX_DIM];       /**< Cell size for uniform mapping */
    //TODO: Consider mapping on non-uniform spacing
};

typedef ops_particle_mapping_core *ops_particle_mapping;

/*--------------------------------------------------------------------------------*/
/* Define a local box based on ops_dat structures */
/*--------------------------------------------------------------------------------*/


/**
 * Define the part of the simulation box owned by the given process when the
 * simulation box is given
 *
 * @param block  pointer to an ops_block_structure
 * @param xmin   the coordinates of the (xmin, ymin, zmin) local point
 * @param xmax   the coordinates of the (xmax, ymin, ymax) local point
 * @param xglb_min
 * @param xglb_max
 *
 * \return true (false) if block is owned
 */
bool  ops_get_bounding_box_local_to_global(ops_block block, double* xmin, double *xmax,
                                           double* xglb_min, double* xglb_max);


bool ops_bounding_box_global_to_local(const ops_block block, int dim, std::array<ops_point, 2>& globalBoundingBox,
                                      std::array<ops_point, 2>& boundingBox);

/**
 * Create a new BoundingBox object
 * @param dim  spatial size of the simulation box
 *
 * @return  a BoundingBox structure
 */

BoundingBox* ops_create_bounding_box(int dim);

/**
 * Create a new BoundingBox object based on the coordinates of two opposite corners
 *
 *@param block       pointer to an ops_block structure that this BoundingBox is linked to
 *@param dim         spatial size of the simulation box
 *@param point_low   coordinates of the lowest corner point
 *@param point_max   coordinates of the upper-right corner point
 * @return  a BoundingBox structure
 */

BoundingBox* ops_create_bounding_box(const ops_block block, int dim,
                                    ops_point &point_low, ops_point &point_max);

/**
 * Create a new BoundingBox object based on an ops_dat structure
 *
 *@param block   pointer to an ops_block structure that this BoundingBox is linked to
 *@param crds    ops_dat structure used to generate this BoundingBox
 *@param dim     size of the spatial space
 *@param dx      array for increasing the size of the owned box
 *
 * @return  a BoundingBox structure
 */
BoundingBox* ops_create_bounding_box(const ops_block block, const ops_dat crds,
                                     int dim, double *dx);


/**
 * Create a new BoundingBox based on user defined region.
 *
 * @param block   pointer to an ops_block structure that this BoundingBox is related to
 * @param dim     size of the spatial space
 * @param region  Region definition in the form [xmin, ymin, zmin, xmax, ymax, zmax]
 */

BoundingBox* ops_create_bounding_box(const ops_block block, int dim, double *region);

/**
 * Generates a bounding box for the intersection of a BoundingBox with a user
 * defined region
 *
 * @param box     a pointer to a BoundingBox
 * @param region  region in the form [xmin xmax]x[ymin ymax] x [zmin zmax]
 * @param a1      Flag for the state of overlap
 *
 * @return a new BoundingBox structure
 */

//TODO: Further testings
BoundingBox* ops_find_intersection_region(BoundingBox *box, double *region, int &a1);


/**
 * Checks the interesection of two boxes (Allows touching)
 */
int ops_check_box_intersection(int dim, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double *xbox2_hi);

int ops_check_box_intersection(int dim, int idir, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double *xbox2_hi);

int ops_check_box_intersections2(int dim, double *xbox1_lo,
                                double *xbox1_hi, double *xbox2_lo,
                                double *xbox2_hi);

/**
 * Checks the intesesection of two boxes-No overlap is allowed.
 */
int ops_check_box_intersection2(int dim, double *xbox1_lo,
                               double *xbox1_hi, double *xbox2_lo,
                               double* xbox2_hi);

void ops_build_bounding_box(ops_particle particle );


/**
 * Setup the intrablock communications for intrablock particle exchanges
 *
 * @param particle   pointer to an ops_particle_structure for which
 *                   particle intrablock comms will be build
 */
void ops_particle_setup_intrablock_comms(ops_particle particle);

ops_neighbor_history _ops_decl_neigh_history_char(ops_particle particleI,
                                                  ops_particle particleJ,
                                                  const int dofs, const int num_neighsI,
                                                  const int num_neighsJ, int update_type,
                                                  const int type_hist, char* data,
                                                  int size_elem, const char *type,
                                                  const char *name);


/*-------------------------------------------------------------------------------------
 * Auxiliary functions for particle handling into various functions
 *------------------------------------------------------------------------------------*/


/**
 * Definition of a ops_arg_dat structure related to Lagrangian points
 *
 * @param  dat        pointer to an ops_dat structure
 * @param  dim        size of the element per particle point
 * @param  type       type of element
 * @param  particle   the ops_particle structure to which this ops_dat is linked
 * @param  map        map used to handle this particle (depracated)
 * @param  acc        access type
 * @param  stencil    stencil for accessing this point (depracated)
 *
 */
ops_arg ops_arg_dat_particle(ops_dat dat, int dim, char const *type,
                             ops_particle particle, ops_particle_mapping map,
                             ops_access acc, ops_stencil stencil = nullptr); //TODO: Add a stencil to the list

/**
 * Definition of an ops_arg_dat structure for accessing the J particle in
 * an I-J particle interaction
 *
 * @param dat         pointer to an ops_dat structure
 * @param dim         size of the element per particle point
 * @param particle    the ops_particle structure to which this ops_dat is linked to
 * @param acc         access type
 */

ops_arg ops_arg_dat_particleJ(ops_dat, int dim, char const *type,
                              ops_particle particle, ops_access acc);


/**
 * Definition of an ops_arg sturcture for assessing neighbor history data within
 * particle-particle interactions
 *
 * @param history   pointer to an ops_neighbor_history structure
 * @param dim       size of the history data per contact pair
 * @param type      type of element
 * @param access    access type
 */
ops_arg ops_arg_dat_history(ops_neighbor_history history, int dim, char const *type,
                            ops_access acc);


/**
 * Passes a temporary particle structure into particle loop-Used for assigning data
 * into particle structures. Assigns a single value to each particle node
 *
 * @param data  array containing the particle data
 * @param dim   size of data per particle point
 * @param type  type of element
 * @param acc   access type
 *
 **/


template<typename T>
ops_arg ops_arg_gbl_particle(T *data, int dim, char const *type, ops_access acc) {
  (void)type;
  ops_arg temp = ops_arg_gbl_char((char *)data, dim, sizeof(T), acc);
  (&temp)->argtype = OPS_ARG_GBL_PARTICLE;
  return temp;
}

/*--------------------------------------------------------------------------------------*
 *     Function definitions
 *--------------------------------------------------------------------------------------*/


/**
 * This function generates a new ops_particle structure
 *
 * @param block  block to which the generated ops_particle structure belongs
 * @param name   name of the particle structure
 * @param box    box linked to the given particle
 *
 */
OPS_FTN_INTEROP
ops_particle ops_decl_particle(ops_block  block, char const* name, BoundingBox *box = nullptr);

/**
 * This function allocated list of ops_particles that owned by the block to which the given
 * particle belongs
 *
 * @param particle  an ops_particle structure
 */
void ops_particle_realloc_list(ops_particle particle);


/**
 * This function declaires a new ops_dat 1-D structure linked to a given ops_particle
 * structure
 *
 * @param particle   ops_particle structure to which this ops_dat structure belongs
 * @param data_size  dimensionality of data per Lagrangian point
 * @param base       base indices in 1D structure
 * @param data       input data of type @p T
 * @param type       the name of type used for output diagnostics
 *                   (e.g. "double", "float")
 * @param name       a name used for output diagnostics
 * @param assign     flag for assigning the ops_dat structure to the list of
 *                   dynamically allocated ops_dat structures
  */
template <class T>
ops_dat ops_decl_particle_dat(ops_particle particle, int data_size, int *base,
                              T* data, char const *type, char const* name, bool
                              assign = true) {

   if (particle == nullptr)
     throw OPSException(OPS_INVALID_ARGUMENT,"Empty ops_particle structure.");

//   printf("Bool flag = %d\n",assign);
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
   particle_dat->is_exchangable = true;
   /* Add the new ops structure to list */
  // particle->particle_data.push_back(particle_dat);

   if (assign) {
     particle->particle_dat_index++;
     ops_particle_realloc_list( particle);
     particle->particle_dat[particle->particle_dat_index - 1] = particle_dat;
   }
   return particle_dat;
}

/*------------------------------------------------------------------------------------*/
/* Setting ops_dat structure for particle envelope                                    */
/*------------------------------------------------------------------------------------*/

/**
 * This function sets the particle envelope structure (in term of radius)
 *
 * @param particle   ops_particle structure to which this ops_dat structure belongs
 * @param base       base indices in 1D structure
 * @param data       input data of type @p T
 * @param type       the name of type used for output diagnostics
 *                   (e.g. "double", "float")
 * @param name       a name used for output diagnostics
 *
 */
template<class T>
ops_dat ops_decl_particle_envelope(ops_particle particle, int *base, T *data,
                                   char const *type, char const *name) {
  if (particle == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Particle ops_dat structure must be defined before"
                                             "ops_particle definition");

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
  particle_dat->is_exchangable = true;
  particle->particle_envelope = particle_dat;
  return particle_dat;
}

/**
 * This function declaires a new ops_dat 1-D structure which contains the coordinates
 * of each particle
 *
 * @param particle   ops_particle structure to which this ops_dat structure belongs
 * @param data_size  dimensionality of data per Lagrangian point
 * @param base       base indices in 1D structure
 * @param data       input data of type @p T
 * @param type       the name of type used for output diagnostics
 *                   (e.g. "double", "float")
 * @param name       a name used for output diagnostics
  */
template <class T>
ops_dat ops_decl_particle_pos_dat(ops_particle particle, int data_size, int* base,
                                       T* data, char const *type, char const* name)
{
  if (particle == nullptr) {
    throw OPSException(OPS_INVALID_ARGUMENT,"Empty ops_particle structure\n");
  }

  int dim = particle->block->dims;

  if (dim != data_size)
    throw OPSException(OPS_INVALID_ARGUMENT, "Size of particle coordinates must be equal"
                                             "to the size of the physical space");

  if (dim < 2 && dim > 3) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Particle ops_dat structures defined "
                                              "for two or three-dimensional spaces\n");

  }

 if (particle->particle_pos_dat != NULL)
   throw OPSException(OPS_INVALID_ARGUMENT, "Particle position ops dat sturcture is already"
                                            "defined\n");

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
  particle_dat->is_exchangable = true;

  /* Add the new ops structure to list */
  particle->particle_pos_dat = particle_dat;
  return particle_dat;
}

/**
 * This function declaires a new ops_dat 1-D structure which contains the ids
 * of each particle
 *
 * @param particle   ops_particle structure to which this ops_dat structure belongs
 * @param base       base indices in 1D structure
 * @param data       input data of type @p T
 * @param type       the name of type used for output diagnostics
 *                   (e.g. "double", "float")
 * @param name       a name used for output diagnostics
  */
template<typename T>
ops_dat ops_particle_decl_id_dat(ops_particle particle, int *base, T *data, char *type,
                                 char const *name) {

  if (particle == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT,"Empty ops_particle structure\n");



 if (particle->ids != NULL)
   throw OPSException(OPS_INVALID_ARGUMENT, "Particle id ops dat sturcture is already"
                                            "defined\n");

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

  ops_dat particle_ids = ops_decl_dat_char(particle->block, 1, block_size, base,
                                           d_m, d_p, stride, (char *)data, sizeof(T),
                                           type, name);

  particle_ids->is_particle = true;
  particle_ids->is_exchangable = true;
  particle->ids = particle_ids;
  return particle_ids;

}

/** This function declaires an new ops_neighbor_history structure
 *
 * @param particleI        an ops_particle structure that defines particles of type I
 * @param particleJ        an ops_particle structure that defines particles of type J
 * @param dofs             the degrees of freedom of neighbor history
 * @param no_neighborsI    maximum number of no_neighbors for particle type I
 * @oaran no_neighborsJ    maximum number of no_neighbors for particle type J
 * @param update_type      flag for updating the history
 *                         0: History is updated by the single particle
 *                         1: Neighbor history for two different particle types but
 *                            history is updated only via particle I
 *                         2: Neighbor history for two different particles history is updated
 *                            via both particles
 * @param type_history     flag for history type
 *                         0: Both pairs (i,j) point to the same neighbor history points
 *                         1: Both pairs (i, j) points to different neighbor history points
 * @param array            input data of type @p T
 * @param type             the name of type used for output diagnostics
 *                         (e.g. "double", "float")
 * @param name             a name used for output diagnostics
 */

template<typename T>
ops_neighbor_history  ops_decl_neigh_history(ops_particle particleI,
                                             ops_particle particleJ,
                                             int dofs, int no_neighborsI,
                                             int no_neighborsJ, int update_type,
                                             int type_history, T* array,
                                             const char* type, const char* name) {

  return _ops_decl_neigh_history_char(particleI, particleJ, dofs, no_neighborsI,
                                      no_neighborsJ, update_type, type_history,
                                      (char *) array, sizeof(T), type, name);
}

/** This function declaires an new ops_neighbor_history structure for a single particle
 *  type. The type of contact histroy is automatically set to 0.
 *
 * @param particleI        an ops_particle structure that defines particles of type I
 * @param dofs             the degrees of freedom of neighbor history
 * @param no_neighbors     maximum number of no_neighbors for each particle
 * @param array            input data of type @p T
 * @param type             the name of type used for output diagnostics
 *                         (e.g. "double", "float")
 * @param name             a name used for output diagnostics
 */

template<typename T>
ops_neighbor_history ops_decl_neigh_history(ops_particle particleI, int dofs,
                                            int no_neighbors,
                                            int type_hist, T* array, const char* type,
                                            const char* name) {

  int update_type = OPS_HISTORY_UPDATE_UNIQUE;
  return _ops_decl_neigh_history_char(particleI, particleI, dofs, no_neighbors,
                                      no_neighbors, update_type,
                                      type_hist, (char *) array,
                                      sizeof(T), type, name);
}


/**
 * Reallocation of particle ops_dat structures
 * @param particle  ops_particle structure for reallocating
 * @param no_alloc  number of particles added to the list
 */
void ops_particle_realloc_data(ops_particle particle, int no_alloc = 0);


/**
 * This function operates on a given element of the destination array
 *
 * @param dest     pointer to the destination array
 * @param src      pointer to the source array
 * @param dim      dimensionality of the point
 * @param access   Type of operation on the given element
 */
template<typename T>
void  _ops_inc_element(T* dest,  T* src, int dim, ops_access access) {

  for (int i = 0; i < dim; i++) {
    switch (access) {
    case OPS_WRITE:
    case OPS_RW:
      dest[i] = src[i];
      break;
    case OPS_INC:
     dest[i] += src[i];
     break;
    case OPS_MAX:
     dest[i] = MAX(src[i], dest[i]);
     break;
    case OPS_MIN:
     dest[i] = MIN(src[i], dest[i]);
     break;
    default:
      throw OPSException(OPS_RUNTIME_ERROR, "Invalid reduction type");
    }
  }
}

/**
 * This functions marks local particles for deletion
 *
 * @param particle ops_particle structure for which physical elements
 *                 will be marked for deletion
 */
void ops_particle_mark_for_del(ops_particle particle);

void  ops_particle_rearrange_particles_for_removal(ops_particle particle);


/**
 * This function resets the maps for virtual particles
 *
 * @param particle   Particle structure for which particle maps are reset
 */
void  ops_particle_reset_virtual_particles(ops_particle particle);

/**
 * This function setup the particle partition
 */
void ops_particle_setup_partition();

/**
 * This function reset the flags for rebuilding particle maps
 *
 * @param  particle   ops_particle_dat structure for rebuilding mapping structures
 * @param  decide     boolean for enforcing mapping rebuild
 */
void ops_particle_reset_flags(ops_particle particle, bool decide);

/**
 * This function resets history flags to false. TODO: Need to check its needs
 *
 * @param history     an array of ops_neighbor history structures
 * @param nhistories  the size of the history array
 */

void ops_particle_reset_history_flags(ops_neighbor_history *history,
                                      int nhistories);
/*-------------------------------------------------------------------------------------*
 *  Particle halo and particle halo definitions
 *-------------------------------------------------------------------------------------*/

/**
 * This function declaires an ops_particle_halo_data structure
 *
 * @param from         the ops_dat structure which sends data during exchange
 * @param to           the ops_dat structure which receives data during exchange
 * @param orient_flag  Flag for rotating and translating the data during exchange
 */
ops_particle_halo_data ops_particle_decl_data_halo(ops_dat from, ops_dat to,
                                                   ops_part_orient orient_flag);


/**
 * This function declaires an ops_particle_halo_data that used to exchange
 * neighbor history data
 *
 * @param from        the ops_neighbor_history which sends data in the exchange
 * @param to          the ops_neighbor_history that receives data in the exchange
 */
ops_particle_halo_data ops_particle_decl_history_halo(ops_neighbor_history from,
                                                      ops_neighbor_history to);


/**
 * This function generates a new particle_halo
 *
 * @param from            the ops_dat structure which sends data in this exchange
 * @param to              the ops_dat structure which receives data in this exchange
 * @param particle_halos  list of ops_particle_halo_data exchanged by this halo
 * @param nhalos          number of ops_particle_halo_data exchanged by this halo
 * @param critical_length array for changing the Box of each block structure
 * @param dir_from        direction from which data are received
 * @param dir_to          direction to which data are send
 * @param translate       array for translating the simulation box of the sending block
 */
ops_particle_halo ops_particle_decl_halo(ops_particle from, ops_particle to,
                                         ops_particle_halo_data particle_halos[],
                                         int  nhalos, double* critical_length,
                                         int *dir_from, int *dir_to,
                                         double *translate);

//TODO: In the second version, we set directly the sending_region in the form
//      [xmin xmax] x [ymin ymax] x [zmin zmax] - In this case we need to check if the
//      user defined send-receive region intersects the actual region.

ops_particle_halo ops_particle_decl_halo(ops_particle from, ops_particle to,
                                         int nhalos, ops_particle_halo_data particle_halos[],
                                         double sending_region[],
                                         int *dir_from, int *dir_to,
                                         double *translate);


/* This testing function adds automatically a ops_particle_halo structure for the
 * particle positions
 */
ops_particle_halo ops_particle_decl_halo_with_pos(ops_particle from, ops_particle to,
                                                  ops_particle_halo_data particle_halos[],
                                                  int nhalos, double critical_length[],
                                                  int *dir_from, int *dir_to,
                                                  double *translate);


/**
 * Decleration of particle halo group
 *
 * @param particle_halos   List of particle_halos which exchange will performed together
 * @param nhalos           number of ops_particle_halo s that are part of this group
 * @param halo_type        Type of halo exchange (Exchange, border, forward, reverse)
 * @param with_virtual     Flag for inclusion or not of virtual particles in halo
 *                         exchanges
 * @param master           Identification of a halo group that used for the definition
 *                         of data exchange
 *
 */

ops_particle_halo_group ops_particle_decl_halo_group(ops_particle_halo particle_halos[],
                                                     int nhalos,
                                                     ops_part_halo_grp_type halo_type,
                                                     ops_with_virtual with_virtual,
                                                     ops_particle_halo_group master = nullptr);


/**
 * Setup an ops_particle_halo_group. This function must be called before exchanging particle
 * data for the first time.
 *
 * @param halo_grp  Particle halo group
 */
void ops_particle_set_halo_group(ops_particle_halo_group halo_grp); //TODO:

/**
 *  Perform halo transfer for specific type as groups
 *
 * @param exchange_type Particle halo types for which data are exchanged
 * @param exchange      Flag for exchanging data
 */

void ops_particle_halo_transfer_group(ops_part_halo_grp_type exchange_type,
                                      ops_access access, bool exchange = false);

/**
 * Perform halo exchange for a group type. In contrast to traditional particle-halo
 * algorithm, this function exploits a grid-based approach for generating particle-halos
 * for border, forward & backward communications. The function also build the part of
 * the map associated virtual particles.
 *
 * @param exchange_type    an ops_part_halo_grp_type for defining the exchange type
 * @param exchange         exchange flag for defining if a exchange of border type
 *                         communication will be performed.
 */


void ops_particle_halo_transfer_group_map(ops_part_halo_grp_type exchange_type,
                                             ops_access access=OPS_WRITE,
                                             bool exchange = true);

/*--------------------------------------------------------------------------------------*/
/* Neighbor build function declerations
 *--------------------------------------------------------------------------------------*/

/**
 * This function declaires a particle mapping structure
 *
 * @param particle           ops_particle structure that used to map particles
 * @param grid               ops_dat structure that used to map particles upon
 * @param Rp                 an ops_dat structure linked to particle envelope (radius)
 * @param stencil            stencil that used to expand the simulation box
 * @param include_virtual    flag for mapping or not the virtual particles
 * @param particle_changes   flag for particle changing its structure (depracated)
 * @param grid_type          type of grid (uniform or non-uniform)
 * @param skin               critical length for rebuilding the given map
 * @param Ng                 number of grids for mapping particles at non-uniform
 *                           structure grids
 */

ops_particle_mapping  ops_decl_mapping(ops_particle particle, ops_dat grid, ops_dat Rp,
                                       ops_stencil   stencil,
                                       ops_with_virtual include_virtual,
                                       ops_shape_evolve particle_changes,
                                       ops_grid_type grid_type,
                                       double skin, int Ng = 1);


/**
 * This function declaires a mapping structure that is coarser than the associated
 * grid structure
 *
 * @param particle           ops_particle structure that used to map particles
 * @param grid               ops_dat structure that used to map particles upon
 * @param stencil            stencil that used to expand the simulation box
 * @param stride             array containing the ratio of finer to coarser grid sizes
 * @param include_virtual    flag for mapping or not the virtual particles
 * @param particle_changes   flag for particle changing its structure (depracated)
 * @param grid_type          type of grid (uniform or non-uniform)
 * @param skin               critical length for rebuilding the given map
 * @param Ng                 number of grids for mapping particles at non-uniform
 *                           structure grids
 *
 */

ops_particle_mapping ops_decl_mapping(ops_particle particle, ops_dat grid,
                                      ops_stencil stencil, int stride[] ,
                                      ops_with_virtual include_virtual,
                                      ops_shape_evolve particle_changes,
                                      ops_grid_type grid_type,
                                      double skin, int Ng);

ops_particle_mapping ops_decl_mapping(ops_particle particle, double *skin,
                                      ops_with_virtual include_virtual,
                                      int *d_m = nullptr, int *d_p = nullptr);

/**
 * Initialize the maps linked to a given particle structure
 *
 * @param particle  ops_particle structure for which all maps are initialized
 */
void ops_particle_init_maps(ops_particle particle);


/**
 * Initialize histories
 *
 * @param history  an ops_neighbor history structure
 */
void ops_particle_init_histories(ops_neighbor_history history);



int _ops_particle_mapping_decide(ops_particle_mapping map, ops_particle particle,
                                 bool enforce);

/**
 * This function checks if a map structure must be rebuild
 *
 * @param particle   particle structure linked to map
 * @param map        the ops_particle_mapping that checked if it is required to be rebuild
 * @param enforce    Flag that enforces the build of the given map
 */
void ops_particle_map_decide(ops_particle particle, ops_particle_mapping map,
                             bool enforce = false); //OK

/**
 * This function builds a given ops_particle_mapping structure
 *
 * @param particle   particle structure linked to map
 * @param map        the ops_particle_mapping that is rebuild
 */
void ops_particle_map_build(ops_particle particle, ops_particle_mapping map);



/*----------------------------------------------------------------------------------------*/
/* Functions for deciding and building maps for all particle mappings to uniform grids    */
/*----------------------------------------------------------------------------------------*/


bool ops_particle_update_map_lists(ops_particle particle);


/**
 * This function updates map_structures. In practice the function updates the maps for actual
 * & virtual particles.
 */
int ops_particle_update_map_lists_actual_parts(ops_particle particle);

int ops_particle_update_map_lists_actual_hybrid(ops_particle particle);

void ops_particle_remove_particles(ops_particle particle, bool flag = true);

void ops_particle_remove_delete_maps(ops_particle particle, int decide);

void ops_particle_build_maps(ops_particle particle, bool decide_global = true);

/**
 * This function builds and performs a forward communication for a given set
 * of ops_dat structures amd ops_neighbor_histories
 *
 * @param particle   Pointer to an ops_particle structure
 * @param dats       An array of ops_dat structures
 * @param ndats      The number of ops_dat structures
 * @param histories  An array of ops_neighbor_history
 * @param nhis       Number of ops_neighbor_history structures
 */

void ops_particle_intrablock_border(ops_particle particle, ops_dat *dats, int ndats,
                                    ops_neighbor_history *histories = nullptr,
                                    int nhistories = 0);

/**
 * This function builds and performs a forward communication for a given set
 * of ops_dat structures. Finally it generates all map structures for a given
 * set of particle maps.
 *
 * @param particle   Pointer to an ops_particle structure
 * @param dats       An array of ops_dat structures
 * @param ndats      The number of ops_dat structures
 * @param histories  An  array of ops_neighbor_history structures
 * @param nhistories Number of ops_neighbor_history structures
 */

void ops_particle_intrablock_border_map_update(ops_particle particle, ops_dat *dats,
                                               int ndats, ops_neighbor_history *histories = nullptr,
                                               int nhistories = 0);


/**
 * This function performs forward communication for a given set of ops_dat
 * structures
 *
 * @param particle Pointer to an ops_particle structure
 * @param dats     An array of ops_dat structures
 * @param ndats    The number of ops_dat structures
 */

void ops_particle_intrablock_forward(ops_particle particle, ops_dat *dats,
                                     int ndats);

/**
 * This function forward communication for a given set
 * of ops_dat structures. It generates all map structures for the associated
 * ops_particle structure
 *
 * @param particle Pointer to an ops_particle structure
 * @param dats     An array of ops_dat structures
 * @param ndats    The number of ops_dat structures
 */

void ops_particle_intrablock_forward_map_update(ops_particle particle, ops_dat *dats,
                                                int ndats);


/**
 * Function to perform intrablock communications for a given particle structure
 *
 * @param particle   pointer to an ops_particle structure
 * @param dats       an array of ops_dat structures linked to particle
 * @param ndats      the size of the dats array
 * @param access     type of operation for the reverse communication. Currently
 *                   supports the operations OPS_WRITE, OPS_INC, OPS_MIN and OPS_MAX
 */

void ops_particle_intrablock_reverse(ops_particle particle, ops_dat *dats, int ndats,
                                     ops_access access);

/**
 * This functions creates all maps related to the ops_particle structure, particle. This
 * function must be called prior to any time marching scheme. Initialize mapping structures,
 * mapping of actual particles and maps actual particles into grids
 *
 * @param particle   An ops_particle structure for which maps are build
 *
 */
void ops_particle_setup_map_grid(ops_particle particle);


/**
 * This function create the maps linked to the define ops_particle particle structure for
 * the first time. It must be called before any time marching scheme. The function
 * initializes mapping structures, generate maps for particles and perform intra-block
 * particle data exchange for the user define ops_dat structures. The function relies
 * on the new scheme for data exchange via maps.
 *
 * @param particle   Pointer to an ops_particle structure
 * @param dats       Array of ops_dat structures for intra-block communications
 * @param ndats      Number of ops_dat structures
 * @param histories  Array of ops_neighbor_history data
 * @param nhistories Number of history data
 */

void ops_particle_setup_map_grid_dats(ops_particle particle, ops_dat *dats,
                                      int ndats, ops_neighbor_history *histories,
                                      int nhistories);
/**
 * Function that creates all maps for an ops_particle structure. The main difference
 * with the ops_particle_setup_map_grid is that different halo exchange functionality
 * is used.
 */

void ops_particle_setup_map(ops_particle particle);

/**
 * Functions that creeate all maps for an ops_particle structure. The function
 * perform intra-block exchange for a given set of dat structure
 *
  *@param particle   Pointer to an ops_particle structure
 * @param dats       Array of ops_dat structures for intra-block communications
 * @param ndats      Number of ops_dat structures
 * @param histories  Array of ops_neighbor_history structures
 * @param nhistories Number of ops_neighbor_history structures
 */

void ops_particle_setup_maps_with_dats(ops_particle particle, ops_dat *dats, int ndats,
                                       ops_neighbor_history *histories = nullptr,
                                       int nhistories = 0);

/**
 * This functions creates all maps for particle (ops_particle structure). It must be
 * called after setting the maps for the actual particles due to inter-block halos
 *
 * @param particle An ops_particle structure for which maps will be build.
 */
void ops_particle_setup_virtual_particles(ops_particle particle);

bool ops_particle_global_rebuild(bool flag);


void ops_particle_map_get_dx(ops_particle_mapping map, double dx[]);


/*----------------------------------------------------------------------------------------*
 *     Function for writing particle data to txt files
 *----------------------------------------------------------------------------------------*/

void ops_particle_print_data_to_txtfile(ops_particle particle, const char *file_name);

void ops_particle_print_dats_to_txtfile(ops_particle particle, ops_dat *dats,
                                        int ndats,const char *file_name);



/**
 * Returns an integer that has the index of the currently accessed particle, i.e, idp[0]
 * is the current (local) index.
 *
 * @return
 */

OPS_FTN_INTEROP
ops_arg ops_arg_idp();

OPS_FTN_INTEROP
ops_arg ops_arg_idx_map();

/*----------------------------------------------------------------------------------------*/
/* Auxiliarry functions to be moved
 * ---------------------------------------------------------------------------------------*/

void _ops_particle_swap_data(char *data, int i, int j, int elems);

int _ops_particle_owned_dat(ops_particle particle, ops_dat dat); //TODO: Move


#include <ops_particle_internal.h>

#endif /* OPS_C_INCLUDE_OPS_PARTICLES_LIB_CORE_H_ */
