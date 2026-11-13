/*
 * ops_particle_seq.h
 *
 *  Created on: Feb 19, 2025
 *      Author: Valantis Tsinginos
 */

#ifndef __OPS_PARTICLE_SEQ_H
#define __OPS_PARTICLE_SEQ_H

#ifndef OPS_API
#define OPS_API 2
#endif

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#if __cplusplus >= 201103L

#include "ops_seq_v2.h"
#include "ops_exceptions.h"

inline int multipl(const int size[], int dir) {
  int prod{1};



  for (int i = 0; i <= dir; i++) {
    prod *= size[i];
  }

  return prod;
}

template<typename T>
inline int  _point_in_range_v1(const T *range, T *coords, const int dim) {

  for (int is = 0; is < dim; is++) {
    if (coords[is] < range[2 * is] || coords[is] > range[2 * is + 1]) return 0;
  }

  return 1;
}


inline int particle_in_group_ps(ops_particle particle, int i, ops_particle_group group) {
  if (particle->index != group->particle_index)
    throw OPSException(OPS_RUNTIME_ERROR,"Error: The given ops_particle_group not "
                                         "linked to the given ops_particle_structure");

  if (((int *) group->group_dat->data)[i] != 0) return 1;

  return 0;
}

inline int get_address(int loc[], const int mdim, const int dim, const int d_m[],
                const int size[]) {

#ifdef OPS_SOA
  int size0 = 1;
#else
  int size0 = mdim;
#endif

  int sizeL[dim + 1];
  sizeL[0] = size0;
  for (int i = 1; i < dim + 1; i++)
    sizeL[i] = size[i-1];


  int address{0};
  for (int i = 0; i < dim; i++) {
    address += (-d_m[i] + loc[i]) * multipl(sizeL, i);
  }

  return address;
}


inline void get_local_point(const int point,const int size[],
                            const int d_m[],const int dim, int grid[]) {

  int address = point;
  for (int i = dim - 1; i >= 0; i--) {
      int prod = 1;
      for (int j = 0; j < i; j++)
        prod *= size[j];

      grid[i] = address / prod;
      address -= grid[i] * prod;
      grid[i] += d_m[i];
  }
}

inline void   init_off_grids(ops_arg &arg, const int dim, const int iPart,
                             int &first_point, ops_particle_mapping map) {

  if (arg.argtype == OPS_ARG_DAT) {
    int *part2grid = (int *) map->parts_to_grid->data;
    int ifirst = part2grid[0];
    int local_point[OPS_MAX_DIM] = {};
    int d_mb[OPS_MAX_DIM] = {};
    int d_m[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
  for (int d = 0; d < dim; d++){
    d_mb[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
    d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];}
#else
  for (int d = 0; d < dim; d++) {
    d_mb[d] =  map->binhead->d_m[d];
    d_m[d] = arg.dat->d_m[d];
  }
#endif

  int address_0 = part2grid[iPart];
  int loc_index[OPS_MAX_DIM] = {};

  get_local_point(address_0, map->binhead->size,map->binhead->d_m, dim, loc_index);

  first_point  = get_address(loc_index, 1, dim, d_m, arg.dat->size);

  }

}

inline void compute_offsets(ops_arg &arg,const int dim, const int iP, const int ip_last,
                            ops_particle_mapping map, int  &prev_address, int &offset,
                            OPS_instance  *instance) {
  if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
    offset = iP - ip_last;
  }
  else if (arg.argtype == OPS_ARG_DAT) {
    int *part2grid = (int *) map->parts_to_grid->data;

    int d_mb[OPS_MAX_DIM] = {};
    int d_m[OPS_MAX_DIM] = {};

#ifdef OPS_MPI
    for (int d = 0; d < dim; d++) {
      d_mb[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
      d_m[d] = map->binhead->d_m[d];
    }
#else
    for (int d = 0; d < dim; d++) {
      d_m[d] = arg.dat->d_m[d];
      d_mb[d] = map->binhead->d_m[d];
    }
#endif

    //Get point on local map grid
    int cur_address = part2grid[iP];
    int loc_point[OPS_MAX_DIM] = {};
    get_local_point(cur_address, map->binhead->size,map->binhead->d_m, dim, loc_point);


    cur_address = get_address(loc_point, 1, dim, d_m, arg.dat->size);

    offset = (cur_address - prev_address) * ((arg.dat->block->instance->OPS_soa) ? 1 : arg.dat->dim);

    prev_address = cur_address;

  }
  else if (arg.argtype == OPS_ARG_IDX) {
     int *part2grid = (int *) map->parts_to_grid->data;

     int d_mb[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
    for (int d = 0; d < dim; d++) {
      d_mb[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
    }
#else
    for (int d = 0; d < dim; d++) {
      d_mb[d] = arg.dat->d_m[d];
    }
#endif

    int cur_address = part2grid[iP];
    int loc_point[OPS_MAX_DIM] ={};
    get_local_point(cur_address, map->binhead->size, map->binhead->d_m, dim, loc_point);

    for (int d = 0; d < dim; d++)
      instance->arg_idx[d] = loc_point[d];
    for (int d = dim; d < OPS_MAX_DIM; d++)
      instance->arg_idx[d] = 0;
  }
  else if (arg.argtype == OPS_ARG_IDP) {
    instance->arg_idp[0] = iP;
  }
}

/*------------------------------------------------------------------------------------*/
/* \struct particle_param_handler
 *         Handles data for particle structures
 *
 * \param[in] arg          pointer to an ops_arg structure
 * \param[in] dim          size of the spatial space
 * \param[in] ndim         size of particle structure
 * \param[in] map          a mapping structure related to the given particle data
 * \param[in] n_particles  number of particles to loop over
 * \param[in] n_particles
 *
 */


template<typename ParamT> struct particle_param_handler {
  static char* construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, size_t n_particles) {

    if (arg.argtype == OPS_ARG_GBL || arg.argtype == OPS_ARG_GBL_PARTICLE) {
      if (arg.acc == OPS_READ) return arg.data;
      else
#ifdef OPS_MPI
        return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * block->index;
#else
        return ((ops_reduction)arg.data)->data;
#endif
    }
    else if (arg.argtype == OPS_ARG_IDX) {
      int d_m[OPS_MAX_DIM];
#ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
#else
      for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d];
#endif

      int *part2grid = (int *)map->parts_to_grid->data;
      int address = part2grid[0]; //First pount

      int map_loc[OPS_MAX_DIM] = {};

      get_local_point(address, map->binhead->size,map->binhead->d_m, dim,map_loc);

#ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[block->index]; //TODO: Multigrid
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d]
             = OPS_sub_dat_list[map->binhead->index]->decomp_disp[d] + map_loc[d];
#else //OPS_MPI
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = map_loc[d];
#endif

      return (char *)block->instance->arg_idx;
    }
    else if (arg.argtype == OPS_ARG_IDP) {
      block->instance->arg_idp[0] = 0;

      return (char *) block->instance->arg_idp;
    }

    return nullptr;
  }

  static ParamT get(char *data) { //TODO: Shift it for satefyt
    return (ParamT)data;
  }

  static void shift_arg(const ops_arg &arg, char *p, const int offs,
                        OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_IDP) {
      instance->arg_idp[0] += offs;
    }
    else if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      p += offs * arg.elem_size;
    }
  }

  static void free(char *) { }

};

template <typename T>struct particle_param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, size_t n_particles) {
    if (arg.argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM] = {};
      int d_mb[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
    for (int d = 0; d < dim; d++)  {d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
                                    d_mb[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];}
#else //OPS_MPI
    for (int d = 0; d < dim; d++) {d_m[d] = arg.dat->d_m[d];
                                   d_mb[d] = map->binhead->d_m[d];
    }
#endif

    int start[OPS_MAX_DIM];
    for (int i = 0; i < dim; i++) start[i] = 0;

    int *part2bin = (int *) map->parts_to_grid->data;
    int addressx = part2bin[0];
    get_local_point(addressx, map->binhead->size, d_mb, dim, start);


#ifdef OPS_1D
    return (char *) new ACC<T>(arg.dim, arg.dat->size[0], (T*)(arg.data //base of 2D array
#elif defined(OPS_2D)
    return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], (T*)(arg.data //base of 2D array
#elif defined(OPS_3D)
    return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2], (T*)(arg.data //base of 3D array
#elif defined(OPS_4D)
    return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2], arg.dat->size[3], (T*)(arg.data //base of 3D array
#else
      return (char *) ((arg.dat->data //TODO
#endif
      + address(ndim, arg.dat->block->instance->OPS_soa ? arg.dat->type_size : arg.dat->elem_size, &start[0],
        arg.dat->size, arg.stencil->stride, arg.dat->base,
        d_m))); //TODO

    }

    return nullptr;
  }

  static ACC<T>& get(char *data) { return *((ACC<T> *) data);}

  //offset contains the calculated shift
  static void shift_arg(const ops_arg &arg, char *p,  const int offset,
                        OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT) {
      ((ACC<T>*)p)->next(offset);
    }
  }

  static void free(char *data) { delete (ACC<T> *)data; }
};

template <typename T>struct particle_param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, size_t n_particles) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      return (char *) new ACCP<T>(arg.dim, n_particles, (T *)arg.dat->data);
    }

    return nullptr;
  }

  static ACCP<T>& get(char *data) { return *((ACCP<T> *)data); }

  static void shift_arg(const ops_arg &arg, char *p, const int offs,
                        OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {

      int offset = offs * arg.dat->dim;
      ((ACCP<T>*)p)->next(offset);
    }
  }

  static void free(char *data) { delete (ACCP<T> *)data;}

};

template<typename T>
struct param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[], ops_block) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      return (char *) new ACCP<T>(arg.dim, arg.dat->size[0], (T *)arg.dat->data); //TODO: Need to pass size into
      //acc structure
    }

    return nullptr;
  }

  static ACCP<T>& get(char *data) {return *((ACCP<T> *)data);}

  static void shift_arg(const ops_arg &arg, char *p, int m, const int* start,
                        const int offs[], OPS_instance *instance) {
    //TODO See what we need here

  }

  static void free(char *data) { delete (ACCP<T> *)data;}

};

/*template<typename... ParamType, typename... OPSARG, size_t... J>
void ops_particle_par_loop_impl(indices<J...>, void (*kernel)(ParamType...),
                                char const *name, ops_particle particle,
                                int dim, double *range_particles, OPSARG... arguments)
{
  if (particle->no_particles == 0) return;
  constexpr int N = sizeof...(OPSARG);

  int count[OPS_MAX_DIM]={0};

  //Perform a sanity check if ops_arg

}
*/

template <typename T, typename... ParamType, typename... OPSARG, size_t... J>
void ops_particle_par_loop_impl(indices<J...>, void (*kernel)(ParamType...),
                                char const *name, ops_particle particle,
                                ops_particle_group group,
                                int dim, ops_particle_iterate_type iterate_type,
                                T *range_particles, ops_particle_mapping map,
                                OPSARG... arguments)
{
  constexpr int N = sizeof...(OPSARG);

  if (sizeof(T) != particle->particle_pos_dat->type_size)
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Inconsistent types for"
                                         " particle positions and simulation"
                                         " region" );

//  int count[OPS_MAX_DIM] = {0};

  /* In case of empty set default list */ //TODO:
  ops_particle_mapping map_part
     = (map != nullptr) ? map : nullptr; //particle->mapping_list[particle->def_list];
  ops_block block = particle->block;
//  size_t Nlist{0}; //, Nmax{OPS_MAX_PART};


  ops_arg args[N] = {arguments...};

  size_t no_particles = particle->no_particles + particle->no_virtual;




  /* Get range of elements */
  BoundingBox<T> *box = (BoundingBox<T> *) particle->box_block;


  /* First identify intersection region */

  //Init offsets to the points

  size_t n_loop_particles = 0;
  /* looping_particles is set to it self */
  switch (iterate_type) {
  case OPS_PARTICLE_ITERATE_LOCAL:
    n_loop_particles = particle->no_particles;
    break;
  case OPS_PARTICLE_ITERATE_ALL:
  case OPS_PARTICLE_ITERATE_RANDOM: //TODO: We need to expand the region to virtual
    n_loop_particles = particle->no_particles + particle->no_virtual;
    break;
  default:
    throw OPSException(OPS_RUNTIME_ERROR,"Invalid iteration type. Please check your "
                        "settings");

  }

  if (n_loop_particles == 0) return;

  int prev_grid_point[N] = {};
  std::initializer_list<int>{(init_off_grids(arguments, dim, 0, prev_grid_point[J],
                             map), 0)...};

  /* Set start for grid points */
  int ndim = particle->block->dims;


  //  int start[OPS_MAX_DIM];
  //for (int n = 0; n < particle->block->dims; n++)  start[n] = 0;

  char *p_a[N] =
   {particle_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim,
                                                                       ndim, block,
                                                                       map_part, no_particles)...};//TODO: SAnity check

  /* Loop over all particles */

  int firstPart = 0; //Initialize to zero
  int offset[N] = {}; //offsets with respect to previous point
  T *coords = (T *) particle->particle_pos_dat->data;
  for (size_t iPart = 0; iPart < n_loop_particles; iPart++) {

    if (group != nullptr)
      if (!particle_in_group_ps(particle, iPart, group)) continue;

    if (iterate_type == OPS_PARTICLE_ITERATE_RANDOM) {
      int point_within= (range_particles != nullptr) ?
         _point_in_range_v1(range_particles, coords + dim * iPart, dim) : 1;

      if (!point_within) continue;
    }

    int curPart = iPart;

    (void) std::initializer_list<int>{(compute_offsets(arguments, dim, curPart, firstPart,
                                                       map, prev_grid_point[J],
                                                       offset[J], block->instance), 0)...};


    (void) std::initializer_list<int>{(
      particle_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], offset[J],
                                                                         particle->block->instance), 0)...};

    firstPart = curPart;

    //TODO: Modify shift_arg to inlcude

     kernel((particle_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
  }

  (void) std::initializer_list<int>{
    (particle_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...};

  //TODO: Set functionality
}
/**
 * Perform a particle-based loop, executing the user defined function kernel,
 * passing data as specified by the list arguments. The iteration is over
 * a specific set of particles
 *
 * Arguments to kernel are passed through ACC<datatype>& references for grid
 * and particle related ops_dat structures. Data accessed via overload
 * operators. For other types of arguments, pointers passed than can be
 * dereferenced directly
 *
 * @param kernel            user kernel # of arguments must match the ops_arg
 *                          parameters
 * @param name              name of the loop
 * @param particle          an ops_particle structure for which the loop is
 *                          performed
 * @param dim               dimensionality of the particle block
 * @param iterate_type      an ops_particle_iteration_types
 *                          OPS_PARTICLE_ITERATE_LOCAL: Iterate only actual particles
 *                          OPS_PARTICLE_ITERATE_ALL:  Iterate over actual & virtual particles
 *                          OPS_PARTICLE_ITERATE_RANDOM: Iterate over a random array of
 *                          particles. Currently, it is supported only for particles within
 *                          a user defined region
 * @param particle_range    Bounding box of the simulation loop. Defined as
 *                          [xmin xmax ymin ymax zmin zmax]
 * @param map               An ops_particle_map structure used for mapping particles into grids
 * @param arguments         a list of ops_arg arguments
 *
 * NOTE: We can remove later on the particle_range and assume that particles
 *       are within a box.
 *
 * TODO: Remove any form of ops_dat structures
 */

template <typename T, typename... ParamType, typename... OPSARG>
void ops_particle_par_loop(void (*kernel)(ParamType...), char const *name,
                       ops_particle particle, ops_particle_group group, int dim,
                       ops_particle_iterate_type iterate_type,
                       T *particle_range, ops_particle_mapping map,
                       OPSARG... arguments) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters should match the number of ops_arg");

  ops_particle_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                             particle, group, dim, iterate_type, particle_range, map, arguments...);

}

/*template<typename... ParamType, typename... OPSARG>
void ops_particle_par_loop(void (*kernel)(ParamType...), char const *name,
                           ops_particle particle, int dim, double *range,
                           OPSARG... arguments) {

  static_assert(sizeof...(ParamType)==sizeof...(OPSARG),
                "Number of kernel parameters should match the number of ops_arg");

  ops_particle_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel ame,
                            particle, dim, range, arguments);
}
*/


#endif /* C++11 */

#endif /* OPS_C_INCLUDE_OPS_PARTICLE_SEQ_H_ */
