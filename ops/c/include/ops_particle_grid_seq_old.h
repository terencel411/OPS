/*
 * ops_particle_grid_seq.h
 *
 *  Created on: Sep 17, 2025
 *      Author: valantis
 */

#ifndef __OPS_PARTICLE_GRID_SEQ_H_
#define __OPS_PARTICLE_GRID_SEQ_H_

#ifndef OPS_API
#define OPS_API 2
#endif

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "ops_seq_v2.h"
#include "ops_exceptions.h"

inline void get_local_point(int *grid, int address, int dim, int *size, int *d_m) {

  for (int i = dim = 1; i >= 0; i--) {
    int prod = 1;
    for (int j = 0; j < i; j++)
      prod *= size[j];
    grid[i] = address / prod;
    address -= grid[i] * prod;
    grid[i] += d_m[i];
  }

}

template<typename T>
inline int  _point_in_range(const T *range,const T *coords, const int dim) {

  for (int i = 0; i < dim; i++) {
    if (coords[i] < range[2 * i] || coords[i] > range[2 * i + 1]) return 0;
  }

  return 1;
}

/* Perform correction due to transition from coarse to fine mesh*/
template<typename T>
inline void  correction_due_to_grid(int *grid_point,const ops_stencil stencil,
                                    const double *x_p,const int dim,
                                    const ops_point<T> xmin, T *dx) {
  if (stencil->type == 2) {
    grid_point[0] += (x_p[0] - xmin.x - grid_point[0] * dx[0]) / dx[0];
    grid_point[1] += (x_p[1] - xmin.y - grid_point[1] * dx[1]) / dx[1];
    if (dim >= 3)
      grid_point[2] += (x_p[2] - xmin.z - grid_point[2] * dx[2]) / dx[2];
  }
}

template <typename T>
inline int box_intersect(const int dim, const T *A, const T *B,
                         T *region) {

  for (int i = 0; i < dim; i++) {
    region[2 * i] = (A[2 * i] > B[2 * i]) ? A[2 * i] : B[2 * i];
    region[2 * i + 1] = (A[2 * i + 1] < B[2 * i + 1]) ? A[2 * i + 1] : B[2 * i + 1];

    if (region[2 * i] >=region[2 * i + 1] )
      return 0;
  }

  return 1;
}

//Computes intersection range in map points and shift them into a block based grid
template<typename T>
inline int get_interaction_range(int map_range[], ops_dat binhead, T range[],
                                 const int dim,  T *xmin, T *xmax, T *dx,
                                 ops_particle_iterate_type type) {

  int d_m[OPS_MAX_DIM], d_p[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
    d_m[i] = OPS_sub_dat_list[binhead->index]->d_im[i] + binhead->d_m[i];
    d_p[i] = OPS_sub_dat_list[binhead->index]->d_ip[i] + binhead->d_p[i];
#else
    d_m[i] = binhead->d_m[i];
    d_p[i] = binhead->d_p[i];
#endif
  }

  if (type ==  OPS_PARTICLE_ITERATE_LOCAL) {
    map_range[0] = 0;
    map_range[1] = binhead->size[0] + d_m[0] - d_p[0];

    map_range[2] = 0;
    map_range[3] = binhead->size[1] + d_m[1] - d_p[1];

    map_range[4] = 0;
    map_range[5] = (dim == 3) ? binhead->size[2] + d_m[2] - d_p[2] : 1;
    return 1;
  }
  else if (type == OPS_PARTICLE_ITERATE_ALL) {

    map_range[0] = d_m[0];
    map_range[1] = binhead->size[0];

    map_range[2] = d_m[1];
    map_range[3] = binhead->size[1];

    map_range[4] = (dim == 3) ? d_m[2] : 0;
    map_range[5] = (dim == 3) ? binhead->size[2] : 1;

    return 1;
  }
  else { //Random location
    int inter_range[2 * OPS_MAX_DIM];
    //Part I: Check for intersection between
    T region_map[2 * OPS_MAX_DIM];
    for (int i = 0; i < dim; i++) {
      region_map[2 * i] = xmin[i] + static_cast<T>(d_m[i]) * dx[i];
      region_map[2 * i + 1] = xmax[i] + static_cast<T>(d_p[i]) * dx[i];
    }

    int intersect = box_intersect(dim, region_map, range, inter_range);

    if (intersect == 0) return 0;

    for (int i = 0; i < dim; i++) {
      map_range[2 * i] = ops_floor((inter_range[2 * i] - map_range[2 * i])/dx[i]) - d_m[i];
      map_range[2 * i + 1] = ops_floor((inter_range[2 * i + 1] - map_range[2 * i]) / dx[i]) - d_m[i];
    }

    return 1;

  }


  return  0;
}

//This function translates the particle map point into grid map point-This can be
//also smaller
inline void _get_grid_point(int grid_point[],const int map_point[],
                            ops_stencil map_stencil, const int dim) {

  for (int i = 0; i < dim; i++) {
    if (map_stencil->type == 0) grid_point[i] = map_point[i];
    else if (map_stencil->type == 1) grid_point[i] = map_point[i]
                                                   / map_stencil->mgrid_stride[i];
    else if (map_stencil->type == 2) grid_point[i] = map_point[i]
                                                   * map_stencil->mgrid_stride[i];
  }

}

/* The function return the address of the grid point */
inline int  get_first_interaction_grid_point(int map_type,const int grid_map_point[],
                                             const int *stencil, const int dim,
                                             ops_dat grid_head, ops_dat id_grid,
                                             int &ilocal) {
  int address_grid = -1;
  int next_point[OPS_MAX_DIM]={};
  int d_m[OPS_MAX_DIM];
  for (int  i = 0; i < dim; i++) {
    next_point[i] = grid_map_point[i] + stencil[i];
#ifdef OPS_MPI
    int d_m[i] = OPS_sub_dat_list[grid_head->index]->d_im[i] + grid_head->d_m[i];
#else
    int d_m[i] = grid_head->d_m[i];
#endif
    if (next_point[i] < d_m[i] || next_point[i] > grid_head->size[i])
      return -1;
  }

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    d_m[i] = 0;
    next_point[i] = 0;
  }


  //return address
  int address_map = (next_point[0] - d_m[0])
                  + (next_point[1] - d_m[1]) * grid_head->size[0]
                  + (next_point[2] - d_m[2]) * grid_head->size[0] * grid_head->size[1];

  ilocal = (map_type < 2) ? 0 : grid_head[address_map];
  address_grid = (map_type < 2) ? address_map : ((int *) id_grid->data)[ilocal];

  return address_grid;
}

inline int get_next_grid_point(int &ilocal,  ops_dat id_grid,
                               ops_dat grid_bin) {

  ilocal = ((int *) grid_bin->data)[ilocal];

  return ((int *) id_grid->data)[ilocal];
}



#if __cplusplus >= 201103L

template<typename T>
struct part_grid_param_handler2 {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle) {
    if (arg.argtype == OPS_ARG_GBL || arg.argtype == OPS_ARG_GBL_PARTICLE) {
      if (arg.acc == OPS_READ) return arg.dat;
      else
#ifdef OPS_MPI
      return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * block->index;
#else //OPS_MPI
      return ((ops_reduction)arg.data)->data;
#endif //OPS_MPI

    } else if (arg.argtype == OPS_ARG_IDX) {
#ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[block->index];
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++)
        block->instance->arg_idx[d] = sb->decomp_disp[d]; //TODO: Be careful how you assign
#else
      for (int d = 0; d < dim; && d < OPS_MAX_DIM; d++)
        block->instance->arg_idx[d] = 0;
#endif
      return (char *) block->instance->arg_idx;
    }
    else if (arg.argtype == OPS_ARG_IDP) {
      block->instance->arg_idp[0] = 0; //First point
      return (char *)block->instance->arg_idp;
    }

    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int icurr, int ilast, const int dim,
                              OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_IDP) {
      instance->arg_idp[0] = icurr;
    }
    else if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      p+= (icurr - ilast) * arg.elem_size;
    }
  }

  static void shift_arg(const ops_arg &arg, char *p_a, const int grid_point[],
                        int &address_old, const int dim, OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_IDX) {
      for (int i = 0; i < dim; i++) {
        instance->arg_idp[i] = grid_point[i];
      }
    }
  }

  static T get(char *data) {
    return (T) data;
  }

  static void free(char *data) { }

};

template<typename T>
struct part_grid_param_handler2<ACC<T>> {
  static char *construct(const ops_arg &arg, char *p, int dim, int ndim, ops_block block,
                         ops_particle particle) {
    if (arg.argtype == OPS_ARG_DAT) {
#ifdef OPS_1D
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], (T* ) arg.data);
#elif defined(OPS_2D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], (T *) arg.data);
#elif defined(OPS_3D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2], (T *) arg.data);
#elif defined(OPS_4D)
      return (char *) new ACC<T>(arg.din, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2],
                                 arg.dat->size[3], (T *) arg.data);
#else
      return (char *) new ACC<T>(arg.dat->data);
#endif
    }

    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int icurr, int ilast, const int dim,
                              OPS_instance *instance) { }

  static void shift_arg(const ops_arg &arg, char *p_a, const int grid_point[],
                        int &address_old, const int dim, OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_DAT) {
      int address = 0;
      size_t prod = (instance->OPS_soa) ? 1 : arg.dat->dim;
      for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
        int d_m = OPS_sub_dat_list[arg.dat->index]->d_im[i] + arg.dat->d_m[i];
#else
        int d_m = arg.dat->d_m[i];
#endif
        address += (grid_point[i] - d_m) * prod;
        for (int j = 0; j <= i; j++ ) prod *= arg.dat->size[i];
      }
    }

    int offset = address - address_old;
    ((ACC<T> *) p_a)->next(offset);
    address_old = address;
  }

  static ACC<T>& get(char *data) {return *((ACC<T> *) data); }

  static void free(char *data) {delete (ACC<T> *) data;}
};

template<typename T>
struct part_grid_param_handler2<ACCP<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle particle) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      size_t no_particles = particle->no_particles + particle->no_virtual;
      return (char *) new ACCP<T>(arg.dim,  no_particles, (T *) arg.dat->data);
    }

    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int icurr, int ilast,
                              const int dim, OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      int offset = (icurr - ilast) * arg.dat->dim;
      ((ACCP<T> *) p)->next(offset);
    }
  }

  static void shift_arg(const ops_arg &arg, char *p_a, const int grid_point[],
                        int &address_old, const int dim, OPS_instance *instance) {

  }

  static ACCP<T>& get(char *data) {return *((ACCP<T> *) data); }

  static void free(char *data) {delete (ACCP<T> *) data;}
};




template <typename T, typename... ParamType, typename... OPSARG, size_t ...J>
void ops_par_particle_grid_loop_impl2(indices<J...>, void (*kernel) (ParamType...),
                                      char const *name, ops_particle particle,
                                      ops_particle_mapping map, int dim,
                                      ops_particle_iterate_type iter_type,
                                      T * range, ops_stencil map_stencil,
                                      OPSARG... arguments) {
  constexpr int N = sizeof...(OPSARG);
  ops_arg args[N] = {arguments...};
  int count[OPS_MAX_DIM] = {};
  ops_block block = particle->block;

  //Sanity checks
  if (particle->type_box != sizeof(T)) {
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Incompatible types for position "
                                          "data and iteration region\n");
  }

  if (map == nullptr) {
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty map structure");
  }

  if (map->binhead == nullptr || map->bin == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty binhead and bin data structures");

  if (map->map_type > 1 & (map->grid_bin == nullptr || map->id_grid == nullptr
                           || map->grid_head == nullptr))
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Map structures for grid maps/\n");

  //Algorithm need equal size elements for non-conforming and non-uniform grids
  if (    map->grid_type == OPS_NON_UNI_STAG
       || map->grid_type == OPS_NON_UNI_COLL) {
    for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
      int size_grid = OPS_sub_dat_list[map->grid->index]->gbl_size[i]
                    + OPS_sub_dat_list[map->grid->index]->gbl_base[i]
                    + OPS_sub_dat_list[map->grid->index]->gbl_d_m[i]
                    - OPS_sub_dat_list[map->grid->index]->gbl_d_p[i];
#else
      int size_grid = map->grid->size[i] + map->grid->base[i]
                    + map->grid->d_m[i] - map->grid->d_p[i];
#endif

      for (int iarg = 0; iarg < N; iarg++) {
        if (args[iarg].argtype != OPS_ARG_DAT) continue;
#ifdef OPS_MPI
        int size_arg = OPS_sub_dat_list[args[iarg].dat->index]->gbl_size[i]
                     + OPS_sub_dat_list[args[iarg].dat->index]->gbl_base[i]
                     + OPS_sub_dat_list[args[iarg].dat->index]->gbl_d_m[i]
                     - OPS_sub_dat_list[args[iarg].dat->index]->gbl_d_p[i];
#else
        int size_arg = args[iarg].dat->size[i] + args[iarg].dat->base[i]
                     + args[iarg].dat->d_m[i] - args[iarg].dat->d_p[i];
#endif
        if (size_arg != size_grid)
          throw OPSException(OPS_RUNTIME_ERROR,
                             "Error: Interaction between particles and grid require dat structures of same grid\n");
        //TODO: Relax the assumption
      }

    }
  }

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
#endif

  //Getting particle for iteration
  T *dx = (T *)map->dx;
  T dx_map_grid[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++)
    dx_map_grid[i] = (map_stencil->type == 2)
                   ? dx[i] /map_stencil->mgrid_stride[i] : dx[i];


  size_t n_loop_particles = (iter_type == OPS_PARTICLE_ITERATE_LOCAL) ? particle->no_particles
                                                                      : particle->no_particles
                                                                        + particle->no_virtual;
  if (n_loop_particles == 0) return;
  int map_range[2 *OPS_MAX_DIM];
  //Part II: Obtain interaction range in terms of map
  int intersect = get_interaction_range(map_range, map->binhead, range, dim, iter_type);

  if (intersect == 0) return; //No-intersection

  //Data structure exchange
  /* Exchange grid dat structures if necessary */

  int dmap_m[OPS_MAX_DIM], dgrid_m[OPS_MAX_DIM];
  int map_point[OPS_MAX_DIM], grid_map_point[OPS_MAX_DIM], grid_point[OPS_MAX_DIM] ;

  for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
    dmap_m[i] = OPS_sub_dat_list[map->binhead->index]->d_im[i] + map->binhead->d_m[i];
    dgrid_m[i] = OPS_sub_dat_list[map->grid->index]->d_im[i] + map->grid->d_m[i];
#else
    dmap_m[i] = map->binhead->d_m[i];
    dgrid_m[i] = map->grid->d_m[i];
#endif
  }

  //TODO: Upgrade size updating
  int range_base[2 * OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
    range_base[2 * i] = OPS_sub_dat_list[map->grid->index]->gbl_d_m[i];
    range_base[2 * i + 1] = OPS_sub_dat_list[map->grid->index]->gbl_size[i]
                          + OPS_sub_dat_list[map->grid->index]->gbl_base[i];
#else
    range_base[2 * i] = map->grid->d_m[i];
    range_base[2 * i + 1] = map->grid->size[i] + map->grid->base[i];
#endif
  }

  ops_H_D_exchanges_host(args, N);
  ops_halo_exchanges(args,N,range_base);
  ops_H_D_exchanges_host(args, N);

  //construct data
  char *p_a[N] =
       {part_grid_param_handler2<param_remove_cvref_t<ParamType>>::construct(arguments, dim, dim,
                                                                             block, particle)...};
  int *address_old[N] = {};
  size_t ilast = 0;
  //Loop over particles
  T* coords = (T *) particle->particle_pos_dat->data;
  int *part_to_map = (int *) map->parts_to_grid->data;
  for (size_t i = 0; i < n_loop_particles; i++) {

    if (iter_type == OPS_PARTICLE_ITERATE_RANDOM) {
      //check that point is within range
      int point_within = _point_in_range(range, coords + dim * i, dim);
      if (!point_within) continue;
    }

    (void) std::initializer_list<int>{(
        part_grid_param_handler2<param_remove_cvref_t<ParamType>>::shift_part_args(arguments, p_a, i, ilast,
                                                                                   dim, block->instance), 0)...};

    //Now to find the bin-point
    int address = part_to_map[i];

    if (address < 0) continue;

    //Get local point of address
    get_local_point(map_point, address, dim, map->binhead->size, dmap_m);

    //Translate map to grid address (Need address and grid_point)
    _get_grid_point(grid_map_point, map_point, map_stencil, dim);
    if (map->mapping_type < 2 && map_stencil->type == 2)
       correction_due_to_grid(grid_map_point, map_stencil, coords + i * dim,
                              dim, (T *) map->xmin, dx_map_grid);

    //Part II: Loop over grid_map_point with stencil
    for (int isten = 0; isten < map_stencil->points; isten++) {
      int ilocal_grid;
      int grid_address = get_first_interaction_grid_point(map->map_type, grid_map_point,
                                                    map_stencil->stencil + isten * dim, dim,
                                                    map->grid_head, map->id_grid, ilocal_grid);

      while (grid_address > - 1) {

        //Get local point
        get_local_point(grid_point, grid_address, dim, map->grid->size, dgrid_m);
        //Shift data to the proper grid point and compute next point
        (void) std::initializer_list<int>{(part_grid_param_handler2<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J],
                                           grid_point, address_old[J], dim, block->instance), 0)...};



        kernel((part_grid_param_handler2<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);


        //Get new address
         grid_address = (map->map_type < 2) ? -1 : get_next_grid_point(ilocal_grid, map->id_grid,
                                                                       map->grid_bin);
      }
    }

    ilast = i;
  }


  //Additional operations and closure
  (void) std::initializer_list<int>{
    (part_grid_param_handler2<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...};
}


/**
 * Perform a particle/grid interaction loop. Exterior loop operates on particles while
 * the interior loop operators on grid points defined by a given stencil. The main
 * assumption is that the grid structures conform to the first point of the binhead
 * structure.
 *
 * Arguments to kernel are passed through ACCP<T>& and ACC<T>& datatypes for particle
 * and grid based ops_dat structures. Data accessed via overload
 * operators. For other types of arguments, pointers passed than can be
 * dereferenced directly
 *
 * @param kernel         user kernel # of arguments must match the ops_arg parameters
 * @param name           name of the loop
 * @param particle       an ops_particle structure for which the loop is performed
 * @param map            the ops_particle_mapping structure that used to map the
 *                       particles within the block
 * @param dim            dimensionality of the particle block
 * @param iterate_type   Definition of iteration range of particles. Currently the
 *                       following three types are supported
 *                         OPS_PARTICLE_ITERATE_LOCAL: Iterate only actual particles
 *                         OPS_PARTICLE_ITERATE_ALL:  Iterate over actual & virtual particles
 *                         OPS_PARTICLE_ITERATE_RANDOM: Iterate over a random array of
 *                         particles. Currently, it is supported only for particles within
 *                         a user defined region
 * @range                Range for identifying particles for iterating upon. The range
 *                       is given in the form [xmin xmax ymin ymax zmin zmax].
 * @param map_stencil    Stencil that defined to a user defined grid used that
 *                       projects the mapping grid.
 * @param arguments      a list of ops_arg_ arguments. The stencils of grid structures
 *                       are defined with respect to the projection grid (coarsest) grid
 *                       used in the simulation.
 *
 */
template <typename T, typename... ParamType, typename... OPSARG>
void ops_par_particle_grid_loop(void (*kernel)(ParamType...), char const *name,
                                ops_particle particle, ops_particle_mapping map,
                                int dim, ops_particle_iterate_type iterate_type,
                                T *range, ops_stencil map_stencil,
                                OPSARG... arguments) {

  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters do not match the number of ops_arg");

  ops_par_particle_grid_loop_impl2(build_indices<sizeof...(ParamType)>{}, kernel, name,
                                  particle, map, dim, iterate_type, range, map_stencil,
                                  arguments...);

}
#endif /* C++11 */

#endif /* OPS_C_INCLUDE_OPS_PARTICLE_GRID_SEQ_H_ */
