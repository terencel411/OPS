/*
 * ops_grid_part_seq_v2.h
 *
 *  Created on: Sep 15, 2025
 *      Author: valantis
 */

#ifndef __OPS_GRID_PART_SEQ_V2_H
#define __OPS_GRID_PART_SEQ_V2_H

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "ops_seq_v2.h"


inline void find_old_indices(const ops_arg &arg, int  *indices, const int  &dim, int  *start) {

  if (arg.argtype == OPS_ARG_DAT) {
    switch(arg.stencil->type) {
    case 0:
      for (int i = 0; i < dim; i++) indices[i] = start[i] * arg.stencil->stride[i];
      break;
    case 1: //Prolong
      for (int i = 0; i < dim; i++) indices[i] = start[i] * arg.stencil->stride[i] / arg.stencil->mgrid_stride[i];
      break;
    case 2: //Restrict
      for (int i = 0; i < dim; i++) indices[i] = start[i] * arg.stencil->stride[i] * arg.stencil->mgrid_stride[i]; //TODO: Check the stride
      break;
    default:
      OPSException(OPS_RUNTIME_ERROR, "This type of stencil is not supported\n");
    }
  }
  else {
    for (int i = 0; i < dim; i++) indices[i] = -1;
  }
}

inline void find_offs(const ops_arg &arg, int &offs, const int *old_ind, const int dim,
                      const int *point) {

  int ielem[3] = {point[0], point[1], point[2]};
  if (arg.argtype == OPS_ARG_DAT) {
    switch(arg.stencil->type) {
    case 0:
      for (int i = 0; i < dim; i++) ielem[i]  *= arg.stencil->stride[i]; //TODO: Check
      break;
    case 1:
      for (int i = 0; i < dim; i++) ielem[i]  *= arg.stencil->stride[i] / arg.stencil->mgrid_stride[i];
      break;
    case 2:
      for (int i = 0; i < dim; i++) ielem[i]  *= arg.stencil->stride[i] * arg.stencil->mgrid_stride[i];
      break;
    default:
      OPSException(OPS_RUNTIME_ERROR, " This type of stencil is not supported\n");
    }

    offs = 0;
    int prod = 1; //arg.dat->type_size;
    for (int idim = 0; idim < dim; idim++) {
      offs += (ielem[idim] - old_ind[idim]) * prod;
      prod *=  arg.dat->size[idim];
    }
  }
  else
    offs = 0;
}

//This is problematic

inline  void get_point_in_map(int *point2map, int *point, int dim,
                              ops_stencil stencil) {

  //Get actual grid point in stencil

  switch(stencil->type) {
  case 0:
    for (int n = 0; n < dim; n++) point2map[n] = point[n];
    break;
   case 1:
    for (int n = 0; n < dim; n++)
      point2map[n] = point[n] / stencil->mgrid_stride[n];
    break;
  case 2:
    for (int n = 0; n < dim; n++)
      point2map[n] = point[n] * stencil->mgrid_stride[n];
    break;
  }
}

inline int get_mapping_address(int  *point,int dim, ops_dat binhead) {

  int address = 0;
  int prod = 1;
  for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
    int d_m = OPS_sub_dat_list[binhead->index]->d_im[i] + binhead->d_m[i];
#else
    int d_m = binhead->d_m[i];
#endif
    int ix = point[i] - d_m;
    if (ix < 0 || ix > binhead->size[i] - 1) //TODO: Check this option
      return -1; //Safety
    address += ix * prod;
    prod *= binhead->size[i];
  }

  return address;
}

inline int get_first_point(const int start[], const int dim,const ops_particle_mapping map,
                           const ops_stencil  map_stencil) {

  //Transfer point to map

  int map_point[OPS_MAX_DIM] = {};
  switch (map_stencil->type) {
  case 0:
    for (int n = 0; n < dim; n++) map_point[n] = start[n];
    break;
  case 1:
    for (int d = 0; d < dim; d++)
      map_point[d] = start[d] / map_stencil->mgrid_stride[d];
    break;
  case 2:
    for (int d = 0; d < dim; d++)
      map_point[d] = start[d] * map_stencil->mgrid_stride[d];
    break;
  }

  ops_dat binhead = map->binhead;
  int *bin_head = (int *) map->binhead->data;
  int address = get_mapping_address(map_point, dim, binhead);

  return (address > -1) ? bin_head[address] : 0;

}


#if __cplusplus >= 201103L

//find_offs(arguments, offs[J], old_indices, dim, i, j, k), 0)

template<typename ParamT> struct grid_part_param_handler {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[],
                         int &ifirst, ops_particle particle, ops_particle_mapping map,
                         ops_stencil map_stencil) {
    if (arg.argtype == OPS_ARG_GBL) {
      if (arg.acc == OPS_READ) return arg.data;
      else
#ifdef OPS_MPI
      return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * particle->block->index;
#else
      return ((ops_reduction)arg.data)->data;
#endif
    }
    else if (arg.argtype == OPS_ARG_IDX) {
#ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[particle->block->index]; //TODO: Multigrid
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) particle->block->instance->arg_idx[d] = sb->decomp_disp[d] + start[d];
#else
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) particle->block->instance->arg_idx[d] = start[d];
#endif
      return (char *)particle->block->instance->arg_idx;
    }
    else if (arg.argtype == OPS_ARG_IDP) {


      particle->block->instance->arg_idp[0] = ifirst;

      return (char *)particle->block->instance->arg_idp;
    }
//    else if (arg.argtype == OPS_ARG_IDX_MAP) {
//      //tODO
//    }

    return nullptr;
  }

  static ParamT get(char *data) { return (ParamT) data;}

  static void shift_arg(const ops_arg &arg, const ops_particle_mapping map, const char *p,
                        const int offs, int dim, const int *point,
                        OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_IDX) {
#ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[map->particle->block->index]; //TODO: Multigrid
      for (int i = 0; i < dim; i++) instance->arg_idx[i] = point[i]  + sb->decomp_disp[i]; //TODO: Need to add the initial point

#else
      for (int i = 0; i < dim; i++) instance->arg_idx[i] = point[i]; //TODO: Need to add the initial point
#endif
      for (int i = dim; i < OPS_MAX_DIM; i++) instance->arg_idx[i] = 0;
    }
  }

  static void shift_point_arg(const ops_arg &arg, char *p, int point, int &ifirst,
                              OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_IDP) {
      instance->arg_idp[0] = point;
    }
  }

  //shift_arg(arguments, map, p_a[J],  offs, point, block->instance)

  //TODO: Additional functions
  static void free(char *) { }
};

template <typename T> struct grid_part_param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[],
                         int &ifirst, ops_particle particle, ops_particle_mapping map,
                         ops_stencil map_stencil) {

    if (arg.argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
#else
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d];
#endif

      int start_dat[OPS_MAX_DIM];
      switch (arg.stencil->type) {
      case 0:
        for (int i = 0; i < dim; i++) start_dat[i] = start[i];
        break;
      case 1: //Prolong stencil
        for (int i = 0; i < dim; i++) start_dat[i] = start[i] / arg.stencil->mgrid_stride[i]; //This needs sanity checks
        break;
      case 2: // Restrict stencil
        for (int i = 0; i < dim; i++) start_dat[i] = start[i] * arg.stencil->mgrid_stride[i];
        break;
      default:
        OPSException(OPS_RUNTIME_ERROR,"This stencil is not supported\n");
      }

      long int address = start_dat[0] * arg.stencil->stride[0] - arg.dat->base[0] - d_m[0];
      if (dim >= 2) address +=  (   start_dat[1] *  arg.stencil->stride[1] - arg.dat->base[1]
                                  - d_m[1]) * arg.dat->size[0];
      if (dim == 3) address += (    start_dat[2] * arg.stencil->stride[2] - arg.dat->base[2]
                                  - d_m[2]) * arg.dat->size[0] * arg.dat->size[1];


#ifdef OPS_1D
      return (char *) new ACC<T>(arg.dim, arg.dat>size[0], (T *)(arg.data
#elif defined(OPS_2D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], (T *)(arg.data
#elif defined(OPS_3D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2],
                                 (T *)(arg.data
#else
      return (char *) (arg.dat->data
#endif
      + address * (arg.dat->block->instance->OPS_soa ? arg.dat->type_size : arg.dat->elem_size)));
    }

    return nullptr;
  }

  static ACC<T>& get(char *data) { return *((ACC<T> *)data); }

  static void shift_arg(const ops_arg &arg, const ops_particle_mapping map, const char *p,
                        const int offs, int dim, const int *point, OPS_instance *instance)
  {
    if (arg.argtype == OPS_ARG_DAT) {
      int offset = (arg.dat->block->instance->OPS_soa) ? offs : arg.dat->dim * offs;

      ((ACC<T> *)p)->next(offset);
    }
  }

  static void shift_point_arg(const ops_arg &arg, char *p, int point, int &ifirst,
                              OPS_instance *instance) { }

  static void free(char *data) { delete (ACC<T> *)data;}

};

template <typename T> struct grid_part_param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, int start[],
                         int &ifirst, ops_particle particle, ops_particle_mapping map,
                         ops_stencil map_stencil) {

    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      /*
      //Shift start to actual grid
      int start_map[OPS_MAX_DIM];
      switch(map_stencil->type) {
      case 0: //Default:
        for (int i = 0; i < dim; i++) start_map[i] = start[i];
        break;
      case 1: // Prolong: Fine to coarse
        for (int i = 0; i < dim; i++) start_map[i] = start[i] / map_stencil->mgrid_stride[i];
        break;
      case 2: // Restrict
        for (int i = 0; i < dim; i++) start_map[i] = start[i] * map_stencil->mgrid_stride[i];
        break;
      default:
        OPSException(OPS_RUNTIME_ERROR,"This stencil is not supported\n");
      }

      long int address = start_map[0] * map_stencil->stride[0] - binhead->base[0] + d_m[0];
      if (dim >= 2)
        address += ( start_map[1] * map_stencil->stride[1]  - binhead->base[1] + d_m[1])
                 * binhead->size[0];
      if (dim == 3)
        address += (  start_map[1] * map_stencil->stride[2] - binhead->base[2] + d_m[2])
                 * binhead->size[1] * binhead->size[2];

      ifirst = (binhead_data[address] < 0) ? 0 : binhead_data[address]; */
      ACCP<T> *datap = new ACCP<T>(arg.dim, particle->no_particles + particle->no_virtual,
                                   (T *)arg.dat->data + ifirst * arg.dim); //TODO: Check

    //datap->bin_address = address;
    return (char *) datap;
    }

    return nullptr;
  }

  static ACCP<T>& get(char *data) { return *((ACCP<T> *)data); }

  static void shift_arg(const ops_arg &arg, const ops_particle_mapping map,
                        const char *p, const int offs, int dim, const int *point,
                        OPS_instance *instance) { }

  static void shift_point_arg(const ops_arg &arg, char *p, int point, int &ifirst,
                              OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ACCP<T> *datap = (ACCP<T> *)p;
      datap->next((point - ifirst) * arg.dat->dim);

     // ifirst = point;
    }
  }

  static void free(char *data) {delete (ACCP<T> *)data;};


 //TODO: Additional functions
};

template<typename... ParamType, typename... OPSARG, size_t... J>
void ops_par_loop_impl(indices<J...>, void (*kernel)(ParamType ...),
                       char const *name, ops_particle particle,
                       ops_particle_mapping map, ops_stencil map_stencil,
                       int dim, int *range, OPSARG... arguments) {

  constexpr int N = sizeof...(OPSARG);
  ops_arg args[N] = {arguments...};
//  int count[OPS_MAX_DIM] = {0};
  ops_block block = particle->block;

  if (map_stencil == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty mapping stencil\n");

  if (map_stencil->dims != dim)
    throw OPSException(OPS_INVALID_ARGUMENT, "The block dimensionality "
                        "is different from the dimensional of the mapping "
                        "stencil\n");

  if (dim > 3)
    throw OPSException(OPS_RUNTIME_ERROR, " Grid/Particle parallel "
                       "loops are supported up to three dimensional "
                       "spaces");

  if (map == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty mapping structure");

  if (map->binhead == nullptr || map->bin == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty mapping elements");

#ifdef CHECKPOINTING
  if (!ops_checkpointing_name_before(args,N,range,name)) return;
#endif

  /* Set start and end of loop */
  int start[OPS_MAX_DIM];
  int end[OPS_MAX_DIM];

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  //compute locally allocated range for the sub-block
  int ndim = sb->ndim;
  for (int n=0; n<ndim; n++) {
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) start[n] = 0;
    else start[n] = range[2*n] - start[n];
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];
    else end[n] = sb->decomp_size[n];
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }

  #else //!OPS_MPI
  int ndim = block->dims;
  for (int n=0; n<ndim; n++) {
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
#endif //OPS_MPI
  for (int n = ndim; n < OPS_MAX_DIM; n++) {
    start[n] = 0; end[n] = 1;
  }

  /*
  printf("Rank %d: Start [%d %d] End [%d %d]\n", ops_get_proc(), start[0], start[1], end[0], end[1]);
  printf("Rank %d Binhead size [%d %d]\n", ops_get_proc(),
         map->binhead->size[0] +  (OPS_sub_dat_list[map->binhead->index]->d_im[0] +
         map->binhead->d_m[0]) - (OPS_sub_dat_list[map->binhead->index]->d_ip[0] + map->binhead->d_p[0]),
         map->binhead->size[1] + (OPS_sub_dat_list[map->binhead->index]->d_im[1] + map->binhead->d_m[1])
         - (OPS_sub_dat_list[map->binhead->index]->d_ip[1] + map->binhead->d_p[1]));
         */

#ifdef OPS_DEBUG
  ops_register_args(block->instance, args, name);
#endif

  //Perform data exchange as necessary for comms if needed
  ops_H_D_exchanges_host(args, N);
  ops_halo_exchanges(args,N,range);
  ops_H_D_exchanges_host(args, N);

  //Construction of needed structures
  int ifirst[N] = {};

  int first_point = get_first_point(start, dim, map, map_stencil);
  char *p_a[N] = {grid_part_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim, ndim, start,
                                                                                      first_point, particle, map,
                                                                                      map_stencil)...};

 // printf("Passed construct\n");

  int offs[N] = {};
  int old_indices[N][OPS_MAX_DIM];
//  int igrid_to_map[OPS_MAX_DIM];

  ops_dat binhead = map->binhead;
  ops_dat bins = map->bin;

  int *bin_head = (int *)map->binhead->data;
  int *bin_data = (int *)bins->data;

  //TODO: Proper access first point. Need that for mpi

  int point2map[OPS_MAX_DIM]={};
  int point[OPS_MAX_DIM]={};

  (void) std::initializer_list<int>{(find_old_indices(arguments, old_indices[J], dim, start), 0)...}; //TODO

  //printf("Rank %d [%d %d]x[%d %d]x[%d %d]\n", ops_get_proc(), start[0], end[0], start[1], end[1], start[2], end[2]);

  int stencil_point[OPS_MAX_DIM];
  for (int k = start[2]; k < end[2]; k++) {
    for (int j = start[1]; j < end[1]; j++) {
      for (int i = start[0]; i < end[0]; i++) {
        point[0] = i;
        point[1] = j;
        point[2] = k;

        (void) std::initializer_list<int>{(find_offs(arguments, offs[J], old_indices[J], dim, point), 0)...};


  //      printf("offs[0] = %d offs[1] = %d\n", offs[0], offs[1]);

        (void) std::initializer_list<int>{ (grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, map, p_a[J],
                                            offs[J], dim, point, block->instance),0)...};

        get_point_in_map(point2map, point, dim, map_stencil); //TODO

        for (int isou = 0; isou < map_stencil->points; isou++) {
          //Get stencil point
          for (int d = 0; d < dim; d++) {
            stencil_point[d] = map_stencil->stencil[isou * dim + d] + point2map[d];
          }



          int address = get_mapping_address(stencil_point, dim, binhead); //TODO
/*
          printf("Rank %d: Point [%d %d %d] (%d %d) stencil [%d %d] yields address %d\n",
                 ops_get_proc(), i, j, k, point2map[0], point2map[1],
                 stencil_point[0], stencil_point[1], address);
*/
          if (address < 0) continue; //no-address out of scope

          int ipart = bin_head[address];

/*          if (ipart != -1) printf("Rank %d: Particle %d in bin [%d %d] stencil [%d %d]\n", ops_get_proc(), ipart,
                                  point2map[0], point2map[1], stencil_point[0], stencil_point[1]); */

          while (ipart != - 1) {

            //Shift particle data & update the previous point
            (void) std::initializer_list<int>{
             (grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_point_arg(arguments, p_a[J],
                                                                                        ipart, first_point,
                                                                                        block->instance),0)...};
           first_point = ipart;

           //Compute the Kernel
           kernel((grid_part_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
           ipart = bin_data[ipart]; //TODO: Remove ifirst;


          }

        }

        //Get old indices
        (void) std::initializer_list<int>{(find_old_indices(arguments, old_indices[J], dim, point), 0)...};

      }
    }
  }

#ifdef OPS_DEBUG_DUMP
  (void) std::initializer_list<int>{(
    arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ? ops_dump3(arguments.dat,name),0:0)...};
#endif
  (void) std::initializer_list<int>{(
     (arguments.argtype == OPS_ARG_DAT && arguments.acc != OPS_READ)?  ops_set_halo_dirtybit3(&arguments,range),0:0)...};
  ops_set_dirtybit_host(args, N);

  (void) std::initializer_list<int>{
    (grid_part_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]),0)...};

  //FREE VARIOUS STRUCTURES
}

/**
 * Perform a parallel loop to handle the interaction between a particle and a grid point
 * executing the user-defined function kernel, passing data as specified by the list arguments.
 * The iteration space is dim dimensional, and the bounds are specified by range. The general
 * stencil is used to handle the interaction of the grid with the particles via mapping structures
 *
 * Arguments to kernel are passed through as ACC<datatype>& references
 * for grid ops_dats. Meanwhile, particle ops_dats are passed as ACCP<T>& references. Both
 * accessed via the overload operator.  For other types of arguments, pointers are passed
 * that can be dereferenced directly
 *
 * @param kernel       user kernel, the number of arguments must match the number of
 *                     ops_arg parameters
 * @param name         a name for the parallel grid/particle interaction loop
 * @param block        the ops_block to iterate on
 * @param map          the ops_particle_mapping structure that handles the interaction
 *                     of the grid with a given ops_particle structure
 * @param map_stencil  the stencil that links the map to the ops_dat structure
 * @param dim          dimensionality of the block
 * @param range        loop bounds in the following order: {dim0_lower,
 *                     dim0_upper_exclusive, dim1_lower, ...}
 *
 * @param arguments a list of ops_arg arguments
 */
template<typename... ParamType, typename... OPSARG>
void ops_par_loop(void (*kernel)(ParamType...), char const *name,
                  ops_particle particle, ops_particle_mapping map, ops_stencil map_stencil,
                  int dim, int *range, OPSARG... arguments) {

  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "number of kernel parameter do not match the number of ops_arg");

  ops_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                    particle, map, map_stencil, dim, range, arguments...);

}

#endif
#endif /* OPS_C_INCLUDE_OPS_GRID_PART_SEQ_V2_H_ */
