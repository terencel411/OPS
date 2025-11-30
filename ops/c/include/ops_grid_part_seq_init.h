#ifndef __OPS_GRID_PART_SEQ_H
#define __OPS_GRID_PART_SEQ_H

#ifndef OPS_API
#define OPS_API 2
#endif

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

//#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if __cplusplus >= 201103L

#include "ops_seq_v2.h"

template<typename ParamT> struct grid_part_param_handler {
  static char *construct(const ops_arg &arg, int dim, int ndim,
                         int start[], int &first, ops_block block, ops_particle_mapping map) {
    if (arg.argtype == OPS_ARG_GBL) {
      if (arg.acc == OPS_READ) return arg.data;
      else
#ifdef OPS_MPI
        return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * block->index;
#else
        return ((ops_reduction)arg.data)->data;
#endif
    }
    else if (arg.argtype == OPS_ARG_IDX) {
#ifdef OPS_MPI
      sub_block_list sb = OPS_sub_block_list[block->index]; //TODO: Multigrid
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = sb->decomp_disp[d] + start[d];
#else
      for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = start[d];
#endif
      return (char *)block->instance->arg_idx;
    }

    return nullptr;
  }

  static ParamT get(char *data) {return (ParamT) data;};

#ifdef OPS_MPI
  static void shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m, int* start,
                        const int offs[], int &first, const sub_block_list &sb, OPS_instance *instance)
#else
  static void shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m, int *start,
                       const int offs[], int &first, OPS_instance *instance)
#endif
  {
    if (arg.argtype == OPS_ARG_IDX && m < OPS_MAX_DIM) {
      instance->arg_idx[m]++;
#ifdef OPS_MPI
      for (int d = 0; d < m && d < OPS_MAX_DIM; d++) instance->arg_idx[d] = sb->decomp_disp[d] + start[d];
#else //OPS_MPI
      for (int d = 0; d < m && d < OPS_MAX_DIM; d++) instance->arg_idx[d] = start[d];
#endif
    }
  }

  static void shift_point_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int dim, int point,
                              int &ifirst) { }

  static void free(char *) {}
};

template <typename T> struct grid_part_param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim,
                         int start[], int &ifirst, ops_block block,
                         ops_particle_mapping map) {

    if (arg.argtype == OPS_ARG_DAT) {
 //     printf("arg_name = %s\n", arg.dat->name);
      int d_m[OPS_MAX_DIM] = {};
  #ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
  #else //OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d];
  #endif //OPS_MPI
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
    //assert(false && "Arg must be OPS_ARG_DAT if accessed by ACC");
    return nullptr;
  }

  static ACC<T>& get(char *data) { return *((ACC<T> *)data); }

#ifdef OPS_MPI
  static int shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m, int* start,
                       const int offs[], int &first,const sub_block_list &sb, OPS_instance *instance)
#else
  static void shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m, int *start,
                       const int offs[], int &first, OPS_instance *instance)
#endif
  {
    if (arg.argtype == OPS_ARG_DAT) {
      int offset = (arg.dat->block->instance->OPS_soa ? 1 : arg.dat->dim) * offs[m];
      ((ACC<T> *)p)->next(offset);
    }

  }

  static void shift_point_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int dim, int point, int &ifirst) { }

  static void free(char *data) { delete (ACC<T> *)data;}
};

template <typename T> struct grid_part_param_handler<ACCP<T>> {

  static char *construct (const ops_arg &arg, int dim, int ndim,
                          int start[], int &ifirst, ops_block,
                          ops_particle_mapping map) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ops_dat bins = map->binhead;
      int *binhead =(int *)map->binhead->data;

      int d_m[OPS_MAX_DIM] ={};
      int stride[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
      for (int d = 0; d < dim; d++) {
        d_m[d] =  bins->d_m[d] + OPS_sub_dat_list[bins->index]->d_im[d];
        stride[d] = 1;
      }
#else
      for (int d = 0; d < dim; d++) { d_m[d] = bins->d_m[d]; stride[d] = 1;}
#endif

      int address_bin = address(ndim, bins->block->instance->OPS_soa ? bins->type_size : bins->elem_size, &start[0],
                                bins->size, stride, bins->base, d_m);

      //ifirst = binhead[address_bin]; //TODO: Check if this element is an actuall point

      ifirst = (binhead[address_bin] <= 0) ? 0 : binhead[address_bin];
      ACCP<T> *datap = new ACCP<T>(arg.dim, map->nParticles, (T* ) (arg.dat->data + ifirst));

      datap->bin_address = address_bin;
      return (char *) datap;
    }

    return nullptr;
  }

#ifdef OPS_MPI
  static int shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m,  int* start,
                       const int offs[], int &first, const sub_block_list &sb, int &ifirst, OPS_instance *instance)
#else
  static void shift_arg(const ops_arg &arg, ops_particle_mapping map, char *p, int m,  int* start,
                       const int offs[], int &ifirst, OPS_instance  *instance)
#endif
  {
//    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {

//      ACCP<T> *datap = (ACCP<T> *)p;

 //     ops_dat binhead = map->binhead;

 //     int *bin_head = (int *)binhead->data;
/*
      int dim = arg.dat->block->dims;
      int d_m[OPS_MAX_DIM] ={};
      int stride[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
      for (int d = 0; d < dim; d++) {
        d_m[d] =  binhead->d_m[d] + OPS_sub_dat_list[binhead->index]->d_im[d];
        stride[d] = 1;
      }
#else
      for (int d = 0; d < dim; d++) { d_m[d] = binhead->d_m[d]; stride[d] = 1;}
#endif
*/
      //Get address
      /*
      datap->bin_address += offs[m];

      int icurr = bin_head[datap->bin_address];
      datap->next((icurr-ifirst) * arg.dat->dim); //DATA shifted
      ifirst = icurr;
      */
//    }

  }

  static ACCP<T>& get(char *data) { return *((ACCP<T> *) data);};

  static void shift_point_arg(const ops_arg &arg, ops_particle_mapping map, char *p,int dim, int ipoint,
                              int &ifirst) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ACCP<T> *datap = (ACCP<T> *)p;

      datap->next((ipoint - ifirst) * arg.dat->dim);
      ifirst = ipoint;
    }
  }


  static void free(char *data) {delete (ACCP<T> *)data;};

};
//offsets for particle structures computed for the same grid
//Assumption herein-Assume that grid and particle mapping have similar sizes
static void initoffs_grid_part(const ops_arg &arg,const ops_particle_mapping map,
                               int *offs, const int &ndim, int *start, int *end) {
  if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
    int stride[OPS_MAX_DIM] = {};
    for (int i = 0; i < ndim; i++) stride[i] = 1; //TODO: Replace with stencil on accessing data

    offs[0] = stride[0] * 1;
    for (int n = 1; n < ndim; n++)
      offs[n] = off(ndim, n, start, end, map->binhead->size, stride); //Uniform grid assumption
  }
  else {
    if(arg.stencil == nullptr) return;

    offs[0] = arg.stencil->stride[0] * 1;
    for (int n = 1; n<ndim; ++n)
      offs[n] = off(ndim, n, start, end, arg.dat->size, arg.stencil->stride);
  }

}


template <typename... ParamType, typename... OPSARG, size_t... J>
void  ops_par_loop_impl(indices<J...>, void (*kernel)(ParamType...),
                        char const *name, ops_block block,
                        ops_particle_mapping map, ops_stencil part_stencil, int dim,
                        int *range, OPSARG... arguments) {

  constexpr int N = sizeof...(OPSARG);
  ops_arg args[N] = {arguments...};
  int  count[OPS_MAX_DIM] = {0};



  if (part_stencil == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty particle stencil\n");

  if (map == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "An empty ops_particle_mapping "
                       "structure was found!\n");
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


#ifdef OPS_DEBUG
  ops_register_args(block->instance, args, name);
#endif

  int ifirst[N] = {};

  //Shift data to first point and construct associated structures
  char *p_a[N] = {grid_part_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim, ndim, start,
                                                                                     ifirst[J], block, map)...};

  ops_arg arg_bin = ops_arg_dat(map->binhead, 1, part_stencil, "int", OPS_READ);

  int igrid_first{-1};
  char *p_grid = {grid_part_param_handler<param_remove_cvref_t<ACC<int>>>::construct(arg_bin, dim, ndim, start,
                                                                                      igrid_first, block, map)};


  int offs[N][OPS_MAX_DIM] = {};
  (void) std::initializer_list<int>{(initoffs_grid_part(arguments, map, offs[J], ndim,start, end), 0)...};

  int offs_grid[OPS_MAX_DIM] = {};
  (void) std::initializer_list<int>{(initoffs_grid_part(arg_bin, map, offs_grid, ndim, start, end), 0)};


  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = end[n]-start[n];  // number in each dimension
    total_range *= count[n];
    total_range *= (count[n]<0?0:1);
  }

  count[dim-1]++;     // extra in last to ensure correct termination
  //Perform exchanges
  ops_H_D_exchanges_host(args, N); //TODO-Check the type
  ops_halo_exchanges(args,N,range); //TODO-Check the type
  ops_H_D_exchanges_host(args, N); //TODO check the type


  int *bins = (int *) map->bin->data;


  for (int nt = 0; nt < total_range; nt++) {
    //TODO: Loop over points in stencil
    //LOOP OVER POINTS FINDING THEM IF NECESSARY
    for (int ipoints  = 0; ipoints < part_stencil->points; ipoints++) {

 //     printf("%d: Stencil points [%d %d %d]\n", ipoints, part_stencil->stencil[3 * ipoints],
  //                                                 part_stencil->stencil[3 * ipoints + 1],
  //                                                 part_stencil->stencil[3 * ipoints + 2]);
#if defined(OPS_1D)
      ACC<int>& binpoint = grid_part_param_handler<int>::get(p_grid);
      int ipart =  binpoint(0, part_stencil->stencil[ipoints]);
#elif defined(OPS_2D)
      ACC<int>& binpoint = grid_part_param_handler<ACC<int>>::get(p_grid);
      int ipart = binpoint(0, part_stencil->stencil[2 * ipoints], part_stencil->stencil[2 * ipoints + 1]); //TODO
#elif defined(OPS_3D)
      ACC<int>& binpoint = grid_part_param_handler<ACC<int>>::get(p_grid); //Here we get the binpoint
 //     printf("ACC(0, 0, 0) = %d\n", binpoint(0, 0, 0, 0));
      int ipart = binpoint(0, part_stencil->stencil[3 * ipoints], part_stencil->stencil[3 * ipoints + 1],
                                                                  part_stencil->stencil[3 * ipoints + 2]);

     // printf("ipart = %d\n", ipart);

#else
     int ipart;
     throw OPSException(OPS_RUNTIME_ERROR, "ops-grid-particle loop supported only for 1D-2D or 3D grids");
#endif


     while (ipart != -1) { //TODO: WE need to change it to sth different for CUDA
   //   printf("Entered to loop for %d - %d\n", ipoints, ipart);
      //TODO: Shift into a variadic function need to loop
      (void) std::initializer_list<int>{(grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_point_arg(arguments, map, p_a[J], dim,
                                                                                                                   ipart, ifirst[J]), 0)...};

      //`
//TODO: Need to change the name just use ACCP.
      kernel((grid_part_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
      ipart = bins[ipart];
     }

    }
    count[0]--; //decrement counter
    int m = 0;
    while (count[m]==0) {
      count[m] =  end[m]-start[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }
    //if (nt == 3) break;
    //shift bin data to data
#ifdef OPS_MPI
  (void) std::initializer_list<int>{(
  grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, map, p_a[J], m, start,
                                                                      offs[J], ifirst[J], sb, block->instance),
                                                                      0)...};
  (void) std::initializer_list<int>{(grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arg_bin, map, p_grid, m,
                                                                                                         start, offs_grid, sb,
                                                                                                         block->instance), 0)};

#else
  (void) std::initializer_list<int>{(grid_part_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, map, p_a[J],
                                                                                                         m, start, offs[J], ifirst[J],
                                                                                                         block->instance), 0)...};

  grid_part_param_handler<ACC<int>>::shift_arg(arg_bin, map, p_grid, m,
                                                start, offs_grid, igrid_first,
                                                block->instance);//, 0)};
#endif
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

  grid_part_param_handler<param_remove_cvref_t<ACC<int>>>::free(p_grid);


}
//const ops_arg &arg, ops_particle_mapping map, char *p, int m,  int* start,
//const int offs[], int &first, const sub_block_list &sb, int &ifirst, OPS_instance *instance)
template <typename... ParamType, typename... OPSARG>
void ops_par_loop(void (*kernel)(ParamType...), char const *name,
                  ops_block block, ops_particle_mapping map, ops_stencil part_stencil, int dim,
                  int *range, OPSARG... arguments) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
      "number of parameters of the kernel shoud match the number of ops_arg");

  //ops_printf("Entering into ops_par_loop for particles-grid interaction\n");

  ops_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                    block, map, part_stencil, dim, range, arguments...);
}

#endif /* C++11 */

#endif /* OPS_C_INCLUDE_OPS_GRID_PART_SEQ_H_ */
