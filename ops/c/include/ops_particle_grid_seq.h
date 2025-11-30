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

inline int multiply(const int size[], int dir) {
  int prod{1};



  for (int i = 0; i <= dir; i++) {
    prod *= size[i];
  }

  return prod;
}

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

//TODO: Expand to MPI case need to address decomposition approach
inline void shift_from_grid_to_grid(int *transf, int *init, int dim, ops_stencil stencil) {


  for (int n = 0; n < dim; n++) {
    if (stencil->type == 0) transf[n] = init[n];
    else if (stencil->type == 1) transf[n] = init[n] / stencil->mgrid_stride[n];
    else if (stencil->type == 2) transf[n] = init[n] * stencil->mgrid_stride[n];
  }
}


inline int get_address2(int loc[], const int mdim, const int dim, const int d_m[],
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
    address += (-d_m[i] + loc[i]) * multiply(sizeL, i);
  }

  return address;
}

inline void get_coord_point2(const double *coords, const int size[],const int d_m[],
                            const int ilocal[], const int dim, double xlocal[]) {

#ifdef OPS_SOA
  if (dim == 2) {
    xlocal[0] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]);
    xlocal[1] = *(coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                  + size[0] * size[1]);
  }
  else {
    xlocal[0] = *(  coords + (ilocal[0] - d_m[0]) + (ilocal[1] - d_m[1]) * size[0]
                  + (ilocal[2] - d_m[2]) * size[0] * size[1]);
    xlocal[1] = *(  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                  + (ilocal[2] - d_m[2]) * size[0] * size[1] +
                  size[1] * size[2] * size[0]);
    xlocal[2] = (  coords + (ilocal[0] - d_m[0])+ (ilocal[1] - d_m[1]) * size[0]
                  + (ilocal[2] - d_m[2]) * size[0] * size[1]
                  + 2 * size[1] * size[2] * size[0]);
  }
#else
  if (dim == 2) {
    xlocal[0] = *(coords + dim * (ilocal[0] - d_m[0])
              + dim * (ilocal[1] - d_m[0]) * size[0]);
    xlocal[1] = *(coords +  dim * (ilocal[0] - d_m[0])
                  + dim * (ilocal[1] - d_m[0]) * size[0] + dim + 1);
  }
  else {
    xlocal[0] = *( coords + dim * (ilocal[0] - d_m[0])
                  + dim * (ilocal[1] - d_m[1]) * size[0]
                  + dim * (ilocal[2] - d_m[2]) * size[0] * size[1]);
    xlocal[1] =  *( coords + 1 + dim * (ilocal[0] - d_m[0])
                    + dim * (ilocal[1] - d_m[1]) * size[0]
                    + dim * (ilocal[2] - d_m[2]) * size[0] * size[1]);
    xlocal[2] =  *( coords + 2 + dim * (ilocal[0] - d_m[0])
                    + dim * (ilocal[1] - d_m[1]) * size[0]
                    + dim * (ilocal[2] - d_m[2]) * size[0] * size[1]);
  }
#endif
}

inline size_t get_grid_size2(BoundingBox *intersection, BoundingBox *box,
                            ops_particle_mapping map, int dim, int  local_grid[]) {
  size_t nelems{1};

  //At first assume uniform grid
  double dx[OPS_MAX_DIM], xfirst[OPS_MAX_DIM], xlast[OPS_MAX_DIM];
  int iloc[OPS_MAX_DIM], inext[OPS_MAX_DIM];
  for (int i = 0; i < dim; i++) {
    inext[i] = 1;
    iloc[i] = 0;
  }
  get_coord_point2((double *) map->grid->data, map->grid->size, map->grid->d_m,
                  iloc, dim, xfirst);
  get_coord_point2((double *) map->grid->data, map->grid->size, map->grid->d_m,
                  inext, dim, xlast);

  for (int i = 0; i < dim; i++)
    dx[i] = xlast[i] - xfirst[i];

  for (int i = 0; i < OPS_MAX_DIM; i++) {
    if (i < dim) {
      local_grid[2 * i] = floor((intersection->getMinCoordDir(i) - box->getMinCoordDir(i)) / dx[i]);
      local_grid[2 * i + 1] =  ceil((intersection->getMaxCoordDir(i) - box->getMinCoordDir(i)) / dx[i]) + 1;

      if (local_grid[2 * i + 1] > map->grid->size[i] - map->grid->d_p[i] + map->grid->d_m[i])
        local_grid[2 * i + 1] = map->grid->size[i] - map->grid->d_p[i] + map->grid->d_m[i];
    }
    else {
      local_grid[2 * i] = 0;
      local_grid[2 * i + 1] = 1;
    }

    nelems *= local_grid[2 * i + 1] - local_grid[2 * i];
  }

  return nelems;
}

inline long int get_direct_iteration_points2(BoundingBox *intersection,ops_dat crd_parts,
                                            int nParticles, int  dim,
                                            long int  *&looping_particles) {

  int nmax = OPS_MAX_PART;
  long int nloop{0};

  looping_particles = (long int *) ops_malloc(sizeof(long int) * nmax);

  double *coords = (double *)crd_parts->data;
  for (int i = 0; i < nParticles; i++) {
    bool is_in = intersection->isCoordinateInBoundingBox(coords + dim * i);
    if (is_in) {
      nloop++;
      if (nloop > nmax) {
        looping_particles = (long int  *)ops_realloc(looping_particles,
                                                     sizeof(long int) * (nloop + OPS_MAX_PART));
        nmax = nloop + OPS_MAX_PART;
      }
      looping_particles[nloop - 1] = i;
    }
  }

  return nloop;

}

inline long int  get_particles_from_grid2(ops_particle_mapping map, int local_grid[], int dim,
                                         long int *&looping_particles) {

  long int nloop{0};
  int nmax{OPS_MAX_PART};

  int *binhead = (int *)map->binhead->data;
  int *bins = (int *)map->bin->data;

  int localPoint[OPS_MAX_DIM];
  looping_particles = (long int *)ops_malloc(sizeof(long int) * nmax);
  for (int i = local_grid[0]; i < local_grid[1]; i++)
    for (int j = local_grid[2] ; j < local_grid[3]; j++)
      for (int k = local_grid[4] ; k < local_grid[5]; k++) {
        localPoint[0] = i; localPoint[1] = j; localPoint[2] = k;
        int address = get_address2(localPoint, 1,dim, map->binhead->d_m,
                                  map->binhead->size);
        int a1 = binhead[address]; //TODO
        while (a1 != -1) {
          nloop++;
          if (nloop > nmax) {
            looping_particles = (long int *)ops_realloc(looping_particles,
                                                        sizeof(long int) * (nloop + OPS_MAX_PART));
            nmax = nloop + OPS_MAX_PART;
          }
          looping_particles[nloop - 1] = a1;
          a1 = bins[a1];

        }
      }

  return nloop;

}

inline size_t get_particles_in_range(size_t nParticles, ops_particle particle,
                                    BoundingBox *box, ops_particle_mapping map,
                                    int dim, double *range, long int *&loop_particles) {
  int inters = 2;

  BoundingBox *intersection = ops_find_intersection_region(box, range, inters);

  int nsize = 0;
  if (inters == 2) {delete intersection; return 0;}
  else if (inters == 0) {
    loop_particles = (long int *)ops_malloc(sizeof(long int) * nParticles);
    for (size_t i = 0; i < nParticles; i++) loop_particles[i] = i;
    return nParticles;
  }
  else {
    int local_grid[2 * OPS_MAX_DIM];
    size_t ngrid_points{1};
    if (map != nullptr) {
      ngrid_points = get_grid_size2(intersection, box, map, dim, local_grid);
    }

    if (ngrid_points > nParticles || map == nullptr) {
      nsize =  get_direct_iteration_points2(intersection, particle->particle_pos_dat,
                                         nParticles, dim, loop_particles);
    }
    else
      nsize =  get_particles_from_grid2(map, local_grid, dim, loop_particles);
  }

  delete intersection;
  return nsize;
}

inline void init_offs_particles(int dim, size_t n_particles, long int *looping_particles,
                                int *off_parts) {
  int ifirst{0};
  for (size_t i = 0; i < n_particles; i++) {
    off_parts[i] = (looping_particles[i] - ifirst);
    ifirst = looping_particles[i];
  }
}

inline int check_bounds(int dim, int point[], int size[], int  d_m[], int d_p[]) {

  for (int d = 0; d < dim; d++) {
    if ((point[d] < d_m[d]) || (point[d] > size[d] + d_p[d] - d_m[d] -1)) return -1;
  }

  return 1;
}

inline void find_offs2(const ops_arg &arg, int &offs, const int *prev_point,
                      const int dim,   const int *point) {

  int ielem[dim];
  if (arg.argtype == OPS_ARG_DAT) {
    for (int i = 0; i < dim; i++) {

      ielem[i] = point[i] * arg.stencil->stride[i];
      if (arg.stencil->type == 1) ielem[i] /= (arg.stencil->mgrid_stride[i]);
      else if (arg.stencil->type == 2) ielem[i] *= arg.stencil->mgrid_stride[i];


      if (ielem[i] < arg.dat->d_m[i]) ielem[i] = arg.dat->d_m[i];
      if (ielem[i] > arg.dat->size[i] - 1) ielem[i] = arg.dat->size[i] - 1;
    }

    offs = 0;
    int prod = 1;
    for (int idim = 0; idim < dim; idim++) {
      offs += (ielem[idim] - prev_point[idim]) * prod;
      prod *= arg.dat->size[idim];
    }
  }
}

inline void find_old_indices2(const ops_arg &arg, int *old_point, const int dim,
                             const int *point) {

  if (arg.argtype == OPS_ARG_DAT) {
    for(int i = 0; i < dim; i++) {
      old_point[i] = point[i] * arg.stencil->stride[i];
      if (arg.stencil->type == 1) old_point[i] /= arg.stencil->mgrid_stride[i];

      if (old_point[i] < arg.dat->d_m[i]) old_point[i] = arg.dat->d_m[i];
      if (old_point[i] > arg.dat->size[i] - 1) old_point[i] = arg.dat->size[i] - 1;
    }

  }
  else if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
    for (int i = 0; i < dim; i++)
      old_point[i] = -100;
  }

}

inline void compute_finer_grid_size(double *dx_f, const int dim,
                                    const BoundingBox *box,const int *size_proj) {

  dx_f[0] = (box->getLocalMax().x - box->getLocalMin().x)
          / ( static_cast<double>(size_proj[0]));
  dx_f[1] = (box->getLocalMax().y - box->getLocalMin().y)
          / ( static_cast<double>(size_proj[1]));

  dx_f[2] = (dim == 3) ?
      (box->getLocalMax().z - box->getLocalMin().z) / (static_cast<double>(size_proj[2])) : 0.0;

}


/* Perform correction due to transition from coarse to fine mesh*/
inline void     correction_due_to_grid(int *grid_point,const ops_stencil stencil,
                                       const double *x_p,const int dim,
                                       const ops_point xmin, double *dx) {
  if (stencil->type == 2) {
    grid_point[0] += (x_p[0] - xmin.x - grid_point[0] * dx[0]) / dx[0];
    grid_point[1] += (x_p[1] - xmin.y - grid_point[1] * dx[1]) / dx[1];
    if (dim >= 3)
      grid_point[2] += (x_p[2] - xmin.z - grid_point[2] * dx[2]) / dx[2];
  }
}


#if __cplusplus >= 201103L

template<typename ParamT>
struct part_grid_param_handler {
  static char* construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, ops_stencil map_stencil,
                         size_t no_particles) {

    if (arg.argtype == OPS_ARG_GBL || arg.argtype == OPS_ARG_GBL_PARTICLE) {
      if (arg.acc == OPS_READ || arg.acc == OPS_WRITE)
        return arg.data;
      else
#ifdef OPS_MPI
        return ((ops_reduction).arg.data)->data + ((ops_reduction)arg.data)->size * block->index;
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

      //Get local address
      int *part2grid = (int *)map->parts_to_grid->data;
      int address = part2grid[0];

      int map_loc[OPS_MAX_DIM];
      get_local_point(map_loc, address, dim, map->binhead->size, d_m); //tODO and modify


      for (int n  = 0 ; n < dim; n++) {
        if (map_stencil->type == 1)
          map_loc[n] /= map_stencil->mgrid_stride[n];
        else if (map_stencil->type == 2)
        map_loc[n] *= map_stencil->mgrid_stride[n];
      }

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
      block->instance->arg_idp[0] = 0; //First point

      return (char *)block->instance->arg_idp;
    }
    else if (arg.argtype == OPS_ARG_IDX_MAP) {
      int d_m[OPS_MAX_DIM];
#ifdef OPS_MPI
      for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
#else
      for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d];
#endif

      //Get local address
      int *part2grid = (int *) map->parts_to_grid->data;
      int address = part2grid[0];

      int map_loc[OPS_MAX_DIM];
      get_local_point(map_loc, address, dim, map->binhead->size, d_m);

      for (int d = 0; d < dim; d++) block->instance->arg_idx_map[d] = map_loc[d];
      for (int d = dim; d < dim; d++) block->instance->arg_idx_map[d] = 0;

      return (char *)block->instance->arg_idx_map;
    }

    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int off_parts,
                              const int *bin_point, const int dim,
                              OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      p += off_parts * arg.elem_size;
    }
    else if (arg.argtype == OPS_ARG_IDP) {
      instance->arg_idp[0] += off_parts;
    }
    else if (arg.argtype == OPS_ARG_IDX_MAP) {
      for (int d = 0; d < dim; d++) instance->arg_idx_map[d] = bin_point[d];
      for (int d = dim ; d < OPS_MAX_DIM; d++) instance->arg_idx_map[d] = 0;
    }
  }

  static void shift_arg(const ops_arg &arg, char* p_a, int off,
                         int dim, int map_point[], OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_IDX) {
      for (int d = 0; d < dim; d++) instance->arg_idx[d] = map_point[d];
      for (int d = dim; d < OPS_MAX_DIM; d++) instance->arg_idx[d] = 0;
    }

  }

  static ParamT get(char *data) {
    return (ParamT) data;
  }

  static void free(char *data) { }

};

template<typename T>
struct part_grid_param_handler<ACC<T>> {
  static char* construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, ops_stencil map_stencil,
                         size_t no_particles) {


    if (arg.argtype == OPS_ARG_DAT) {

      int d_m[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
    for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d] + OPS_sub_dat_list[map->binhead->index]->d_im[d];
#else //OPS_MPI
    for (int d = 0; d < dim; d++) d_m[d] = map->binhead->d_m[d];
#endif

      int *part2grid = (int *)map->parts_to_grid->data;

      int addressx = part2grid[0];

      int map_loc[OPS_MAX_DIM];
      get_local_point(map_loc, addressx, dim, map->binhead->size, d_m); //TODO:

      //Shift mapping data to finer/coarser grid
      switch(map_stencil->type) {
      case 1:
        for (int i = 0; i < dim; i++) map_loc[i] /= map_stencil->mgrid_stride[i];
        break;
      case 2:
        for (int i = 0; i < dim; i++) map_loc[i] *= map_stencil->mgrid_stride[i];
        break;
      }

      //Shift from finer/coarse to actual grid
      switch (arg.stencil->type) {
      case 1:
        for (int i = 0; i < dim; i++) map_loc[i] /= arg.stencil->mgrid_stride[i];
        break;
      case 2:
        for (int i = 0; i < dim; i++) map_loc[i] *= arg.stencil->mgrid_stride[i];
        break;
      }

      addressx = address(ndim, arg.dat->block->instance->OPS_soa ? arg.dat->type_size : arg.dat->elem_size, &map_loc[0],
                         arg.dat->size, arg.stencil->stride, arg.dat->base, arg.dat->d_m);


#ifdef OPS_1D
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], (T*)(arg.data
#elif defined(OPS_2D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], (T *) (arg.data
#elif defined(OPS_3D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2],
                                 (T *) (arg.data
#elif defined(OPS_4D)
      return (char *) new ACC<T>(arg.dim, arg.dat->size[0], arg.dat->size[1], arg.dat->size[2],
                                 arg.dat->size[3], (T *) (arg.data
#else
      return (char *) ((arg.dat->data
#endif
      + addressx));
    }
    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int off_parts,
                              const int *bin_point, const int dim,
                              OPS_instance *instance) { }

  static void shift_arg(const ops_arg &arg, char* p, int off,
                        int dim, int *map_point, OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT) {
      int offset = (arg.dat->block->instance->OPS_soa) ? off : arg.dat->dim * off;

      ((ACC<T> *)p)->next(offset);
    }
  }

  static ACC<T>& get(char *data) {return *((ACC<T> *) data);}

  static void free(char *data) {delete (ACC<T> *) data;}


};

template <typename T>
struct part_grid_param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map, ops_stencil map_stencil,
                         size_t no_particles) {

    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {

      return (char *) new ACCP<T>(arg.dim,  no_particles, (T *) arg.dat->data);
    }
    return nullptr;
  }

  static void shift_part_args(const ops_arg &arg, char *p, int off_parts,
                              const int *bin_point, const int dim,
                              OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      int offset = arg.dat->dim * off_parts;
      ((ACCP<T> *)p)->next(offset);
    }
  }

  static void shift_arg(const ops_arg &arg, char* p_a, int off,
                        int dim, int *map_point, OPS_instance *instance) { }

  static ACCP<T>& get(char *data) { return *((ACCP<T> *) data);}

  static void free(char *data) {delete (ACCP<T> *) data;}


};

//TODO: Need to check limits for the operation of transition to different grids
template <typename...ParamType, typename... OPSARG, size_t ...J>
void ops_par_particle_grid_loop_impl(indices<J...>, void (*kernel)(ParamType... ),
                                     char const *name, ops_particle particle,
                                     ops_particle_mapping map, int dim,
                                     ops_particle_iterate_type iter_type,
                                     double *range, ops_stencil map_stencil,
                                     int *d_mg, int *d_pg,
                                     OPSARG... arguments) {

  constexpr int N = sizeof...(OPSARG);
  ops_arg args[N] = {arguments...};
  int count[OPS_MAX_DIM] = {0};
  ops_block block = particle->block;

  BoundingBox *box = particle->box_block;



  if (map == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty mapping structure");

  if (map->binhead == nullptr || map->bin == nullptr)
    throw OPSException(OPS_INVALID_ARGUMENT, "Empty mapping elements");

  long int *looping_particles = nullptr;
  size_t n_loop_particles{0};

  switch(iter_type) {
  case OPS_PARTICLE_ITERATE_LOCAL:
    n_loop_particles = particle->no_particles;
    looping_particles = (long int *)ops_malloc(sizeof(long int) * n_loop_particles);
    for (size_t i = 0; i < n_loop_particles; i++) looping_particles[i] = i;
    break;
  case OPS_PARTICLE_ITERATE_ALL:
    n_loop_particles = particle->no_particles + particle->no_virtual;
    for (size_t i = 0; i < n_loop_particles; i++) looping_particles[i] = i;
    break;
  case OPS_PARTICLE_ITERATE_RANDOM:
    n_loop_particles = get_particles_in_range(particle->no_particles + particle->no_virtual, particle,
                                             box, map, dim, range, looping_particles);
    break;
  default:
    throw OPSException(OPS_RUNTIME_ERROR, "Invalid iteration type. Please check your "
                       "settings");
  }

  if (n_loop_particles == 0) return;


  int d_mb[OPS_MAX_DIM], d_pb[OPS_MAX_DIM];
#ifdef OPS_MPI
  for (int d = 0; d < dim; d++) {
    d_mb[d] = map->binhead->d_m[d]
            + OPS_sub_dat_list[map->binehead->index]->d_im[d];
    d_pb[d] = map->binhead->d_p[d]
            + OPS_sub_dat_list[map->binhead->index]->d_ip[d];
  }
#else
  for (int d = 0; d < dim; d++) {
    d_mb[d] = map->binhead->d_m[d];
    d_pb[d] = map->binhead->d_p[d];
  }
#endif





  size_t no_particles = particle->no_particles + particle->no_virtual;
  int *off_parts = (int *)ops_malloc(sizeof(int) * n_loop_particles);
  init_offs_particles(dim, n_loop_particles, looping_particles, off_parts);

  //int start[OPS_MAX_DIM];
  int ndim = particle->block->dims;
  //for (int n = 0; n < particle->block->dims;n++) start[n] = 0;

  char *p_a[N] =
      {part_grid_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim,
                                                                           ndim, block, map,
                                                                           map_stencil,
                                                                           no_particles)...};

  int old_indices[N][OPS_MAX_DIM];
  int offs[N] ={};
  /* After construction get first point in grid-Links to 1st element in the list  */
  int *part2grid = (int *)map->parts_to_grid->data;
  int address = part2grid[0];
  int grid_point[OPS_MAX_DIM] = {};
  int bin_point[OPS_MAX_DIM] = {};
  get_local_point(bin_point, address, dim, map->binhead->size, d_mb); //TODO:
  shift_from_grid_to_grid(grid_point, bin_point, dim, map_stencil);

  (void) std::initializer_list<int>{(find_old_indices2(arguments, old_indices[J], dim, grid_point), 0)...};



  int stencil_point[OPS_MAX_DIM]={};

  int size_proj[OPS_MAX_DIM];


  for (int d = 0; d < dim; d++) {
    size_proj[d] = map->binhead->size[d] + d_mb[d] - d_pb[d] - 1;
    if (map_stencil->type == 1) size_proj[d] /= map_stencil->mgrid_stride[d];
    else if (map_stencil->type == 2) size_proj[d] *= map_stencil->mgrid_stride[d];

  }

  /**
   * Getting grid size of finer grid from binhead & exploited stencil
   * Assumption: Uniform grids
   */
  double dx_f[OPS_MAX_DIM] = {};
  compute_finer_grid_size(dx_f, dim, box, size_proj);

  for (size_t i = 0; i < n_loop_particles; i++) {


    int address = part2grid[looping_particles[i]];
    get_local_point(bin_point, address, dim, map->binhead->size, d_mb); //TODO:

    //TODO: Shift particle structures to actual point using offset
    (void) std::initializer_list<int>{(
        part_grid_param_handler<param_remove_cvref_t<ParamType>>::shift_part_args(arguments, p_a[J],
                                                                                 off_parts[i],
                                                                                 bin_point, dim,
                                                                                 block->instance), 0)...};

    /* Transfer to the grid */

    //Part I: Particle position  not corrected (OK for finer to coarse and same grid)
    shift_from_grid_to_grid(grid_point, bin_point, dim, map_stencil);


    //Part II: Correct due to coarse->fine
    correction_due_to_grid(grid_point, map_stencil,
                           (double *)(particle->particle_pos_dat->data
                            + particle->particle_pos_dat->elem_size * looping_particles[i]),
                            dim, box->getLocalMin(), dx_f);



    for (int ipoint = 0; ipoint < map_stencil->points; ipoint++) {

      //Get looping point-Assume that point out of user bound
      for (int d = 0; d < dim; d++)
        stencil_point[d] = map_stencil->stencil[dim * ipoint + d]
                         + grid_point[d];

  //    printf("Stencil point %d [%d %d]\n", ipoint, stencil_point[0], stencil_point[1]);
      //TODO: Get sanity checks for distances in this grid
      int a1 = check_bounds(dim, stencil_point, size_proj, d_mg, d_pg);

   //   printf("Stencil point after bound check [%d %d] (a1 = %d)\n", stencil_point[0], stencil_point[1], a1);
      if (a1 < 0) continue;

      (void) std::initializer_list<int>{(find_offs2(arguments, offs[J], old_indices[J], dim, stencil_point), 0)...}; //TODO:Check

 //     printf("Offs[0] = %d offs[1] = %d\n", offs[0], offs[1]);


      (void) std::initializer_list<int>{
         (part_grid_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], offs[J],
                                                                              dim, stencil_point,
                                                                              block->instance), 0)...};

      kernel((part_grid_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);

      (void) std::initializer_list<int>{(find_old_indices2(arguments, old_indices[J], dim, stencil_point), 0)...};

      //TODO:Copy old po
    }

  }

  //Free structures
  (void) std::initializer_list<int>{
    (part_grid_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...}; //TODO

  ops_free(off_parts);
  ops_free(looping_particles);

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
template <typename... ParamType, typename... OPSARG>
void ops_par_particle_grid_loop(void (*kernel)(ParamType...), char const *name,
                                ops_particle particle, ops_particle_mapping map,
                                int dim, ops_particle_iterate_type iterate_type,
                                double *range, ops_stencil map_stencil, int d_mg[],
                                int d_pg[], OPSARG... arguments) {

  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters do not match the number of ops_arg");

  ops_par_particle_grid_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                                  particle, map, dim, iterate_type, range, map_stencil,
                                  d_mg, d_pg, arguments...);

}
#endif /* C++11 */

#endif /* OPS_C_INCLUDE_OPS_PARTICLE_GRID_SEQ_H_ */
