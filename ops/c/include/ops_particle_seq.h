/*
 * ops_particle_seq.h
 *
 *  Created on: Feb 19, 2025
 *      Author: valantis
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

inline int multipl(const int size[], int dir) {
  int prod{1};



  for (int i = 0; i <= dir; i++) {
    prod *= size[i];
  }

  return prod;
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

inline void  get_grid_point_coords(const int igrid,const int dims,const int *size, int *loc_point) {

  int accum{igrid};


  for (int i = dims - 1; i > 0; i--) {

    int sizeL = multipl(size, i - 1);

    loc_point[i] = accum / sizeL;
    accum -= loc_point[i] * sizeL;

  }

  loc_point[0] = accum;
}

/*----------------------------------------------------------------------------*/
/* \brief Finds particles within the iterative region. Particles to iterate for
 *        stored as distances from the previous element.
 *
 * \param[in] nParticles          number of actual particles
 *                                within a bounding region
 * \param[in] box                 pointer to the BoundingBox
 *                                for which a loop must
 *                                will be performed
 * \param[in] map                 an ops_particle_mapping
 *                                object
 * \param[in] dim                 spatial size
 * \param[in] range               the box for performing
 *                                a simulation
 * \param[out] looping_particles  list of particles within range. The particles
 *                                are stored as distance from the previous
 *                                element of the list
 *
 * \return                        number of particles
 *
 */
inline size_t _get_iteration_range(size_t nParticles, BoundingBox *box,
                            ops_particle_mapping map, int dim, double *range,
                            long int *&looping_particles) {


  int inters{0};
  BoundingBox *intersection = ops_find_intersection_region(box, range, inters);

  size_t nloop{0};
  if (inters == 0 ) { //Bounding Box within range
    nloop = nParticles;
    looping_particles = (long int *)ops_malloc(sizeof(long int) * nloop);
    for (size_t i = 0; i < nloop; i++)
      looping_particles[i] = i;
  }
  else if (inters == 1) { //overlapping area
    double *xcrd = (double *)map->grid->data;
    int *bin_head = (int *)map->binhead->data;
    int *bins = (int *)map->bin->data;

    int size_bin{1};
    for (int i = 0; i < dim; i++) size_bin *= map->binhead->size[i];

    size_t nmax{10};
    looping_particles = (long int *)ops_malloc(sizeof(size_t) * nmax);
    size_t ilast{0};

    for (int i = 0; i < size_bin; i++) { //TODO: Modify for other MPI

      //FOR MPI PERFORM SHIFT

      bool is_in = intersection->isCoordinateInBoundingBox(xcrd + dim * i); //TODO: Shift with or without staggering
      if (is_in) {
        int a1 = bin_head[i];
        while (a1 != -1) {
          nloop++;
          if (nloop > nmax) {
            nmax += 100;
            looping_particles = (long int *) ops_realloc(looping_particles, sizeof(long int) * nmax);
          }

          looping_particles[nloop-1] = a1 - ilast;
          ilast = a1;
          a1 = bins[a1];
        }
      }


    }
  }

  delete intersection;
  return nloop;
}

/*------------------------------------------------------------------------------------*/
/* Local function for getting particles within the user defined range when the number
 * of particles does not exceed the number of grid points within the simulation
 *
 * \param[in]        nParticles        total number of particles with the given block
 * \param[in]        particle          pointer to an ops_particle structure
 * \param[in]        box               pointer to the BoundingBox of the given block
 * \param[in]        dim               size of the spatial space
 * \param[in]        range             user defined region
 * \param[in, out]   looping_particles array containing all particles within the user
 *                   defined range
 * \return  the total number of particles within the user defined region
 */
/*------------------------------------------------------------------------------------*/

inline size_t _get_iteration_range_direct(size_t nParticles, ops_particle particle,
                                          BoundingBox *box, int dim, double *range,
                                        long int *&looping_particles) {

  int inters{0};
  BoundingBox *intersection = ops_find_intersection_region(box, range, inters);

  size_t nloop{0};

  if (inters == 0 ) { //Bounding Box within range
    nloop = nParticles;
    looping_particles = (long int *)ops_malloc(sizeof(long int) * nloop);
    for (size_t i = 0; i < nloop; i++)
      looping_particles[i] = i;
  }
  else if (inters == 1) {
    double *xcrds = (double *) particle->particle_pos_dat->data;

    size_t nmax{10};
       looping_particles = (long int *)ops_malloc(sizeof(size_t) * nmax);

    for (int i = 0; i <  nParticles; i++) {
      bool is_in = intersection->isCoordinateInBoundingBox(xcrds + dim * i);
      if (is_in) {
        nloop++;
        if (nloop > nmax) {
          nmax += 100;
          looping_particles = (long int *) ops_realloc(looping_particles, sizeof(long int));
        }

        looping_particles[nloop-1] = i;
      }
    }
  }

  delete intersection;
  return nloop;
}

/*---------------------------------------------------------------------------------------*/
/*
 * \brief Identifying particles within a user defined region. The function identifies
 *        particles within a given region via two different algorithms based on the
 *        ratio of particles to grid_cell and if a map is defined by the user
 *
 * \param[in]     nParticles           number of particles within the given block
 * \param[in]     particle             an ops_particle structure
 * \param[in[     box                  a BoundingBox structure associated with
 *                                     the given block
 * \param[in]     dim                  size of the spatial space
 * \param[in]     range                user defined region for particle looping given in
 *                                     the form [xmin xmax] x [ymin ymax] x [zmin zmax]
 * \param[in,out] looping_particles    array containing all particles within the loop
 */
/*---------------------------------------------------------------------------------------*/

inline size_t getting_looping_particles(size_t nParticles, ops_particle  particle,
                                        BoundingBox *box, ops_particle_mapping map,
                                        int dim, double *range,
                                        long int *&looping_particles) {

  //TODO: Get particle size
  int size[OPS_MAX_DIM];
  for (int i = 0; i < OPS_MAX_DIM; i++)
    size[i]= map->bin->size[i];

  size_t size_tot{1};

  for (int i = 0; i < dim; i++)
    size_tot *= size[i];

  if (size_tot < nParticles && map != nullptr)
    return _get_iteration_range(nParticles, box, map, dim, range,
                                looping_particles);
  else
    return _get_iteration_range_direct(nParticles, particle, box, dim,
                                       range, looping_particles);
}
/*-----------------------------------------------------------------------------*/
/* Computes the first point within the loop for ops_dat structures related
 * to various grid structures
 *
 * \param[in] ops_arg          an ops_arg structure associated with
 *                             grid structures
 * \param[in] offs             pointer shifts based on particle location
 *                             within the current particle structure
 * \param[in] map              map associated with the given grid
 * \param[in] n_loop_particles number of particles for which to loop
 * \param[in] loop_particles   shifter to particle locations in particle
 *                             lists
 */
/*-----------------------------------------------------------------------------*/

inline void  initoffs_particles(const ops_arg &arg, int *&offs, const int ndim,
                                const ops_particle_mapping map,
                                const size_t n_loop_particles,
                                const long int *loop_particles) {
  if (arg.argtype == OPS_ARG_DAT) {
    offs = (int *)ops_malloc((int) n_loop_particles * sizeof(int));

    if (map == nullptr)
      throw OPSException(OPS_RUNTIME_ERROR,"Particle map is not defined\n");

    if (map->binhead == nullptr)
      throw OPSException(OPS_RUNTIME_ERROR, "ops_dat for grid structure is not defined\n");

    int d_mb[OPS_MAX_DIM];
    int size_b[OPS_MAX_DIM];

    for (int i = 0; i < OPS_MAX_DIM; i++) {
      d_mb[i] = map->binhead->d_m[i];
      size_b[i] = map->binhead->size[i];
    }

    int d_m[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++) d_m[i]= arg.dat->d_m[i];
//    int d_p[OPS_MAX_DIM];
//   for (int i = 0; i < OPS_MAX_DIM; i++) d_p[i]= arg.dat->d_p[i];
    int size[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++) size[i]= arg.dat->size[i];

    int zeros[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++)
      zeros[i] = 0;

    int ifirst = get_address(zeros, arg.dat->dim, ndim, d_m, size);

    //Get offset based
    int iPart{0};
    int loc_point[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++)
      loc_point[i] = 0;

    int *part_to_grid = (int *)map->parts_to_grid->data;

    /* Loop over all particles */
    for (size_t i = 0; i < n_loop_particles; i++) {

      /* Get actual particle +*/
      iPart = loop_particles[i];

      /* Get bin and [i,j,k] for the bin where particle projected*/
      int igrid = part_to_grid[iPart];
      get_grid_point_coords(igrid, ndim, size_b, loc_point);

      /* Shift particle to ops_dat structure */
      int igrid_to_dat  = get_address(loc_point, arg.dat->dim, ndim, d_m, size);

      /* Get offs for the given particle */

      offs[i] = igrid_to_dat - ifirst;
      ifirst = igrid_to_dat;
    }
  }
  else if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
    offs = (int *)ops_malloc(n_loop_particles * sizeof(int));
    int ifirst{0};
    int mdim = arg.dat->dim;
    for (size_t i = 0; i < n_loop_particles; i++) {
      offs[i] = mdim * (loop_particles[i] - ifirst);
      ifirst = loop_particles[i];

    }
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
                         ops_particle_mapping map,
                         size_t n_particles) {

    if (arg.argtype == OPS_ARG_GBL || arg.argtype == OPS_ARG_GBL_PARTICLE) {
      if (arg.acc == OPS_READ) return arg.data;
      else
#ifdef OPS_MPI
        return ((ops_reduction).arg.data)->data + ((ops_reduction)arg.data)->size *__OPS_PARTICLE_SEQ_H block->index;
#else
        return ((ops_reduction)arg.data)->data;
#endif
    }
    if (arg.argtype == OPS_ARG_IDX) {

#ifdef OPS_MPI
    sub_block_list sb = OPS_sub_block_list[block->index]; //TODO: Multigrid
    for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = sb->decomp_disp[d];
#else //OPS_MPI
    for (int d = 0; d < dim && d < OPS_MAX_DIM; d++) block->instance->arg_idx[d] = 0;
#endif
    }

    return nullptr;
  }

  static ParamT get(char *data) { //TODO: Shift it for satefyt
    return (ParamT)data;
  }

  static void shift_arg(const ops_arg &arg, char *p, const int offs) {
    //NOT SUPPORTED FOR THE MOMENT
  }

  static void shift_address(const ops_arg &arg, char *&p ,const int offs) {
    if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      p += offs * arg.elem_size;
    }
  }

  static void free(char *) { }

};

template <typename T>struct particle_param_handler<ACC<T>> {
  static char *construct(const ops_arg &arg, int dim, int ndim, ops_block block,
                         ops_particle_mapping map,
                         size_t n_particles) {
    if (arg.argtype == OPS_ARG_DAT) {
      int d_m[OPS_MAX_DIM] = {};
#ifdef OPS_MPI
    for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d] + OPS_sub_dat_list[arg.dat->index]->d_im[d];
#else //OPS_MPI
    for (int d = 0; d < dim; d++) d_m[d] = arg.dat->d_m[d];
#endif

    int start[OPS_MAX_DIM];
    for (int i = 0; i < OPS_MAX_DIM; i++) start[i] = 0;
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
  static ACC<T>& get(char *data) {return *((ACC<T> *) data);}

  //offset contains the calculated shift
  static void shift_arg(const ops_arg &arg, char *p,  const int offset) {
    if (arg.argtype == OPS_ARG_DAT) {
      ((ACC<T>*)p)->next(offset);
    }
  }

  static void shift_address(const ops_arg &arg, char *&p ,const int offs) { }

  static void free(char *data) { delete (ACC<T> *)data;}
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

  static void shift_arg(const ops_arg &arg, char *p, const int offset) {
    ((ACCP<T>*)p)->next(offset);
  }

  static void shift_address(const ops_arg &arg, char *&p ,const int offs) { }


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


template <typename... ParamType, typename... OPSARG, size_t... J>
void ops_particle_par_loop_impl(indices<J...>, void (*kernel)(ParamType...),
                                char const *name, ops_particle particle,
                                ops_particle_mapping map,
                                int dim, double *range_particles, OPSARG... arguments)
{
  constexpr int N = sizeof...(OPSARG);

  int count[OPS_MAX_DIM] = {0};

  if (particle->no_particles == 0) return;

  /* In case of empty set default list */ //TODO:
  ops_particle_mapping map_part
     = (map != nullptr) ? map : nullptr; //particle->mapping_list[particle->def_list];
  ops_block block = particle->block;
  size_t Nlist{0}, Nmax{10};

  size_t nParticles = map_part->nParticles; //TODO: Vrf what is stored herein
  size_t no_particles = particle->no_particles;

  /* Get range of elements */
  BoundingBox *box = particle->box_block;


  /* First identify intersection region */
  long int *looping_particles = nullptr;


  //Init offsets to the points


  /* looping_particles is set to it self */

  size_t  n_loop_particles = _get_iteration_range(particle->no_particles, box, map_part,
                                                  dim, range_particles, looping_particles);



  //Add here the mapping offs
  int *offs[N];
  for (int i = 0; i < N; i++)
    offs[i] = nullptr;
//TODO: Remove some structures prior to loop keep only for elements-we can use only the particle.
  (void) std::initializer_list<size_t>{(initoffs_particles(arguments, offs[J], particle->block->dims,
                                                           map_part, n_loop_particles,
                                                           looping_particles), 0)...};

  /* Set start for grid points */
  int start[OPS_MAX_DIM];
  int ndim = particle->block->dims;
  for (int n = 0; n < particle->block->dims; n++)
    start[n] = 0;

  char *p_a[N] =
   {particle_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, dim,
                                                                       ndim, block,
                                                                       map_part, no_particles)...};

  //We may need the offset for computing data into different structures

//TODO: For the data, we need a different process to access data for stencil.

  //TODO: Exchange data for grid structures prior to loop if necessary

  /* Loop over all particles */
  for (size_t iPart = 0; iPart < n_loop_particles;iPart++) {
    (void) std::initializer_list<int>{(
      particle_param_handler<param_remove_cvref_t<ParamType>>::shift_arg(arguments, p_a[J], offs[J][iPart]), 0)...};

     kernel((particle_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
  }

  (void) std::initializer_list<int>{
    (particle_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...};

  for (int i = 0; i < N; i++)
    ops_free(offs[i]);

  //TODO: Set functionality
  ops_free(looping_particles);
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
 * @param particle_range    Bounding box of the simulation loop. Defined as
 *                          [xmin xmax ymin ymax zmin zmax]
 * @param arguments         a list of ops_arg arguments
 *
 * NOTE: We can remove later on the particle_range and assume that particles
 *       are within a box.
 */

template <typename... ParamType, typename... OPSARG>
void ops_particle_par_loop(void (*kernel)(ParamType...), char const *name,
                       ops_particle particle, ops_particle_mapping map,
                       int dim, double *particle_range,
                       OPSARG... arguments) {
  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters should match the number of ops_arg");

  ops_particle_par_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                             particle, map, dim, particle_range, arguments...);
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
