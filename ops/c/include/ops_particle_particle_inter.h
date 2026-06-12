/*
 * ops_particle_particle_inter.h
 *
 *  Created on: Feb 9, 2026
 *      Author: valantis
 */

#ifndef __OPS_PARTICLE_PARTICLE_INTER_H_
#define __OPS_PARTICLE_PARTICLE_INTER_H_

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "ops_seq_v2.h" //TODO: Check if wew need

inline int get_global_bin(double xmin, double xmax, double dx, double point) {

  int ix = -1;
  double epsilon = 1.e-12;

  if ((point >= xmin - epsilon) && (point <= xmax + epsilon)) {
    ix = ops_floor((point - xmin) / dx);
  }

  if (ix < 0)
    ix = (int) ops_floor((point - xmin) / dx + epsilon);


  return ix;

}

inline int get_map_address(const int point[], int dim, ops_dat dat) {

  int address = 0;
  int prod = 1;
  for (int i = 0; i < dim; i++) {
#ifdef OPS_MPI
    int d_m = OPS_sub_dat_list[dat->index]->d_im[i] + dat->d_m[i];
#else
    int d_m = dat->d_m[i];
#endif

    int ix = point[i]  - d_m;
    if (ix < 0 || ix > dat->size[i] - 1)
      return -1; //Safety
    address += ix * prod;
    prod *= dat->size[i];
  }

  return address;
}

//TODO: Need check
inline void shift_point_to_other_grid(int *point2mapJ, int *point2mapI,
                                      int dim, ops_stencil stencil) {
  switch (stencil->type) {
  case 0:
    for (int i = 0; i < dim; i++) point2mapJ[i] = point2mapI[i];
    break;
  case 1:
    for (int i = 0; i < dim; i++) point2mapJ[i] = point2mapI[i] / stencil->mgrid_stride[i];
    break;
  case 2:
    for (int i = 0; i < dim; i++) point2mapJ[i] = point2mapI[i] * stencil->mgrid_stride[i];
    break;
  }
}



inline int get_first_point(int range[], int dim, ops_particle_mapping map) {

  int point[OPS_MAX_DIM] = {};
  for (int ip = 0; ip < dim; ip++)
    point[ip] = range[2 * ip];

  int *bin_head = (int *) map->binhead->data;
  //Get  location of start at the binhead list
  int address = get_map_address(point, dim, map->binhead);

  return (address > -1) ? bin_head[address] : 0;

}

inline int get_first_point_transform(int range[], int dim, ops_particle_mapping map,
                                     ops_stencil stencil) {

  int map_point[OPS_MAX_DIM];
  switch (stencil->type) {
  case 0: //Projection to the same cell
    for (int i = 0; i < dim; i++) map_point[i] = range[2 * i];
    break;
  case 1: //Shift to a coarser grid: Use prolong stencil
    for (int i = 0; i < dim; i++) map_point[i] = range[2 * i] / stencil->mgrid_stride[i];
    break;
  case 2: //Shift to finer grid: Use restrict stencil (or shift them around)
    for (int i = 0; i < dim; i++) map_point[i] = range[2 * i] * stencil->mgrid_stride[i];
    break;
  default:
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: This type of stencil is not supported\n");
  }

  int address = get_map_address(map_point, dim, map->binhead);

  int *binhead = (int *)map->binhead->data;
  return (address > -1) ? binhead[address] : 0;
}

inline int get_mapping_address(int *point, int *d_m,int  *size, int dim) {

  int address = 0;
  int prod = 1;
  for (int i = 0; i < dim; i++) {
    int ix = point[i]- d_m[i];

    if (ix < 0 || ix  > size[i] - 1) return -1;

    address += ix * prod;
    prod *= size[i];
  }

  return address;
}


inline void find_local_intersection(ops_block block, BoundingBox *box, ops_particle_mapping map, int dim,
                                    double  range[], int range_map[]) {

  int range_gl[2 * OPS_MAX_DIM];
#ifdef OPS_MPI
  sub_dat *sd = OPS_sub_dat_list[map->binhead->index];
  sub_block *sb = OPS_sub_block_list[block->index];
#endif

  for (int  i = 0; i < dim; i++) {
    range_gl[2 * i] = get_global_bin(box->getGlobalMin(i), box->getGlobalMax(i),
                                      map->dx[i], range[2 * i]);
    range_gl[2 * i + 1] = get_global_bin(box->getGlobalMin(i), box->getGlobalMax(i),
                                         map->dx[i], range[2 * i + 1]);

//TODO: Need to ensure that we get all
#ifdef OPS_MPI
    //Init data structures->Rmv virtual cells
    range_map[2 * i] = sd->decomp_disp[i] - (map->binhead->base[i] + map->binhead->d_m[i]);
    range_map[2 * i + 1] = range_map[2 * i] + (sd->decomp_size[i] + map->binhead->base[i]
                                               + map->binhead->d_m[i] - map->binhead->d_p[i]);
    int decomp_disp = range_map[2 * i];
    int decomp_size = (  sd->decomp_size[i] + map->binhead->base[i]
                       + map->binhead->d_m[i] - map->binhead->d_p[i]);
    if (range_map[2 * i] >= range_gl[2 * i]) range_map[2 * i] = map->binhead->d_m[i] + map->binhead->base[i]
                                                              + sd->d_im[i];
    else range_map[2 * i] = range_gl[2 * i] - range_map[2 * i];
    if (sb->id_m[i] == MPI_PROC_NULL && range_gl[2 * i] < 0) range_map[2 * i] = range_gl[2 * i];

    //Set end of loop
    if (range_map[2 * i + 1] >= range_gl[2 * i + 1])
      range_map[2 * i + 1] = range_gl[2 * i + 1] - decomp_disp;
    else
      range_map[2 * i + 1] = map->binhead->size[i];

    if (sb->id_p[i] == MPI_PROC_NULL && (range_gl[2 *i + 1] > decomp_disp + decomp_size))
      range_map[2 * i + 1] += (range_gl[2 * i + 1] - decomp_disp - decomp_size);
#else
    range_map[2 * i] = (range_gl[2 * i] == 0) ? -1 : range_gl[2 * i];
    range_map[2 * i + 1] = (range_gl[2 * i + 1] == map->binhead->size[i] + map->binhead->d_m[i] - map->binhead->d_p[i]) ? map->binhead->size[i] : range_gl[2 * i + 1];
#endif
  }

  if (block->dims < 3) {
    range_map[2 * 2] = 0;
    range_map[2 * 2 + 1] = 1;
  }
}


#if __cplusplus >= 201103L

template<typename ParamT> struct part_inter_param_handler {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         int dim, int ifirst, int jfirst) {

    if (arg.argtype == OPS_ARG_GBL) {
      if (arg.acc == OPS_READ) return arg.data;
      else
#ifdef OPS_MPI
      return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * particleI->block->index;
#else
      return ((ops_reduction)arg.data)->data;
#endif
    }
    else if (arg.argtype == OPS_ARG_IDP) {

      particleI->block->instance->arg_idp[0] =
          (particleI->ids != nullptr) ? ((int *)particleI->ids->data)[ifirst] : ifirst;
      return  (char *) particleI->block->instance->arg_idp;
    }
    else if (arg.argtype == OPS_ARG_GBL_PARTICLE) {
      throw OPSException(OPS_RUNTIME_ERROR, "ERROR: This type of arg is not currently"
                         " supported");
    }
    else if (arg.argtype == OPS_ARG_IDJ) {
      particleJ->block->instance->arg_idj[0] =
          (particleJ->ids != nullptr) ?  ((int *)particleI->ids->data)[jfirst] : jfirst;
      return (char *) particleJ->block->instance->arg_idj;
    }

    return nullptr;
  }

  static void shift_point_arg(const ops_arg &arg,ops_particle particle, char *p,
                              int ishift, int ifirst, OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_IDP) {
      int iloc = ishift + ifirst;
      instance->arg_idp[0] =
          (particle->ids != nullptr) ? ((int *)particle->ids->data)[iloc] : iloc;
    }
  }

  static void shift_point_j_arg(const ops_arg &arg, ops_particle particle, char *p,
                                int ishift, int ifirst, OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_IDJ) {
      int iloc = ishift + ifirst;
      instance->arg_idj[0] =
          (particle->ids != nullptr) ? ((int *)particle->ids->data)[iloc] : iloc;
    }

  }

  static void shift_history(const ops_arg &arg, char *&p, const int ip,
                            const int tagJ) { }

  static ParamT get(char *data) { return (ParamT) data;}

  static void free(char *) { }

};

template<typename T> struct part_inter_param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         int dim, int ifirst, int jfirst) {

    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ACCP<T> *datap = new ACCP<T>(arg.dim, particleI->no_particles + particleI->no_virtual,
                                   (T *)arg.dat->data + ifirst * arg.dim);

      return (char *)datap;
    }

    return nullptr;
  }

  static void shift_point_arg(const ops_arg &arg, ops_particle particle, char *p,
                              int ishift, int ifirst, OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      int offset = ishift * arg.dat->dim;
      ((ACCP<T> *)p)->next(offset);
    }
  }

  static void shift_point_j_arg(const ops_arg &arg, ops_particle particle, char *p,
                                int ishift, int ifirst, OPS_instance *instance) {

  }

  static void shift_history(const ops_arg &arg, char *&p, const int ip,
                            const int tagJ) { }

  static ACCP<T>& get(char * data) {return *((ACCP<T> *)data); }

  static void free(char * data) {
    delete (ACCPJ<T> *)data;
  }
};

template<typename T> struct part_inter_param_handler<ACCPJ<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE_J) {
      ACCPJ<T> *datap = new ACCPJ<T>(arg.dim, particleJ->no_particles + particleJ->no_virtual,
                                     (T *)arg.dat->data + jfirst * arg.dim);

      return (char *)datap;
    }

    return nullptr;
  }

  static void shift_point_arg(const ops_arg &arg, ops_particle particle, char *p,
                              int ishift, int ifirst, OPS_instance *instance) {

  }

  static void shift_point_j_arg(const ops_arg &arg, ops_particle particle, char *p,
                                int ishift, int ifirst, OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_DAT_PARTICLE_J) {
      int offset = ishift * arg.dat->dim;
      ((ACCPJ<T> *) p)->next(offset);
    }

  }

  static void shift_history(const ops_arg &arg, char *&p, const int ip,
                            const int tagJ) { }

  static ACCPJ<T>& get(char * data) {return *((ACCPJ<T> *)data); }

  static void free(char * data) {
    delete (ACCPJ<T> *)data;
  }
};

template<typename T> struct part_inter_param_handler<ACC_HIS<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_DAT_HISTORY) {
      printf("arg.dat name %s:\n", arg.dat->name);
      ACC_HIS<T> *data = new ACC_HIS<T>(arg.dat->dim, (T *) arg.history->data->data); //TODO Points no where

      return (char *)data;
    }

    return nullptr;
  }

  static void shift_point_arg(const ops_arg &arg, ops_particle particle, char *p,
                              int ishift, int ifirst, OPS_instance *instance) { }

  static void shift_point_j_arg(const ops_arg &arg, ops_particle particle, char *p,
                                int ishift, int ifirst, OPS_instance *instance) { }

  static void shift_history(const ops_arg &arg, char *&p, const int ip,
                            const int tagJ) {
    if (arg.argtype == OPS_ARG_DAT_HISTORY) {
      int offset = 0;
      ops_neighbor_history history = arg.history;
      int index;
      int nmax = history->num_neighsI;
      for (int i = 0; i < ((int *) history->n_partnersI->data)[ip]; i++) {
        if (tagJ == ((int *) history->partnersI->data)[ip * nmax + i]) {
 //         printf("tagJ = %d found in %d\n", tagJ, i);
          offset = ((int *) history->indexI->data)[ip * nmax + i] * arg.dat->dim;
          break;
        }
      }

      ((ACC_HIS<T> *)p)->next(offset);
      ((ACC_HIS<T> *)p)->update_old_offset(offset);

    }
  }

  static ACC_HIS<T>& get (char *data) {return *((ACC_HIS<T> *)data); }

  static void free(char *data) {
    delete (ACC_HIS<T> *) data;
  }
};

template<typename... ParamType, typename... OPSARG, size_t... J>
void ops_part_inter_loop_impl(indices<J...>, void (*kernel)(ParamType...), char const *name,
                              ops_particle particleI, ops_particle_mapping mapI,
                              ops_particle particleJ, ops_particle_mapping mapJ,
                              ops_stencil stencil, ops_particle_iterate_type iter_type,
                              int dim, double *range, OPSARG... arguments) {

  /* Part I: Sanity checks */

  //Check I: Particle I and Particle J do not owned by the same block
  if (particleI->block->index != particleJ->block->index)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ParticleI and particleJ are not assigned "
                       "to the same block (region");

  if (mapI->particle->index != particleI->index)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Map for particle I not related to particle I");

  if (mapJ->particle->index != particleJ->index)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Map for particle J does not particles J");

  if (stencil->points <= 1)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Stencil for accessing interactions between "
                                          "I and J is local");

#ifdef OPS_MPI
  sub_block *sb = OPS_sub_block_list[particleI->index];
  if (!sb->owned) return;
  if (particleI->no_particles + particleI->no_virtual == 0) return;
  if (particleJ->no_particles + particleJ->no_virtual == 0) return;
#else
  if (particleI->no_particles + particleI->no_virtual == 0) return;
  if (particleJ->no_particles + particleJ->no_virtual == 0) return;
#endif
  constexpr int N = sizeof...(OPSARG);
  ops_arg args[N] = {arguments ...};
  ops_neighbor_history history = nullptr;
  int nhists = 0;
  for (int i = 0; i < N; i++) {
    if (args[i].argtype == OPS_ARG_DAT_HISTORY) {
      nhists += 1;
      int hist_index = args[i].hist_index;
      history =
          args[i].dat->block->instance->OPS_block_list[particleI->block->index].histories[hist_index];
      printf("history->index = %d\n", history->index);
    }
  }

  if (nhists > 1)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: The number of assigned histories exceed the "
                                          "required number for particle/particle loops ");

  if (nhists == 1 && (particleI->index != history->particleI->index))
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ParticleI is not assinged as the first "
                                          " particle of the neighbor history");
  if (nhists == 1 && (particleJ->index != history->particleJ->index))
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: ParticleJ is not assinged as the second "
                                          " particle of the neighbor history");

  if (particleI->ids == nullptr && nhists == 1)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Neighbor history requires particle ids");

  if (particleJ->ids == nullptr && nhists == 1)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Neighbor history requires particle ids");


  int *tagsJ = (int *)particleJ->ids->data;
  //Check III: For MPI
#ifdef OPS_MPI
  //Check if we have virtual maps for both (or at least of particle J)
  if (mapI->mapping_type != OPS_WITH_VIRTUAL ||
      mapJ->mapping_type != OPS_WITH_VIRTUAL)
    throw OPSException(OPS_RUNTIME_ERROR, "ERROR: Map of particles I  and/or of particles J does not "
                                          "support virtual particles\n");
#endif

  /* Part II: Find loop bounds */
  int range_map[2 * OPS_MAX_DIM] = {};

  int d_mI[OPS_MAX_DIM] = {};
  int d_pI[OPS_MAX_DIM] = {};
  int d_mJ[OPS_MAX_DIM] = {};
  int d_pJ[OPS_MAX_DIM] = {};

  int sizeI[OPS_MAX_DIM] = {};
  int sizeJ[OPS_MAX_DIM] = {};
  //Get d_m d_p
//#ifdef OPS_MPI

  //TODO: Add the first element directly on the list and shift with bins
  if (particleI->no_particles + particleI->no_virtual == 0) return;
  if (particleJ->no_particles + particleJ->no_virtual == 0) return;

#ifdef OPS_MPI

    for (int i = 0; i < dim; i++) {
    d_mI[i] = mapI->binhead->d_m[i] + OPS_sub_dat_list[mapI->binhead->index]->d_im[i];
    d_pI[i] = mapI->binhead->d_p[i] + OPS_sub_dat_list[mapI->binhead->index]->d_ip[i];
    d_mJ[i] = mapJ->binhead->d_m[i] + OPS_sub_dat_list[mapJ->binhead->index]->d_im[i];
    d_pJ[i] = mapJ->binhead->d_p[i] + OPS_sub_dat_list[mapJ->binhead->index]->d_ip[i];

  }
#else
  for (int i = 0; i < dim; i++) {
    d_mI[i] = mapI->binhead->d_m[i];
    d_pI[i] = mapI->binhead->d_p[i];
    d_mJ[i] = mapJ->binhead->d_m[i];
    d_pJ[i] = mapJ->binhead->d_p[i];
  }
#endif

  for (int i = 0; i < dim; i++) {
    sizeI[i] = mapI->binhead->size[i];
    sizeJ[i] = mapJ->binhead->size[i];
  }
  for (int i = dim; i < OPS_MAX_DIM; i++) {
    sizeI[i] = mapI->binhead->size[i];
    sizeJ[i] = mapJ->binhead->size[i];
  }

  //Setting up range
  if (iter_type != OPS_PARTICLE_ITERATE_RANDOM) {
    for (int i = 0; i < dim; i++) {
      range_map[2 * i] = d_mI[i];
      range_map[2 * i + 1] = mapI->binhead->size[i] + d_mI[i]; //- d_pI[i]; //TODO: check
    }


  }
  else {
    //Get xlo xmax of bounding box
    find_local_intersection(particleI->block, particleI->box_block, mapI, dim, range, range_map);
  }

  for (int i = dim; i < OPS_MAX_DIM; i++) {
    range_map[2 * i] = 0;
    range_map[2 * i + 1] = 1;
  }


  /* Part III: Initialize OPSARG arguments*/

  //Get the first-particle for mapI
  int ifirst = get_first_point(range_map, dim, mapI);

  //Need to find the first particle for mapJ
  int jfirst = get_first_point_transform(range_map, dim, mapJ, stencil); //TODO


  char *p_a[N] = {part_inter_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, particleI, particleJ,
                                                                                       dim, ifirst, jfirst)...};

  int *binI = (int *) mapI->bin->data;
  int *binJ = (int *) mapJ->bin->data;



  /* Part IV: Perform inner loop */


  int ipart = ifirst;
  int jpart = jfirst;

  int point2mapI[OPS_MAX_DIM] = {};
  int point2mapJ[OPS_MAX_DIM] = {};
  int map_pointJ[OPS_MAX_DIM] = {};
  for (int k = range_map[2 * 2]; k < range_map[2 * 2 + 1]; k++) {
    for (int j = range_map[2 * 1]; j < range_map[2 * 1 + 1]; j++) {
      for (int i = range_map[0]; i < range_map[1]; i++) {
        int address = (i - d_mI[0]) + (j - d_mI[1]) * sizeI[0]
                    + (k - d_mI[2]) * sizeI[0] * sizeI[1];


        int ipart = ((int *) mapI->binhead->data)[address];

        point2mapI[2] = k;
        point2mapI[1] = j;
        point2mapI[0] = i;

        shift_point_to_other_grid(point2mapJ, point2mapI, dim, stencil);

        while (ipart != - 1) {
           int ishift = ipart - ifirst;

            (void) std::initializer_list<int>{
              (part_inter_param_handler<param_remove_cvref_t<ParamType>>::shift_point_arg(arguments, particleI, p_a[J],
                                                                                          ishift, ifirst,
                                                                                          particleI->block->instance), 0)...};
            //TODO: Inner loops
            for (int isten = 0; isten < stencil->points; isten++) {

              //Get inner point
              for (int d = 0; d < dim; d++)
                map_pointJ[d] = stencil->stencil[isten * dim + d] + point2mapJ[d];

              int addressJ = get_mapping_address(map_pointJ, d_mJ, mapJ->binhead->size, dim);

              if (addressJ < 0) continue;

              int jpart = ((int *)mapJ->binhead->data)[addressJ];


              while (jpart != - 1) {

  //              printf("Ipart = %d and j = %d\n", ipart, jpart);

                int jshift = jpart - jfirst;


                if (ipart == jpart) { jpart = binJ[jpart]; continue;}


                //Get tags for particle (I, J) get history
                (void) std::initializer_list<int>{
                  (part_inter_param_handler<param_remove_cvref_t<ParamType>>::shift_history(arguments,
                                                                                          p_a[J], ipart, tagsJ[jpart]),0)...};
                //Shift j-arguments
                (void) std::initializer_list<int>{
                  (part_inter_param_handler<param_remove_cvref_t<ParamType>>::shift_point_j_arg(arguments, particleJ, p_a[J],
                                                                                               jshift, jfirst,
                                                                                               particleJ->block->instance), 0)...};

                //Perform Kernel
                kernel((part_inter_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);

                jfirst = jpart;

                jpart = binJ[jpart];
              }
            }


            ifirst = ipart;

            ipart = binI[ipart];
          }

      }
    }
  }

  /* Part V: Clean-up */
  (void) std::initializer_list<int>{
    (part_inter_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...};
}

/**
 * Perform an iteration loop to handle the interaction between a particle of type main with particles
 * of other types as defined by secondary. The function executes the user-efined kernel passing data
 * as specified by the list arguments. The particles that will interact with particles of second are
 * defined based on the ops_particle_iterate_type or the range. Material properties can be given
 * based on ops_arg_gbl structues
 *
 * Arguments to kernels are passed as ACCP& references for the dat structures linked to particle
 * and ACCPJ& references for particle of second type.
 *
 * @param kernel        user kernel for computing the interaction between particles of type main
 *                      nad particle of type second.
 * @param name          a name of the particle/particle interaction loop
 * @param particleI     an ops_particle structure that defines the I particle in the interaction
 *                      kernel
 * @param mapI          an ops_particle_mapping structure that yields access to particles of type
 *                      I
 * @param particleJ     an ops_particle structure that define J particles in the interaction
 *                      kernel
 * @param map_sec       an ops_particle_mapping structure that sets the particles of J type
 *                      within the given block
 * @param stencil       stencil for the particle of type I to access particles of type J
 * @param iter_type     mode for accessing particles of type I and type J
 * @param dim           size of physical space
 * @param range         loop bounds in case of random selection in the form {xlow, xmax}x{ylow, ymax} ....
 *
 * @param arguments     a list of ops_arg arguments
 */

template<typename... ParamType, typename... OPSARG>
void ops_particle_inter_loop(void (*kernel)(ParamType...), char const *name,
                             ops_particle particleI, ops_particle_mapping mapI,
                             ops_particle particleJ, ops_particle_mapping mapJ,
                             ops_stencil stencil, ops_particle_iterate_type iter_type,
                             int dim, double *range, OPSARG... arguments) {

  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters do not match the number of ops_arg");

  ops_part_inter_loop_impl(build_indices<sizeof...(ParamType)>{}, kernel, name,
                           particleI, mapI, particleJ, mapJ, stencil,
                           iter_type, dim, range, arguments...); //TODO
}
#endif
#endif /* OPS_C_INCLUDE_OPS_PARTICLE_PARTICLE_INTER_H_ */
