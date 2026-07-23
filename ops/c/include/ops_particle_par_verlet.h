/*
 * ops_particle_par_verlet.h
 *
 *  Created on: Jun 30, 2026
 *      Author: valantis
 */

#ifndef __OPS_PARTICLE_PAR_VERLET_H_
#define __OPS_PARTICLE_PAR_VERLET_H_

#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "ops_seq_v2.h" //TODO: Check if wew need

#if __cplusplus >= 201103L

template<typename ParamT> struct verlet_param_handler {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         const int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_GBL) {
      if (arg.acc == OPS_READ) return arg.data;
#ifdef OPS_MPI
      return ((ops_reduction)arg.data)->data + ((ops_reduction)arg.data)->size * particle->block->index;
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
          (particleJ->ids != nullptr) ?  ((int *)particleJ->ids->data)[jfirst] : jfirst;
      return (char *) particleJ->block->instance->arg_idj;
    }

    return nullptr;
  }

  static void shift_point(const ops_arg &arg, ops_particle particleI,
                          ops_particle particleJ, char *p, int ipart, int ifirst,
                          int jpart, int jfirst, int ihist_shift,  OPS_instance *instance) {

    if (arg.argtype == OPS_ARG_IDP) {
      instance->arg_idp[0] =
         (particleI->ids != nullptr) ? ((int *)particleI->ids->data)[ipart] : ipart;
    }
    else if (arg.argtype == OPS_ARG_IDJ) {
      instance->arg_idj[0] =
          (particleJ->ids != nullptr) ? ((int *)particleJ->ids->data)[jpart] : jpart;
    }
  }

  static void free(char *) { }

  static ParamT get(char *data) { return (ParamT) data;}
};

template<typename T> struct verlet_param_handler<ACCP<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         const int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      ACCP<T> *data = new ACCP<T>(arg.dim, particleI->no_particles + particleI->no_virtual,
                                  (T *)arg.dat->data + ifirst * arg.dim);
      return (char *) data;
    }

    return nullptr;
  }

  static void shift_point(const ops_arg &arg, ops_particle particleI,
                          ops_particle particleJ, char *p, int ipart, int ifirst,
                          int jpart, int jfirst, int ihist_shift,  OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE) {
      int offset = (ipart- ifirst) * arg.dat->dim;
      ((ACCP<T> *)p)->next(offset);
    }
  }

  static ACCP<T>& get(char *data) {return   *((ACCP<T> *)data);}

  static void free(char *data) {
    delete (ACCP<T> *)data;
  }
};

template<typename T>
struct verlet_param_handler<ACCPJ<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         const int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE_J) {
      ACCPJ<T> *data = new ACCPJ<T>(arg.dim, particleI->no_particles + particleJ->no_virtual,
                                   (T *)arg.dat->data + jfirst * arg.dim);
       return (char *) data;
    }

    return nullptr;
  }

  static void shift_point(const ops_arg &arg, ops_particle particleI,
                          ops_particle particleJ, char *p, int ipart, int ifirst,
                          int jpart, int jfirst, int ihist_shift,  OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_PARTICLE_J) {
      int offset = (jpart- jfirst) * arg.dat->dim;

      ((ACCPJ<T> *)p)->next(offset);
    }
  }

  static ACCPJ<T>& get(char * data) {return *((ACCPJ<T> *)data); }

  static void free(char * data) {
    delete (ACCP<T> *)data;
  }

};

template<typename T>
struct verlet_param_handler<ACC_HIS<T>> {
  static char *construct(const ops_arg &arg, ops_particle particleI, ops_particle particleJ,
                         const int dim, int ifirst, int jfirst) {
    if (arg.argtype == OPS_ARG_DAT_HISTORY) {
      ACC_HIS<T> *data = new ACC_HIS<T>(arg.dat->dim, (T *) arg.history->data->data); //TODO Points no where

      return (char *)data;
    }

    return nullptr;
  }

  static void shift_point(const ops_arg &arg, ops_particle particleI,
                          ops_particle particleJ, char *p, int ipart, int ifirst,
                          int jpart, int jfirst, int ihist_shift,  OPS_instance *instance) {
    if (arg.argtype == OPS_ARG_DAT_HISTORY) {
      int offset =  ihist_shift * arg.dim;
      ((ACC_HIS<T> *)p)->next(offset);
    }
  }

  static ACC_HIS<T>& get (char *data) {return *((ACC_HIS<T> *)data); }

  static void free(char *data) {
    delete (ACC_HIS<T> *) data;
  }
};


template<typename... ParamType, typename... OPSARG, size_t... J>
void ops_part_par_verlet_imp(indices<J...>, void (*kernel)(ParamType...), char const *name,
                             ops_particle particle, ops_particle particleJ, int dim,
                             ops_neighbor_history history_user, OPSARG... arguments) {

  constexpr int N = sizeof...(OPSARG);

  int count[OPS_MAX_DIM] = {0};

  ops_arg args[N] = {arguments...};

  if (particle->nhistories == 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: Verlet iteration requires at least "
                                          "one ops_neighbor_history structure to be defined"
                                          "for the given particle\n");
  ops_neighbor_history history = history_user;
  int nhists = (history_user == nullptr) ? 0 : 1;
  int nops_dat = 0;
  for (int i = 0; i < N; i++) {
    if (args[i].argtype == OPS_ARG_DAT_HISTORY) {
      if (args[i].history->particleI->index != particle->index
          && args[i]. history->particleJ->index != particleJ->index)
        throw OPSException(OPS_RUNTIME_ERROR,"Error: For verlet list defined must be "
                                             " of the same type as the actual particle\n");

      if (history->index != args[i].history->index) {
        history= args[i].history;
        nhists++;
      }
    }
    if (args[i]. argtype == OPS_ARG_DAT) {
      nops_dat++;
    }
  }

  if (nhists == 0)
    throw OPSException(OPS_RUNTIME_ERROR,"Error: Verlet particle loop requires an ops_arg_dat of history type");

  if (nops_dat > 0)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: ops_dat structures not supported for the given loop");

  if (nhists > 1)
    throw OPSException(OPS_RUNTIME_ERROR, "Error: A single neighbor history is supported");

  //Initialize arguments

  int ifirst = 0;
  int jfirst = 0;

  char *p_a[N]
     = {verlet_param_handler<param_remove_cvref_t<ParamType>>::construct(arguments, particle, particleJ,
                                                                         dim, ifirst, jfirst)...};

  int ninters = history->nconts;
  int ipart, jpart;
  int *parts_in_cont = (int *) history->indexing_local->data; //TODO
  //Loop over all contacts
  int iold = 0;

  for (int i = 0; i < ninters; i++) {
    ipart = parts_in_cont[2 * i];
    jpart = parts_in_cont[2 * i + 1];

    (void) std::initializer_list<int>{
      (verlet_param_handler<param_remove_cvref_t<ParamType>>::shift_point(arguments, particle, particleJ, p_a[J],
                                                                          ipart, ifirst, jpart, jfirst, 1,
                                                                          particle->block->instance), 0)...};

    kernel((verlet_param_handler<param_remove_cvref_t<ParamType>>::get(p_a[J]))...);
    iold++;
    ifirst = ipart;
    jfirst = jpart;
  }

  (void) std::initializer_list<int>{
    (part_inter_param_handler<param_remove_cvref_t<ParamType>>::free(p_a[J]), 0)...};

  //
}

template<typename... ParamType, typename... OPSARG>
void ops_particle_par_verlet(void (*kernel)(ParamType...), char const *name,
                             ops_particle particle, ops_particle particleJ, int dim,
                             ops_neighbor_history history,
                             OPSARG... arguments) {

  static_assert(sizeof...(ParamType) == sizeof...(OPSARG),
                "Number of kernel parameters do not match the number of ops_arg");

  ops_part_par_verlet_imp(build_indices<sizeof...(ParamType)>{}, kernel, name,
                           particle, particleJ, dim, history, arguments...);
}

#endif
#endif /* OPS_C_INCLUDE_OPS_PARTICLE_PAR_VERLET_H_ */
