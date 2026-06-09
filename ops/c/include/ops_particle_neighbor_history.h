/*
 * ops_particle_neighbor_history.h
 *
 *  Created on: May 19, 2026
 *      Author: valantis
 */

#ifndef _OPS_PARTICLE_NEIGHBOR_HISTORY_H_
#define _OPS_PARTICLE_NEIGHBOR_HISTORY_H_

#include "ops_lib_core.h"


/**
 * Function that updates neighbor history after maps are build
 *
 * @param kernel    Kernel function for initializing points within the contact history
 * @param name      name of the kernel function for diagnostic inputs
 * @param history   An ops_neighbor_history_object which will be updated at this point
 * @param mapI      An ops_particle_mapping linked to first particle type
 * @param mapJ      An ops_particle_mapping linked to second particle type
 * @param stencil   An ops_stencil used to access cells of mapJ from cells of mapI
 * @param dim       Size of the physical problem
 * @param arguments Variables to initialize the neighbor history
 */

template<typename T, typename... ParamType, typename... args>
void ops_particle_update_neighbor_history(void (*kernel)(T *, ParamType...), char const *name,
                                          ops_neighbor_history history, ops_particle_mapping mapI,
                                          ops_particle_mapping mapJ, ops_stencil stencil,
                                          int dim, args... argument);

/**
 * Function that setup the neighbor history for particle simulations
 * @param kernel    Kernel function for initializing points within the contact history
 * @param name      name of the kernel function for diagnostic inputs
 * @param history   An ops_neighbor_history_object which will be updated at this point
 * @param mapI      An ops_particle_mapping linked to first particle type
 * @param mapJ      An ops_particle_mapping linked to second particle type
 * @param stencil   An ops_stencil used to access cells of mapJ from cells of mapI
 * @param dim       Size of the physical problem
 * @param arguments Variables to initialize the neighbor history
 */
template<typename T, typename... ParamType, typename... args>
void ops_particle_setup_neighbor_history(void (*kernel)(T *, ParamType...), char const *name,
                                         ops_neighbor_history history, ops_particle_mapping mapI,
                                         ops_particle_mapping mapJ, ops_stencil stencil,
                                         int dim, args... argument);

#ifdef OPS_MPI
  #include "ops_particle_neighbor_history_mpi.tpp"
#else
#include "ops_particle_neighbor_history_seq.tpp"
#endif


#endif /* OPS_C_INCLUDE_OPS_PARTICLE_NEIGHBOR_HISTORY_H_ */
