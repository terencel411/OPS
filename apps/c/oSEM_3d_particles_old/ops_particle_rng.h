/*
 * ops_particle_rng.h -- ops_fill_random_uniform for particle dats
 *
 * ==================================================================== *
 *  THIS IS A CANDIDATE FOR THE OPS LIBRARY, NOT APPLICATION CODE.
 * ==================================================================== *
 *
 * It lives here only because the OPS tree is read-only in this workspace.
 * Nothing in it is specific to eddies or to the SEM. It is meant to be moved
 * verbatim into ops/c/src/core/ops_lib_core.cpp, next to
 * ops_fill_random_uniform_host, with the declarations going into
 * ops/c/include/ops_lib_core.h beside ops_fill_random_uniform -- at which point
 * `inline` comes off and this file is deleted.
 *
 *
 * WHY IT HAS TO EXIST
 *
 * ops_fill_random_uniform() cannot be called on a particle dat. Its last act is
 * ops_set_halo_dirtybit3, and to make the argument for it, it builds an
 * ops_arg_dat (ops_lib_core.cpp:2618):
 *
 *     ops_arg arg = ops_arg_dat(dat, dat->dim, ..., OPS_WRITE);
 *
 * ops_arg_dat rejects any dat with is_particle set (ops_lib_core.cpp:1229-1233):
 *
 *     Error: ops_arg_dat_opt cannot be called for particle_ops_dat structure
 *
 * That tail is the ONLY incompatibility. Everything above it already works:
 *
 *   - the fill is a flat walk over cumsize = dat->dim * prod(dat->size), and
 *     ops_dat_init_metadata_core sets size[n] = 1 for n >= block->dims, so a
 *     particle dat is {Nmax,1,1,1,1} and cumsize comes out as exactly
 *     dim * Nmax -- which is the allocation
 *     (ops_decl_particle_dat_char: type_size * dim * dataset_size[0]);
 *   - by the time the exception is raised the numbers are written and
 *     dirty_hd is set.
 *
 * And halo bookkeeping is meaningless for particle storage in any case: a
 * particle dat has no block halo to dirty. So these functions are the existing
 * bodies with the tail removed, and nothing else changed. The bodies are kept
 * deliberately close to the originals, repetition and all, so that the diff
 * against ops_fill_random_uniform_host is easy to read.
 *
 *
 * THE GENERATOR IS AN ARGUMENT, AS IT IS IN THE HOST VERSIONS
 *
 * ops_fill_random_uniform_host takes std::mt19937& (ops_internal2.h:380); only
 * the public ops_randomgen_init fixes the seeding. That matters, because
 * ops_randomgen_init_host (ops_lib_core.cpp:2570-2579) IGNORES its `options`
 * argument and unconditionally applies
 *
 *     seed + my_global_rank * 2654435761u        when comm size > 1
 *
 * so anything built on it is rank-count dependent. ops_particle_randomgen_init
 * below gives `options` the meaning the existing signature already declares.
 */

#ifndef _OPS_PARTICLE_RNG_H_
#define _OPS_PARTICLE_RNG_H_

#include <climits>
#include <cstring>
#include <random>

/**
 * Seed a generator for filling particle dats.
 *
 * @param seed     base seed
 * @param options  0 -- every rank gets the SAME stream
 *                 1 -- seed + my_global_rank * 2654435761u, i.e. what
 *                      ops_randomgen_init_host does unconditionally
 * @param gen      generator to seed
 *
 * Which one a particle set wants depends on what its particles ARE:
 *
 *   options = 0  when every rank holds the same logical set -- e.g. a global
 *                list instantiated everywhere and then culled to the rank that
 *                owns each member. All ranks must agree on the values or they
 *                disagree about the set itself.
 *
 *   options = 1  when each rank owns distinct particles and the streams should
 *                be independent.
 *
 * The existing ops_randomgen_init takes an `options` argument and discards it,
 * so there is no way to ask for the first. That is the reason this exists
 * rather than the caller simply calling ops_randomgen_init.
 */
inline void ops_particle_randomgen_init(unsigned int seed, int options,
                                        std::mt19937 &gen) {
  if (options == 0) {
    gen.seed(seed);
    return;
  }

  const int comm_global_size = ops_num_procs();
  const int my_global_rank   = ops_get_proc();

  if (comm_global_size == 1)
    gen.seed(seed);
  else
    gen.seed(seed + my_global_rank * 2654435761u);
}

/**
 * Fill a particle ops_dat with uniform random numbers.
 *
 * The particle counterpart of ops_fill_random_uniform_host
 * (ops_lib_core.cpp:2581). Doubles and floats are drawn from [0,1).
 *
 * @param dat  a particle ops_dat -- dat->is_particle must be set
 * @param gen  a generator, seeded by the caller (see
 *             ops_particle_randomgen_init)
 */
inline void ops_fill_random_uniform_particle(ops_dat dat, std::mt19937 &gen) {

  /* The mirror of the guard in ops_arg_dat that makes this function necessary:
     that one rejects particle dats, this one rejects grid dats. */
  if (!dat->is_particle) {
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: ops_fill_random_uniform_particle expects a "
                       "particle ops_dat; use ops_fill_random_uniform for a "
                       "grid ops_dat");
  }

  size_t cumsize = dat->dim;
  const char *type = dat->type;

  /* Identical to the grid version, deliberately. For a particle dat
     size = {Nmax,1,1,...}, so this is dim * Nmax -- the whole allocation. */
  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < cumsize; i++) {
      ((double *)dat->data)[i] = distribution(gen);
    }
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < cumsize; i++) {
      ((float *)dat->data)[i] = distribution(gen);
    }
  }
  else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0 || strcmp(type, "integer(kind=4)") == 0) {
    /* DELIBERATE DIVERGENCE FROM ops_fill_random_uniform_host, which uses
       std::uniform_int_distribution<int>(0, INT_MAX) and therefore never
       produces a negative value. Kernels reasonably read an int dat as a
       full-range signed int; ../oSEM_3d does exactly that
       (opensbliblock00_kernels.h:39-48) and gets eddies in an eighth of the box
       with every sign +1 -- see finding 1 in the README.

       Full range is what the name promises. Flagged rather than silently
       matched, because changing it upstream is a behaviour change for any
       existing app that has adapted to the truncated range. */
    std::uniform_int_distribution<int> distribution(INT_MIN, INT_MAX);
    for (size_t i = 0; i < cumsize; i++) {
      ((int *)dat->data)[i] = distribution(gen);
    }
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: uniform random number generation not implemented for data type: " << dat->type;
    throw ex;
  }

  dat->dirty_hd = 1;

  /* NO ops_set_halo_dirtybit3 HERE. That is the whole difference from the grid
     version: a particle dat has no block halo, and building the ops_arg_dat the
     call needs is what throws. */
}

/**
 * Fill a particle ops_dat with normally distributed random numbers, mean 0
 * standard deviation 1. The particle counterpart of
 * ops_fill_random_normal_host (ops_lib_core.cpp:2628), which -- like its
 * uniform sibling -- supports floating point types only.
 */
inline void ops_fill_random_normal_particle(ops_dat dat, std::mt19937 &gen) {

  if (!dat->is_particle) {
    throw OPSException(OPS_INVALID_ARGUMENT,
                       "Error: ops_fill_random_normal_particle expects a "
                       "particle ops_dat; use ops_fill_random_normal for a "
                       "grid ops_dat");
  }

  size_t cumsize = dat->dim;
  const char *type = dat->type;

  for (int d = 0; d < OPS_MAX_DIM; d++) {
    cumsize *= dat->size[d];
  }

  if (strcmp(type, "double") == 0 || strcmp(type, "real(8)") == 0 || strcmp(type, "real(kind=8)") == 0) {
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < cumsize; i++) {
      ((double *)dat->data)[i] = distribution(gen);
    }
  }
  else if (strcmp(type, "float") == 0 || strcmp(type, "real") == 0 || strcmp(type, "real(4)") == 0 ||
             strcmp(type, "real(kind=4)") == 0) {
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < cumsize; i++) {
      ((float *)dat->data)[i] = distribution(gen);
    }
  }
  else {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "Error: normal random number generation not implemented for data type: " << dat->type;
    throw ex;
  }

  dat->dirty_hd = 1;
}

#endif /* _OPS_PARTICLE_RNG_H_ */
