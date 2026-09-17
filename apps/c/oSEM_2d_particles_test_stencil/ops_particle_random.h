#ifndef _OPS_PARTICLE_RANDOM_H_
#define _OPS_PARTICLE_RANDOM_H_

#include <cstdint>
#include <random>

enum ops_particle_rng_method {
  OPS_PRNG_MT19937 = 0,
  OPS_PRNG_MINSTD = 1
};

// per-particle engine seeded from (seed, gid, counter)
template <typename Engine>
static inline Engine ops_prandom_engine_t(unsigned int seed, int gid,
                                          unsigned int counter) {
  std::seed_seq seq{static_cast<std::uint32_t>(seed),
                    static_cast<std::uint32_t>(gid),
                    static_cast<std::uint32_t>(counter)};
  return Engine(seq);
}

static inline double ops_prandom_uniform(unsigned int seed, int gid,
                                         unsigned int counter, int component) {
  std::mt19937 gen = ops_prandom_engine_t<std::mt19937>(seed, gid, counter);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int c = 0; c < component; c++) distribution(gen);
  return distribution(gen);
}

// include_ghosts fills the virtual particles too. Values are a pure function of
// (seed, gid, counter), so a ghost reproduces what its owner computes.
inline void ops_fill_random_uniform_particle(ops_particle particle, ops_dat dat,
                                             ops_dat gid_dat, unsigned int seed,
                                             unsigned int counter,
                                             int method = OPS_PRNG_MT19937,
                                             bool include_ghosts = false) {
  const int n = include_ghosts
                  ? (int)(particle->no_particles + particle->no_virtual)
                  : (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  if (method == OPS_PRNG_MINSTD) {
    for (int i = 0; i < n; i++) {
      std::minstd_rand gen =
          ops_prandom_engine_t<std::minstd_rand>(seed, gid[i], counter);
      for (int c = 0; c < d; c++) out[d * i + c] = distribution(gen);
    }
    return;
  }

  for (int i = 0; i < n; i++) {
    std::mt19937 gen = ops_prandom_engine_t<std::mt19937>(seed, gid[i], counter);
    for (int c = 0; c < d; c++) out[d * i + c] = distribution(gen);
  }
}

inline std::mt19937 ops_prandom_gen;

// set seed before performing the random fill
inline void ops_prandom_init(unsigned int seed) {
#ifdef OPS_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ops_prandom_gen.seed(seed + (unsigned int)rank * 2654435761u);
#else
  ops_prandom_gen.seed(seed);
#endif
}

inline void ops_fill_random_uniform_particle_v1(ops_particle particle,
                                               ops_dat dat) {
  const int total = (int)particle->no_particles * dat->dim;
  double *out = (double *)dat->data;

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < total; i++) out[i] = distribution(ops_prandom_gen);
}

#endif /* _OPS_PARTICLE_RANDOM_H_ */
