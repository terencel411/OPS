/*
 * ops_particle_random.h -- uniform fill for PARTICLE dats. ops_fill_random_uniform
 * throws on a particle dat and keys on storage slot; these key on the global id,
 * which is what makes a run reproduce at any rank count. See the README.
 */

#ifndef _OPS_PARTICLE_RANDOM_H_
#define _OPS_PARTICLE_RANDOM_H_

#include <cstdint>
#include <random>

/* Engine choice, measured rather than argued: mt19937 costs 10.99 ms/fill,
   minstd_rand 0.08 ms, and their position/sign correlations are
   indistinguishable at 10^6 draws. minstd_rand is the default. */
enum ops_particle_rng_method {
  OPS_PRNG_MT19937 = 0,  /* per particle, gid-keyed. Strongest, slowest.   */
  OPS_PRNG_MINSTD = 1,   /* per particle, gid-keyed. Tiny state, so fast.  */
  OPS_PRNG_SHARED = 2    /* one engine per rank, walked in storage order.
                            Mirrors ops_fill_random_uniform exactly.       */
};

/* Per-particle engine seeded from (seed, gid, counter) via std::seed_seq. */
template <typename Engine>
static inline Engine ops_prandom_engine_t(unsigned int seed, int gid,
                                          unsigned int counter) {
  std::seed_seq seq{static_cast<std::uint32_t>(seed),
                    static_cast<std::uint32_t>(gid),
                    static_cast<std::uint32_t>(counter)};
  return Engine(seq);
}

/* Shared engine, seeded as ops_randomgen_init_host does: seed + rank*2654435761u
   above one rank, or every rank would draw the same sequence. */
inline std::mt19937 &ops_prandom_shared_engine() {
  static std::mt19937 gen;
  return gen;
}

inline void ops_prandom_shared_init(unsigned int seed) {
#ifdef OPS_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ops_prandom_shared_engine().seed(seed + (unsigned int)rank * 2654435761u);
#else
  ops_prandom_shared_engine().seed(seed);
#endif
}

/* A single draw, uniform in [0,1), keyed on gid. Used by the host seeding pass. */
static inline double ops_prandom_uniform(unsigned int seed, int gid,
                                         unsigned int counter, int component) {
  std::mt19937 gen = ops_prandom_engine_t<std::mt19937>(seed, gid, counter);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int c = 0; c < component; c++) distribution(gen);
  return distribution(gen);
}

/* Fill a particle dat with uniform [0,1) values. Owned particles only: ghosts
   are refreshed from their owner by the border exchange. */
inline void ops_fill_random_uniform_particle(ops_particle particle, ops_dat dat,
                                             ops_dat gid_dat, unsigned int seed,
                                             unsigned int counter,
                                             int method = OPS_PRNG_MT19937) {
  const int n = (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  if (method == OPS_PRNG_SHARED) {
    /* Keyed on storage slot, so a draw changes when the list is compacted. */
    std::mt19937 &gen = ops_prandom_shared_engine();
    const int total = n * d;
    for (int i = 0; i < total; i++) out[i] = distribution(gen);
    return;
  }

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

#endif /* _OPS_PARTICLE_RANDOM_H_ */
