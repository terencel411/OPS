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

// the d uniforms belonging to (seed, gid, counter): the only rng this app needs,
// and independent of rank, of the local index and of how many particles precede it
inline void ops_prandom_uniform_gid(unsigned int seed, int gid, unsigned int counter,
                                    int method, int d, double *out) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  if (method == OPS_PRNG_MINSTD) {
    std::minstd_rand gen = ops_prandom_engine_t<std::minstd_rand>(seed, gid, counter);
    for (int c = 0; c < d; c++) out[c] = distribution(gen);
    return;
  }
  std::mt19937 gen = ops_prandom_engine_t<std::mt19937>(seed, gid, counter);
  for (int c = 0; c < d; c++) out[c] = distribution(gen);
}

#endif /* _OPS_PARTICLE_RANDOM_H_ */
