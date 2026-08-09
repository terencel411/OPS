/*
 * ops_particle_random.h
 *
 *  Random-number fill for PARTICLE dats, in the style of OPS's
 *  ops_fill_random_uniform(): call it from the driver before the kernels, and
 *  the kernels just read a dat.
 *
 *  Kept inside the app for now. It is written to be liftable into ops/ later --
 *  it uses only the public particle API plus dat->data -- but it should earn
 *  that by being confirmed correct here first.
 *
 * ------------------------------------------------------------------
 * WHY NOT JUST USE ops_fill_random_uniform ON A PARTICLE DAT
 * ------------------------------------------------------------------
 * Two reasons, and the first is fatal for particles.
 *
 * 1. IT WOULD BE KEYED ON STORAGE POSITION. A particle dat is indexed by local
 *    slot, and a particle's local slot changes when it migrates between ranks
 *    or when the list is compacted. So the same eddy would draw from a
 *    different stream after moving, and two runs at different rank counts
 *    would give different answers. The whole point of this app being
 *    rank-invariant is that an eddy's randomness belongs to the EDDY.
 *
 *    So the fill below is keyed on the particle's GLOBAL ID, which is stable
 *    for the life of the run and identical at any decomposition.
 *
 * 2. ops_fill_random_uniform ON AN INT DAT NEVER RETURNS A NEGATIVE VALUE.
 *    ops_lib_core.cpp:2605 builds std::uniform_int_distribution<int>(0,
 *    INT_MAX). oSEM relies on `(rng < 0) ? -1 : 1` for its eddy signs, so in
 *    that code every sign is +1. Drawing doubles in [0,1) and thresholding at
 *    0.5 avoids the whole question.
 *
 * ------------------------------------------------------------------
 * COUNTER-BASED, NOT A RUNNING STREAM
 * ------------------------------------------------------------------
 * There is no state anywhere. Each value is a pure hash of
 *
 *     (seed, global id, counter, component)
 *
 * where `counter` is normally the timestep. That gives three properties worth
 * having:
 *
 *   - no state dat to declare, migrate or keep in the border list
 *   - reproducible: ask for step 137's draw at any time and get the same value
 *   - INDEPENDENT ACROSS COMPONENTS BY CONSTRUCTION. This one matters. The
 *     previous implementation advanced one LCG per eddy and took successive
 *     states for successive quantities, which made an eddy's eps_x a
 *     deterministic function of its x -- and since compute_fluct selects
 *     eddies by x, it selected a biased set of signs. Measured: rms 8.85 /
 *     4.89 / 6.41 when all three must be equal, with each sign individually
 *     unbiased so no test of the sign distribution would have caught it.
 *     Here each component is a separate hash input, so that failure mode
 *     cannot recur.
 */

#ifndef _OPS_PARTICLE_RANDOM_H_
#define _OPS_PARTICLE_RANDOM_H_

#include <cstdint>
#include <random>

/* ------------------------------------------------------------------ *
 * Bit source: counter-based, keyed on (seed, gid, counter, component)
 * ------------------------------------------------------------------ *
 * Deliberately separated from the DISTRIBUTION below. The keying is what this
 * header exists for; the mapping from bits to a distribution is standard and
 * should look like everybody else's.
 *
 * This satisfies the C++ UniformRandomBitGenerator requirements, so it can be
 * handed to any std:: distribution exactly as OPS hands std::mt19937 to
 * std::uniform_real_distribution in ops_fill_random_uniform_host
 * (ops_lib_core.cpp:2581).
 *
 * Unlike mt19937 it carries no meaningful state between calls: every value is
 * a fresh hash of the key plus a local call count, so an engine constructed
 * with the same four inputs always produces the same sequence, wherever and
 * whenever it is constructed.
 */
static inline uint64_t ops_prandom_mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

class ops_prandom_engine {
public:
  typedef uint64_t result_type;

  ops_prandom_engine(unsigned int seed, int gid, unsigned int counter,
                     int component)
      : n_(0) {
    uint64_t k = (uint64_t)seed;
    k = ops_prandom_mix64(k ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(uint32_t)gid));
    k = ops_prandom_mix64(k ^ (0xC2B2AE3D27D4EB4FULL * (uint64_t)counter));
    key_ = ops_prandom_mix64(k ^
                             (0x165667B19E3779F9ULL * (uint64_t)(uint32_t)component));
  }

  static result_type min() { return 0; }
  static result_type max() { return UINT64_MAX; }

  result_type operator()() {
    return ops_prandom_mix64(key_ ^ (0x9E3779B97F4A7C15ULL * (++n_)));
  }

private:
  uint64_t key_;
  uint64_t n_;
};

/* ------------------------------------------------------------------ *
 * Distributions: the same std:: objects OPS uses
 * ------------------------------------------------------------------ *
 * ops_fill_random_uniform_host builds
 *     std::uniform_real_distribution<double> distribution(0.0, 1.0);
 * and that is what is used here, so the bits-to-value mapping is identical to
 * every other OPS fill. Swapping in std::normal_distribution or anything else
 * is now a one-line change, which is the point of keeping the engine separate.
 *
 * One caveat worth knowing: std::uniform_real_distribution is not specified to
 * give identical values across standard library implementations, so results are
 * reproducible for a given toolchain rather than universally. That is the price
 * of matching OPS rather than hand-rolling the conversion, and it does not
 * affect rank invariance -- every rank in a run uses the same library.
 */
static inline double ops_prandom_uniform(unsigned int seed, int gid,
                                         unsigned int counter, int component) {
  ops_prandom_engine gen(seed, gid, counter, component);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(gen);
}

/** Normal(mean, stddev), for symmetry with ops_fill_random_normal. */
static inline double ops_prandom_normal(unsigned int seed, int gid,
                                        unsigned int counter, int component,
                                        double mean, double stddev) {
  ops_prandom_engine gen(seed, gid, counter, component);
  std::normal_distribution<double> distribution(mean, stddev);
  return distribution(gen);
}

/**
 * Fill a particle dat with uniform [0,1) values, one per component per
 * particle, keyed on global id.
 *
 * Call it from the driver before the kernels that consume it -- the same shape
 * of call as oSEM's ops_fill_random_uniform(d_y_rng).
 *
 * @param particle  the particle set
 * @param dat       destination, a double particle dat of any dim
 * @param gid_dat   the int particle dat holding global ids
 * @param seed      run seed; changing it changes the whole realisation
 * @param counter   normally the timestep, so successive steps draw afresh
 *
 * Only OWNED particles are filled (0 .. no_particles). Ghost copies are not:
 * they are refreshed from their owner by the border/forward exchange, so
 * filling them here would overwrite the owner's values with a second draw --
 * the same double-counting trap as iterating ITERATE_ALL in a publish kernel.
 */
inline void ops_fill_random_uniform_particle(ops_particle particle, ops_dat dat,
                                             ops_dat gid_dat, unsigned int seed,
                                             unsigned int counter) {
  const int n = (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  for (int i = 0; i < n; i++)
    for (int c = 0; c < d; c++)
      out[d * i + c] = ops_prandom_uniform(seed, gid[i], counter, c);
}

/**
 * As above, but +1 / -1 with equal probability -- what eddy signs actually
 * want, and it keeps the 0.5 threshold in one place rather than in every
 * kernel that needs a sign.
 */
inline void ops_fill_random_sign_particle(ops_particle particle, ops_dat dat,
                                          ops_dat gid_dat, unsigned int seed,
                                          unsigned int counter) {
  const int n = (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  for (int i = 0; i < n; i++)
    for (int c = 0; c < d; c++)
      out[d * i + c] =
          (ops_prandom_uniform(seed, gid[i], counter, c) < 0.5) ? -1.0 : 1.0;
}

/**
 * Normal(mean, stddev) fill, mirroring ops_fill_random_normal.
 */
inline void ops_fill_random_normal_particle(ops_particle particle, ops_dat dat,
                                            ops_dat gid_dat, unsigned int seed,
                                            unsigned int counter, double mean,
                                            double stddev) {
  const int n = (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  for (int i = 0; i < n; i++)
    for (int c = 0; c < d; c++)
      out[d * i + c] =
          ops_prandom_normal(seed, gid[i], counter, c, mean, stddev);
}

#endif /* _OPS_PARTICLE_RANDOM_H_ */
