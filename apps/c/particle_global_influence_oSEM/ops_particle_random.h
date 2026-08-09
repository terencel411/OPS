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
 *   - components are successive draws from a TEMPERED engine (mt19937), so
 *     they are decorrelated. This one matters. The
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
 * Everything below is standard library
 * ------------------------------------------------------------------ *
 * ops_fill_random_uniform_host (ops_lib_core.cpp:2581) is
 *
 *     std::mt19937 gen;                                    // seeded once
 *     std::uniform_real_distribution<double> dist(0,1);
 *     data[i] = dist(gen);
 *
 * and this is the same three lines, with one difference: the engine is seeded
 * per PARTICLE, from (seed, global id, counter), instead of once for the whole
 * dat. That is the entire point of the header -- see the note above on why the
 * keying has to be the global id rather than the storage slot.
 *
 * std::seed_seq is the standard library's own facility for turning several
 * values into a well-scrambled engine state, so no hand-written mixing is
 * needed. It also avoids the known weakness of scalar-seeding mt19937 with
 * nearby values, which can leave the first outputs of neighbouring seeds
 * correlated -- and with one engine per eddy, the seeds ARE neighbouring.
 *
 * Components are successive draws from that engine. Safe here in a way it was
 * not for the hand-rolled LCG this replaced: mt19937 tempers its output, so
 * successive values are decorrelated. The earlier bug came from using a raw
 * LCG state directly as the value, where state n+1 is a simple affine function
 * of state n.
 */

/** Build the per-particle engine. One place, so the fills and the
 *  single-value helper below cannot drift apart. */
static inline std::mt19937 ops_prandom_engine(unsigned int seed, int gid,
                                              unsigned int counter) {
  std::seed_seq seq{static_cast<std::uint32_t>(seed),
                    static_cast<std::uint32_t>(gid),
                    static_cast<std::uint32_t>(counter)};
  return std::mt19937(seq);
}

/**
 * A single draw, uniform in [0,1), for code outside a fill -- the seeding pass
 * uses it so the initial positions come from exactly the stream the fill would
 * have produced. `component` is the position in the engine's sequence, matching
 * how the fills below lay components out.
 */
static inline double ops_prandom_uniform(unsigned int seed, int gid,
                                         unsigned int counter, int component) {
  std::mt19937 gen = ops_prandom_engine(seed, gid, counter);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int c = 0; c < component; c++) distribution(gen);
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

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < n; i++) {
    std::mt19937 gen = ops_prandom_engine(seed, gid[i], counter);
    for (int c = 0; c < d; c++) out[d * i + c] = distribution(gen);
  }
}

/**
 * As above, but +1 / -1 with equal probability -- what eddy signs actually
 * want, and it keeps the threshold in one place rather than in every kernel.
 */
inline void ops_fill_random_sign_particle(ops_particle particle, ops_dat dat,
                                          ops_dat gid_dat, unsigned int seed,
                                          unsigned int counter) {
  const int n = (int)particle->no_particles;
  const int d = dat->dim;

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < n; i++) {
    std::mt19937 gen = ops_prandom_engine(seed, gid[i], counter);
    for (int c = 0; c < d; c++)
      out[d * i + c] = (distribution(gen) < 0.5) ? -1.0 : 1.0;
  }
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

  std::normal_distribution<double> distribution(mean, stddev);

  for (int i = 0; i < n; i++) {
    std::mt19937 gen = ops_prandom_engine(seed, gid[i], counter);
    for (int c = 0; c < d; c++) out[d * i + c] = distribution(gen);
  }
}

#endif /* _OPS_PARTICLE_RANDOM_H_ */
