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

/* ------------------------------------------------------------------ *
 * Three fill strategies, so they can be compared rather than argued about
 * ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ *
 * WHICH ONE TO USE -- measured, not argued
 * ------------------------------------------------------------------ *
 *   method                    fill      rank-invariant   notes
 *   mt19937 per particle      10.99 ms  yes              624-word seed_seq
 *                                                        per eddy per step
 *   minstd_rand per particle   0.08 ms  yes              DEFAULT
 *   shared engine              0.13 ms  NO               mirrors OPS exactly
 *
 * minstd_rand is the default. The obvious objection -- that a bare LCG's
 * consecutive outputs lie on a lattice, so an eddy's position could couple to
 * its signs, which is exactly the bug that skewed an earlier version -- was
 * tested rather than assumed. Drawing the app's 6-tuple for 10^6 (gid,counter)
 * pairs:
 *
 *     mt19937      corr(x, eps) = +0.00115 +0.00115 -0.00177   (1 sigma = 0.00100)
 *     minstd_rand  corr(x, eps) = -0.00150 +0.00126 -0.00132
 *
 * Indistinguishable, and minstd's are no larger. The lattice problem needs a
 * CONTINUING stream; here every eddy gets a fresh seed_seq-scrambled state and
 * six draws, which is far too short for the structure to appear. mt19937's
 * tempering solves a problem this usage does not have.
 *
 * Caveat: that tests the one coupling known to have caused a real bug, not
 * general RNG quality. If a future kernel draws LONG sequences from a single
 * engine, mt19937 becomes the right choice again.
 */
enum ops_particle_rng_method {
  OPS_PRNG_MT19937 = 0,  /* per particle, gid-keyed. Strongest, slowest.   */
  OPS_PRNG_MINSTD = 1,   /* per particle, gid-keyed. Tiny state, so fast.  */
  OPS_PRNG_SHARED = 2    /* one engine per rank, walked in storage order.
                            Mirrors ops_fill_random_uniform exactly.       */
};

/** Per-particle engine seeded from (seed, gid, counter) via std::seed_seq.
 *  Templated on the engine so the two per-particle methods share one body. */
template <typename Engine>
static inline Engine ops_prandom_engine_t(unsigned int seed, int gid,
                                          unsigned int counter) {
  std::seed_seq seq{static_cast<std::uint32_t>(seed),
                    static_cast<std::uint32_t>(gid),
                    static_cast<std::uint32_t>(counter)};
  return Engine(seq);
}

/**
 * The shared engine, seeded exactly as ops_randomgen_init_host does
 * (ops_lib_core.cpp:2570): `seed + rank * 2654435761u` when there is more than
 * one rank. Without the rank offset every rank would draw the SAME sequence,
 * so the k-th eddy on every rank would get identical signs -- spatially
 * unrelated eddies with correlated randomness, which a mean-and-variance check
 * would not catch.
 */
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

/**
 * A single draw, uniform in [0,1), keyed on gid. Used by the seeding pass so
 * the initial eddy positions are identical for all three methods and at any
 * rank count -- that keeps the comparison to the per-step fill alone.
 */
static inline double ops_prandom_uniform(unsigned int seed, int gid,
                                         unsigned int counter, int component) {
  std::mt19937 gen = ops_prandom_engine_t<std::mt19937>(seed, gid, counter);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int c = 0; c < component; c++) distribution(gen);
  return distribution(gen);
}

/**
 * Fill a particle dat with uniform [0,1) values.
 *
 * @param method  which strategy; see ops_particle_rng_method
 *
 * Only OWNED particles are filled. Ghosts are refreshed from their owner by
 * the border/forward exchange, so filling them here would overwrite the
 * owner's values with a second draw.
 */
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
    /* Structurally identical to ops_fill_random_uniform_host: one engine,
       walk the array. Values are keyed on STORAGE SLOT, so an eddy's draw
       changes when it migrates or the list is compacted. */
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
