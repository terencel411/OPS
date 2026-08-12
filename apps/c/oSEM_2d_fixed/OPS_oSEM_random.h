/*
 * OPS_oSEM_random.h  --  rank-invariant random fill for the eddy dats
 *
 * Same idea as particle_global_influence_oSEM/ops_particle_random.h, adapted
 * from particle dats to an ordinary decomposed grid dat.
 *
 * ------------------------------------------------------------------
 * WHY ops_fill_random_uniform IS NOT ENOUGH
 * ------------------------------------------------------------------
 * Two problems, one fixed already and one not.
 *
 * 1. FIXED, by declaring the rng dats double instead of int.
 *    ops_fill_random_uniform on an INT dat draws
 *    std::uniform_int_distribution<int>(0, INT_MAX) (ops_lib_core.cpp:2605),
 *    never negative, so the reference's `(rng < 0) ? -1 : 1` gave every eddy a
 *    +1 sign. On a double dat it draws uniform_real[0,1) and the sign is a
 *    fair coin at 0.5.
 *
 * 2. NOT FIXED, and the reason this header exists: IT IS NOT RANK-INVARIANT.
 *    ops_randomgen_init_host (ops_lib_core.cpp:2570) seeds
 *
 *        gen.seed(seed + my_global_rank * 2654435761u)
 *
 *    when there is more than one rank, then ops_fill_random_uniform_host walks
 *    the LOCAL array with that engine. So a draw belongs to a (rank, storage
 *    slot) pair rather than to an eddy. Change the rank count and every eddy
 *    gets different randomness: np=1 and np=4 are different realisations of
 *    the same statistics, not the same run. Measured before this header:
 *    u' rms 30.72 at np=1 against 29.15 at np=4.
 *
 * ------------------------------------------------------------------
 * COUNTER-BASED, KEYED ON THE GLOBAL EDDY ID
 * ------------------------------------------------------------------
 * There is no stream and no state. Every value is a pure function of
 *
 *     (seed, global eddy id, counter, component)
 *
 * with `counter` the timestep. The global id comes from a `gid` dat filled once
 * by instantiate_gid through ops_arg_idx -- the grid-dat equivalent of the
 * p_gid dat the particle app carries. Eddy 137 therefore draws the same six
 * numbers at step 42 on one rank as on sixteen.
 *
 * The six components of one eddy are successive draws from ONE engine seeded
 * by std::seed_seq. That mirrors the particle app exactly, including the
 * reason: an earlier version there advanced a raw LCG per eddy and used
 * successive STATES as successive quantities, which made eps_x a deterministic
 * function of x. compute_fluct selects eddies by x, so it selected a biased set
 * of signs -- measured as rms 8.85 / 4.89 / 6.41 where all three must be equal,
 * with each sign individually unbiased so no test of the sign distribution
 * would have caught it. seed_seq scrambling plus six draws from a tempered or
 * freshly-seeded engine cannot reproduce that.
 *
 * ------------------------------------------------------------------
 * AS SIMPLE AS THE LIBRARY'S OWN FILL
 * ------------------------------------------------------------------
 * Write dat->data directly, exactly as ops_fill_random_uniform_host and
 * ops_particle_random.h both do. No extents, no slab setter, no range
 * conversion.
 *
 * An earlier version of this header keyed the fill on the rank's displacement
 * from ops_dat_get_extents and wrote back through
 * ops_dat_set_data_slab_memspace. That was the long way round, and it cost a
 * bug: the slab setter wants a LOCAL range and was handed a global one, which
 * is harmless on rank 0 (displacement 0) and heap corruption everywhere else.
 * np=1 and np=2 completed and compared bit-identical; np>=4 took a SIGSEGV in
 * free(). Carrying a gid dat -- one int per eddy, one kernel, filled once --
 * removes the whole problem.
 */

#ifndef OPS_OSEM_RANDOM_H
#define OPS_OSEM_RANDOM_H

#include <cstdint>
#include <random>
#include <vector>
#include <ops_lib_core.h>

/* Component indices. One engine per (eddy, counter) yields six successive
   draws; these say which draw feeds which dat, and they must stay fixed or a
   run's realisation changes. */
enum osem_rng_component {
  OSEM_RNG_X = 0,
  OSEM_RNG_Y = 1,
  OSEM_RNG_Z = 2,
  OSEM_RNG_EPS_X = 3,
  OSEM_RNG_EPS_Y = 4,
  OSEM_RNG_EPS_Z = 5,
  OSEM_RNG_NCOMP = 6
};

/** Engine for one eddy at one counter, seeded through std::seed_seq.
 *  seed_seq is the standard library's own facility for turning several values
 *  into a well-scrambled state, which also avoids the known weakness of
 *  scalar-seeding an engine with nearby values -- and with one engine per eddy
 *  the seeds ARE nearby. */
template <typename Engine>
static inline Engine osem_rng_engine(unsigned int seed, int gid,
                                     unsigned int counter) {
  std::seed_seq seq{static_cast<std::uint32_t>(seed),
                    static_cast<std::uint32_t>(gid),
                    static_cast<std::uint32_t>(counter)};
  return Engine(seq);
}

/** One draw, uniform [0,1), for (eddy, counter, component). Pure function --
 *  no state, so it is identical at any rank count and can be asked for any
 *  step at any time. minstd_rand for the same reason the particle app defaults
 *  to it: measured indistinguishable from mt19937 on the one correlation known
 *  to have caused a real bug, corr(x, eps), and far cheaper to seed. */
static inline double osem_rng_uniform(unsigned int seed, int gid,
                                      unsigned int counter, int component) {
  std::minstd_rand gen = osem_rng_engine<std::minstd_rand>(seed, gid, counter);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int c = 0; c < component; c++) distribution(gen);
  return distribution(gen);
}

/**
 * Fill one eddy rng dat with uniform [0,1) values keyed on the GLOBAL eddy id.
 *
 * Structurally this is ops_fill_random_uniform_host (ops_lib_core.cpp:2581):
 *
 *     std::uniform_real_distribution<double> dist(0,1);
 *     ((double*)dat->data)[i] = dist(gen);
 *
 * with one difference -- the engine is seeded per EDDY from
 * (seed, gid, counter) rather than once per rank. That is the whole change,
 * and it is the same change ops_particle_random.h makes for particle dats.
 *
 * gid comes from a dat filled once by instantiate_gid via ops_arg_idx, which
 * is the grid-dat equivalent of the p_gid dat the particle app carries. It has
 * the same layout as `dat`, so index i lines up in both.
 *
 * The whole local allocation is filled, halo cells included, exactly as the
 * library does. Their contents do not matter: every eddy kernel uses the
 * self-only S2D_00 stencil, so nothing ever reads a halo cell.
 */
inline void osem_fill_random_uniform(ops_dat dat, ops_dat gid_dat,
                                     unsigned int seed, unsigned int counter,
                                     int component) {
  size_t cumsize = dat->dim;
  for (int d = 0; d < OPS_MAX_DIM; d++) cumsize *= dat->size[d];

  double *out = (double *)dat->data;
  const int *gid = (const int *)gid_dat->data;
  if (out == NULL || gid == NULL) return;   /* rank owns none of this block */

  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (size_t i = 0; i < cumsize; i++) {
    std::minstd_rand gen =
        osem_rng_engine<std::minstd_rand>(seed, gid[i], counter);
    for (int c = 0; c < component; c++) distribution(gen);
    out[i] = distribution(gen);
  }

  /* Same bookkeeping ops_fill_random_uniform_host does after writing. */
  dat->dirty_hd = 1;
}

#endif /* OPS_OSEM_RANDOM_H */
