/*
 * ==================================================================== *
 *  oSEM_3d eddy initialisation, as OPS particles
 * ==================================================================== *
 *
 *  SCOPE: the instantiate_eddies and convect_eddies kernels, and nothing else.
 *
 *  ../oSEM_3d/opensbliblock00_kernels.h:39-59 instantiates its eddies into ten
 *  ops_dats of shape {eddies,1,1} declared on the FLUID block, then convects
 *  them along x every step and re-injects whatever leaves. This app does the
 *  same with each eddy as an OPS particle instead, then reports what was
 *  produced and writes it out to be looked at.
 *
 *  Each is one kernel run by one loop, as it is there -- convect_eddies is
 *  instantiate_eddies behind an `if`, and stays that way here.
 *
 *  Not here, deliberately: the fluctuation field, the Reynolds stresses, the
 *  coupling to the inlet BC.
 *
 *  WHY PARTICLES AT ALL. In ../oSEM_3d the eddy dats live on the fluid block, so
 *  MPI decomposes them by the fluid partitioning -- an eddy list sliced along x
 *  by a decomposition that has nothing to do with eddies. Kernel030 then needs
 *  the whole list on every rank and gets it through seven
 *  ops_arg_gbl(..., OPS_READ) arrays, which OPS does not communicate, so the app
 *  carries a hand-rolled MPI_Allgatherv (opensbli.cpp:456-531) every timestep.
 *  As particles, an eddy is simply owned by the rank whose subdomain it occupies.
 *
 *  Build and run:
 *      make oSEM_3d_particles_dev_seq && ./oSEM_3d_particles_dev_seq
 *      make oSEM_3d_particles_dev_mpi && mpirun -np 4 ./oSEM_3d_particles_dev_mpi
 *      python3 plot_osem3d_h5.py     # frames/*.png
 *      python3 make_xdmf.py          # osem3d_eddies.xmf, for ParaView
 *
 *  Options:
 *      -ngrid NX NY NZ   eddy-box grid resolution
 *      -seed N           base seed
 *      -nsteps N         convection steps (default 400 = 2.1 box flushes)
 *      -nout M           write every M steps (default 25)
 *      -rng-selftest     test ops_particle_rng.h and exit
 *      -noh5             skip the HDF5 output
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <random>
#include <vector>

#define OPS_3D

#include <ops_seq_v2.h>
#include <ops_particle_seq.h>

#ifdef OPS_MPI
#include <mpi.h>
#endif

#include "sem3d_constants.h"
#include "eddy_kernels.h"
#include "sem3d_stats.h"
#include "sem3d_io.h"
#include "ops_particle_rng.h"

/* ================================================================== *
 *  Are the eddies actually being convected?
 * ================================================================== *
 *
 * report_eddy_field() below cannot answer this and should not be asked to. It
 * tests the eddy LIST -- count, containment, uniformity, sign balance -- and the
 * list starts uniform, so it stays uniform whatever convection does or does not
 * do. Freeze every eddy by replacing `pos(0) + inc(0)` with `pos(0)` and it
 * still reports PASS on all four targets, with the field bit-identical to the
 * initial one. Only containment says anything at all: without re-injection the
 * eddies would pile up past eddy_x_max and it would fail.
 *
 * So this checks the TRAJECTORY, against a closed form rather than against a
 * second copy of the kernel. Given a starting position x0 and n steps of
 * `x += inc; if (x > x_max) x = x_min`:
 *
 *   k1 = floor((x_max - x0)/inc) + 1        steps to the first re-injection
 *   P  = floor((x_max - x_min)/inc) + 1     steps between re-injections after
 *
 *   n <  k1 :  x(n) = x0 + n*inc            and y, z, eps are UNTOUCHED
 *   n >= k1 :  x(n) = x_min + ((n-k1) mod P) * inc
 *
 * The second line is what catches a stalled or mis-scaled advance; the first is
 * what catches convection corrupting an eddy it should only have slid along x,
 * which is the failure a position check alone misses (../particle_tutorial_1_
 * drift_3d/drift3d.cpp:284-288 makes the same point about y and z).
 *
 * The eddies are snapshotted at step 0 and compared slot by slot, which is sound
 * because nothing is created, destroyed or migrated after startup: slot p is the
 * same eddy for the whole run.
 *
 * ONE STEP OF SLACK ON k1. The kernel accumulates x one increment at a time
 * while the closed form multiplies, so the two disagree by ~n*eps and an eddy
 * landing within that of the face can cross a step either side of where the
 * formula says. The three candidates k1-1, k1, k1+1 are tried and the best
 * match taken; how often the slack was needed is reported, and it should be
 * rare (0 of 267 at 400 steps).
 */
struct eddy_snapshot {
  std::vector<double> pos;
  std::vector<int>    eps;
};

void snapshot_eddies(ops_particle particle, ops_dat pos, ops_dat peps,
                     eddy_snapshot &s) {
  const size_t n = particle->no_particles;
  s.pos.assign((const double *)pos->data,  (const double *)pos->data  + 3 * n);
  s.eps.assign((const int *)peps->data,    (const int *)peps->data    + 3 * n);
}

int check_convection(ops_particle particle, ops_dat pos, ops_dat peps,
                     const eddy_snapshot &s, int nstep, int verbose) {

  const int    n   = (int)particle->no_particles;
  const double inc = u0 * dt;
  const int    P   = (int)floor((eddy_x_max - eddy_x_min) / inc) + 1;

  const double *d_pos = (const double *)pos->data;
  const int    *d_eps = (const int *)peps->data;

  /* Accumulating inc nstep times loses precision, so scale with nstep as
     drift3d does rather than demanding exactness. */
  const double tol = 1e-12 * (nstep + 1);

  double worst = 0.0;
  int bad_x = 0, bad_carry = 0, slack = 0, still_original = 0;

  for (int p = 0; p < n; p++) {
    const double x0 = s.pos[3 * p];
    const int    k1 = (int)floor((eddy_x_max - x0) / inc) + 1;

    /* k1 is tried FIRST and ties are kept, so an alternative wins only when it
       is strictly better. Ordering this the other way round counts every eddy
       still on its first pass as "slack", because before the first re-injection
       all three candidates predict the same x. */
    int    best_k = k1;
    double best   = fabs(d_pos[3 * p] - ((nstep < k1)
                                         ? x0 + nstep * inc
                                         : eddy_x_min + ((nstep - k1) % P) * inc));

    for (int k = k1 - 1; k <= k1 + 1; k += 2) {
      const double xe = (nstep < k) ? x0 + nstep * inc
                                    : eddy_x_min + ((nstep - k) % P) * inc;
      const double e = fabs(d_pos[3 * p] - xe);
      if (e < best) { best = e; best_k = k; }
    }

    if (best > worst) worst = best;
    if (best > tol) bad_x++;
    if (best_k != k1) slack++;

    /* Never re-injected yet: convection must not have touched anything but x. */
    if (nstep < best_k) {
      still_original++;
      if (d_pos[3 * p + 1] != s.pos[3 * p + 1] ||
          d_pos[3 * p + 2] != s.pos[3 * p + 2] ||
          d_eps[3 * p + 0] != s.eps[3 * p + 0] ||
          d_eps[3 * p + 1] != s.eps[3 * p + 1] ||
          d_eps[3 * p + 2] != s.eps[3 * p + 2])
        bad_carry++;
    }
  }

  int ntot = n;
#ifdef OPS_MPI
  /* Same pattern as report_eddy_field(): a diagnostic, reduced with MPI
     directly. Nothing in the convection path itself does this. */
  int g[5]; double gw;
  int l[5] = {n, bad_x, bad_carry, slack, still_original};
  MPI_Allreduce(l, g, 5, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&worst, &gw, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  ntot = g[0]; bad_x = g[1]; bad_carry = g[2]; slack = g[3];
  still_original = g[4]; worst = gw;
#endif

  const int ok = (bad_x == 0) && (bad_carry == 0);

  if (verbose || !ok) {
    /* "not yet recycled" = has never passed the outlet face, so it is still on
       the leg it was instantiated on. Those are the eddies whose y, z and signs
       are also being held to their instantiation values just above. */
    ops_printf("  convection  step %5d : max |x - exact| = %.3e (tol %.1e)"
               "   %d/%d not yet recycled\n",
               nstep, worst, tol, still_original, ntot);
    if (slack)
      ops_printf("              %d eddy(s) crossed the face a step off the "
                 "closed form (round-off)\n", slack);
    if (bad_x)
      ops_printf("  FAIL: %d of %d eddies are not on their trajectory\n",
                 bad_x, ntot);
    if (bad_carry)
      ops_printf("  FAIL: %d of %d eddies that have not been re-injected had "
                 "y, z or a sign changed\n", bad_carry, ntot);
  }

  return ok;
}

/* ================================================================== *
 *  Did the instantiation produce what it should have?
 * ================================================================== *
 *
 * Every check here is a property of the eddy LIST -- no shape function, no
 * Reynolds stresses, no physics. Four questions:
 *
 *   is the eddy count consistent with the box?
 *   is every eddy inside it?
 *   are the positions uniformly distributed over it?
 *   are the signs balanced and independent?
 *
 * The uniformity tests are chi-square rather than KS because chi-square needs
 * only BIN COUNTS, which reduce with one MPI_SUM. KS would need the globally
 * sorted sample, i.e. a position gather, and would give different answers at
 * different rank counts -- a diagnostic whose verdict depends on the
 * decomposition is worse than no diagnostic.
 */
int report_eddy_field(ops_particle particle, ops_dat pos, ops_dat peps) {

  const int n = (int)particle->no_particles;
  const double *d_pos = (const double *)pos->data;
  const int    *d_eps = (const int *)peps->data;

  double lo[3] = { 1e30,  1e30,  1e30};
  double hi[3] = {-1e30, -1e30, -1e30};
  int pos_count[3] = {0, 0, 0};

  for (int p = 0; p < n; p++) {
    for (int d = 0; d < 3; d++) {
      const double v = d_pos[3 * p + d];
      if (v < lo[d]) lo[d] = v;
      if (v > hi[d]) hi[d] = v;
    }
    for (int d = 0; d < 3; d++) if (d_eps[3 * p + d] > 0) pos_count[d]++;
  }

  int ntot = n;
#ifdef OPS_MPI
  double glo[3], ghi[3];
  int gpos[3], gn;
  MPI_Allreduce(lo, glo, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(hi, ghi, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(pos_count, gpos, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int d = 0; d < 3; d++) { lo[d] = glo[d]; hi[d] = ghi[d]; pos_count[d] = gpos[d]; }
  ntot = gn;
#endif

  const double blo[3] = {eddy_x_min, eddy_y_min, eddy_z_min};
  const double bhi[3] = {eddy_x_max, eddy_y_max, eddy_z_max};
  const char  *nm[3]  = {"x", "y", "z"};

  ops_printf("\n--- eddy field (%d eddies) ---\n", ntot);
  int ok = (ntot == eddies);
  if (!ok)
    ops_printf("  MISMATCH: %d eddies present, %d expected\n", ntot, eddies);

  /* One eddy per r^3 of box, which is how oSEM_3d picks the count. Assumed
     everywhere else; asserted here. */
  {
    const double vol  = (bhi[0]-blo[0]) * (bhi[1]-blo[1]) * (bhi[2]-blo[2]);
    const int    want = (int)trunc(vol / (radius * radius * radius));
    ops_printf("  density   N == trunc(V/r^3) = %d : %s\n",
               want, (eddies == want) ? "yes" : "NO");
    ok = ok && (eddies == want);
  }

  for (int d = 0; d < 3; d++) {
    const double span = 100.0 * (hi[d] - lo[d]) / (bhi[d] - blo[d]);
    ops_printf("  %s  min=% .6e max=% .6e   box [% .6e, % .6e]  spans %5.1f%%\n",
               nm[d], lo[d], hi[d], blo[d], bhi[d], span);
    /* Containment only. Span is reported because a degenerate z range is the
       signature of the _ops_construct_local_box_from_dat() regression
       (ops_particle_box_host_funcs.h:45), which until 2026-07-30 left xmax[2]
       unassigned in the 3D array-of-structs branch, silently collapsing every
       particle onto z = 0 while still reporting PASS.

       Span is NOT a uniformity criterion: a lattice spans 99.9% per axis and a
       half-interval field spans 50%, so the number cannot separate them. The
       chi-square tests below do that. */
    ok = ok && (lo[d] >= blo[d] - 1e-12) && (hi[d] <= bhi[d] + 1e-12);
  }

  /* ---- uniformity, per axis and jointly ---------------------------- *
   * TWO-SIDED. A chi-square far BELOW its degrees of freedom is as damning as
   * one far above: it means the points are spread more evenly than chance
   * allows, which is the signature of a lattice or a stratified sample. */
  {
    const double p_fail = 0.001;
    const int nb1 = sem_nbins_for(ntot);      /* 53 bins at 267 eddies */

    for (int d = 0; d < 3; d++) {
      std::vector<long> cnt(nb1, 0);
      for (int p = 0; p < n; p++) {
        int b = (int)((d_pos[3 * p + d] - blo[d]) / (bhi[d] - blo[d]) * nb1);
        b = std::min(std::max(b, 0), nb1 - 1);
        cnt[b]++;
      }
#ifdef OPS_MPI
      std::vector<long> g(nb1);
      MPI_Allreduce(cnt.data(), g.data(), nb1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      cnt.swap(g);
#endif
      int df; double pv;
      const double chi2 = sem_chi2_uniform(cnt, ntot, &df, &pv);
      const char *verdict = (pv < p_fail)       ? "<-- CLUSTERED"
                          : (pv > 1.0 - p_fail) ? "<-- TOO REGULAR (lattice?)"
                                                : "";
      ops_printf("  uniform %s   chi2 = %8.2f  df = %3d   p = %.4f  %s\n",
                 nm[d], chi2, df, pv, verdict);
      ok = ok && (pv > p_fail) && (pv < 1.0 - p_fail);
    }

    /* 2-D in (y, z): the test for lattice structure. A lattice has perfectly
       uniform marginals, so the three checks above can all pass on one. */
    int nby, nbz;
    sem_split_bins_2d(sem_nbins_for(ntot),
                      (bhi[2] - blo[2]) / (bhi[1] - blo[1]), &nby, &nbz);

    std::vector<long> cnt2((size_t)nby * nbz, 0);
    for (int p = 0; p < n; p++) {
      int by = (int)((d_pos[3 * p + 1] - blo[1]) / (bhi[1] - blo[1]) * nby);
      int bz = (int)((d_pos[3 * p + 2] - blo[2]) / (bhi[2] - blo[2]) * nbz);
      by = std::min(std::max(by, 0), nby - 1);
      bz = std::min(std::max(bz, 0), nbz - 1);
      cnt2[(size_t)by * nbz + bz]++;
    }
#ifdef OPS_MPI
    std::vector<long> g2(cnt2.size());
    MPI_Allreduce(cnt2.data(), g2.data(), (int)cnt2.size(), MPI_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    cnt2.swap(g2);
#endif
    int df2; double pv2;
    const double chi2 = sem_chi2_uniform(cnt2, ntot, &df2, &pv2);
    const char *v2 = (pv2 < p_fail)       ? "<-- LATTICE / CLUSTERING"
                   : (pv2 > 1.0 - p_fail) ? "<-- TOO REGULAR (lattice?)"
                                          : "";
    ops_printf("  uniform yz  chi2 = %8.2f  df = %3d   p = %.4f  "
               "(%d x %d bins)  %s\n", chi2, df2, pv2, nby, nbz, v2);
    ok = ok && (pv2 > p_fail) && (pv2 < 1.0 - p_fail);
  }

  /* ---- sign balance, tolerance derived rather than guessed ---------- *
   * Each component is ntot fair draws, so the +1 fraction has
   * sigma = 0.5/sqrt(ntot) -- 3.06% at 267 eddies. A fixed 45-55% band is only
   * +/-1.63 sigma there and rejects 29% of correct seeds. Four sigma leaves a
   * per-component false-failure rate near 6e-5 and still rejects an all-one-sign
   * field, which sits ~16 sigma out. */
  {
    const double sigma = sem_sign_sigma_pct(ntot);
    const double tol   = 4.0 * sigma;
    for (int d = 0; d < 3; d++) {
      const double frac = 100.0 * pos_count[d] / ntot;
      const double z    = (frac - 50.0) / sigma;
      ops_printf("  eps_%s  +1: %5d (%5.1f%%)   -1: %5d (%5.1f%%)   "
                 "[50.0 +/- %.1f%%, %+.1f sigma]  %s\n",
                 nm[d], pos_count[d], frac, ntot - pos_count[d], 100.0 - frac,
                 tol, z, (fabs(frac - 50.0) < tol) ? "" : "<-- IMBALANCED");
      ok = ok && (fabs(frac - 50.0) < tol);
    }
  }

  /* ---- sign independence ------------------------------------------- *
   * Balance alone is blind to correlation. oSEM_3d's host_rng_sign() takes
   * signs from the parity of s = (5s+3) mod 2^29, which strictly alternates, so
   * every eddy would get (a, -a, a): perfectly balanced, perfectly
   * anti-correlated, and invisible to the test above. */
  {
    long sp[3] = {0, 0, 0};    /* xy, xz, yz */
    for (int p = 0; p < n; p++) {
      sp[0] += (long)d_eps[3 * p]     * d_eps[3 * p + 1];
      sp[1] += (long)d_eps[3 * p]     * d_eps[3 * p + 2];
      sp[2] += (long)d_eps[3 * p + 1] * d_eps[3 * p + 2];
    }
#ifdef OPS_MPI
    long gsp[3];
    MPI_Allreduce(sp, gsp, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    for (int k = 0; k < 3; k++) sp[k] = gsp[k];
#endif
    const char *pair[3] = {"xy", "xz", "yz"};
    for (int k = 0; k < 3; k++) {
      double z;
      const double c = sem_sign_correlation(sp[k], ntot, &z);
      ops_printf("  corr eps_%s  %+.4f  (%+.1f sigma)  %s\n",
                 pair[k], c, z, (fabs(z) < 4.0) ? "" : "<-- CORRELATED");
      ok = ok && (fabs(z) < 4.0);
    }
  }

  return ok;
}

/* ================================================================== *
 *  Self-test for ops_particle_rng.h  (-rng-selftest)
 * ================================================================== *
 *
 * ops_particle_rng.h is destined for the OPS library, so it gets tested as a
 * library function rather than only through the eddies. Six checks, and two of
 * them mean nothing except under MPI.
 *
 * The int check is the important one: it asserts the exact behaviour where
 * ops_fill_random_uniform_particle DIVERGES from ops_fill_random_uniform_host,
 * which draws int dats from (0, INT_MAX) and would fail it.
 */

/* Order-independent digest of a dat's raw bytes, so ranks can be compared
   without gathering the data itself. FNV-1a. */
static unsigned long long rng_digest(const ops_dat dat, size_t nelems) {
  const unsigned char *b = (const unsigned char *)dat->data;
  const size_t nbytes = nelems * dat->type_size;
  unsigned long long h = 1469598103934665603ULL;
  for (size_t i = 0; i < nbytes; i++) {
    h ^= (unsigned long long)b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void rng_check(const char *name, int pass, const char *detail, int *all) {
  ops_printf("  %-46s %s   %s\n", name, pass ? "PASS" : "FAIL", detail);
  *all = *all && pass;
}

static int rng_selftest(ops_dat d_uni, ops_dat d_int, ops_dat d_nrm,
                        ops_dat grid_dat, unsigned int seed) {

  int all = 1;
  char detail[256];

  const size_t n_uni = (size_t)d_uni->size[0] * d_uni->dim;
  const size_t n_int = (size_t)d_int->size[0] * d_int->dim;
  const size_t n_nrm = (size_t)d_nrm->size[0] * d_nrm->dim;

  ops_printf("\n--- ops_particle_rng.h self-test ---\n");

  std::mt19937 gen;

  /* 1. uniform, double: range and mean. */
  ops_particle_randomgen_init(seed, 0, gen);
  ops_fill_random_uniform_particle(d_uni, gen);
  {
    const double *v = (const double *)d_uni->data;
    double lo = 2.0, hi = -1.0, sum = 0.0;
    for (size_t i = 0; i < n_uni; i++) {
      if (v[i] < lo) lo = v[i];
      if (v[i] > hi) hi = v[i];
      sum += v[i];
    }
    const double mean = sum / (double)n_uni;
    snprintf(detail, sizeof detail, "n=%zu range [%.6f, %.6f] mean %.4f",
             n_uni, lo, hi, mean);
    rng_check("uniform double in [0,1), mean ~ 0.5",
              lo >= 0.0 && hi < 1.0 && fabs(mean - 0.5) < 0.02, detail, &all);
  }

  /* 2. uniform, int -- THE DIVERGENCE. ops_fill_random_uniform_host would
        produce nothing below zero here and fail this outright. */
  ops_particle_randomgen_init(seed, 0, gen);
  ops_fill_random_uniform_particle(d_int, gen);
  {
    const int *v = (const int *)d_int->data;
    int lo = INT_MAX, hi = INT_MIN;
    size_t nneg = 0;
    for (size_t i = 0; i < n_int; i++) {
      if (v[i] < lo) lo = v[i];
      if (v[i] > hi) hi = v[i];
      if (v[i] < 0) nneg++;
    }
    const double frac_neg = (double)nneg / (double)n_int;
    /* Both halves populated, and the extremes reached to within 1% of the
       range -- i.e. the distribution really is over the full signed int. */
    const double reach = 0.01 * 4294967296.0;
    snprintf(detail, sizeof detail, "n=%zu range [%d, %d] %.1f%% negative",
             n_int, lo, hi, 100.0 * frac_neg);
    rng_check("uniform int over FULL signed range",
              nneg > 0 && nneg < n_int && fabs(frac_neg - 0.5) < 0.02 &&
              (double)lo < (double)INT_MIN + reach &&
              (double)hi > (double)INT_MAX - reach, detail, &all);
  }

  /* 3. normal, double: mean and standard deviation. */
  ops_particle_randomgen_init(seed, 0, gen);
  ops_fill_random_normal_particle(d_nrm, gen);
  {
    const double *v = (const double *)d_nrm->data;
    double sum = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < n_nrm; i++) { sum += v[i]; sum2 += v[i] * v[i]; }
    const double mean = sum / (double)n_nrm;
    const double sd   = sqrt(sum2 / (double)n_nrm - mean * mean);
    snprintf(detail, sizeof detail, "n=%zu mean %+.4f sd %.4f", n_nrm, mean, sd);
    rng_check("normal double, mean ~ 0, sd ~ 1",
              fabs(mean) < 0.05 && fabs(sd - 1.0) < 0.05, detail, &all);
  }

  /* 4. The is_particle guard: a grid dat must be refused. */
  {
    int threw = 0;
    try {
      ops_fill_random_uniform_particle(grid_dat, gen);
    } catch (OPSException &) {
      threw = 1;
    }
    snprintf(detail, sizeof detail, "grid dat \"%s\" %s", grid_dat->name,
             threw ? "rejected" : "ACCEPTED");
    rng_check("is_particle guard rejects a grid dat", threw, detail, &all);
  }

  /* 5. options = 0: reproducible, and identical on every rank. */
  {
    ops_particle_randomgen_init(seed, 0, gen);
    ops_fill_random_uniform_particle(d_uni, gen);
    const unsigned long long a = rng_digest(d_uni, n_uni);

    ops_particle_randomgen_init(seed, 0, gen);
    ops_fill_random_uniform_particle(d_uni, gen);
    const unsigned long long b = rng_digest(d_uni, n_uni);

    int same_across_ranks = 1;
#ifdef OPS_MPI
    unsigned long long dmin, dmax;
    MPI_Allreduce(&b, &dmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&b, &dmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    same_across_ranks = (dmin == dmax);
#endif
    snprintf(detail, sizeof detail, "reseed reproduces: %s; all ranks equal: %s",
             (a == b) ? "yes" : "NO", same_across_ranks ? "yes" : "NO");
    rng_check("options=0 -- same stream on every rank",
              (a == b) && same_across_ranks, detail, &all);
  }

  /* 6. options = 1: the rank offset, i.e. the flag actually selects. Vacuous on
        one rank, where ops_randomgen_init_host also skips the offset. */
  {
    ops_particle_randomgen_init(seed, 1, gen);
    ops_fill_random_uniform_particle(d_uni, gen);
    const unsigned long long b = rng_digest(d_uni, n_uni);

    int differs = 1;
    const char *note = "single rank, vacuous";
#ifdef OPS_MPI
    if (ops_num_procs() > 1) {
      unsigned long long dmin, dmax;
      MPI_Allreduce(&b, &dmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&b, &dmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
      differs = (dmin != dmax);
      note = differs ? "ranks differ, as intended" : "RANKS IDENTICAL";
    }
#endif
    snprintf(detail, sizeof detail, "%s", note);
    rng_check("options=1 -- per-rank stream", differs, detail, &all);
  }

  ops_printf("\nOPS_PARTICLE_RNG SELF-TEST : %s\n", all ? "PASS" : "FAIL");
  return all;
}

/* ================================================================== */

int main(int argc, char **argv) {

  ops_init(argc, argv, 1);

  /* ---- Parameters, from ../oSEM_3d/opensbli.cpp:73, 112-121 -------- */

  u0     = 1.0;               /* non-dimensional freestream */
  dt     = 0.025;
  delta  = 11.6973525411;
  radius = 0.2 * delta;

  span_z = 40.0;              /* block0np2 * Delta2block0 = 150 * (40/150) */

  eddy_x_min = -radius;
  eddy_x_max =  radius;
  eddy_y_min =  0.0    - radius;
  eddy_y_max =  delta  + radius;
  eddy_z_min = -radius;
  eddy_z_max =  span_z + radius;

  eddy_vol = fabs((eddy_x_max - eddy_x_min) * (eddy_y_max - eddy_y_min)
                                            * (eddy_z_max - eddy_z_min));
  calc_eddies(eddies, eddy_vol, radius);

  eddy_seed = 182383739;      /* oSEM_3d's seed_gbl, opensbli.cpp:135 */
  int write_h5 = 1;
  int selftest = 0;
  int nsteps   = 400;   /* 2.1 flushes: (x_max-x_min)/(u0*dt) = 188 steps each */
  int nout     = 25;

  /* The eddy-box grid is not a physical grid -- nothing is solved on it. It
     defines the bounding box and the bins, so the only sensible scale is the
     eddy radius. Half a radius per cell is fine enough to be useful and coarse
     enough to stay cheap. delta == 5*radius exactly, so x and y come out as
     whole cell counts (4 and 14). */
  const double cell = 0.5 * radius;
  int nex = (int)ceil((eddy_x_max - eddy_x_min) / cell) + 1;   /*  5 nodes */
  int ney = (int)ceil((eddy_y_max - eddy_y_min) / cell) + 1;   /* 15 nodes */
  int nez = (int)ceil((eddy_z_max - eddy_z_min) / cell) + 1;   /* 40 nodes */

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ngrid") == 0 && i + 3 < argc) {
      nex = atoi(argv[++i]); ney = atoi(argv[++i]); nez = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-noh5") == 0) {
      write_h5 = 0;
    } else if (strcmp(argv[i], "-nsteps") == 0 && i + 1 < argc) {
      nsteps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nout") == 0 && i + 1 < argc) {
      nout = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-rng-selftest") == 0) {
      selftest = 1;
    } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
      eddy_seed = (int)strtoul(argv[++i], NULL, 10);
    }
  }

  if (nex < 2 || ney < 2 || nez < 2) {
    ops_printf("FATAL: the eddy-box grid needs at least 2 nodes per direction, "
               "got %d x %d x %d\n", nex, ney, nez);
    ops_exit();
    return 1;
  }

  /* A translator-generated kernel file sees ops_decl_const variables and nothing
     else from these headers. KerInstantiateEddies reads u0, dt, radius and the
     six box bounds; the rest are registered for the kernels that come later,
     exactly as ../oSEM_3d registers them. eddy_seed is NOT among them -- it is
     consumed host-side by ops_particle_randomgen_init, not by any kernel. */
  ops_decl_const("u0", 1, "double", &u0);
  ops_decl_const("dt", 1, "double", &dt);
  ops_decl_const("delta", 1, "double", &delta);
  ops_decl_const("radius", 1, "double", &radius);
  ops_decl_const("span_z", 1, "double", &span_z);
  ops_decl_const("eddy_x_min", 1, "double", &eddy_x_min);
  ops_decl_const("eddy_x_max", 1, "double", &eddy_x_max);
  ops_decl_const("eddy_y_min", 1, "double", &eddy_y_min);
  ops_decl_const("eddy_y_max", 1, "double", &eddy_y_max);
  ops_decl_const("eddy_z_min", 1, "double", &eddy_z_min);
  ops_decl_const("eddy_z_max", 1, "double", &eddy_z_max);
  ops_decl_const("eddies", 1, "int", &eddies);

  /* ---- 1. Block and the eddy-box coordinate dat -------------------- *
   *
   * The eddies get their own block rather than riding on oSEM_3d's
   * opensbliblock00. Two properties of oSEM_3d force it:
   *
   *   THE EDDY BOX OVERHANGS THE FLUID DOMAIN ON ALL THREE LOW FACES. x, y and
   *   z all start at -radius, while the fluid block spans
   *   [0,375] x [0,100] x [0,40]. The bounding box is derived from a coordinate
   *   dat and is exactly [first node .. last node], so a fluid-derived box would
   *   put every eddy at x<0, y<0 or z<0 outside it -- and outside means deleted.
   *
   *   THE FLUID WALL-NORMAL GRID IS SINH-STRETCHED (opensbli.cpp:15,
   *   x1 = Lx1*sinh(by*Delta1*j/Lx1)/sinh(by), by = 4.0), which the uniform
   *   mapping used here does not model. OPS_NON_UNI_STAG exists for that, and
   *   coupling to the real block will have to use it.
   *
   * Spacing is extent/(nodes-1), so the first and last nodes land exactly on
   * the faces of the eddy box.
   */

  ops_block block = ops_decl_block(3, "eddy_box");

  int grid_size[] = {nex, ney, nez};
  int base[]      = {0, 0, 0};
  int d_m[]       = {0, 0, 0};
  int d_p[]       = {0, 0, 0};

  double *null_dbl = NULL;
  int    *null_int = NULL;

  /* dim 3: each node stores its own (x, y, z). */
  ops_dat d_coords = ops_decl_dat(block, 3, grid_size, base, d_m, d_p,
                                  null_dbl, "double", "eddy_coords");

  /* ---- 2. Stencils ------------------------------------------------- */

  int s3d_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3d_000, "0,0,0");

  int s3d_27pt[3 * 27];
  {
    int k = 0;
    for (int i = -1; i <= 1; i++)
      for (int j = -1; j <= 1; j++)
        for (int l = -1; l <= 1; l++) {
          s3d_27pt[3 * k] = i; s3d_27pt[3 * k + 1] = j; s3d_27pt[3 * k + 2] = l;
          k++;
        }
  }
  ops_stencil S3D_27pt = ops_decl_stencil(3, 27, s3d_27pt, "27pt");

  /* ---- 3. Bounding box, particle set, dats ------------------------- *
   *
   * ops_create_bounding_box() stores only a POINTER to the coordinate dat; the
   * bounds are computed later, inside ops_particle_setup_partition(). Its dx
   * argument is ignored.
   *
   * Taking the coordinate-dat route rather than declaring the region explicitly
   * is deliberate: the explicit-region overload reaches a branch that needs the
   * mapping's cell size, which a grid+stencil mapping derives back through the
   * box, so map->dx comes out {0,0,0} and it fails a step later
   * (../particle_tutorial_1_drift_3d/drift3d.cpp:421-427).
   */

  double dx_box[] = {0.0, 0.0, 0.0};
  BoundingBox<double> *box = ops_create_bounding_box(block, d_coords, 3, dx_box);

  ops_particle eddy_parts = ops_decl_particle(block, "eddies", box);

  /* The position dat is privileged: exactly one per particle set, dim must
     equal the block dimension, and it must be declared before any mapping.
     Every part of the library -- binning, migration, deletion -- reads
     positions from it.

     The rest are ../oSEM_3d's eddy variables one for one: eddy_x/y/z become the
     position, eddy_r the radius, eddy_increment the per-step convection
     distance, and eddy_eps_x/y/z the three signs, gathered into one dim-3 dat. */
  ops_dat p_pos = ops_decl_particle_pos_dat(eddy_parts, 3, base, null_dbl,
                                            "double", "eddy_pos");

  ops_dat p_r   = ops_decl_particle_dat(eddy_parts, 1, base, null_dbl,
                                        "double", "eddy_r");
  ops_dat p_inc = ops_decl_particle_dat(eddy_parts, 1, base, null_dbl,
                                        "double", "eddy_increment");
  ops_dat p_eps = ops_decl_particle_dat(eddy_parts, 3, base, null_int,
                                        "int", "eddy_eps");
  /* Not in oSEM_3d, and unread here: the global eddy index, which is how runs at
     different rank counts are compared eddy by eddy. */
  ops_dat p_id  = ops_decl_particle_dat(eddy_parts, 1, base, null_int,
                                        "int", "eddy_id");
  /* Scratch: KerMarkUnownedEddies' verdict, read once by the removal below.
     A particle dat rather than a plain array because it is written by a kernel,
     and only a particle dat can be. */
  ops_dat p_del = ops_decl_particle_dat(eddy_parts, 1, base, null_int,
                                        "int", "eddy_unowned");

  /* -ops-rng only: six random ints per eddy, filled by
     ops_fill_random_uniform_particle(). This is ../oSEM_3d's eddy_x_rng (dim 1,
     for x) and eddy_bulk_rng (dim 5, for y, z and the three signs) merged into
     one particle dat, and it is int for the same reason theirs is -- so that
     KerInstantiateEddiesOpsRng can be their kernel verbatim.

     int also exercises the branch of ops_fill_random_uniform_particle that
     DIVERGES from ops_fill_random_uniform_host: full signed range rather than
     (0, INT_MAX). See -rng-selftest. */
  ops_dat p_rng = ops_decl_particle_dat(eddy_parts, 6, base, null_int,
                                        "int", "eddy_rng");

  /* Scratch for -rng-selftest, declared only in that mode so the normal path
     carries no extra particle dats. */
  ops_dat t_uni = NULL, t_int = NULL, t_nrm = NULL;
  if (selftest) {
    t_uni = ops_decl_particle_dat(eddy_parts, 4, base, null_dbl, "double", "rngtest_uniform");
    t_int = ops_decl_particle_dat(eddy_parts, 4, base, null_int, "int",    "rngtest_int");
    t_nrm = ops_decl_particle_dat(eddy_parts, 4, base, null_dbl, "double", "rngtest_normal");
  }

  /* ---- 4. Mapping -------------------------------------------------- *
   *
   * A cell-linked list binning eddies onto cells. S3D_27pt here is NOT an
   * interpolation stencil -- it only widens the halo depth of the bin array,
   * i.e. declares "track eddies up to one cell outside my subdomain", which is
   * what sizes the ghost band used during migration.
   *
   * One bin per cell, no stride. A strided mapping becomes necessary once a
   * search radius spans many cells, and it brings a divisibility rule the two
   * backends disagree about -- the sequential build computes the checked size
   * as grid->size + d_m - d_p (ops_particle_host_single_node.cpp:3053) while
   * MPI subtracts one more (ops_mpi_particle_core.cpp:441), and consecutive
   * integers are coprime, so no single dat size satisfies both. Instantiation
   * needs no stride and so builds on every target.
   */

  ops_particle_mapping map = ops_decl_mapping(eddy_parts, d_coords, S3D_27pt,
                                              OPS_WITH_VIRTUAL,
                                              OPS_UNIFORM_STAG, 1);

  /* border  : exchanged when an eddy changes rank owner -- every dat whose
   *           value must survive migration belongs here
   * forward : the cheap per-step refresh of the existing ghost layer
   * Counts computed, never hand-typed. */
  ops_dat dat_border[]  = {p_pos, p_r, p_inc, p_eps, p_id, p_rng};
  ops_dat dat_forward[] = {p_pos, p_r, p_eps};

  const int nborder  = sizeof(dat_border)  / sizeof(dat_border[0]);
  const int nforward = sizeof(dat_forward) / sizeof(dat_forward[0]);
  (void)nforward;   /* used by the migration cycle, once there is one */

  /* No particle halo groups. Nothing migrates yet, and the single-node backend
     rejects halo groups outright ("Particle Halo exchange must be set after
     block boxes are set", ops_particle_host_single_node.cpp:1908), so declaring
     them would cost the serial build. */

  /* ---- 5. Partition, fill the grid, instantiate -------------------- */

  ops_partition("");

  int grid_range[] = {0, nex, 0, ney, 0, nez};

  const double gdesc[6] = {eddy_x_min, eddy_y_min, eddy_z_min,
                           (eddy_x_max - eddy_x_min) / (double)(nex - 1),
                           (eddy_y_max - eddy_y_min) / (double)(ney - 1),
                           (eddy_z_max - eddy_z_min) / (double)(nez - 1)};

  ops_par_loop(KerInitEddyGrid, "KerInitEddyGrid", block, 3, grid_range,
               ops_arg_dat(d_coords, 3, S3D_000, "double", OPS_WRITE),
               ops_arg_gbl((double *)gdesc, 6, "double", OPS_READ),
               ops_arg_idx());

  /* Only now do real coordinates exist, so only now can the box be derived.
     Calling this before the loop above gives a box of all zeros and
     "Defined bounding box of non-positive volume". */
  ops_particle_setup_partition();

  if (selftest) {
    const int ok_rng = rng_selftest(t_uni, t_int, t_nrm, d_coords,
                                    (unsigned int)eddy_seed);
    ops_exit();
    return ok_rng ? 0 : 1;
  }

  /* Storage and count for the whole eddy list, on every rank. OPS infers
     neither: a particle loop iterates over particles that already exist, so the
     slots have to be there before the kernel can write into them. The tutorials
     set both by hand for the same reason
     (../particle_tutorial_1_drift_3d/drift3d.cpp:126-128). */
  if (eddies > (int)eddy_parts->Nmax)     /* Nmax is OPS_MAX_PART = 1000, and
                                             realloc rejects a shrink */
    ops_particle_realloc_data(eddy_parts, eddies);
  eddy_parts->no_particles = eddies;

  /* Only consulted for OPS_PARTICLE_ITERATE_RANDOM, but always required. */
  double part_range[] = {eddy_x_min, eddy_x_max, eddy_y_min, eddy_y_max,
                         eddy_z_min, eddy_z_max};

  /* THE instantiation. One kernel, run by OPS over the eddies, writing every
     field ../oSEM_3d/opensbliblock00_kernels.h:39-48 writes. ops_arg_idp() hands
     it the eddy index. */
  /* Fill the random dat, then read it from the kernel -- ../oSEM_3d's structure,
     and the only one available: no OPS random generator can be called from
     inside a kernel (ops_lib_core.h:1391-1397 are all whole-dat host fills).

     ops_fill_random_uniform() cannot do this fill. It throws on a particle dat,
     from the ops_arg_dat it builds for ops_set_halo_dirtybit3
     (ops_lib_core.cpp:2618, guard at :1229-1233), and its stream is rank-seeded.
     ops_fill_random_uniform_particle() (ops_particle_rng.h) is that function
     without the halo tail; taking the generator as an argument is what lets
     options = 0 give every rank the same stream, which is what this app needs
     since every rank instantiates the whole list and then culls. */
  std::mt19937 gen;
  ops_particle_randomgen_init((unsigned int)eddy_seed, 0, gen);
  ops_fill_random_uniform_particle(p_rng, gen);

  ops_particle_par_loop(KerInstantiateEddies, "instantiate_eddies", eddy_parts, 3,
                        OPS_PARTICLE_ITERATE_LOCAL, part_range, map,
                        ops_arg_dat_particle(p_pos, 3, "double", eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_r,   1, "double", eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_inc, 1, "double", eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_eps, 3, "int",    eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_id,  1, "int",    eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_rng, 6, "int",    eddy_parts, map, OPS_READ),
                        ops_arg_idp());

  /* ---- 6. Ownership ------------------------------------------------ *
   *
   * Every rank has just instantiated the whole list, so all but its own share
   * have to go. A kernel cannot change the particle count, which is why this is
   * a separate step rather than part of the instantiation: the first kernel
   * decides what an eddy IS, the second decides who keeps it. That is the same
   * split OPS makes between the insert kernel and the decide kernel of
   * ops_particle_insert.
   *
   * REMOVE BEFORE THE MAPS EXIST. There is a second removal path,
   * ops_particle_rearrange_particles_for_removal(), which also repairs the
   * cell-linked list as it compacts. It cannot be used here: it walks the bin
   * chains looking for each moved particle, while the generic swap ahead of it
   * has already swapped map->bin and map->parts_to_grid -- both are registered
   * particle dats (ops_particle_lib_core.cpp:1693-1695, assign = true) -- so the
   * chain it walks no longer matches the indices it is looking for. Under MPI
   * with 125 of 267 eddies removed the walk runs off the end of bins[] and
   * hangs. ops_particle_remove_particles() leaves the maps alone, and
   * ops_particle_setup_maps_with_dats() below builds them from scratch anyway.
   *
   * The bounds are the library's own, so this app cannot disagree with the
   * migration machinery about where a subdomain ends. */
  double box_lo[OPS_MAX_DIM], box_hi[OPS_MAX_DIM];
  box->getLocalMaxMin(box_lo, box_hi);

  ops_particle_par_loop(KerMarkUnownedEddies, "mark_unowned_eddies", eddy_parts, 3,
                        OPS_PARTICLE_ITERATE_LOCAL, part_range, map,
                        ops_arg_dat_particle(p_del, 1, "int",    eddy_parts, map, OPS_WRITE),
                        ops_arg_dat_particle(p_pos, 3, "double", eddy_parts, map, OPS_READ),
                        ops_arg_gbl(box_lo, 3, "double", OPS_READ),
                        ops_arg_gbl(box_hi, 3, "double", OPS_READ));

  /* Hand the kernel's verdict to the library and let it compact the list.
     ops_particle_remove_particles() swaps the surviving particles down over the
     removed ones and resets the flags, and -- the reason it is the right call
     here -- it does not touch the cell-linked list, which has not been built
     yet. ops_particle_setup_maps_with_dats() below builds it from scratch on
     what survives.

     The memcpy is the one host line in the sequence, and it decides nothing:
     the decision is the kernel's, and this copies its output into the array the
     removal reads. ops_particle_user_delete() is the call that would fold the
     two together and remove it, but ops_particle_insert_del.h does not compile
     -- line 108 declares `BoundingBox *boxBlock = particle->box_block`, and
     BoundingBox is a class template (ops_bounding_box.h:63) while box_block is
     a char* (ops_particles_lib_core.h:145). See the README. */
  memcpy(eddy_parts->mark_deletion, p_del->data,
         eddy_parts->no_particles * sizeof(int));

  ops_particle_remove_particles(eddy_parts, true);

  ops_particle_setup_maps_with_dats(eddy_parts, dat_border, nborder);

  /* ---- 7. Report --------------------------------------------------- */

  ops_printf("\noSEM_3d eddy initialisation as OPS particles\n");
  ops_printf("seed  = %d, filled by ops_fill_random_uniform_particle()\n",
             eddy_seed);
  ops_printf("delta = %.10f, radius = %.10f\n", delta, radius);
  ops_printf("eddy box   x [% .6e, % .6e]\n", eddy_x_min, eddy_x_max);
  ops_printf("           y [% .6e, % .6e]\n", eddy_y_min, eddy_y_max);
  ops_printf("           z [% .6e, % .6e]\n", eddy_z_min, eddy_z_max);
  ops_printf("volume = %.6e, eddies = %d\n", eddy_vol, eddies);
  ops_printf("eddy-box grid %d x %d x %d nodes, cells %.4f x %.4f x %.4f "
             "(%.2f x %.2f x %.2f radii)\n",
             nex, ney, nez, gdesc[3], gdesc[4], gdesc[5],
             gdesc[3] / radius, gdesc[4] / radius, gdesc[5] / radius);

  int ok = report_eddy_field(eddy_parts, p_pos, p_eps);

  /* ---- 8. Output --------------------------------------------------- */

  sem3d_io_params params;
  params.eddies = eddies;
  params.nex = nex;  params.ney = ney;  params.nez = nez;
  params.delta = delta;  params.radius = radius;
  params.dt = dt;  params.u0 = u0;  params.span_z = span_z;
  params.domain[0] = eddy_x_min;  params.domain[1] = eddy_x_max;
  params.domain[2] = eddy_y_min;  params.domain[3] = eddy_y_max;
  params.domain[4] = eddy_z_min;  params.domain[5] = eddy_z_max;

  ops_dat dat_output[] = {p_pos, p_r, p_eps, p_id};
  const int noutput = sizeof(dat_output) / sizeof(dat_output[0]);

  if (write_h5)
    HDF5_IO_Write_eddy_box(block, "osem3d_eddies", 0, d_coords, eddy_parts,
                           dat_output, noutput, params);

  /* ---- 9. Convection ----------------------------------------------- *
   *
   * ../oSEM_3d's per-timestep eddy work, opensbli.cpp:399-411: refill the random
   * dat, then one ops_par_loop over convect_eddies. The same two lines here.
   *
   * Nothing else happens per step. No eddy is created or destroyed, so the count
   * is conserved without being managed, and no eddy moves between ranks, so
   * nothing has to migrate: the kernel changes an eddy's coordinates, not who
   * holds it.
   *
   * WHAT THAT COSTS. A re-injected eddy keeps the rank that owned it even though
   * its new y and z may lie in another rank's subdomain, so the rank-to-position
   * correspondence decays over a flush of the box, and the bins built at startup
   * go stale with it. Nothing here reads either -- the checks and the output are
   * over the whole list -- and coupling Kernel030 needs the whole list on every
   * rank anyway (opensbli.cpp:554-569 passes it seven ops_arg_gbl arrays). If a
   * later stage does need ownership to track position, that is the point to add
   * a re-cull, and it belongs there rather than here.
   */

  eddy_snapshot snap;
  snapshot_eddies(eddy_parts, p_pos, p_eps, snap);

  for (int step = 1; step <= nsteps; step++) {

    ops_fill_random_uniform_particle(p_rng, gen);

    ops_particle_par_loop(KerConvectEddies, "convect_eddies", eddy_parts, 3,
                          OPS_PARTICLE_ITERATE_LOCAL, part_range, map,
                          ops_arg_dat_particle(p_pos, 3, "double", eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_inc, 1, "double", eddy_parts, map, OPS_READ),
                          ops_arg_dat_particle(p_eps, 3, "int",    eddy_parts, map, OPS_RW),
                          ops_arg_dat_particle(p_id,  1, "int",    eddy_parts, map, OPS_READ),
                          ops_arg_gbl((int *)p_rng->data, 6 * eddies, "int", OPS_READ));

    /* Every step, not just output steps: a trajectory that went wrong once and
       recovered still went wrong. */
    ok = check_convection(eddy_parts, p_pos, p_eps, snap, step,
                          step % nout == 0 || step == nsteps) && ok;

    if (step % nout == 0 || step == nsteps) {
      if (write_h5)
        HDF5_IO_Write_eddy_box(block, "osem3d_eddies", step, d_coords,
                               eddy_parts, dat_output, noutput, params);
    }
  }

  if (nsteps > 0) {
    ops_printf("\n--- after %d steps (%.2f flushes of the box) ---",
               nsteps, nsteps * u0 * dt / (eddy_x_max - eddy_x_min));
    ok = report_eddy_field(eddy_parts, p_pos, p_eps) && ok;
  }

  ops_printf("\nEDDY FIELD : %s\n", ok ? "PASS" : "FAIL");

  ops_exit();
  return ok ? 0 : 1;
}
