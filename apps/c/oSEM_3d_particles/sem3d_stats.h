/*
 * sem3d_stats.h -- statistical tests for the seeded eddy field
 *
 * The point of this header is to replace hand-picked thresholds with tests that
 * report a p-value, so "the eddies look fine" becomes a falsifiable statement.
 *
 * Two failure modes motivated it, both real and both invisible to the obvious
 * checks:
 *
 *   - a field confined to the upper half of every interval still spans 50% per
 *     axis and merely looks mediocre;
 *   - a field on a lattice spans 99.9% per axis and looks perfect, while
 *     leaving diagonal bands of the inlet plane empty.
 *
 * The first is caught by a per-axis uniformity test, the second ONLY by a joint
 * test in (y, z). Per-axis tests cannot see a lattice: its marginals are
 * perfectly uniform. Both known-bad seeding modes in sem3d.cpp exist so these
 * tests can be shown to have teeth.
 *
 * WHY CHI-SQUARE RATHER THAN KOLMOGOROV-SMIRNOV.
 * KS needs the globally sorted sample, which under MPI means gathering every
 * position onto one rank. Chi-square needs only BIN COUNTS, which reduce with a
 * single MPI_SUM -- so the test costs no position gather and, more importantly,
 * gives bit-identical answers at every rank count. A diagnostic whose verdict
 * depends on the decomposition is worse than no diagnostic.
 */

#ifndef _SEM3D_STATS_H_
#define _SEM3D_STATS_H_

#include <algorithm>
#include <cmath>
#include <vector>

/* ------------------------------------------------------------------ *
 *  Regularised incomplete gamma Q(a,x), for chi-square p-values
 * ------------------------------------------------------------------ *
 * Series below a+1, continued fraction above -- the standard split, since each
 * converges quickly only on its own side.
 */
inline double sem_gamma_q(double a, double x) {
  if (x < 0.0 || a <= 0.0) return 0.0;
  if (x == 0.0) return 1.0;

  const double gln = lgamma(a);
  const double eps = 1e-14;

  if (x < a + 1.0) {                    /* series for P(a,x) */
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; n++) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (fabs(del) < fabs(sum) * eps) break;
    }
    const double p = sum * exp(-x + a * log(x) - gln);
    return (p > 1.0) ? 0.0 : 1.0 - p;
  }

  /* continued fraction for Q(a,x), modified Lentz */
  const double tiny = 1e-300;
  double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (int i = 1; i <= 1000; i++) {
    const double an = -(double)i * ((double)i - a);
    b += 2.0;
    d = an * d + b;  if (fabs(d) < tiny) d = tiny;
    c = b + an / c;  if (fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (fabs(del - 1.0) < eps) break;
  }
  return exp(-x + a * log(x) - gln) * h;
}

/** p-value of a chi-square statistic with df degrees of freedom. */
inline double sem_chi2_pvalue(double chi2, int df) {
  if (df <= 0) return 1.0;
  return sem_gamma_q(0.5 * (double)df, 0.5 * chi2);
}

/* ------------------------------------------------------------------ *
 *  Uniformity from bin counts
 * ------------------------------------------------------------------ *
 * `counts` is any binning of `n` points into equal-measure bins -- 1-D per
 * axis, or a flattened 2-D binning of (y, z). Equal measure is what makes the
 * expected count uniform, so the caller must bin over the box, not over the
 * data range.
 *
 * Returns the chi-square statistic and sets df and the p-value. A small p-value
 * means the points are NOT uniformly distributed over the bins.
 */
inline double sem_chi2_uniform(const std::vector<long> &counts, long n,
                               int *df_out, double *pvalue) {
  const int nb = (int)counts.size();
  if (nb < 2 || n <= 0) { *df_out = 0; *pvalue = 1.0; return 0.0; }

  const double expected = (double)n / (double)nb;
  double chi2 = 0.0;
  for (int k = 0; k < nb; k++) {
    const double d = (double)counts[k] - expected;
    chi2 += d * d / expected;
  }
  *df_out = nb - 1;
  *pvalue = sem_chi2_pvalue(chi2, nb - 1);
  return chi2;
}

/**
 * Bin counts for a chi-square test need an expected count of at least ~5, or
 * the chi-square approximation degrades. Given n points, this is the largest
 * sensible number of bins.
 */
inline int sem_nbins_for(long n, double min_expected = 5.0) {
  const int nb = (int)floor((double)n / min_expected);
  return std::max(2, nb);
}

/**
 * Split a total bin budget into (na, nb) so the bins stay roughly square in
 * physical space, given the box aspect ratio (extent_b / extent_a).
 */
inline void sem_split_bins_2d(int budget, double aspect, int *na, int *nb) {
  int a = (int)floor(sqrt((double)budget / std::max(aspect, 1e-12)));
  a = std::max(2, a);
  int b = std::max(2, budget / a);
  *na = a; *nb = b;
}

/* ------------------------------------------------------------------ *
 *  Sign statistics
 * ------------------------------------------------------------------ *
 * A balance test alone is not enough. oSEM_3d's host_rng_sign() would produce a
 * perfectly balanced stream whose signs strictly alternate -- balance sees
 * nothing; the correlation test sees it immediately.
 */

/** Standard deviation, in percent, of the +1 fraction of n fair draws. */
inline double sem_sign_sigma_pct(long n) {
  if (n <= 0) return 0.0;
  return 100.0 * 0.5 / sqrt((double)n);
}

/**
 * Sample correlation of two sign sequences in {-1,+1}, given the sum of their
 * products and the count. For independent signs the correlation is zero with
 * standard error 1/sqrt(n), so the z-score is how many standard errors away
 * from independent the sample lies.
 */
inline double sem_sign_correlation(long sum_products, long n, double *zscore) {
  if (n < 2) { *zscore = 0.0; return 0.0; }
  const double corr = (double)sum_products / (double)n;
  *zscore = corr * sqrt((double)n);
  return corr;
}

#endif /* _SEM3D_STATS_H_ */
