#ifndef OPS_OSEM_EDDY_GATHER_H
#define OPS_OSEM_EDDY_GATHER_H

/*
 * Fetching the eddy ops_dats into plain host arrays for ops_arg_gbl.
 *
 * WHY THIS EXISTS
 * ---------------
 * ops_dat_fetch_data() is NOT a global gather.  In the MPI backend
 * (ops/c/src/mpi/ops_mpi_rt_support.cpp:2125-2147) it copies only *this rank's
 * local extent* of the dat, packed at offset 0 of the destination buffer.  The
 * rest of the destination buffer is left untouched.
 *
 * "eddy_block" is an ops_block of shape {eddies, 1}, so OPS decomposes it like
 * any other block.  Measured behaviour before this fix:
 *
 *   ranks  rank  local extent   entries actually written into x_gbl[1718]
 *   -----  ----  -------------  ---------------------------------------
 *     1      0   [0 .. 1718)    1718  (correct)
 *     2      0   [0 .. 0)          0  -> whole array stayed zero
 *     2      1   [0 .. 1718)    1718
 *     4      0   [0 .. 0)          0
 *     4      1   [0 .. 0)          0
 *     4      2   [0 .. 859)      859  at indices 0..858
 *     4      3   [859 .. 1718)   859  ALSO at indices 0..858
 *
 * compute_fluct then loops `for (i < eddies)` over that buffer, so for any
 * np > 1 every rank summed a different, partly-zeroed eddy set -- and since
 * inlet_block is decomposed too, the assembled inlet plane was stitched
 * together from mutually inconsistent halves.  Silently wrong, no error.
 *
 * WHAT THIS DOES
 * --------------
 * Fetch the local slice, then MPI_Allgatherv it back into its correct global
 * position so that every rank ends up holding the identical, complete eddy
 * array.  Each element is contributed by exactly one owner, so the result is
 * well defined regardless of how OPS chose to decompose the block.
 *
 * NOTE ON REPRODUCIBILITY: ops_fill_random_uniform() draws per rank, so the
 * eddy field still differs between different rank counts.  That is a change of
 * random realisation, not of correctness -- statistics are unaffected.  Results
 * are reproducible for a fixed number of ranks.
 */

#ifdef OPS_MPI
#include <mpi.h>
#include <vector>

inline void fetch_eddy_dat(ops_dat dat, void *global, size_t elem_bytes) {
  int nranks = 1, myrank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int ldisp[OPS_MAX_DIM] = {0};
  int lsize[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(dat, 0, ldisp, lsize);

  /* eddy dats are {eddies, 1}: a rank owns lsize[0] entries starting at
   * ldisp[0], or nothing at all if it was handed an empty slab in either
   * direction. */
  int my_count = (lsize[0] > 0 && lsize[1] > 0) ? lsize[0] : 0;
  int my_disp = (my_count > 0) ? ldisp[0] : 0;

  std::vector<char> local(static_cast<size_t>(my_count > 0 ? my_count : 1) * elem_bytes);
  if (my_count > 0) ops_dat_fetch_data(dat, 0, local.data());

  std::vector<int> counts(nranks), displs(nranks);
  int my_bytes = my_count * static_cast<int>(elem_bytes);
  int my_off = my_disp * static_cast<int>(elem_bytes);
  MPI_Allgather(&my_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my_off, 1, MPI_INT, displs.data(), 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Allgatherv(local.data(), my_bytes, MPI_BYTE, global, counts.data(),
                 displs.data(), MPI_BYTE, MPI_COMM_WORLD);
}

/*
 * Diagnostic: assert that every rank now holds a bit-identical copy of a
 * gathered eddy array.  This is the exact invariant compute_fluct depends on
 * and that was violated before the gather was introduced.  Returns 0 on
 * success, non-zero on mismatch.  Cheap enough to leave in; only called when
 * asked for.
 */
inline int check_eddy_replication(const void *global, size_t nbytes,
                                  const char *name) {
  const unsigned char *p = (const unsigned char *)global;
  unsigned long long h = 1469598103934665603ULL; /* FNV-1a */
  for (size_t i = 0; i < nbytes; i++) {
    h ^= (unsigned long long)p[i];
    h *= 1099511628211ULL;
  }
  unsigned long long hmin = 0, hmax = 0;
  MPI_Allreduce(&h, &hmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&h, &hmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (hmin != hmax) {
    ops_printf("EDDY REPLICATION CHECK: *** FAILED *** for %s "
               "(ranks disagree: %llx vs %llx)\n", name, hmin, hmax);
    return 1;
  }
  ops_printf("EDDY REPLICATION CHECK: ok  %-10s hash=%llx\n", name, hmin);
  return 0;
}

/* Range summary of a gathered double array -- catches eddies collapsing into a
 * sub-region of the box, which is what a mis-scaled RNG looks like. */
inline void report_eddy_range(const double *a, int n, const char *name,
                              double lo, double hi) {
  double mn = a[0], mx = a[0], sum = 0.0;
  for (int i = 0; i < n; i++) {
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
    sum += a[i];
  }
  ops_printf("  %-6s min=% .6e max=% .6e mean=% .6e   (box [% .6e, % .6e], "
             "coverage %.1f%%)\n",
             name, mn, mx, sum / n, lo, hi,
             100.0 * (mx - mn) / (hi - lo));
}

/* Sign balance of an epsilon array -- SEM needs these to be ~50/50 and
 * mutually independent, otherwise the fluctuations are biased. */
inline void report_eps_balance(const int *a, int n, const char *name) {
  int pos = 0;
  for (int i = 0; i < n; i++) if (a[i] > 0) pos++;
  ops_printf("  %-6s  +1: %5d (%5.1f%%)   -1: %5d (%5.1f%%)\n", name, pos,
             100.0 * pos / n, n - pos, 100.0 * (n - pos) / n);
}

#else /* !OPS_MPI -- single node, the local slice is the whole dat */

inline void fetch_eddy_dat(ops_dat dat, void *global, size_t elem_bytes) {
  (void)elem_bytes;
  ops_dat_fetch_data(dat, 0, (char *)global);
}

inline int check_eddy_replication(const void *global, size_t nbytes,
                                  const char *name) {
  (void)global; (void)nbytes; (void)name;
  return 0;
}

inline void report_eddy_range(const double *a, int n, const char *name,
                              double lo, double hi) {
  double mn = a[0], mx = a[0], sum = 0.0;
  for (int i = 0; i < n; i++) {
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
    sum += a[i];
  }
  ops_printf("  %-6s min=% .6e max=% .6e mean=% .6e   (box [% .6e, % .6e], "
             "coverage %.1f%%)\n",
             name, mn, mx, sum / n, lo, hi,
             100.0 * (mx - mn) / (hi - lo));
}

inline void report_eps_balance(const int *a, int n, const char *name) {
  int pos = 0;
  for (int i = 0; i < n; i++) if (a[i] > 0) pos++;
  ops_printf("  %-6s  +1: %5d (%5.1f%%)   -1: %5d (%5.1f%%)\n", name, pos,
             100.0 * pos / n, n - pos, 100.0 * (n - pos) / n);
}

#endif /* OPS_MPI */

#endif /* OPS_OSEM_EDDY_GATHER_H */
