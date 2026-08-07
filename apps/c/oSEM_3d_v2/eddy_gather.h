#ifndef EDDY_GATHER_H
#define EDDY_GATHER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef OPS_MPI
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
// Replicating the eddy arrays onto every MPI rank.
//
// The eddy dats are declared on opensbliblock00 with size {eddies, 1, 1}, so
// OPS marks them as edge datasets in y and z: every rank allocates the whole
// (y=0, z=0) pencil, and only dimension 0 is genuinely decomposed.  Two things
// follow from that, and both have to be handled here.
//
//  1. instantiate_eddies / convect_eddies run over {0,eddies, 0,1, 0,1}.  OPS
//     intersects that range with the *block* decomposition (sub_block, not the
//     dat), so the kernel body only executes on ranks whose sub-domain contains
//     global y=0 and z=0.  Every other rank holds a stale copy of the pencil,
//     even though it has storage for it.
//
//  2. Among the ranks that do own the pencil, each one holds only the x-slab
//     [disp0, disp0+size0) of the eddy array.
//
// opensbliblock00Kernel030 reads the eddy arrays as ops_arg_gbl, and OPS does
// not communicate gbl READ arguments - each rank uses its own host buffer
// verbatim.  So every rank needs the complete array, or the
// "for (int i = 0; i < eddies; i++)" loop in that kernel reads whatever malloc
// left behind.
//
// The gather is therefore: pencil owners contribute their x-slab, everyone else
// contributes nothing, and one MPI_Allgatherv per element type rebuilds the
// full array everywhere.  The pencil owners' slabs tile [0, eddies) exactly, so
// the result is complete and free of duplicates.
//
// Selecting the contributors by pencil ownership - rather than by de-duplicating
// on disp0 and letting the lowest rank of each x-slab win - is what makes this
// correct rather than lucky.  The lowest-numbered rank of an x-slab is only the
// rank that executed the kernel because of how OPS happens to order its
// Cartesian communicator; nothing in the API promises it.
//
// Usage:
//     eddy_gather_init(eddy_x, rho_B0, eddies);     // once, after ops_partition
//     ...
//     ops_par_loop(instantiate_eddies, ...);
//     eddy_gather(...);                             // after instantiate
//     for (iter ...) {
//       ops_par_loop(convect_eddies, ...);
//       eddy_gather(...);                           // after every convect
//       ops_par_loop(opensbliblock00Kernel030, ...);
//     }
//     eddy_gather_free();
// ---------------------------------------------------------------------------

// Number of double / int fields packed per eddy, in the order they are gathered.
#define EG_NDOUBLE 5   // x, y, z, r, increment
#define EG_NINT    3   // eps_x, eps_y, eps_z

static int eg_n = 0;          // total number of eddies
static int eg_nranks = 1;     // ranks in MPI_COMM_WORLD
static int eg_myrank = 0;
static int eg_local_disp = 0; // first global eddy index stored on this rank
static int eg_local_n = 0;    // eddies stored on this rank
static int eg_send_n = 0;     // eddies contributed by this rank (0 if not a pencil owner)
static int eg_fetch_cap = 1;  // elements ops_dat_fetch_data writes here

static double *eg_fetch_d = NULL;  // staging buffer for one double dat
static int    *eg_fetch_i = NULL;  // staging buffer for one int dat
static double *eg_send_d = NULL;   // EG_NDOUBLE interleaved doubles per contributed eddy
static int    *eg_send_i = NULL;   // EG_NINT interleaved ints per contributed eddy
static double *eg_recv_d = NULL;   // EG_NDOUBLE * eg_n
static int    *eg_recv_i = NULL;   // EG_NINT * eg_n
static int    *eg_cnt_d = NULL, *eg_dsp_d = NULL;  // Allgatherv layout, doubles
static int    *eg_cnt_i = NULL, *eg_dsp_i = NULL;  // Allgatherv layout, ints

// Set to 1 to have eddy_gather_check() compare the gathered arrays across ranks.
static int eddy_gather_verify = 0;

static void eg_fail(const char *msg) {
  fprintf(stderr, "eddy_gather: %s\n", msg);
  fflush(stderr);
#ifdef OPS_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  exit(1);
}

// Rank/size bookkeeping that eddy_gather_check() and eddy_report() need in both
// modes.  eddy_gather_init() calls this too, but the broadcast path never goes
// near the dats, so it has to be callable on its own - otherwise eg_n stays 0
// and the cross-rank check silently compares nothing.
static void eddy_check_init(int n_eddies) {
  eg_n = n_eddies;
#ifdef OPS_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &eg_nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &eg_myrank);
#endif
}

// eddy_ref  : any one of the eddy dats, used for the x-decomposition
// block_ref : a full-size dat on the same block (e.g. rho_B0), used to find out
//             whether this rank owns the global (y=0, z=0) pencil, i.e. whether
//             it actually executes the eddy kernels
// n_eddies  : the global eddy count
static void eddy_gather_init(ops_dat eddy_ref, ops_dat block_ref, int n_eddies) {
  eddy_check_init(n_eddies);
  if (eg_n <= 0) eg_fail("initialised with a non-positive eddy count");

#ifndef OPS_MPI
  (void)block_ref;
  int d[OPS_MAX_DIM] = {0}, s[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(eddy_ref, 0, d, s);
  eg_local_disp = 0;
  eg_local_n = eg_n;
  eg_send_n = eg_n;
  eg_fetch_cap = (s[0] > eg_n) ? s[0] : eg_n;
  eg_fetch_d = (double *)malloc(eg_fetch_cap * sizeof(double));
  eg_fetch_i = (int *)malloc(eg_fetch_cap * sizeof(int));
#else
  MPI_Comm_size(MPI_COMM_WORLD, &eg_nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &eg_myrank);

  // What slice of the eddy array does this rank store?  ops_dat_get_extents
  // reports the owned (halo-free) global range, and ops_dat_fetch_data writes
  // exactly size[0]*size[1]*size[2] elements into the caller's buffer.
  int d[OPS_MAX_DIM] = {0}, s[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(eddy_ref, 0, d, s);
  if (d[0] < 0) eg_fail("negative displacement on an eddy dat");

  int cap = 1;
  for (int i = 0; i < 3; i++) cap *= (s[i] > 0 ? s[i] : 0);
  eg_fetch_cap = (cap > 0) ? cap : 1;
  eg_fetch_d = (double *)malloc(eg_fetch_cap * sizeof(double));
  eg_fetch_i = (int *)malloc(eg_fetch_cap * sizeof(int));

  eg_local_disp = d[0];
  eg_local_n = (s[0] > 0) ? s[0] : 0;
  if (eg_local_disp >= eg_n)                  eg_local_n = 0;
  else if (eg_local_disp + eg_local_n > eg_n) eg_local_n = eg_n - eg_local_disp;

  // Does this rank own global (y=0, z=0)?  That is the pencil the eddy kernels
  // are iterated over, so it is exactly the set of ranks that ran them.
  int bd[OPS_MAX_DIM] = {0}, bs[OPS_MAX_DIM] = {0};
  ops_dat_get_extents(block_ref, 0, bd, bs);
  int owns_pencil = (bd[1] <= 0 && bd[1] + bs[1] > 0) &&
                    (bd[2] <= 0 && bd[2] + bs[2] > 0);
  eg_send_n = owns_pencil ? eg_local_n : 0;

  int *cnt = (int *)malloc(eg_nranks * sizeof(int));
  int *dsp = (int *)malloc(eg_nranks * sizeof(int));
  int my_disp = (eg_send_n > 0) ? eg_local_disp : 0;
  MPI_Allgather(&eg_send_n, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my_disp,   1, MPI_INT, dsp, 1, MPI_INT, MPI_COMM_WORLD);

  // The contributions must tile [0, eddies) exactly: no gaps (garbage in the
  // result) and no overlaps (an eddy written twice, the second write winning).
  // Cheap to check once, and it is the assumption the whole scheme rests on.
  {
    char *cover = (char *)calloc(eg_n, 1);
    for (int r = 0; r < eg_nranks; r++)
      for (int i = 0; i < cnt[r]; i++) {
        int g = dsp[r] + i;
        if (g < 0 || g >= eg_n) eg_fail("a rank contributes eddies outside [0, eddies)");
        cover[g]++;
      }
    for (int i = 0; i < eg_n; i++) {
      if (cover[i] == 0) eg_fail("no rank contributes some eddy index - gathered array would contain garbage");
      if (cover[i] > 1)  eg_fail("more than one rank contributes the same eddy index");
    }
    free(cover);
  }

  eg_cnt_d = (int *)malloc(eg_nranks * sizeof(int));
  eg_dsp_d = (int *)malloc(eg_nranks * sizeof(int));
  eg_cnt_i = (int *)malloc(eg_nranks * sizeof(int));
  eg_dsp_i = (int *)malloc(eg_nranks * sizeof(int));
  for (int r = 0; r < eg_nranks; r++) {
    eg_cnt_d[r] = EG_NDOUBLE * cnt[r];
    eg_dsp_d[r] = EG_NDOUBLE * dsp[r];
    eg_cnt_i[r] = EG_NINT * cnt[r];
    eg_dsp_i[r] = EG_NINT * dsp[r];
  }
  free(cnt);
  free(dsp);

  eg_send_d = (double *)malloc((eg_send_n > 0 ? EG_NDOUBLE * eg_send_n : 1) * sizeof(double));
  eg_send_i = (int *)malloc((eg_send_n > 0 ? EG_NINT * eg_send_n : 1) * sizeof(int));
  eg_recv_d = (double *)malloc(EG_NDOUBLE * eg_n * sizeof(double));
  eg_recv_i = (int *)malloc(EG_NINT * eg_n * sizeof(int));

  printf("[rank %d] eddy_gather: stores [%d, %d), contributes %d (pencil owner: %s)\n",
         eg_myrank, eg_local_disp, eg_local_disp + eg_local_n, eg_send_n,
         owns_pencil ? "yes" : "no");
  fflush(stdout);
#endif

  if (!eg_fetch_d || !eg_fetch_i) eg_fail("out of memory");
}

// Push a globally-indexed per-eddy array into an eddy rng dat: the mirror of the
// fetch in eddy_gather().  `global` holds ncomp values for every one of the
// eddies and must already be identical on all ranks; each rank writes only the
// slab it stores.  Collective, like the fetch - all ranks must call it.
//
// Use the *slab* form, not ops_dat_set_data.  The two disagree about the index
// origin: ops_dat_set_data builds its range from sd->gbl_d_m (halo-inclusive,
// ops_mpi_rt_support.cpp:2197), so with halo_m = {-1,0,0} on these dats it lands
// the buffer one cell early and every eddy comes out shifted by one.  The slab
// form indexes as range[2d] + i - (sd->d_im[d] + dat->d_m[d]) (set_loop_slab,
// ops_util.cpp:427), which is the same origin ops_dat_fetch_data uses, so a
// local range starting at 0 is exactly the first owned eddy.
static void eddy_set_rng(ops_dat dat, const int *global, int ncomp) {
  int local_range[6] = {0, (eg_local_n > 0 ? eg_local_n : 0), 0, 1, 0, 1};
  if (eg_local_n <= 0) {
    // Nothing stored here.  Still call in: the slab loop is a no-op for a
    // zero-width range, but ops_execute() inside it is collective.
    int dummy = 0;
    ops_dat_set_data_slab_memspace(dat, 0, (char *)&dummy, local_range, OPS_HOST);
    return;
  }
  int *buf = (int *)malloc(ncomp * eg_local_n * sizeof(int));
  if (!buf) eg_fail("out of memory staging an rng dat");
  memcpy(buf, global + ncomp * eg_local_disp, ncomp * eg_local_n * sizeof(int));
  ops_dat_set_data_slab_memspace(dat, 0, (char *)buf, local_range, OPS_HOST);
  free(buf);
}

// Fetch the eddy dats and leave the complete, identical arrays on every rank.
// All ranks must call this: ops_dat_fetch_data triggers OPS' lazy execution and
// may broadcast a low-dimensional pencil, both of which are collective.
static void eddy_gather(ops_dat d_x, ops_dat d_y, ops_dat d_z, ops_dat d_r,
                        ops_dat d_inc, ops_dat d_ex, ops_dat d_ey, ops_dat d_ez,
                        double *g_x, double *g_y, double *g_z, double *g_r,
                        double *g_inc, int *g_ex, int *g_ey, int *g_ez) {
#ifndef OPS_MPI
  ops_dat_fetch_data(d_x,   0, (char *)g_x);
  ops_dat_fetch_data(d_y,   0, (char *)g_y);
  ops_dat_fetch_data(d_z,   0, (char *)g_z);
  ops_dat_fetch_data(d_r,   0, (char *)g_r);
  ops_dat_fetch_data(d_inc, 0, (char *)g_inc);
  ops_dat_fetch_data(d_ex,  0, (char *)g_ex);
  ops_dat_fetch_data(d_ey,  0, (char *)g_ey);
  ops_dat_fetch_data(d_ez,  0, (char *)g_ez);
#else
  // Pack this rank's slab.  Non-contributors still fetch (collective) but pack
  // nothing, because eg_send_n is 0 for them.
#define EG_PACK_D(dat, field)                                    \
  ops_dat_fetch_data(dat, 0, (char *)eg_fetch_d);                \
  for (int i = 0; i < eg_send_n; i++)                            \
    eg_send_d[EG_NDOUBLE * i + (field)] = eg_fetch_d[i];
#define EG_PACK_I(dat, field)                                    \
  ops_dat_fetch_data(dat, 0, (char *)eg_fetch_i);                \
  for (int i = 0; i < eg_send_n; i++)                            \
    eg_send_i[EG_NINT * i + (field)] = eg_fetch_i[i];

  EG_PACK_D(d_x,   0)
  EG_PACK_D(d_y,   1)
  EG_PACK_D(d_z,   2)
  EG_PACK_D(d_r,   3)
  EG_PACK_D(d_inc, 4)
  EG_PACK_I(d_ex,  0)
  EG_PACK_I(d_ey,  1)
  EG_PACK_I(d_ez,  2)

#undef EG_PACK_D
#undef EG_PACK_I

  MPI_Allgatherv(eg_send_d, EG_NDOUBLE * eg_send_n, MPI_DOUBLE,
                 eg_recv_d, eg_cnt_d, eg_dsp_d, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Allgatherv(eg_send_i, EG_NINT * eg_send_n, MPI_INT,
                 eg_recv_i, eg_cnt_i, eg_dsp_i, MPI_INT, MPI_COMM_WORLD);

  for (int i = 0; i < eg_n; i++) {
    g_x[i]   = eg_recv_d[EG_NDOUBLE * i + 0];
    g_y[i]   = eg_recv_d[EG_NDOUBLE * i + 1];
    g_z[i]   = eg_recv_d[EG_NDOUBLE * i + 2];
    g_r[i]   = eg_recv_d[EG_NDOUBLE * i + 3];
    g_inc[i] = eg_recv_d[EG_NDOUBLE * i + 4];
    g_ex[i]  = eg_recv_i[EG_NINT * i + 0];
    g_ey[i]  = eg_recv_i[EG_NINT * i + 1];
    g_ez[i]  = eg_recv_i[EG_NINT * i + 2];
  }
#endif
}

// Confirm every rank came out of eddy_gather() with bit-identical arrays.
// Collective; call it on all ranks or not at all.
static void eddy_gather_check(const char *where, double *g_x, double *g_y,
                              double *g_z, double *g_r, int *g_ex, int *g_ey,
                              int *g_ez) {
#ifndef OPS_MPI
  (void)where; (void)g_x; (void)g_y; (void)g_z; (void)g_r;
  (void)g_ex; (void)g_ey; (void)g_ez;
#else
  // Bitwise sum, so a single differing bit anywhere changes the result.
  unsigned long long sum = 0;
  for (int i = 0; i < eg_n; i++) {
    unsigned long long b;
    memcpy(&b, &g_x[i], sizeof(b)); sum += b;
    memcpy(&b, &g_y[i], sizeof(b)); sum += b;
    memcpy(&b, &g_z[i], sizeof(b)); sum += b;
    memcpy(&b, &g_r[i], sizeof(b)); sum += b;
    sum += (unsigned long long)(unsigned int)g_ex[i];
    sum += (unsigned long long)(unsigned int)g_ey[i];
    sum += (unsigned long long)(unsigned int)g_ez[i];
  }
  unsigned long long lo = 0, hi = 0;
  MPI_Allreduce(&sum, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&sum, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (lo != hi) {
    if (eg_myrank == 0)
      fprintf(stderr, "eddy_gather_check(%s): ranks disagree (min %llu, max %llu)\n",
              where, lo, hi);
    eg_fail("gathered eddy arrays are not identical on all ranks");
  }
  if (eg_myrank == 0) {
    printf("eddy_gather_check(%s): all %d ranks agree (checksum %llu)\n",
           where, eg_nranks, sum);
    fflush(stdout);
  }
#endif
}

// ---------------------------------------------------------------------------
// Reference mode: rank 0 owns the eddies outright, everyone else is told.
//
// The gather above only exists because the eddy state lives in OPS dats that
// are decomposed in x.  For a reference run that constraint can be dropped: the
// eddy field is a few hundred entries updated by a serial loop, which is nothing
// beside a 750x250x150 timestep, so rank 0 can compute the whole thing on the
// host and broadcast it.
//
// That removes every distributed-memory question from the eddy field at once -
// no decomposition, no pencil ownership, no per-rank RNG streams - so whatever
// ParaView then shows is a property of the SEM algorithm alone, not of the MPI
// plumbing.  Drive it with the host_instantiate_eddies() / host_convect_eddies()
// functions already in opensbli.cpp:
//
//     if (eddy_is_root()) host_convect_eddies();
//     eddy_bcast(...);
// ---------------------------------------------------------------------------

#define EDDY_MODE_KERNEL_GATHER 0  // OPS par_loops + MPI_Allgatherv (the committed path)
#define EDDY_MODE_HOST_BCAST    1  // rank 0 host loops + MPI_Bcast (reference)

static int eddy_mode = EDDY_MODE_KERNEL_GATHER;

// The rank that computes the eddy field in EDDY_MODE_HOST_BCAST.
static int eddy_is_root() {
#ifndef OPS_MPI
  return 1;
#else
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r == 0;
#endif
}

// Replicate the arrays rank 0 just filled onto every other rank.
static void eddy_bcast(double *g_x, double *g_y, double *g_z, double *g_r,
                       double *g_inc, int *g_ex, int *g_ey, int *g_ez, int n) {
#ifndef OPS_MPI
  (void)g_x; (void)g_y; (void)g_z; (void)g_r; (void)g_inc;
  (void)g_ex; (void)g_ey; (void)g_ez; (void)n;
#else
  MPI_Bcast(g_x,   n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(g_y,   n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(g_z,   n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(g_r,   n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(g_inc, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(g_ex,  n, MPI_INT,    0, MPI_COMM_WORLD);
  MPI_Bcast(g_ey,  n, MPI_INT,    0, MPI_COMM_WORLD);
  MPI_Bcast(g_ez,  n, MPI_INT,    0, MPI_COMM_WORLD);
#endif
}

// ---------------------------------------------------------------------------
// Diagnostics: is the eddy field itself sane, independently of how it got here?
// Eddies collapsing into part of the box, or every eddy sharing one sign triple,
// are both invisible in a cross-rank consistency check and both wreck the
// fluctuation field.  Cheap enough to print once per run.
// ---------------------------------------------------------------------------
static void eddy_report_range(const double *a, int n, const char *name,
                              double lo, double hi) {
  double mn = a[0], mx = a[0], sum = 0.0;
  for (int i = 0; i < n; i++) {
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
    sum += a[i];
  }
  ops_printf("  %-5s min=% .5e max=% .5e mean=% .5e  box [% .5e, % .5e]  span %.1f%%\n",
             name, mn, mx, sum / n, lo, hi, 100.0 * (mx - mn) / (hi - lo));
}

static void eddy_report_sign(const int *a, int n, const char *name) {
  int pos = 0;
  for (int i = 0; i < n; i++) if (a[i] > 0) pos++;
  ops_printf("  %-5s  +1: %5d (%5.1f%%)   -1: %5d (%5.1f%%)%s\n", name, pos,
             100.0 * pos / n, n - pos, 100.0 * (n - pos) / n,
             (pos == 0 || pos == n) ? "   <-- degenerate, all one sign" : "");
}

static void eddy_report(const char *where, int n,
                        const double *g_x, const double *g_y, const double *g_z,
                        const int *g_ex, const int *g_ey, const int *g_ez,
                        double xlo, double xhi, double ylo, double yhi,
                        double zlo, double zhi) {
  ops_printf("eddy field after %s (%d eddies, mode %s):\n", where, n,
             eddy_mode == EDDY_MODE_HOST_BCAST ? "host+bcast" : "kernel+allgatherv");
  eddy_report_range(g_x, n, "x", xlo, xhi);
  eddy_report_range(g_y, n, "y", ylo, yhi);
  eddy_report_range(g_z, n, "z", zlo, zhi);
  eddy_report_sign(g_ex, n, "eps_x");
  eddy_report_sign(g_ey, n, "eps_y");
  eddy_report_sign(g_ez, n, "eps_z");
}

static void eddy_gather_free() {
  free(eg_fetch_d); eg_fetch_d = NULL;
  free(eg_fetch_i); eg_fetch_i = NULL;
  free(eg_send_d);  eg_send_d = NULL;
  free(eg_send_i);  eg_send_i = NULL;
  free(eg_recv_d);  eg_recv_d = NULL;
  free(eg_recv_i);  eg_recv_i = NULL;
  free(eg_cnt_d);   eg_cnt_d = NULL;
  free(eg_dsp_d);   eg_dsp_d = NULL;
  free(eg_cnt_i);   eg_cnt_i = NULL;
  free(eg_dsp_i);   eg_dsp_i = NULL;
}

#endif // EDDY_GATHER_H
