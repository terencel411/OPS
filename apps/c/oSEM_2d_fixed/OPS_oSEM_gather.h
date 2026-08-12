/*
 * OPS_oSEM_gather.h  --  a correct global gather of the eddy dats
 *
 * WHAT THIS REPLACES
 *
 * The reference drives compute_fluct from seven host arrays refreshed once a
 * step, between convect_eddies and compute_fluct:
 *
 *     ops_dat_fetch_data(d_x_gbl, 0, (char*)x_gbl);      // ... and six more
 *
 * That is correct in serial and wrong under MPI. `ops_dat_fetch_data` computes
 * the rank's global displacement and then never applies it
 * (ops/c/src/mpi/ops_mpi_rt_support.cpp:2125-2146):
 *
 *     int ldisp[OPS_MAX_DIM] = {0};
 *     ops_dat_get_extents(dat, part, ldisp, lsize);      // ldisp filled here
 *     ...
 *     memcpy(&data[k*lsize[0]*lsize[1] + j*lsize[0]],    // ... and never used
 *            &dat->data[...], lsize[0]);
 *
 * So every rank writes ITS OWN slice starting at offset 0 of the destination.
 * On n ranks, x_gbl[0 .. eddies/n) holds that rank's eddies and the remaining
 * entries are whatever malloc last left there, while compute_fluct loops over
 * the full global `eddies`. Each rank therefore computes a different inlet
 * plane from a different set of mostly-uninitialised eddies. It does not
 * crash, and the output looks plausible, which is what makes it dangerous.
 *
 * WHAT THIS DOES INSTEAD
 *
 * The displacement the library discards is exactly what an allgather needs, and
 * ops_dat_get_extents is public API (ops_lib_core.h:1193). So: ask for the
 * rank's (displacement, count), fetch the local slice -- offset 0 is now the
 * correct destination, because the buffer IS local -- and MPI_Allgatherv it
 * into the global array using those displacements.
 *
 * WHY NOT AN ops_reduction, AS particle_global_influence_oSEM USES
 *
 * That app gathers with an array-valued ops_reduction indexed by global eddy
 * id: ops_arg_reduce(h_all, eddies * NCOMP, "double", OPS_INC) inside an
 * ops_particle_par_loop, which reduces to an MPI_Allreduce over the whole
 * buffer. It works there for a reason that does not carry over -- the
 * translator does not parse ops_particle_par_loop at all. A grid ops_par_loop
 * IS parsed, and the reduction path has two constraints that kill it here:
 *
 *   1. the dimension must be a bare integer literal (parseIntLiteral,
 *      ops_translator/ops-translator/cpp/parser.py:353), so `eddies * NCOMP`
 *      fails the translator build while the dev builds compile happily;
 *   2. the generated reduction is UNROLLED, one scalar local and one
 *      write-back per slot. 1718 eddies x 7 components = 12026 slots. A
 *      measured 6150-slot reduction produced a 24819-line kernel that had not
 *      compiled after 500 s.
 *
 * Both are recorded in particle_global_influence_oSEM/UNDERSTANDING_oSEM.md.
 * MPI_Allgatherv also measured ~4x faster than the reduction in that app, on
 * bit-identical results, so this is not a consolation prize.
 *
 * ASSUMPTION, CHECKED AT RUNTIME
 *
 * The eddy block is declared {eddies, 1}, so its decomposition must be 1-D
 * along dimension 0 for a displacement/count pair to describe the slice. If
 * OPS ever splits dimension 1 as well the slice stops being contiguous in
 * global index order and this gather would silently mis-order. gather_check()
 * verifies it once rather than trusting it.
 */

#ifndef OPS_OSEM_GATHER_H
#define OPS_OSEM_GATHER_H

#include <vector>
#include <ops_lib_core.h>

#ifdef OPS_MPI
#include <mpi.h>

template <typename T> struct osem_mpi_type;
template <> struct osem_mpi_type<double> {
  static MPI_Datatype value() { return MPI_DOUBLE; }
};
template <> struct osem_mpi_type<int> {
  static MPI_Datatype value() { return MPI_INT; }
};

/**
 * Zero the count of any rank whose (displacement, count) a lower-numbered rank
 * already carries, so the surviving slices tile [0, eddies) exactly once.
 *
 * Needed because the eddy block is declared {eddies, 1} and OPS does not
 * always split it 1-D. At np = 8 here it hands each dim-0 slice to TWO ranks,
 * both reporting extent 1 in the degenerate dimension -- measured as slices
 * summing to 3436 for 1718 eddies. Duplicates hold identical data (the kernels
 * are deterministic and the rng is keyed on global id, so both copies of an
 * eddy draw the same numbers), so dropping all but the first is safe. Leaving
 * them in is not: MPI_Allgatherv with overlapping displacements is erroneous.
 */
inline void osem_dedup_slices(std::vector<int> &counts,
                              std::vector<int> &displs) {
  for (size_t r = 0; r < counts.size(); r++)
    for (size_t q = 0; q < r; q++)
      if (displs[q] == displs[r] && counts[q] == counts[r]) {
        counts[r] = 0;
        break;
      }
}
#endif

/**
 * Fetch every element of a 1-D eddy dat onto every rank, in global index order.
 *
 * @param dat   an eddy_block dat of global extent {eddies, 1}
 * @param out   destination, `nglobal` elements, valid on every rank on return
 */
template <typename T>
inline void gather_eddy_dat(ops_dat dat, T *out, int nglobal) {
#ifdef OPS_MPI
  int ldisp[OPS_MAX_DIM] = {0};
  int lsize[OPS_MAX_DIM] = {1};
  ops_dat_get_extents(dat, 0, ldisp, lsize);

  const int n_local = lsize[0];
  const int start = ldisp[0];

  /* fetch_data writes at offset 0; here that is what we want, because the
     destination is this rank's own slice rather than the global array. */
  std::vector<T> local(n_local > 0 ? (size_t)n_local : 1);
  ops_dat_fetch_data(dat, 0, (char *)local.data());

  int nranks = 0, myrank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  std::vector<int> counts(nranks), displs(nranks);
  MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  MPI_Allgather(&start, 1, MPI_INT, displs.data(), 1, MPI_INT, MPI_COMM_WORLD);

  osem_dedup_slices(counts, displs);

  MPI_Allgatherv(local.data(), counts[myrank], osem_mpi_type<T>::value(), out,
                 counts.data(), displs.data(), osem_mpi_type<T>::value(),
                 MPI_COMM_WORLD);
#else
  /* Serial: one rank owns everything, so the reference call was already
     correct and there is nothing to gather. */
  (void)nglobal;
  ops_dat_fetch_data(dat, 0, (char *)out);
#endif
}

/**
 * Verify the eddy decomposition is 1-D and that the slices tile [0, eddies)
 * exactly. Call once after ops_partition(). Cheap, and it turns a silent
 * mis-ordering into a message.
 */
inline void gather_check(ops_dat dat, int nglobal, void *host_array) {
  /* ALIASING CHECK. ops_decl_dat takes ownership of a non-NULL data pointer
     (ops_lib_core.cpp:637-641 reallocs it) and MPI partitioning allocates only
     when dat->data == NULL (ops_mpi_partition.cpp:811). So passing a malloc'd
     array to ops_decl_dat, as the reference does, leaves the dat and the app's
     "global" host array pointing at the SAME memory. Gathering into it would
     then overwrite the dat's own local slab. */
  if ((void *)dat->data == host_array)
    ops_printf("gather_check: WARNING -- the dat and the host array alias at "
               "%p; the gather would overwrite the dat's own storage\n",
               host_array);
  else
    ops_printf("gather_check: dat %p and host array %p are distinct\n",
               (void *)dat->data, host_array);

#ifdef OPS_MPI
  int ldisp[OPS_MAX_DIM] = {0};
  int lsize[OPS_MAX_DIM] = {1};
  ops_dat_get_extents(dat, 0, ldisp, lsize);

  if (lsize[1] != 1 || ldisp[1] != 0) {
    ops_printf("gather_check: FAILED -- the eddy block is split in dimension 1 "
               "(lsize[1] = %d, ldisp[1] = %d). The gather assumes a 1-D "
               "decomposition; results would be mis-ordered.\n",
               lsize[1], ldisp[1]);
    return;
  }

  int nranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  std::vector<int> counts(nranks), displs(nranks);
  MPI_Allgather(&lsize[0], 1, MPI_INT, counts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  MPI_Allgather(&ldisp[0], 1, MPI_INT, displs.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  int raw = 0;
  for (int r = 0; r < nranks; r++) raw += counts[r];
  osem_dedup_slices(counts, displs);

  int total = 0;
  int ok = 1;
  for (int r = 0; r < nranks; r++) {
    if (counts[r] == 0) continue;
    if (displs[r] != total) ok = 0;
    total += counts[r];
  }
  if (raw != total)
    ops_printf("gather_check: eddy block is not split 1-D -- slices sum to %d "
               "for %d eddies; %d duplicate ranks dropped from the gather\n",
               raw, nglobal, raw / (nglobal ? nglobal : 1) - 1);
  if (!ok || total != nglobal) {
    ops_printf("gather_check: FAILED -- slices do not tile [0, %d): total %d\n",
               nglobal, total);
    return;
  }
  ops_printf("gather_check: %d ranks tile [0, %d) contiguously\n", nranks,
             nglobal);
#else
  (void)dat;
  ops_printf("gather_check: serial build, gather is a direct fetch of %d\n",
             nglobal);
#endif
}

#endif /* OPS_OSEM_GATHER_H */
