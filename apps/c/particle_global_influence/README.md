# All-to-all particle interaction in OPS Particles, without application MPI

Each particle needs a quantity summed over **every other particle in the
domain** — no cut-off, no neighbour list. The particles are decomposed across
ranks, so no rank holds the data it needs.

```
phi_i = sum_{j != i}  m_j / (eps + |r_i - r_j|)
```

## Why the particle library alone does not do it

| Mechanism | Reach |
|---|---|
| border exchange / virtual particles | the halo band sized by the stencil given to `ops_decl_mapping` — a cell or two |
| `ops_particle_inter_loop` | pairs found by walking that same map with a stencil |
| `ops_particle_halo_group` | a translated band across a periodic seam or block interface |
| `ops_particle_intrablock_reverse` | ghosts back to their owners |

Every one of them is short-range by construction. There is no
`ops_particle_allgather`, and `particle->global_particles` is only ever set to
the local count (`ops/c/src/core/ops_particle_lib_core.cpp:204`).

## The mechanism that does

An `ops_reduction` is already a global collective, and it is **element-wise
over an array of any length**:

- `ops_decl_reduction_handle(size, ...)` takes `size` in **bytes**
- `ops_reduction_result()` → `ops_checkpointing_reduction` →
  `ops_execute_reduction` → `MPI_Allreduce(..., dim, ..., OPS_MPI_GLOBAL)`
  (`ops/c/src/mpi/ops_mpi_rt_support.cpp:1281`), so the reduced array comes
  back on **every** rank
- `ops_arg_reduce` zeroes the buffer for `OPS_INC` each time it is used
  (`ops/c/src/core/ops_lib_core.cpp:966`)
- particle loops accept it: `particle_param_handler` routes a non-`OPS_READ`
  `OPS_ARG_GBL` to the reduction storage
  (`ops/c/include/ops_particle_seq.h:634`)

Give every particle a globally unique id dense in `[0, N)`, let each rank
`+=` its own particles into their own slots, and every slot has exactly one
non-zero contributor — so `SUM` returns the value itself, bit for bit.
**A sum-reduction over a disjointly filled array is an allgather.**

Kernel C becomes two loops with the collective between them:

```
C1  ops_particle_par_loop(KerPublishState, ..., ops_arg_reduce(h_all, N*NCOMP, "double", OPS_INC))
    ops_reduction_result(h_all, all_state.data());          <-- the Allreduce
C2  ops_particle_par_loop(KerInfluence,    ..., ops_arg_gbl(all_state.data(), N*NCOMP, "double", OPS_READ))
```

`NCOMP` is per-particle payload — here `{x, y, strength}`; add whatever else
the interaction law needs.

## Rules it depends on

1. **Global ids, unique and dense in `[0,N)`.** Here every rank walks the same
   seeding RNG sequence, so the sequence index *is* the id, assigned with zero
   communication. If you seed only your own share per rank, build the same
   numbering with a prefix sum — itself an OPS `int` reduction of `dim =
   nranks`, still no MPI.
2. **C1 must be `OPS_PARTICLE_ITERATE_LOCAL`.** A ghost carries its owner's
   gid; `ITERATE_ALL` would add that particle in twice and double its
   gathered position.
3. **`ops_reduction_result` is collective** — every rank must reach it the
   same number of times. `ops_particle_par_loop` returns early on a rank that
   owns no particles, which is safe: `ops_arg_reduce()` is evaluated at the
   call site, so the handle is still zeroed and registered.
4. **Fixed population.** Insertion/deletion moves ids. If the population
   changes, first gather the global count with an `int` `OPS_INC` reduction,
   size the buffer to a maximum, and renumber — the gather itself is unchanged.

## What it costs

- one global collective per timestep — a hard synchronisation point
- `N * NCOMP * 8` bytes per rank per step
- `O(N)` work per particle → `O(N²)` per step
- the MPI backend allocates `dim * nranks` scratch elements inside the reduce
  (`ops_mpi_rt_support.cpp:1282`), i.e. `N * NCOMP * 8 * nranks` bytes
  momentarily

Fine for thousands of particles. For millions the answer is a different
algorithm, not a better gather: project the particle strengths onto the grid
with the particle overload of `ops_par_loop`, combine ghost contributions with
`ops_particle_intrablock_reverse(..., OPS_INC)`, smooth/solve on the grid with
ordinary `ops_par_loop`s (where OPS's halo exchange does all the communication),
and interpolate back with `ops_par_particle_grid_loop`. That is the
particle-mesh route: `O(N)`, no collective, and approximate.

## Visualisation

One self-contained HDF5 file per output step, then a plotting script:

```
./influence_dev_seq                  # writes 50 x influence_output_??????.h5
python3 plot_influence_h5.py         # -> frames/*.png and frames/influence.gif
python3 plot_influence_h5.py --quiver --no-gif
```

The left panel is the cloud coloured by influence (marker area = strength); the
right panel tracks min/mean/max phi over the run. The colour scale is fixed
across frames on purpose — a per-frame autoscale would hide the very change you
are trying to see.

**The writer is also MPI-free, for the same reason the physics is.** The
particle API has no HDF5 path, so apps roll their own;
`particle_tutorial_1_drift/drift_io.h` does it with `MPI_Allgatherv` over the
per-rank slices. Here the gather has already happened, so `influence_io.h` just
writes the reduction buffers. Two consequences worth having:

- the arrays are ordered by **global id**, not by rank, so row `i` is gid `i` in
  every frame and particle tracks are trivial to follow
- the files are **byte-identical at any rank count** (verified: 1071 datasets
  across 51 files, serial vs. np=8, zero differences). A rank-ordered
  `Allgatherv` cannot give you that.

`h_phi` is a second gather, because phi does not exist until C2 has run and so
cannot ride in the C1 buffer. `influence_io.h` also shows an OPS-only
collective barrier (`ops_sync_barrier`), used to separate "every rank deletes
stale files" from "one rank starts creating new ones".

### Why the initial condition has a swirl

Kernel A gives each particle `v = omega * (-(y-cy), (x-cx))`. Without it — with
every particle sharing one velocity — the cloud translates **rigidly**, every
inter-particle distance is constant, and phi is frozen for the entire run. There
would be nothing to watch. The swirl (with no centripetal force to balance it)
rotates and spreads the cloud, so phi genuinely evolves: it falls from ~2429 to
~1097 over the run as the cloud dilutes, with the dense core staying brightest.
The motion is still a closed-form recurrence, so the exact check is unaffected.

## Files

| File | Contents |
|---|---|
| `influence.cpp` | driver: declarations, seeding, time loop, verification |
| `particle_kernels.h` | kernels A, B, C1, C2, the phi gather and the check kernel |
| `grid_kernels.h` | coordinate-grid fill only |
| `influence_io.h` | per-step HDF5 frames, written from the gather buffers |
| `plot_influence_h5.py` | frames + animated GIF from the .h5 files |

## Build and run

```
source /home/terence411/proj5/installations/configure_env_vars.sh
make influence_dev_seq        # serial, no translator
make influence_dev_mpi        # MPI, no translator
make influence_mpi            # MPI, via the OPS translator

./influence_dev_seq
mpirun -np 4 ./influence_dev_mpi
python3 plot_influence_h5.py
```

The run is self-checking: the influence is recomputed on the host from the
known exact trajectory of all `NPART` particles and compared inside a kernel,
with the error and the particle count reduced through OPS. A correct run prints
`PASS` and prints the same numbers at any rank count. There is no `#include
<mpi.h>` anywhere in the application.
