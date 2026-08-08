# All-to-all influence with an MPI_Allgatherv gather

Same problem and same physics as
[`../particle_global_influence`](../particle_global_influence): every particle
needs a quantity summed over every other particle in the domain, so the whole
population must be on every rank once per step.

That app gathers with an array-valued `ops_reduction` — each rank INCs its
particles into slots picked by global id, and the `MPI_Allreduce` inside
`ops_reduction_result` hands the array back. Elegant, and no MPI in the
application at all.

**This app does the same gather with `MPI_Allgatherv`**, which is the primitive
that actually matches the operation. An Allreduce ships the full `N*NCOMP` array
from every rank and sums it; the data is disjoint, so a concatenation is what is
wanted, and each rank sends only its own particles.

```
make influence_gather_dev_seq
./influence_gather_dev_seq -npart 2000 -nsteps 100 -check

make influence_gather_dev_mpi
OMP_NUM_THREADS=1 mpirun -np 4 ./influence_gather_dev_mpi -npart 5000
```

---

## How it compares

Gather cost per timestep, ms, N = 5000, measured against the reduction version.
(Those numbers came from an earlier revision of this app that compiled both
gathers into one binary and ran them in the same timestep — same machine, same
decomposition, same positions. That build also confirmed the two produce
**bit-identical** buffers, `max|diff| = 0.000e+00` at np = 1, 2, 4, 8, which is
the only acceptable answer since both are pure data movement.)

| ranks | `ops_reduction` | `MPI_Allgatherv` | speedup |
|---:|---:|---:|---:|
| 1 | 0.2446 | 0.1000 | 2.4× |
| 2 | 0.4769 | 0.1337 | 3.6× |
| 4 | 0.9734 | 0.2232 | 4.4× |
| 8 | 1.8100 | 0.4159 | 4.4× |

And with N, at np = 4:

| N | `ops_reduction` | `MPI_Allgatherv` | speedup |
|---:|---:|---:|---:|
| 1 000 | 0.1597 | 0.0385 | 4.1× |
| 5 000 | 0.7796 | 0.1854 | 4.2× |
| 20 000 | 4.0389 | 0.8590 | 4.7× |

The physics check still passes bit-exactly here — `max influence error
0.000e+00` at np = 1 and np = 4.

## Reading the numbers

Both methods scale linearly in N, which is just data volume. The interesting
axis is **rank count**: at fixed N = 5000 the reduction goes 0.24 → 0.48 → 0.97
→ 1.81 ms for 1 → 2 → 4 → 8 ranks, i.e. almost exactly proportional to P, while
Allgatherv grows about four times more slowly.

That is partly inherent — an Allreduce has every rank contribute the full array
— and partly an OPS implementation cost. Two allocations in the reduce path are
larger than they need to be:

- `ops_mpi_rt_support.cpp:1282` — `std::vector<type> result(arg->dim *
  ops_comm_global_size)`, allocated per call. For `OPS_INC` only `dim` of it is
  ever used, and the size grows with rank count.
- `ops_mpi_rt_support.cpp` `ops_reduce_exec` — `std::vector<type>
  local(handle->size)`, where `handle->size` is in **bytes**, so for doubles it
  over-allocates 8×.

At N = 5000 that is a few MB of scratch allocated and copied on every step, and
it scales the wrong way. It is consistent with the reduction being slower even
at **np = 1**, where there is no communication at all to blame. I have not
isolated how much of the 4× is allocation and how much is the collective itself,
so treat that as a contributing factor rather than the whole story.

## What each method actually has to do

|  | `ops_reduction` | `MPI_Allgatherv` |
|---|---|---|
| collectives | 1 (Allreduce of `N*NCOMP`) | 2 (Allgather of P counts, then Allgatherv) |
| bytes sent per rank | the whole array | only its own particles |
| result ordering | already gid-ordered — the slot **is** the id | rank order; must carry gids and be permuted |
| packing | none | host-side pack of a contiguous send buffer |
| application MPI | none | one function |

Everything in the right-hand column is timed as part of the Allgatherv route —
the count exchange, the pack, the collective, the permute. Anything else would
flatter it.

The count exchange cannot be cached: particles migrate between ranks, so the
per-rank counts change.

## Why the pack is host code

The OPS-native way to fill a per-particle buffer would be
`ops_arg_gbl_particle`, which exists for exactly this. **It does not work.**
In `ops_particle_seq.h:681-689`:

```c
static void shift_arg(const ops_arg &arg, char *p, const int offs, ...) {
  if (arg.argtype == OPS_ARG_IDP) { instance->arg_idp[0] += offs; }
  else if (arg.argtype == OPS_ARG_GBL_PARTICLE) { p += offs * arg.elem_size; }
}
```

`p` is taken **by value**, so the increment is dead code and the pointer never
advances — every particle would write to slot 0. Compare the `IDP` branch
directly above it, which mutates `instance->arg_idp[0]`, and the `ACCP` handler,
which calls `next()` on the pointed-to accessor; those mutate something that
outlives the call, and they work.

So the pack reads the particle dats' contiguous AoS storage directly, the same
way seeding and the HDF5 writers in these apps already do. Everything else —
loops, kernels, migration cycle, verification — stays OPS. Raw MPI appears in
exactly one function, `gather_state_mpi()`.

## So should the other apps switch?

Only if the gather is actually costing you. Put it in proportion: at N = 5000
on 8 ranks the gather is 1.81 ms against an O(N²) influence sum of roughly
300 ms — **under 1 %** of the step. Saving 1.4 ms of it changes nothing, and
costs you the property that makes `particle_global_influence` valuable: no MPI
in the application at all.

Where it would matter is the regime where the sum is no longer the bottleneck —
a cheaper interaction, a cutoff, or a mesh method whose per-step cost is already
down at ~10 ms. There the gather stops being a rounding error and the 4× is
worth having.

## Visualisation

```
OMP_NUM_THREADS=1 mpirun -np 4 ./influence_gather_dev_mpi \
    -npart 5000 -nsteps 400 -nprint 8
python3 plot_gather_h5.py          # -> frames/*.png and frames/gather.gif
python3 plot_gather_h5.py --no-gif
```

Off by default (`-nprint 0`) so timing runs pay nothing for I/O, and the frame
write is outside both gather timers.

Two panels:

- **left** — the particle cloud coloured by influence. Physics identical to
  `particle_global_influence`; the gather method cannot change it, and the run
  asserts bit-for-bit agreement every step.
- **right** — the reason this app exists: both gathers, per step, as the run
  proceeds, with a marker on the frame being shown. Log y-axis, because the two
  differ by a factor rather than an offset.

The right panel draws the raw per-step trace faint and a **rolling median**
bold. Single-step timings are jittery — one slow step is scheduling noise, not
a property of the method, and on individual steps the Allgatherv trace does
occasionally spike above the reduction. The median is what carries the result,
and the raw data stays visible rather than being smoothed away.

Each frame carries `t_reduce_ms` and `t_mpi_ms` for that step, so the timing
panel is measured data travelling with the run, not a separate benchmark.

Measured on the run above (np = 4, N = 5000, 400 steps): `ops_reduction`
median **1.2494 ms/step**, `MPI_Allgatherv` median **0.3034 ms/step** — 4.3x,
matching the whole-run averages (2.3159 vs 0.5962 ms).

## Options

```
-npart N     particle count                    default 2000
-nsteps N    timesteps                         default 200
-nprint N    write an HDF5 frame every N steps  default 0 (off)
-check       verify against a host reference (O(N^2) on the host)
```

## Files

| File | Contents |
|---|---|
| `influence_gather.cpp` | driver, both gathers, timing and cross-check |
| `particle_kernels.h` | unchanged from `particle_global_influence` |
| `grid_kernels.h` | unchanged |
| `gather_io.h` | per-step HDF5 frames, carrying the gather timing |
| `plot_gather_h5.py` | two-panel frames + animated GIF |
| `particle_kernels.h` | no publish kernel — the gather is host-packed |
