# oSEM with the eddies as OPS particles

A port of [`../oSEM`](../oSEM) (2-D inlet plane) in which the synthetic eddies
are **OPS particles** rather than grid dats on a second block, structured like
[`../particle_global_influence`](../particle_global_influence).

```
make influence_osem_dev_seq
./influence_osem_dev_seq -niter 400

make influence_osem_dev_mpi
OMP_NUM_THREADS=1 mpirun -np 2 ./influence_osem_dev_mpi -niter 150
```

## The mapping

| oSEM | here | role |
|---|---|---|
| `instantiate_grid` / `instantiate_RST` / `instantiate_eddies` | `KerInitGrid` / `KerInitRST` / `KerInitEddy` | before the loop |
| `convect_eddies` | `KerConvectEddies` | the **advance** kernel — purely per-eddy |
| `ops_dat_fetch_data` × 7 | the `ops_reduction` allgather | **the fix, see below** |
| `compute_fluct` | `KerComputeFluct` | the **influence** kernel — every node sums over every eddy |

The eddy **position is (y, z)** — the inlet plane, which is also what OPS
decomposes. The streamwise coordinate `x` rides as an ordinary particle dat,
because it is not a spatial dimension of this problem: nothing is decomposed
along it and no neighbour search uses it.

One block, not two. The eddies live inside the inlet block, whose grid already
spans the full eddy box — oSEM's `instantiate_grid` runs from `z_min - r_max` to
`z_max + r_max` for exactly that reason, and it matters here because the
bounding box OPS derives from the coordinate dat is what decides whether an eddy
is inside the domain.

## Why this is a fix, not just a restructuring

oSEM hands the eddy state to `compute_fluct` by calling `ops_dat_fetch_data` on
seven eddy dats and passing the host arrays as `ops_arg_gbl`. **Under MPI that
is wrong, not merely partial.** `ops_dat_fetch_data`
(`ops_mpi_rt_support.cpp:2125`) copies only *this rank's* slice of the
decomposed eddy block, and writes it starting at offset 0 — it computes a
displacement `ldisp` and then never uses it in the `memcpy`. `compute_fluct`
then loops over the **global** eddy count, so every index past the local slice
reads uninitialised heap on the first step and stale values afterwards. Each
rank builds the inlet from a different, partly garbage eddy set. It is correct
only at np = 1.

The reduction allgather returns the **complete** eddy list, in id order, on
every rank — the same array shape `compute_fluct` already wanted, so the kernel
body ports across unchanged.

## Results

Serial, 1718 eddies, 101 × 151 inlet, target `u0·TI = 8.236`:

| niter | rms u' | rms v' | rms w' |
|---|---|---|---|
| 100 | 8.6629 | 9.1126 | 8.6609 |
| 250 | 8.6529 | 8.4060 | 8.9659 |
| 400 | 8.4340 | 8.5848 | 8.7259 |

(measured on the teleporting version; the continuous one gives 8.6893 / 8.8646
/ 8.8594 at 150 steps, identically at np = 1, 2, 4 and 8)

All three cluster on the target with no component systematically largest, which
is what `a11 = a22 = a33` requires. Eddy population conserved for the whole run.

Cost per step (serial): convect + migrate 0.098 ms, gather 0.043 ms,
`compute_fluct` **108 ms** — the all-to-all dominates completely, as in the
influence app.

## A bug worth knowing about, because it looked like physics

The first working version gave rms u' = 8.85–9.79 across realisations while v'
and w' sat at 4.8–6.7. Those three **must** be statistically equal. The signs
were individually unbiased — mean eps was `+0.007, -0.005, 0.000` over 1718
eddies — so a test of the signs alone would have passed.

The cause: `KerInitEddy` draws the eddy's `x` from LCG state s₁ and `eps_x` from
s₂ = next(s₁). A bare LCG state is a deterministic function of the previous one,
so **eps_x was a deterministic function of x** — and since `compute_fluct`
selects eddies by x (the `|x| < r` test), it selected a *biased* set of eps_x.
Position–sign correlation, invisible to any check on the sign distribution.

Fixed by putting a murmur3-style finalizer on the output (`lcg_mix` in
`osem_common.h`) so consecutive states produce decorrelated draws. oSEM does not
have this problem because it draws each quantity from a separate
`ops_fill_random_uniform` call rather than from one running stream.

## MPI: continuous re-injection

oSEM recycles an exiting eddy by assigning it a **new random (y, z)**. As a
particle operation that is a jump to an arbitrary point in the plane, and OPS
particle migration only hands a particle to a **neighbouring** rank. Measured on
the faithful port, 150 steps:

| ranks | with random re-injection | with continuous drift |
|---|---|---|
| 1 | works | works |
| 2 | works | works |
| 4 | **fails** | works |
| 8 | — | works |

A controlled experiment pinned the cause: removing *only* the two position
assignments — same eddies, same dats, same gather, same convection, same random
stream consumption — made np = 3 and np = 4 run clean with the population
conserved. So it was never "eddies vs particles". It was **discontinuous
re-injection vs continuous motion**. `particle_global_influence` moves its
particles by `x += v*dt` and never jumps, which is why it works at every rank
count; this app jumped, and did not.

### The fix

Only one of the three coordinates was ever a problem. **x is not a spatial
dimension of this block** — it is an ordinary particle dat, nothing is
decomposed along it, so resetting `x = x_min` involves no migration at all. It
was purely the (y,z) assignment that jumped.

So the eddy gets a transverse **velocity** instead, drawn once at
initialisation, and reflects off the eddy-box faces. Per-step displacement is
0.29 cells in y and 0.43 in z at the default resolution — small and local, so
migration only ever sees a move into an adjacent subdomain.

The statistical refreshment survives, because the part that matters is not
spatial: the **signs are still re-drawn on every recycle**, and they are what
randomises u', v', w'. Positions decorrelate by drift instead of by teleport, on
a timescale set so an eddy crosses the box in about one flow-through time.

### Result

At np = 1, 2, 4, 8 the run is not merely stable but **rank-invariant** — rms
u' = 8.6893, v' = 8.8646, w' = 8.8594 identically at every rank count, with the
eddy population conserved throughout. That falls out of three things together:
each eddy carries its own LCG stream so randoms do not depend on which rank
owns it, the gather is bit-exact, and the motion is deterministic per eddy.

This is a deliberate change to the model, not a bug fix — the eddy positions now
evolve continuously rather than being resampled. If exact fidelity to oSEM's
resampling matters more than running past two ranks, the teleporting version is
in the git history.

## Files

| File | Contents |
|---|---|
| `osem_common.h` | gather layout, parameter block, the mixed LCG |
| `particle_kernels.h` | init / convect / publish / count — the eddies |
| `grid_kernels.h` | grid, RST, `compute_fluct` — the inlet plane |
| `influence_osem.cpp` | driver |
