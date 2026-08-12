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

> **New to the method?** [`UNDERSTANDING_oSEM.md`](./UNDERSTANDING_oSEM.md)
> explains what SEM is doing from first principles — what the eddies are, why
> their positions are 3-D when the output is a 2-D plane, how they traverse the
> domain, and what each constant means — then sets out how this app's approach
> differs. Start there if the code below is not self-explanatory.

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

## Verifying the shear stress, and what it turned up

With the tabulated RST (`-rst tbl`) the Cholesky factor gains an off-diagonal
term `a21`, and nothing else in the app tests it: the rms values depend on `a21`
only through `a22 = sqrt(R22 - a21^2)`, so `a21^2 + a22^2` collapses to `R22`
whatever `a21` is. An error there would pass every other check silently. The
shear stress `<u'v'>` is the one observable that sees it.

Two things were needed to make that check mean anything.

**Time-averaging.** A single snapshot is hopeless: `<u'v'>` depends on the
cancellation `<S_x S_y> -> 0`, which at one instant scatters by far more than
the signal. The deviation from `R21` swung 64 / 52 / 154 % across snapshots of
identical code. The profile accumulator now sums over every step.

**Dividing out the normalisation.** Even time-averaged, the deviation flattened
at ~17 % and stayed there under ensemble averaging over 12 seeds, which makes it
systematic. It is not `a21`. Every stress carries a common factor `<S^2>`, the
variance of the raw eddy sum, measured at **1.16** where it should be 1:

```c
vol    = (x_max-x_min) * (y_max-y_min + 2*r_max) * (z_max-z_min + 2*r_max);
eddy_y_min = y_min;            /* no bottom padding */
eddy_y_max = y_max + r_max;    /* top padding only  */
eddies = vol / eddy_radius^3;
```

`vol` pads y by `2*r_max`, the eddy box pads it only on top — 0.01474 against
0.01187, so the eddy count is **1.24x** too high for the box the eddies occupy.
Predicted `<S^2> = 0.949 * 1.242 = 1.179` against 1.16 measured. This is
inherited: `apps/c/oSEM/OPS_oSEM.cpp:43-49` has the identical pair, and it is
left alone because the port holds the reference's invariants. The one-term
correction was tried and reverted — it removes the density excess (1718 eddies
-> 1384, `<S^2>` 1.16 -> ~0.93) but does not reach 1 on its own, and keeping
parity with oSEM matters more for now. `UNDERSTANDING_oSEM.md` Part 3 has the
measurements.

The correlation coefficient divides that factor out:

```
rho = <u'v'> / sqrt(<u'u'> <v'v'>)   ->   R21 / sqrt(R11 R22)
```

Over 12 realisations, **slope 1.0090** — `a21` is exact to 0.9 %, row-by-row
ratios 0.98–1.03. The free cross-terms `<u'w'>` and `<v'w'>` converge to zero as
they must (0.0667 -> 0.0048 from 200 to 12800 steps).

A by-product worth knowing: `<u'u'>/R11` drops to ~0.87 at the first interior
row and ~0.59 at the top row, because the plane spans the full eddy box, so
nodes at either y extreme are surrounded by eddies on one side only. Also
inherited — oSEM's grid does the same.

`-seed N` selects the realisation and `-dumpshear FILE` writes the per-row
profile, since these statistics are only interpretable against their own
realisation-to-realisation scatter (single-run `rho` scatters +/- 7 %).

## Files

| File | Contents |
|---|---|
| `osem_common.h` | gather layout and parameter block |
| `osem_constants.h` | every constant and run option; valued at the top of `main` |
| `ops_particle_random.h` | `ops_fill_random_uniform_particle()` — the RNG fill |
| `particle_kernels.h` | init / convect / publish / count — the eddies |
| `grid_kernels.h` | grid, RST, `compute_fluct` — the inlet plane |
| `influence_osem.cpp` | driver |
| `osem_io.h` | per-step HDF5 frames, written to `h5files/` |
| `UNDERSTANDING_oSEM.md` | the method explained, and how this app differs |

Four plot scripts, each answering a different question of the same frames:

| Script | Question | Needs |
|---|---|---|
| `plot_osem_h5.py` | is the **amplitude** right? rms and the wall-normal profile against the tabulated stresses | any frame |
| `plot_hdf5_files.py` | what does the plane **look like**? u'/v'/w' maps, without opening ParaView | any frame |
| `plot_structure.py` | is the **structure** right? in-plane `(w',v')` vectors, streamwise vorticity, divergence | one frame, but needs ≳8 cells per eddy radius — it warns below that |
| `plot_correlation.py` | are the structures the **size** they were asked to be? `R_uu(Δy,Δz)` and the integral length scale, against an analytic prediction | many frames, ideally ≥1 flow-through apart |

Two aggregate modes turn a single number into one with an error bar:

- `plot_structure.py … --stats` — the divergence ratio over every frame: mean,
  spread, and a time series. Measured **0.996 ± 0.016** on the isotropic run
  against an incompressible reference of 0.408, with −0.1% drift across 2000
  steps.
- `plot_correlation.py … --split` — the integral length scale on two halves of
  the frames, split **chronologically** (moves if the field is still drifting)
  and **interleaved** (moves only with sampling noise). The interleaved
  difference gives `SE ≈ |a−b|/2` for the full set. Both need enough
  decorrelated realisations per half; below ~8 the script says so and declines
  to give a verdict.
