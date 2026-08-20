# oSEM_2d_particles

[`../oSEM_2d`](../oSEM_2d) (2-D inlet plane) with the synthetic eddies as OPS
particles rather than grid dats on a second block.

Added flags in the code to switch between the control vars

| flag | default | |
|---|---|---|
| `-niter N` | 2000 | timesteps |
| `-ny N` `-nz N` | 100, 150 | inlet plane resolution |
| `-nprint N` | 100 | print iter info and HDF5 frame interval |
| `-rst tbl\|iso` | `tbl` | tabulated boundary-layer profile, or isotropic |
| `-rng mt19937\|minstd\|shared` | `minstd` | specify how the random fill gets executed |
| `-seed N` | 2893328493 | seed |

## Kernel Replacements

| oSEM | here | role |
|---|---|---|
| `instantiate_grid` / `instantiate_RST` / `instantiate_eddies` | `KerInitGrid` / `KerInitRST` / `KerInitEddy` | executes before the time loop |
| `convect_eddies` | `KerConvectEddies` | executes every iteration |
| Allgather logic missing | `KerGatherEddies` | connects `KerConvectEddies` and `KerComputeFluct` kernels |
| `compute_fluct` | `KerComputeFluct` | executes every iteration |

The eddy position is (y, z), the inlet plane, which is also what OPS decomposes.
The streamwise coordinate `x` rides as an ordinary particle dat, because it is
not a spatial dimension of this problem: nothing is decomposed along it and no
neighbour search uses it.

Only one block is used now. The eddies live inside the inlet block, whose grid already
spans the full eddy box - `z_min - r_max` to `z_max + r_max` for exactly that reason, 
and it matters here because the bounding box OPS derives from the coordinate dat is 
what decides whether an eddy is inside the domain.

## Issue with the eddy dats

oSEM hands the eddy state to `compute_fluct` by calling `ops_dat_fetch_data` on
seven eddy dats and passing the host arrays as `ops_arg_gbl`. That does not work 
because `ops_dat_fetch_data` copies only this rank's slice of the decomposed eddy block 
and writes it starting at offset 0. `compute_fluct` loops over the global eddy count, 
so every index past the local slice reads uninitialised heap on the first step and stale 
values afterwards. Each rank builds the inlet from a different, partly garbage eddy set.
It will only work when a single core is used.

## A bug worth knowing about, because it looked like physics

An early version gave rms u' = 8.85-9.79 across realisations while v' and w' sat
at 4.8-6.7. Those three must be statistically equal. The signs were individually
unbiased - mean eps was `+0.007, -0.005, 0.000` over 1718 eddies - so a test of
the signs alone would have passed.

The cause: `KerInitEddy` drew the eddy's `x` from LCG state s1 and `eps_x` from
s2 = next(s1). A bare LCG state is a deterministic function of the previous one,
so eps_x was a deterministic function of x, and since `compute_fluct` selects
eddies by x (the `|x| < r` test) it selected a biased set of eps_x. A
position-sign correlation, invisible to any check on the sign distribution.

The fix is the structure now in `ops_particle_random.h`: each eddy gets its own
engine seeded from `(seed, gid, counter)` through a `std::seed_seq`, and the six
components of a fill are successive draws from that fresh engine rather than
successive states of one long stream. oSEM does not have this problem because it
draws each quantity from a separate `ops_fill_random_uniform` call.

## Random fill generator options

`-rng` selects how `eddy_particle_rng` is filled. Measured on this app:

| method | fill cost | rank-invariant | notes |
|---|---|---|---|
| `mt19937` | 10.99 ms | yes | 624-word `seed_seq` per eddy per step |
| `minstd` | 0.08 ms | yes | the default |
| `shared` | 0.13 ms | no | one engine per rank, walked in storage order; mirrors OPS exactly |

The obvious objection to `minstd` - that a bare LCG's consecutive outputs lie on
a lattice, so an eddy's position could couple to its signs, which is exactly the
bug described above - was tested rather than assumed. Drawing the 6-tuple for
10^6 `(gid, counter)` pairs:

Indistinguishable, and minstd's are no larger. The lattice problem needs a
continuing stream; here every eddy gets a fresh `seed_seq`-scrambled state and
six draws, far too short for the structure to appear. That tests the one coupling
known to have caused a real bug, not general RNG quality - if a future kernel
draws long sequences from a single engine, `mt19937` becomes the right choice
again.

## Files

| File | Contents |
|---|---|
| `osem_2d_particles.cpp` | driver |
| `osem_common.h` | gather buffer layout |
| `osem_constants.h` | every constant and run option; valued at the top of `main` |
| `ops_particle_random.h` | `ops_fill_random_uniform_particle()` - the gid-keyed RNG fill |
| `particle_kernels.h` | init / convect / publish / count - the eddies |
| `grid_kernels.h` | grid, RST, `compute_fluct` - the inlet plane |
| `osem_io.h` | per-step HDF5 frames, written to `h5files/` |