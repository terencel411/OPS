# Tutorial 3 — Scattering particle data onto the grid

Tutorials 1 and 2 only ever moved data **grid → particle**. This one goes the
other way: particles accumulate onto a grid dat.

That is the direction a synthetic-eddy method, a particle-in-cell deposition, or
any particle source term needs — and it uses a **different loop form** from
either earlier tutorial.

**Prerequisite:** [`../particle_tutorial_1_drift`](../particle_tutorial_1_drift).
Declaration order, seeding and the migration cycle are unchanged and are not
re-explained.
**Reference:** `/home/terence411/proj5/installations/documentation/ops-particles.md`

---

## Build and run

```bash
make tutorial3_dev_seq          # serial, NO translator  <- start here
./tutorial3_dev_seq

make tutorial3_dev_mpi
mpirun -np 4 ./tutorial3_dev_mpi

make tutorial3_mpi              # via the OPS translator (see §4)
mpirun -np 4 ./tutorial3_mpi
```

Options: `-mode all|radius`, `-halo N`, `-nsteps N`, `-npart N`.

---

## 1. The third loop form

There are three ways to pair particles with a grid, and **only one of them can
write to the grid**:

| Form | Outer loop | Grid writable? |
|---|---|---|
| `ops_particle_par_loop` | particles | no grid args at all |
| `ops_par_particle_grid_loop` (tutorial 2) | particles | **no** — throws `OPS_RUNTIME_CONFIGURATION_ERROR` on any non-`OPS_READ` grid arg |
| `ops_par_loop(particle, map, stencil, dim, range, ...)` | **grid** | **yes** |

The third is declared in `ops_grid_part_seq_v2.h:562`. Its structure is:

```
for each grid node (i,j):
    for each offset s in map_stencil:
        for each particle in bin(node -> map + s):
            kernel(grid args at (i,j), particle args at that particle)
```

Your kernel is the innermost body. It runs **once per (node, nearby particle)
pair**, so it must accumulate — which is why `KerZeroDensity` is a separate loop
immediately before. Fold the zeroing in and each node keeps only the last
particle's contribution.

Grid writes are permitted because this loop sets halo dirtybits for non-`OPS_READ`
args (`ops_grid_part_seq_v2.h:527`) and has no read-only check.

---

## 2. The check

`-mode all` (the default) deposits each particle's weight at **every stencil
point it is reachable from**, with no geometric test. A particle in bin `b` is
visited once per offset `s`, by grid node `b - s`, so

```
sum(rho) == N_particles * stencil_points
```

**exactly** — an integer identity independent of grid spacing, particle
positions and rank count. One number that covers bin traversal, ghost particles
and the halo.

It holds only while particles stay at least `HALO` cells from the grid boundary,
otherwise some source nodes `b - s` fall outside the iteration range. That is
why `SEED_BOX` is a tight central box: it keeps the identity valid even at
half-width 12.

`-mode identity` is the same unconditional deposit, but each particle carries a
**distinct** weight (its global lattice index + 1) instead of 1.0. The expected
total becomes

```
sum(rho) == stencil_points * sum(1..N) == stencil_points * N*(N+1)/2
```

This is the important one. With every particle carrying the same weight, a loop
that visits the right *number* of particles but reads the *wrong* particle's
data still passes — `-mode all` is blind to it. Distinct weights close that gap:
the total can only come out right if every visit reads the correct particle.

`-mode radius` deposits only within `HALO*dx`. No closed form, so it is checked
by requiring the total to match between serial and MPI (it does: 1240 at
np = 1, 3 and 4).

---

## 3. What this app established

Measured with `-nsteps 50`, error = `|sum(rho) - N*stencil_points|`:

| ranks | grid | halo 1 | halo 2 | halo 3 | halo 4 | halo 6 | halo 8 | halo 12 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1×1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | **2×1** | 0 | **50** | **140** | **360** | **260** | **340** | **500** |
| 3 | 3×1 | 0 | — | — | 0 | — | — | — |
| 4 | 2×2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 5×1 | — | — | — | 0 | — | — | — |
| 6 | 3×2 | — | — | — | 0 | — | — | — |
| 8 | 4×2 | — | — | — | 0 | — | — | — |

**The loop is correct.** It is exact in serial at every stencil width, and exact
on 3, 4, 5, 6 and 8 ranks.

**There is one genuine defect: the 2×1 decomposition, for any stencil half-width
greater than 1.** Note that 2×2 passes with the *same* two-way split along
dimension 0, so it is not simply "two ranks in a direction" — a 2×1 grid is the
only case with two ranks in one dimension and none in the other. The errors are
small integers, i.e. a handful of particle-node visits miscounted near the rank
boundary; they grow with stencil width.

**Practical consequence: avoid running on exactly 2 ranks** when using a search
stencil wider than one cell. Everything else measured is exact.

Two other things that were *not* problems, tested here because they looked
suspicious:

- **Exceeding `OPS_MAX_PART`** (1000, `ops_particles_lib_core.h:84`). Running
  1024 and 1600 particles through `ops_particle_realloc_data` is exact at np=1
  and np=4.
- **Stencil half-widths up to 12**, i.e. a 25×25 = 625-point search. Fine
  everywhere except the 2×1 case above, despite MPI halo buffers being sized for
  depth 5 by default (`ops_mpi_partition.cpp:1454`).

### 3a. Ghost particles carry the wrong payload under MPI

Run `-mode identity -halo 2 -nsteps 20`:

| ranks | sum(rho) | expected | |
|---|---|---|---|
| 1 | 126250 | 126250 | **exact** |
| 2 | 124475 | 126250 | short 1.4% |
| 4 | 124688 | 126250 | short 1.2% |
| 8 | 124688 | 126250 | short 1.2% |

**Serial is exact.** That is a strong positive result: the grid-outer loop's
per-particle addressing — `construct()` basing the `ACCP<T>` at
`data + first_point*dim` and `shift_point_arg` advancing by
`(point - ifirst) * dim` (`ops_grid_part_seq_v2.h:433, :331`) — delivers the
correct particle's data. Any application seeing wrong values in serial should
look at its own mapping and stencil setup, not here.

**MPI is not.** The same run is short by ~1.2% from 2 ranks upward. The
identical run under `-mode all` gives error exactly 0 at every rank count, and
the *only* difference is that particles now carry distinct weights. So the right
number of particles is visited, but some of them supply the wrong payload —
i.e. ghost particles, even though `p_wgt` is listed in **both** `dat_border` and
`dat_forward`.

Isolating it against the mapping halo depth gives a sharp answer:

| halo | np=1 | np=2 | np=4 |
|---|---|---|---|
| 1 | 0 | **0** | **0** |
| 2 | 0 | 1.78e3 | 1.56e3 |
| 3 | 0 | 4.27e3 | 2.03e3 |
| 4 | 0 | 1.71e3 | 1.82e3 |

**The particle ghost band is only correct to a depth of one cell.** Serial is
exact at every depth. MPI is exact at depth 1 and wrong at every depth beyond
it, at every rank count. The error is not monotone and changes sign — halo 4 at
np=4 *over*counts — so the ghost set is variously incomplete and duplicated
rather than simply truncated.

**This supersedes the "only 2×1 fails" reading of the table in §3.** That
conclusion came from `-mode all`, whose uniform weights could only expose
miscounts, never wrong payloads. With distinct weights every rank count fails
beyond depth 1. The §3 table is still correct about *counts*; it was just
measuring the less important thing.

### Consequence for real applications

A coupling kernel whose interaction radius spans more than one cell cannot use a
fine mapping under MPI. For the 2-D SEM app that is `r/dy ≈ 12` and `r/dz ≈ 4`,
so a bin-per-cell mapping needs a halo of 12 — well into the broken regime, and
silently so.

The fix is to make the **bins** about one interaction radius across instead of
one cell, via the coarse mapping (`ops_decl_mapping` with a `stride[]`,
`ops_particles_lib_core.h:1282`, plus a prolong stencil — `get_point_in_map`
case 1 divides the grid index by `mgrid_stride`). Then a ±1 stencil covers the
radius and the ghost band stays at the one depth that works. That was previously
noted as a performance optimisation; it is really a **correctness requirement**
for MPI.

`OPS/ops/` is treated as read-only here, so this is reported rather than
patched. Reproducer:
`mpirun -np 4 ./tutorial3_dev_mpi -mode identity -halo 2 -nsteps 20`.

### 3b. The coarse mapping fixes it

`-stride N` makes each bin N cells across, via the strided
`ops_decl_mapping` overload (`ops_particles_lib_core.h:1282`) plus an
`ops_decl_prolong_stencil` carrying the same ratio. Reach is then
`halo * stride` cells while the ghost band stays one *bin* deep.

Identity mode, `-halo 1` throughout:

| stride | reach | np=1 | np=2 | np=4 |
|---|---|---|---|---|
| 1 | 1 cell | 0 | 0 | 0 |
| 2 | 2 cells | 0 | **0** | **0** |
| 4 | 4 cells | 0 | **0** | **0** |

Reaches of 2 and 4 cells are *wrong* under MPI with a bin-per-cell mapping
(§3a) and **exact** with a coarse one. This is the recommended configuration for
any multi-cell interaction radius.

**The stride is not free — it must divide the grid.** `ops_decl_mapping` checks
`size % stride == 0` and throws *"Introduce map do not project properly to block
grid"* otherwise (`ops_particle_lib_core.cpp:1844`). The `size` it tests is
`NX - 1`, i.e. the number of cells, not nodes.

On this 41-node grid (40 cells):

| stride | 40 % stride | bins | result |
|---|---|---|---|
| 2 | 0 | 20 | pass |
| 3 | 1 | — | **rejected: divisibility** |
| 4 | 0 | 10 | pass |
| 5 | 0 | 8 | pass |
| 6 | 4 | — | **rejected: divisibility** |
| 8 | 0 | 5 | **aborts — cause not established** |
| 10 | 0 | 4 | pass |

So when choosing a stride, pick from the divisors of the cell count, at or above
`ceil(radius / spacing)`. If the grid size has no convenient divisor — 100 and
150 cells give 12 and 4 no trouble, but a prime does — size the mapping's
coordinate dat separately, rounding its node count up to a multiple of the
stride. The dat used for the mapping need not be the dat the loop iterates.

Stride 8 passes the divisibility check and still aborts, while stride 10 with
*fewer* bins passes — so this is not simply "too few bins", and the cause is not
yet known. Prefer a stride that has been verified end to end.

---

## 4. Making the translator accept it

The grid-outer loop is an **overload of `ops_par_loop`**, and the translator
scans the source textually for `ops_par_loop(` and tries to code-generate every
match. It reaches this one, finds an `ops_stencil` where it expects `int dim`,
and dies:

```
legacy translator : unknown access type for argument 11..14
modern translator : Parse error ... Expected int expression
```

`scatter_loop.h` fixes this with a one-line rename:

```c
template <typename... ParamType, typename... OPSARG>
inline void ops_par_scatter_loop(void (*kernel)(ParamType...), char const *name,
                                 ops_particle particle, ops_particle_mapping map,
                                 ops_stencil map_stencil, int dim, int *range,
                                 OPSARG... arguments) {
  ops_par_loop(kernel, name, particle, map, map_stencil, dim, range, arguments...);
}
```

The call site now says `ops_par_scatter_loop(...)`, which the scan does not
match, so the ordinary grid loops in the same file still code-generate normally
while this one is left to the C++ templates — which is all it ever needed, since
particle loops are never code-generated anyway.

With that in place all three targets build and agree exactly:
`tutorial3_dev_seq`, `tutorial3_dev_mpi`, and the translator-built
`tutorial3_mpi`.

---

## Files

| File | Contents |
|---|---|
| `deposit.cpp` | The application |
| `grid_kernels.h` | `KerInitGrid`, `KerZeroDensity`, `KerDepositAll`, `KerDepositRadius`, `KerSumDensity` |
| `particle_kernels.h` | `KerUpdatePosition` |
| `scatter_loop.h` | The translator-friendly rename |
| `Makefile` | Targets: `dev_seq`, `dev_mpi`, `seq`, `mpi`, `openmp`, `mpi_openmp` |

---

## Things to try

| Experiment | What you should see |
|---|---|
| Delete the `KerZeroDensity` loop | `sum(rho)` grows without bound — each step adds another full deposit. |
| Run `-halo 4` on exactly 2 ranks | Non-zero error. The defect documented in §3. |
| Widen `SEED_BOX` towards the domain edge with a large `-halo` | Error appears even at np=1: source nodes fall outside the iteration range, so the identity no longer applies. Not a bug. |
| Make a grid arg `OPS_READ` and try to write it | Silent no-op — unlike `ops_par_particle_grid_loop`, this form does not check. |
| Rename `ops_par_scatter_loop` back to `ops_par_loop` and `make tutorial3_mpi` | The translator parse error from §4. |
