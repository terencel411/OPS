# oSEM on OPS Particles (2-D)

The 2-D Synthetic Eddy Method inlet from [`../oSEM`](../oSEM), with each eddy
represented as an **OPS particle** instead of a row in an `ops_dat` on a second
block.

The point is to remove the per-step gather. In `../oSEM` the eddies live on an
`ops_block` of shape `{eddies, 1}`, are fetched to the host every step and
handed to `compute_fluct` as seven `ops_arg_gbl` arrays. Here they are particles
owned by the rank whose subdomain they occupy, found through the mapping's
cell-linked list, and delivered to the kernel as `ops_arg_dat_particle`. No
fetch, no gather, and every construct is a plain OPS call — which matters
because the real target is OpenSBLI-generated code.

Built up from [`../particle_tutorial_1_drift`](../particle_tutorial_1_drift)
(declaration order, seeding, migration) and
[`../particle_tutorial_3_deposit`](../particle_tutorial_3_deposit) (the
grid-outer scatter loop and its constraints).

---

## Status

| stage | what | serial | np=2 | np=4 | np=8 |
|---|---|---|---|---|---|
| A | eddies seeded as particles | PASS | PASS | PASS | — |
| B | convection + recycling | PASS | PASS | PASS | — |
| C | `compute_fluct` as a scatter loop | **PASS** | **PASS** | **PASS** | — |

**Stage C in serial matches the reference to 3.9e-16 relative** over 50 steps,
with `particle hits == reference hits` exactly (73766 = 73766) — the search
finds every true eddy-node pair, and the only difference is summation order.

**Stage C under MPI** is checked with `-nojump`, which disables the respawn so
nothing migrates and the field is fully deterministic; seeding is keyed on the
global eddy index, so the field is identical at any rank count and the whole-
field checksum is a valid cross-rank comparison:

| ranks | `sum|q|^2` | max |
|---|---|---|
| 1 | 2.171922495776301e+07 | 1.450290277361987e+02 |
| 2 | 2.171922495776301e+07 | 1.450290277361987e+02 |
| 4 | 2.171922495776296e+07 | 1.450290277361987e+02 |

np=1 and np=2 are bit-identical; np=4 differs in the 15th digit (2.3e-15
relative), which is reduction ordering. The max is bit-identical throughout.

Not yet done: np = 8, and host-side respawn.

## Build and run

```bash
make oSEM_particles_dev_mpi           # the working target
mpirun -np 1 ./oSEM_particles_dev_mpi -niter 50 -validate
mpirun -np 4 ./oSEM_particles_dev_mpi -niter 400
```

Options: `-niter N`, `-validate`, `-seed N`, `-nojump`, `-globaljump`,
`-halomul N`.

**`make oSEM_particles_dev_seq` does not work** — see the seq/MPI size-rule
mismatch below. Use `mpirun -np 1` for serial runs.

---

## The three constraints that shape this app

### 1. The mapping must be COARSE, for correctness

The eddy radius is `0.2*delta = 1.4e-3`, against `dy = 1.187e-4` and
`dz = 3.716e-4` — so an eddy reaches ~12 cells in y and ~4 in z. A bin-per-cell
mapping would therefore need a ghost band 12 cells deep, and **the particle
ghost band is only correct to depth 1** (see
`../particle_tutorial_3_deposit/README.md` §3a: exact in serial at any depth,
silently wrong under MPI beyond depth 1).

So the bins are made ~one eddy radius across (`stride = {12, 5}`) and the search
stencil is 3×3 bins. Reach is `stride` cells while the ghost band stays one bin
deep — the only configuration that is correct under MPI.

This was also the cause of an earlier serial bug: with a fine mapping and a 25×9
stencil, the scatter found only 26% of the true eddy-node pairs. The coarse
mapping fixed it outright.

### 2. Bin counts must be powers of two

Under MPI the stride must divide **every rank's local subdomain**, not just the
global grid: `ops_mpi_particle_host.cpp:1229` requires
`local_fine_size == stride * local_bin_count` exactly and otherwise throws
*"A non-uniform map grid is generated"*.

Hence `nbin_y = 16`, `nbin_z = 32`, and the coordinate dat is sized **from the
bins** (`nbin * stride + 1` nodes) rather than from the eddy box. It comes out
larger than the eddy range needs, which is harmless — it only has to *span* the
box, keep spacing `dy`/`dz` so bin `k` still groups the same cells as inlet node
`k`, and the surplus bins stay empty.

One consequence to watch: the bounding box now follows this enlarged dat, so the
local-respawn range is intersected with the true eddy box in `main()`. Without
that, eddies scatter into the empty margin.

### 3. The seq and MPI backends disagree on the size rule

`_ops_mapping_def_core` computes the size the divisibility check uses, and the
two backends differ by one:

| backend | `size[i]` | file |
|---|---|---|
| sequential | `grid->size + d_m - d_p` | `ops_particle_host_single_node.cpp:3053` |
| MPI | `grid->size + d_m - d_p - 1` | `ops_mpi_particle_core.cpp:441` |

For a zero-halo dat of N nodes that is `N` versus `N-1`. So the sequential build
wants `stride | N` and the MPI build wants `stride | N-1` — and consecutive
integers are coprime, so **for any stride > 1 no single dat size satisfies
both**. This app is sized for MPI; `dev_seq` therefore throws
*"Introduce map do not project properly to block grid"*.

---

## Known limits

- **np = 8** aborts with SIGFPE on a 4×2 decomposition. Not yet diagnosed; a
  further constraint on how the bins divide across ranks is likely.
- **Respawn is subdomain-local**, not global. Particle migration only delivers
  to an *immediate neighbour* rank, so a textbook SEM respawn — which teleports
  an eddy to a fresh random (y,z) anywhere in the box — deadlocks at np ≥ 4
  (reproduce with `-globaljump`). Local respawn avoids migration entirely, at
  the cost of freezing the eddy count per subdomain, which makes results
  rank-count dependent. **This is a deliberate placeholder**, to be replaced by
  host-side respawn of just the ~1-in-350 eddies that recycle each step.
- **`-validate` is serial-only.** It builds the reference from
  `particle->no_particles` on the calling rank, which is the whole eddy list
  only at np = 1. An MPI check needs either a gathered reference or a
  field-to-field comparison across rank counts.
- The kernels still carry **diagnostic counters** (`visits`, `nhit`) in their
  signatures. They are what localised the 26% bug and are kept until Stage C is
  validated under MPI.

## Files

| File | Contents |
|---|---|
| `sem.cpp` | The application: seeding, migration cycle, stages A-C |
| `sem_constants.h` | Geometry and flow parameters, identical to `../oSEM` |
| `grid_kernels.h` | Inlet grid, particle coords, RST from TBL data, zero, scatter, reference, compare |
| `particle_kernels.h` | `KerConvectEddy` — convection and recycling |
| `scatter_loop.h` | `ops_par_scatter_loop`, the translator-safe rename |
| `TBL_data.h` | Tabulated boundary-layer profile, copied from `../oSEM` |
