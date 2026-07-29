# oSEM_3D → OPS Particles: port design notes

**Status: scaffolding only.** This directory currently holds unmodified copies of
`../oSEM_3D` (the base version) plus `scatter_loop.h`. No particle code has been
written yet. These notes record what the port requires, established by reading
the app and by the 2-D port in `../oSEM_particles`.

Read `../oSEM_particles/README.md` first — the three mapping constraints
documented there apply here unchanged.

---

## Geometry (production dims 750 × 250 × 150)

| | value |
|---|---|
| `delta` | 11.6973525411 |
| `radius` | `0.2*delta` = 2.3395 |
| grid spacing | `Delta0 = 0.5007`, `Delta1 = 0.4016`, `Delta2 = 0.2667` |
| **eddies** | **267** — well under `OPS_MAX_PART` (1000), so no `ops_particle_realloc_data` |
| **reach needed** | **6 cells in y, 9 cells in z** |
| eddy box | x [−2.34, 2.34], y [−2.34, 14.04], z [−2.34, 42.34] |
| grid extent | x [0, 375], y [0, 100], z [0, 40] |

Suggested strides: 8 in y (≥6) and 10 in z (≥9), with power-of-two bin counts.
Derive these from the parameters in code so they recompute if the grid changes —
do not hard-code.

---

## Four things that differ from the 2-D port

### 1. Kernel030 must be SPLIT into two loops

This is the biggest structural change. `opensbliblock00Kernel030` currently
does two jobs in one grid loop: it accumulates `up`, `vp`, `wp` over all eddies,
**and then** uses them to set `rho`, `rhou0..2`, `rhoE`.

In the scatter form the kernel body runs once per *(node, nearby eddy)* pair, so
it can only accumulate — there is no "last eddy" at which to apply the boundary
condition. The loop therefore becomes three:

1. `ops_par_loop` zeroing three new grid dats `up_B0`, `vp_B0`, `wp_B0`
2. `ops_par_scatter_loop` accumulating the eddy contributions into them
3. `ops_par_loop` over the same inlet range applying `up/vp/wp` to
   `rho`, `rhou0..2`, `rhoE` — the second half of the present kernel body,
   unchanged

**This matters for the OpenSBLI generator**: an inlet BC that sums over eddies
cannot be emitted as a single kernel once eddies are particles.

### 2. Spanwise periodicity is already handled IN the kernel — leave it there

`opensbliblock00_kernels.h:322-337` contains a "ghost boundary mirror": an eddy
that fails the radius test is retried against a mirrored z at ±40. So z
periodicity is a property of the *shape function*, not of particle positions.

Consequence: **do not wrap eddy positions in z.** Wrapping would be a
long-distance migration and would hit the neighbour-only limit (defect 2 in
`documentation/ops-particles-defects.md`). Keep the mirror logic verbatim in
the scatter kernel; it works unchanged because it only reads the eddy's own
coordinates.

### 3. `a11`, `a21`, `a22`, `a33` are NOT eddy data

They are y-profile arrays of length `y_cutoff`, indexed by `idx[1]` — the wall-
normal grid index. They stay as `ops_arg_gbl`, exactly like the TBL table in the
2-D port. Only the seven `eddy_*_gbl` arrays become `ops_arg_dat_particle`.

### 4. x is not a search dimension

The particle position dat must be dim 3 (block dims), but eddies span
x ∈ [−2.34, 2.34] while the grid starts at x = 0 — half the eddy x range has no
grid under it. So the position is `(x_fixed, y, z)` with `x_fixed` a constant
plane inside the domain, and the real eddy x rides along as an ordinary particle
dat, read by the shape function as `xtildesq`.

The inlet range is `{-2, 1, ...}` in dim 0, i.e. indices −2, −1, 0. C integer
division truncates toward zero, so with any dim-0 stride ≥ 3 all three map to
bin 0, which is where `x_fixed` bins — a ±1 stencil then reaches them.

---

## Open design question: the box must extend below the grid

Eddies are centred down to −2.34 in **both** y and z, but the OPS bounding box is
exactly `[first node .. last node]` of a coordinate dat
(`_ops_construct_local_box_from_dat`, `ops_particle_box_host_funcs.h:44` —
the `dx` argument is accepted and ignored). Particles outside the box are
deleted.

The agreed approach is to give the mapping its own coordinate dat whose **origin
sits below the grid's**, so the box covers the negative margin. The unverified
part is bin alignment: the grid-outer loop maps grid index `j` to bin
`j / stride`, so an offset origin means bin `k` no longer corresponds to grid
node `k * stride`. Whether `binhead->d_m` absorbs that offset needs testing —
`get_mapping_address` (`ops_grid_part_seq_v2.h:93`) does subtract `d_m`, which
is promising but unproven.

**Test this in `particle_tutorial_3_deposit` before building it here**, with a
coordinate dat offset from the grid: it is a 250-line app with an exact integer
check, and every previous constraint was found faster there than in an
application.

---

## Memory

At 750 × 250 × 150 a single block-sized dim-1 dat is 0.23 GB and the app
declares dozens; a dim-3 coordinate dat adds 0.68 GB. This machine has ~8 GB
available, so **the production size does not fit here**.

Develop and verify at a reduced grid (`block0np0/1/2` are plain variables — try
150 × 100 × 60), with all stride and bin arithmetic derived from the parameters
so it recomputes. Validate at production size on a larger machine.

---

## Suggested order

1. Test the offset-origin coordinate dat in `particle_tutorial_3_deposit`
2. Reduce the grid here; add particle declarations, seeding, count check (Stage A)
3. Convection + migration cycle (Stage B)
4. Split Kernel030 into zero / scatter / apply (Stage C), validated against the
   retained `ops_arg_gbl` path in a `-validate` mode, as in the 2-D app
5. Re-derive strides for production dims and hand over for large-scale testing
