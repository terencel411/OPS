# Particle-mesh vs. direct all-to-all — measured

Companion to [`../particle_global_influence`](../particle_global_influence),
which evaluates a per-particle influence as an exact sum over every other
particle using an array-valued `ops_reduction` as an allgather. That is O(N²)
with a global collective every step. This app implements the particle-mesh (PM)
alternative and runs **both methods in one executable**, so the comparison is
apples to apples rather than across two apps with different physics.

```
make influence_pm_dev_seq
./influence_pm_dev_seq -npart 2000 -check

make influence_pm_dev_mpi
OMP_NUM_THREADS=1 mpirun -np 4 ./influence_pm_dev_mpi -npart 10000
```

---

## The headline

Serial, 129² mesh, `-bc zero`, error is RMS(Δφ)/RMS(φ) at the final step:

| N | direct (ms/step) | mesh (ms/step) | winner | mesh error |
|---:|---:|---:|:---|---:|
| 400 | 1.0 | 6.9 | direct, 7× | 6.6 % |
| 1 000 | 6.4 | 7.6 | **crossover** | 7.9 % |
| 2 000 | 25.5 | 8.0 | mesh, 3.2× | 6.7 % |
| 5 000 | 210 | 11.3 | mesh, 19× | 5.1 % |
| 10 000 | 1 304 | 19.0 | mesh, 69× | 6.1 % |
| 20 000 | 3 441 | 16.1 | mesh, 213× | 7.7 % |
| 50 000 | 17 588 | 16.7 | mesh, ~1050× | — |

The two columns scale completely differently, and that is the whole answer:

- **direct is O(N²)** — 50× more particles cost 3 450× more time
- **mesh is nearly flat** — dominated by the fixed grid work; the
  particle-dependent part (deposit + interpolate) is linear and small
- **the mesh error does not depend on N at all.** It is set by the mesh, so it
  stays ~5–8 % whether there are 400 particles or 20 000

So: **below ~1 000 particles, keep the direct method** — it's exact, simpler,
and faster. Above that the crossover is decisive and grows without limit.

Timings are from a 12-core WSL2 box and have ~10 % run-to-run variance (one
outlier is visible above: mesh at N = 10 000 came out slower than at 20 000).
The order-of-magnitude trend is not in doubt; individual entries are not
precision measurements.

---

## What had to change, and why

**The interaction law.** A mesh can only reproduce a pairwise sum when the
interaction kernel is the Green's function of a local differential operator. In
2-D that is `(1/2π) ln r`, **not** `1/r`. The `1/r` law in
`particle_global_influence` has no local PDE behind it in two dimensions, so it
has no PM equivalent at all. This app therefore uses the softened 2-D Coulomb
potential

```
phi_i = sum_{j != i}  m_j (1/4pi) ln(r_ij^2 + eps^2)
```

for **both** methods, so the only thing that differs is how it is evaluated.
That is the price of admission to PM, and it is worth being blunt about: PM is
not a drop-in accelerator for an arbitrary pairwise law.

**Neutral charges.** Particles are created in +w/−w pairs, so `sum(m) == 0`
exactly. A neutral distribution has a potential that decays, which is what makes
`phi = 0` on a finite boundary usable at all.

---

## The pipeline, and which OPS loop form each stage needs

| Stage | Loop form | Header |
|---|---|---|
| deposit particles → mesh (CIC) | `ops_par_loop` particle overload — outer loop over **grid points** | `ops_grid_part_seq_v2.h:562` |
| boundary values | ordinary `ops_par_loop` on four edge ranges | — |
| solve ∇²φ = ρ | ordinary `ops_par_loop`, red-black SOR | — |
| interpolate mesh → particles | `ops_par_particle_grid_loop` | `ops_particle_grid_seq.h` |

Three details that are not obvious:

**The deposit loop is the only form that may write a grid dat.**
`ops_par_particle_grid_loop` throws on any non-`OPS_READ` grid arg. It is called
here through the `ops_par_scatter_loop` wrapper (borrowed from
`particle_tutorial_3_deposit`) purely so the OPS translator's textual scan for
`ops_par_loop(` does not try to code-generate it and die on the `ops_stencil`
where it expects an `int`.

**A ±1 search stencil is mandatory, not a tuning choice.** The particle ghost
band is only correct to a depth of one cell under MPI — measured in
`particle_tutorial_3_deposit` (README §3a): exact in serial at every depth,
wrong at every rank count beyond depth 1. CIC needs exactly one cell, so this
pipeline sits in the regime that works. A wider interaction radius would need
the coarse (strided) mapping instead.

**The mesh spacing cancels.** Deposit gives the charge `q` at each node, i.e. a
density `q/h²`, and the 5-point Laplacian carries `1/h²` — so the SOR update is
`phi = (phiE + phiW + phiN + phiS - q)/4` with no spacing in it anywhere.

---

## The three approximations, and what the app does about each

### 1. Mesh resolution

Handled by setting `eps = h`, so the pairwise sum is smoothed at the same scale
the mesh can represent. Refine and the difference falls (N = 400, 100 steps):

| grid | error |
|---|---|
| 65² | 10.8 % |
| 129² | 5.5 % |
| 257² | 2.7 % |
| 513² | 1.3 % |

Roughly first order in `h`, which is what CIC gives.

### 2. The boundary condition — the real engineering trade

| `-bc` | what it does | collective? | N = 10 000 |
|---|---|---|---|
| `direct` | boundary nodes from the exact pairwise sum, O(N·N_boundary) | **yes**, keeps the gather | 47.2 ms, 4.1 % error |
| `zero` | φ = 0 on the box edge | **no — none at all** | 10.6 ms, 6.1 % error |

`zero` is the only configuration with *no* array-sized global communication
anywhere: deposit and solve talk to neighbours through the ordinary OPS halo
exchange and nothing else. It pays for that with the image-charge error of a
finite boundary. `direct` removes that error but reinstates one collective per
step — still far cheaper than the O(N²) sum, since `N_boundary ≈ 4·NX ≪ N`.

### 3. Self-energy — the one that does not go away on its own

A particle deposits its own charge and then reads the field back, so it feels
itself; the direct sum excludes `j == i`. The difference is proportional to the
particle's own charge, so with signed charges it biases + and − particles in
opposite directions and does **not** average out or shrink under refinement.
Left uncorrected it is a **55 % error**, i.e. it dominates everything else.

Writing `w_a` for the four CIC weights and `Ĝ` for the discrete Green's
function, `phi_self = m Σ_{a,b} w_a w_b Ĝ(a,b)`, and by symmetry `Ĝ` takes only
three distinct values over a cell — same node, edge-adjacent, diagonal. Grouping
the weight products gives three shape factors `A + B + C = 1` that depend only
on where the particle sits inside its cell, so `phi_self = m(aA + bB + cC)` for
three constants. The app fits them **once, at step 0**, by least squares (11
accumulations in a single `dim`-11 `ops_reduction`, then a 3×3 Cramer solve),
and then freezes them.

Frozen matters. The cloud rotates and spreads for the rest of the run, so the
agreement reported at the **end** is a genuine test, not a fit to the answer.

| correction | error at 513², N = 400 |
|---|---|
| none | 66.0 % |
| one constant | 2.8 % |
| three shapes | **1.3 %** |

---

## Correctness

Everything below is measured, at `-npart 2000 -nsteps 50`:

- **deposit conserves charge exactly** — `sum|q|` on the mesh vs `sum|m|`,
  relative error `0.00e+00` to `3.0e-16` at np = 1, 2, 4. CIC's partition of
  unity holds across rank boundaries, which is the thing most likely to break.
- **the direct method is bit-exact** — `max|Δφ| = 0.000e+00` against a host
  reference replaying the same recurrence, at np = 1, 2, 4.
- **both methods are rank-invariant** — the fitted self-energy coefficients
  `(-1.047766, -0.793302, -0.684389)` and the final error `3.17 %` are identical
  at np = 1, 2 and 4.
- **the solver is not the error.** Pushing the residual from `9.6e-2` down to
  `2.0e-9` (40 → 400 sweeps/step) changes the accuracy by nothing at all: the
  40-sweep warm-started solve is already converged as far as the answer cares.

---

## A trap worth knowing about

**Set `OMP_NUM_THREADS=1` for MPI runs.** The build links OpenMP; with the
variable unset, every rank spawns one thread per core. On this 12-core box np=4
means 48 threads fighting over 12 cores, and the symptom is not a gentle
slowdown:

| SOR solve, 40 sweeps | time |
|---|---|
| serial | 7.5 ms |
| np = 2, `OMP_NUM_THREADS` unset | 2 663 ms |
| np = 4, `OMP_NUM_THREADS` unset | 11 468 ms |
| np = 2, `OMP_NUM_THREADS=1` | 5.8 ms |

A 350× slowdown that grows with rank count reads exactly like a deadlock in the
solver, and it is neither. It bites this app harder than the others because SOR
fires many tiny grid loops per step (80 at the default) rather than a few large
ones.

Related: neither method strong-scales well here, because a 129² mesh split four
ways is only ~4 000 points per rank — too little work to cover the per-loop
overhead. The crossover measured above is algorithmic and shows up at any rank
count; the parallel efficiency numbers on this box are not worth quoting.

---

## Options

```
-npart N     particle count (rounded up to even)     default 2000
-nx N        grid nodes per side                     default 129
-nsteps N    timesteps                               default 500
-sweeps N    red-black SOR sweeps per step           default 40
-cold N      sweeps for the initial cold solve       default 4000
-bc zero|direct                                      default direct
-method both|direct|mesh                             default both
-check       verify `direct` against a host reference (O(N^2) on the host)
```

## Files

| File | Contents |
|---|---|
| `influence_pm.cpp` | driver: both methods, the self-energy fit, timers, reports |
| `grid_kernels.h` | deposit, boundary, SOR, residual |
| `particle_kernels.h` | motion, direct sum, interpolation, self-energy model |
| `scatter_loop.h` | translator-safe alias for the grid-outer loop |
