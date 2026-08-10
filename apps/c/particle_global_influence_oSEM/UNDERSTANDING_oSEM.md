# Understanding oSEM, and how this app differs

Background reading for [`influence_osem.cpp`](./influence_osem.cpp). The first
half explains the method as [`../oSEM`](../oSEM) implements it; the second half
sets out what changes when the eddies become OPS particles.

---

# Part 1 — What oSEM is doing

## The problem it solves

You want to run an LES or DNS of something downstream, and you need to feed its
inlet a velocity field that *looks* like turbulence — correct Reynolds stresses,
correct length scale, correct time correlation — without running an expensive
precursor simulation to generate one.

The Synthetic Eddy Method fakes it. Scatter a cloud of localised velocity blobs
("eddies") in a box around the inlet, give each a random sign, convect them past
the plane, and sum their contributions at every node. The result is random in
space, correlated in time, and statistically matched to a target.

## The geometry: a 2-D output, a 3-D eddy cloud

This is the part that confuses people first, so it is worth being explicit.

```
                eddy box: a 3-D slab, thickness 2·r_max
          ┌───────────────────────────────────────────┐
          │    ○       ○          ○        ○          │
    y ↑   │        ○       ○   ▓▓   ○   ○      ○      │   eddies drift →
          │   ○         ○      ▓▓        ○            │   at speed u0
          │        ○        ○  ▓▓    ○        ○       │
          └───────────────────────────────────────────┘
         x_min                 ▓▓                   x_max
        (−r_max)          inlet plane               (+r_max)
                          x_plane = 0
                       the 2-D (y,z) face
                            ↑
                  this is where u', v', w' are wanted
```

**The output is 2-D. The eddies have to be 3-D.**

The inlet plane is the (y, z) face where you need `u'`, `v'`, `w'` at every grid
node on every timestep — the 101 × 151 grid in this app. But an eddy only affects
that plane when it is **near it in x**. The x coordinate is not a dimension of
the output at all; it is **the mechanism that makes the signal time-dependent**.

Freeze x, and every node would have a fixed fluctuation forever — useless as an
inlet condition. Let the eddies drift through in x, and the fluctuation at a
fixed (y, z) node rises as an eddy approaches, peaks as it crosses, and falls as
it leaves. That temporal envelope **is** the time correlation of the synthetic
turbulence.

## Why the box has exactly that extent

Every dimension of the box is set by the eddy radius, `r_max = 0.41·δ`:

| bound | value | why |
|---|---|---|
| `x ∈ [−r_max, +r_max]` | ±0.00287 | An eddy's shape function has compact support of radius r. Beyond one radius from the plane it contributes *exactly zero*, so there is no point tracking it. The box is precisely the region where an eddy can matter. |
| `y` padded to `y_max + r_max` | 0 … 0.01187 | An eddy whose **centre** sits just outside the inlet still overlaps nodes near the edge. Without the padding you get an artificial drop in fluctuation intensity along the borders. |
| `z` padded both ends | −0.00287 … 0.05287 | Same reason. |

## The life of one eddy

1. **Born** at a random (x, y, z) in the box, radius `0.2·δ`, and three random
   signs `ε_x, ε_y, ε_z ∈ {−1, +1}`.
2. **Convects**: `x += u0·dt` every step. This is Taylor's frozen-turbulence
   hypothesis — the eddy pattern is simply swept past the inlet at the mean
   speed, rather than being evolved dynamically.
3. **Crosses the plane**, contributing to nearby nodes for as long as `|x| < r`.
4. **Exits** at `x > x_max`, having passed completely through. It can never
   matter again, so it is **recycled**: back to `x_min` with a fresh random
   (y, z) and fresh signs.

The traverse takes

```
2·r_max / (u0·dt)  =  0.00574 / 1.647e−5  ≈  348 steps
```

and that number is the **integral time scale** of the generated signal — how
long the inlet "remembers".

## How many eddies, and how many matter

```
eddies = vol / rep_radius³ = 1718
```

i.e. one eddy per cube of side `0.2·δ`. That density is what makes the blobs
overlap enough to look like a continuous field rather than isolated spots.

At any instant only those with `|x| < r` are active. The fraction is
`2r / (x_max − x_min) ≈ 0.49`, which is why the plots report roughly
**854 of 1718** eddies within a radius of the plane.

## What an eddy actually does to the plane

For each node, for each eddy, **if** `x² + (y_e−y)² + (z_e−z)² < r²`:

```
shape = exp(−½·dx²/r²) · exp(−½·dy²/r²) · exp(−½·dz²/r²) · (1/1.5829045)
```

A 3-D Gaussian in the separation between node and eddy centre, cut off outside
one radius. The `1/1.5829045` is a normalisation chosen so that summing `shape²`
over the eddy density gives **unit variance** — it is what makes the raw signal
come out at rms 1 before any physical scaling.

Note the `dx` term: an eddy some way from the plane in x still contributes, just
weakly. That smooth ramp-up and ramp-down is the shape of the time correlation.

## From random signs to real Reynolds stresses

The three sign-sums are three independent, unit-variance random fields. The
Cholesky factor `a_ij` of the target Reynolds stress tensor `R_ij` turns them
into a correlated set — the standard Lund transformation, since `A·s` has
covariance `A·Aᵀ = R`:

```
u' = a11·Σ(ε_x·shape)
v' = a21·Σ(ε_x·shape) + a22·Σ(ε_y·shape)
w' = a31·Σ(ε_x·shape) + a32·Σ(ε_y·shape) + a33·Σ(ε_z·shape)
```

In the **isotropic** case used here, `a11 = a22 = a33 = u0·TI = 8.236` and the
off-diagonals are zero, so all three components must come out at the same rms.
That is a useful invariant: it is what made the early 8.85 / 4.89 / 6.41 result
diagnosable as a bug rather than as physics (see the README).

oSEM's other variant, `instantiate_RST_TBL`, makes `a_ij` vary with y from
tabulated boundary-layer data, giving a realistic profile with genuine shear
stress `a21 ≠ 0`. This app uses the isotropic one to stay self-contained.

## All the numbers in one place

| quantity | value |
|---|---|
| `u0` (convection speed) | 823.6 |
| `dt` | 2e−8 |
| `δ` | 0.007 |
| `r_max = 0.41·δ` | 0.00287 |
| eddy radius `0.2·δ` | 0.0014 |
| inlet grid | 101 × 151 |
| eddy count | 1718 |
| active at any instant | ≈ 854 |
| traverse / integral time | ≈ 348 steps |
| target rms `u0·TI` | 8.236 |
| grid spacing dy / dz | 0.119 mm / 0.372 mm |
| eddy width in cells (y / z) | ≈ 12 / ≈ 3.8 |

## Resolution versus physical size — easy to mix up

`NY x NZ = 100 x 150` is the number of grid **intervals** (so 101 x 151 nodes).
That is index space: how many points. The plot axes are in **metres**: how big
the thing actually is. Nothing ties those two together.

| | physical extent | nodes | spacing |
|---|---|---|---|
| y | 0 -> 0.01187 m (~12 mm) | 101 | dy = 0.119 mm |
| z | -0.00287 -> 0.05287 m (~56 mm) | 151 | dz = 0.372 mm |

The inlet is about **12 mm x 56 mm**. It is physically small because it is a
boundary-layer inlet and `delta = 7 mm`: `y_max = 9 mm` is a little over one
delta, with `r_max = 2.87 mm` of padding added at the relevant edges. Small
because delta is small, not because the grid is coarse.

The same scale explains the timestep, which looks strange in isolation:
`u0 = 823.6 m/s` with `dt = 2e-8 s` convects the flow **16.5 microns per step**.
Fast flow, fine time resolution, small domain — all consistent.

Two consequences worth carrying around while developing:

**The cells are not square.** `dz/dy ~ 3.1`, so a cell is three times longer in
z than in y. That is why `plot_osem_h5.py` uses `aspect="auto"` rather than
`equal` — at equal aspect the strip is 4.7x wider than tall and unreadable.

**The eddies are far better resolved in y than in z.** With radius
`0.2 delta = 1.4 mm`:

| direction | cells across an eddy |
|---|---|
| y | ~12 |
| z | ~3.8 |

Four cells across a Gaussian is marginal — in z you are close to the limit of
what the grid can represent, which is also why the circles look small relative
to the cell size in that direction. This is inherited from oSEM's choice of
`ny = 100, nz = 150` for a domain 4.7x longer in z; the port did not introduce
it. If the z-direction spectrum ever matters, raise `nz` (the `-nz` option).
It costs nothing in eddy count, which is set by box volume and radius, not by
resolution.

---

# Part 2 — How this app differs

The physics above is unchanged. What changes is the data model: **eddies become
OPS particles instead of grid dats on a second block.**

| | oSEM | this app |
|---|---|---|
| eddy storage | 7 grid dats on a separate `eddy_block` | OPS particles in the inlet block |
| eddy → `compute_fluct` | `ops_dat_fetch_data` × 7 into host arrays | array-valued `ops_reduction` allgather |
| randoms | `ops_fill_random_uniform` into int dats, per step | `ops_fill_random_uniform_particle` into a particle dat, per step, keyed on global id |
| blocks | 2 (`inlet_block`, `eddy_block`) | 1 |
| recycle | new random (y, z) | continuous transverse drift |
| correct under MPI | **no — np = 1 only** | yes, rank-invariant at np = 1, 2, 4, 8 |

## 1. Position is (y, z); x is an ordinary dat

The single most important design decision, and it is not arbitrary.

OPS decomposes the block, and the block **is** the inlet plane. So (y, z) is what
determines which rank owns an eddy, and it is the only part that needs to be a
spatial coordinate. **Nothing is decomposed along x and no neighbour search uses
it**, so x is just a per-eddy scalar.

That distinction has a direct consequence: resetting `x = x_min` on recycle
involves no migration and costs nothing, whereas re-randomising (y, z) is a jump
to an arbitrary rank. Which leads to point 4.

## 2. The gather is a fix, not a restructuring

oSEM hands the eddy state to `compute_fluct` by fetching seven eddy dats into
host arrays. **Under MPI that is wrong, not merely partial.**
`ops_dat_fetch_data` (`ops_mpi_rt_support.cpp:2125`) copies only *this rank's*
slice of the decomposed eddy block and writes it starting at offset 0 — it
computes a displacement `ldisp` and then never uses it in the `memcpy`.
`compute_fluct` then loops over the **global** eddy count, so every index past
the local slice reads uninitialised heap on the first step and stale values
afterwards. Each rank builds the inlet from a different, partly garbage eddy set.

The reason a gather is unavoidable is structural: **an eddy owned by rank 0
affects nodes owned by rank 3.** The interaction is global — every node sums over
every eddy — so ownership of the eddy and ownership of the node it influences
are unrelated. The `ops_reduction` allgather returns the complete eddy list, in
id order, on every rank, which is exactly the array shape `compute_fluct` already
expected, so the kernel body ports across unchanged.

## 3. Randoms are filled from a driver call, keyed on global id

`ops_particle_random.h` provides `ops_fill_random_uniform_particle()`, called
from the driver before the kernels — the same shape of call as oSEM's
`ops_fill_random_uniform(d_y_rng)`, so the kernels just read a dat.

The difference is what it is keyed on. oSEM's fill is keyed on **storage
position**, and a particle's local slot changes when it migrates or when the
list is compacted — so the same eddy would draw from a different stream after
moving. This one is keyed on the particle's **global id**, which is stable for
the run and identical at any decomposition. That is one of the three things
that make the run rank-invariant.

It is also **counter-based**: each value is a pure hash of
`(seed, gid, step, component)`, so there is no state dat to declare, migrate or
put in the border list, and any step's draw is reproducible on demand.

It also sidesteps a defect: `ops_fill_random_uniform` on an int dat never returns
a negative value, so oSEM's `(rng < 0) ? −1 : 1` sign draws are **always +1**.

**A trap an earlier version fell into.** That version advanced one LCG per eddy
and took successive states for successive quantities, which made `ε_x` a
deterministic function of `x` — and since `compute_fluct` selects eddies by x,
it selected a *biased* set of `ε_x`. Result: rms 8.85 / 4.89 / 6.41 when all
three must be equal, with the signs individually unbiased so no test of the sign
distribution would have caught it.

The counter-based fill removes the failure mode rather than patching it: each
component is a separate hash input, so there are no successive states to
correlate. Still worth remembering generally — **a running per-particle stream
combined with any position-dependent selection produces correlation that
checking the random values alone will not find.**

## 4. Continuous drift instead of a teleporting recycle

oSEM recycles an exiting eddy to a **new random (y, z)**. As a particle
operation that is a jump to an arbitrary point in the plane, and OPS particle
migration only hands a particle to a **neighbouring** rank. Measured: works at
np = 1 and 2, fails from np = 3 up. A controlled experiment — removing only the
two position assignments, changing nothing else — made it run clean everywhere,
so the cause is certain.

It was never "eddies versus particles". It was **discontinuous re-injection
versus continuous motion**. `particle_global_influence` moves its particles by
`x += v·dt` and never jumps, which is why it works at every rank count.

So the eddy now carries a transverse **velocity**, drawn once, and reflects off
the box faces — 0.29 cells per step in y, 0.43 in z, small enough that migration
only ever sees a move into an adjacent subdomain.

**The statistical refreshment survives**, because the part that matters is not
spatial: the **signs are still re-drawn on every recycle**, and they are what
randomises `u'`, `v'`, `w'`. Positions now decorrelate by drift rather than by
resampling, over roughly one flow-through time.

This is a deliberate model change, not a bug fix. Resampling gives an instantly
independent position; drift gives a correlated one that decorrelates over ~348
steps. For an inlet generator that is a defensible trade, and arguably more
physical — but it is a choice, and the teleporting version is in git history.

## What you get for it

Rank-invariant to all printed digits at np = 1, 2, 4 and 8:

```
rms u' = 8.6893   v' = 8.8646   w' = 8.8594   (target u0·TI = 8.2360)
```

with the eddy population conserved throughout. That falls out of three things
together: per-eddy random streams, a bit-exact gather, and motion that is
deterministic per eddy.

---

# Part 3 — Verifying the Reynolds stresses, and what it found

A record of the checks that exist, why each is shaped the way it is, and the
two inherited defects they exposed. Dated 2026-08-10.

## Why a shear check was needed at all

With the isotropic RST every off-diagonal of the Cholesky factor is zero, so
nothing exercises them. The tabulated profile (`-rst tbl`) turns on `a21`, and
at that point the app had **no test that could see it**. The reason is worth
stating precisely, because it is not obvious:

```
a22 = sqrt(R22 - a21²)   so   a21² + a22² = R22   for ANY a21
```

The rms of `v'` therefore comes out at `sqrt(R22)` whatever `a21` is — a wrong
`a21` is exactly cancelled by the `a22` that is derived from it. Every check in
the app was an rms check, so `a21` could have been arbitrarily wrong and
everything would still have passed. The shear stress `<u'v'>` is the only
observable that sees it.

## Two things the check needed before it meant anything

**1. Time-averaging.** `<u'v'>` depends on the cancellation `<Sx Sy> -> 0`
between two independent sign fields. At a single instant, with only ~40
independent eddy-sized patches per row, that cancellation scatters by far more
than the signal: the deviation from `R21` swung **64 / 52 / 154 %** across
snapshots of identical, correct code. The profile accumulator now sums every
step, and the deviation falls as it should:

| niter | 200 | 800 | 3200 | 12800 |
|---|---|---|---|---|
| shear deviation | 35.6 % | 32.2 % | 23.3 % | 18.0 % |
| max abs cross terms | 0.0667 | — | 0.0055 | 0.0048 |

**2. Dividing out the normalisation.** Even time-averaged, the deviation
flattened near 17 % and *stayed there* under ensemble averaging over 12 seeds,
which makes it systematic rather than scatter. It is not `a21`. Every stress
carries a common factor `<S²>`, the variance of the raw eddy sum, and the
correlation coefficient cancels it:

```
rho = <u'v'> / sqrt(<u'u'> <v'v'>)   ->   R21 / sqrt(R11 R22)
```

Over 12 realisations: **slope 1.0090**, row-by-row ratios 0.98–1.03. `a21` is
exact to 0.9 %. That is the verification; the raw rms deviation is not.

Single runs are not enough to conclude anything here — `rho` scatters ±7 %
realisation to realisation. Hence `-seed N`, and `-dumpshear FILE` for the
per-row profile.

## Defect 1 — the eddy count is sized for the wrong box

`<S²>` measures **1.16** where it should be 1, so every Reynolds stress the app
produces is ~16 % high in variance, ~8 % high in rms. The cause:

```c
vol    = (x_max-x_min) * (y_max-y_min + 2*r_max) * (z_max-z_min + 2*r_max);
eddy_y_min = y_min;            /* NO bottom padding */
eddy_y_max = y_max + r_max;    /* top padding only  */
eddies = vol / eddy_radius³;
```

`vol` pads y on both sides, the eddy box pads only the top: 0.01474 against
0.01187, a ratio of **1.2418**. So `eddies` is that much too large for the box
the eddies actually occupy, and the density — and every stress with it — comes
out high. Predicted `<S²> = 0.949 × 1.2418 = 1.179` against 1.16 measured.

Inherited, not introduced — `apps/c/oSEM/OPS_oSEM.cpp:43-49` has the identical
pair. The `0.949` is separately interesting: it is `∫shape² dV / (norm² · r³)`
with oSEM's `shape_norm = 1/1.5829045`, so even with a consistent box the raw
sum would be ~5 % low in variance.

### Deliberately NOT fixed — and what fixing it does

**The code keeps oSEM's form on purpose.** Matching the reference matters more
right now than matching the tabulated target, because numbers from the two apps
have to stay comparable. Do not "correct" this without asking first; the source
comment at the `vol` assignment says the same thing.

The correction is one term — `2*r_max` -> `r_max` in the y extent, which makes
`vol` exactly `(x_max-x_min)(eddy_y_max-eddy_y_min)(eddy_z_max-eddy_z_min)`.
It was applied, measured over the full 12-seed ensemble, and reverted.
`-ny 50 -nz 75 -niter 1600`, 12 seeds each:

| | oSEM base | vol fixed |
|---|---|---|
| `eddies` | 1718 | 1384 |
| `<S²>` interior plateau | 1.1558 | 0.9318 |
| `<uu>/R11` slope | 0.9362 | 0.7620 |
| `<vv>/R22` slope | 1.1261 | 0.9059 |
| `<uv>/R21` slope | 1.1005 | 0.8897 |
| `rho` (the a21 test) | 1.0090 ± 0.0128 | 1.0105 ± 0.0106 |

Three things to read out of this.

**The mechanism is confirmed exactly.** The eddy count falls by 1.2413 and
`<S²>` falls by 1.1558/0.9318 = 1.2404 — the same ratio to three digits. The
excess really is the density and nothing else.

**The fix improves absolute agreement but does not reach 1.** `<S²>` goes from
~16 % *high* to ~7 % *low*; in rms, from ~7.5 % high to ~3.5 % low. It cannot
land on 1 on its own, because `shape_norm` alone predicts 0.949 (above) — the
normalisation constant is itself ~5 % off in variance for this shape function
and eddy density. Correcting the box exposes that second, smaller error rather
than cancelling against it. The two errors currently work in opposite
directions, which is why the shipped app looks better on `<S²>` than either
constituent deserves.

**`rho` is untouched, as designed.** 1.0090 -> 1.0105, a shift far inside the
±0.011–0.013 standard error. The `a21` verification is genuinely independent of
the density error, which is the whole reason for using a correlation
coefficient instead of the raw shear.

The earlier version of this table was two seeds and gave a muddled picture
(profile agreement improved on one seed and worsened on the other — pure
scatter). The 12-seed numbers above supersede it.

## Defect 2 — the plane reaches the edge of the eddy box

`<u'u'>/R11` by row, ensemble-averaged over 12 seeds:

```
y = 0.00047  ->  0.867      first interior row
y = 0.00570  ->  1.162      interior, flat
y = 0.01140  ->  0.895
y = 0.01187  ->  0.591      top row
```

Nodes at either y extreme are surrounded by eddies on one side only, because
the plane spans the *whole* eddy box rather than sitting inside it. Also
inherited: oSEM's `instantiate_grid` does the same.

## Three hypotheses the measurements killed

Recorded because each looked convincing and each was wrong:

1. **"Freestream rows dominate the metric."** They sit at 0.1–0.5 % of peak
   `R21`. Splitting the metric changed the number not at all.
2. **"The eddy positions are a frozen lattice."** Drift and reflection are
   deterministic, so only the signs re-randomise — plausible, but the ensemble
   mean over 12 seeds did not collapse (17.2 / 16.3 / 18.6 / 16.7 % for 2 / 4 /
   8 / 12), which rules it out.
3. **"`eps_x` and `eps_y` are correlated."** Measured directly over 4×10⁶
   draws: `<eps_x eps_y> = +0.00025` at 1σ = 0.00050. Clean.

A fourth was my own artefact: an `R11²`-weighted regression appeared to show
`<Sx²> = 0.936` against `<Sy²> = 1.17`, which is impossible by construction.
The weighting simply sampled different parts of the y-dependent curve above.
Per row the two agree exactly.

## A translator constraint worth knowing

`ops_arg_reduce`'s dimension must be a **bare integer literal**: the translator
parses that token with `parseIntLiteral`
(`ops_translator/ops-translator/cpp/parser.py:353`), which accepts an
`INTEGER_LITERAL` only. `6 * (ny + 1)` fails with "Expected int expression" —
and only in the translator build, so the seq/dev builds compile it happily and
the breakage hides until someone builds the MPI target.

It also **unrolls** the reduction: one scalar local and one write-back per
slot, plus an OpenMP clause naming every one. A cap of 6150 generated a
24819-line kernel that had not compiled after 500 s; 606 generates 2643 lines
and builds in 15 s. Hence `PROF_SLOTS = 606`, the `ny <= 100` ceiling under
`-rst tbl`, and the `static_assert` binding the macro to the literal.

## Defects in the reference implementation itself

Established by inspection of `apps/c/oSEM` and the OPS library source — oSEM
was **not run** (out of scope). These matter for one specific reason: they set
a hard limit on what "matching oSEM" can mean.

### 1. Every eddy sign in oSEM is +1

```c
/* OPS_oSEM_kernels.h:82-84, and again at 71-73 in convect_eddies */
eps_x(0, 0) = ((eps_x_rng(0, 0) < 0) ? -1 : 1);
```

fed from `ops_fill_random_uniform(d_eps_x_rng)` on an **int** dat, which is

```c
/* ops_lib_core.cpp:2604 */
std::uniform_int_distribution<int> distribution(0, INT_MAX);
```

Never negative, so the `< 0` test never fires and all three sign fields are
uniformly `+1`. SEM works by *cancellation* between random ±1 signs; with every
sign positive there is none. `S_x` becomes a sum of strictly positive terms, so
the fluctuations are not zero-mean and the Reynolds stresses do not mean
anything.

### 2. oSEM's eddies occupy one eighth of the box

```c
/* OPS_oSEM_kernels.h:80-82 */
x(0,0) = x_min + ((double)x_rng(0,0) + 2147483648.0) / 4294967295.0 * (x_max - x_min);
```

`x_rng` comes from the same fill, so it spans `[0, 2^31-1]`, and

```
(0          + 2147483648) / 4294967295 = 0.5000000001
(2147483647 + 2147483648) / 4294967295 = 1.0
```

The mapped fraction spans **[0.5, 1.0]**, not [0, 1]. Every coordinate is
confined to the upper half of its range — in 3-D, one eighth of the volume.
The same expression is used for the re-seed in `convect_eddies`, so it does not
wash out over time.

### 3. `compute_fluct` reads uninitialised memory under MPI

Documented in Part 2 §2: `ops_dat_fetch_data` computes a displacement `ldisp`
and then never applies it, so each rank gets its own slice written at offset 0
while the kernel loops over the global eddy count.

### What this implies

**Bit-for-bit agreement with oSEM is neither achievable nor desirable.** This
app deliberately differs in four places — random signs, uniform positions, the
allgather, and continuous drift — three of which are fixes for the above. So
"has this been validated against oSEM's output?" is the wrong question to keep
open: the reference output is not a valid target.

What the port *does* keep is oSEM's **physical parameterisation**: box extents,
the `vol`-based density formula, `shape_norm`, the RST construction, the shape
function. That is the sense in which it is a faithful port.

One consequence worth stating plainly. Those constants sat, in oSEM, on top of
a generator with no sign cancellation and eddies in an eighth of the domain —
so they cannot have been calibrated against working physics. That is a
plausible reading of why `shape_norm` matches no natural integral of its own
shape function and arrives through a chain of square roots of `5.01117`
(`1/2.2385651 = 1/sqrt(5.01117)`, then `1/1.5829045 = 1/sqrt(5.01117/2)`, both
still in the source as comments). In *this* app the physics underneath is
correct, so `vol` and `shape_norm` are now the only things between the output
and the tabulated target — they are load-bearing here in a way they never were
in the reference.

They are kept as-is for now by explicit decision. Revisiting them is a matter
of getting the output right on its own merits, not of oSEM parity.

## What is and is not established

| Claim | Status |
|---|---|
| Time-averaging accumulates and converges | verified |
| `a21` is correct | verified — `rho` slope 1.0090 |
| Cross terms `<u'w'>`, `<v'w'>` vanish | verified — 0.0667 -> 0.0048 |
| Rank-invariant | verified — np = 1, 2, 4, 8, 16, both builds |
| App reproduces the target Reynolds stresses | **no** — ~8 % high in rms |
| Eddies uniform in the box, signs unbiased, radius constant | verified — χ² 10.7 / 5.9 / 4.7 over 10 bins |
| Profile *shape* follows the target once scaled | verified — flat to 0.9 % (`uu`), 0.7 % (`vv`) over interior rows |
| App agrees with reference oSEM's own output | **not a valid target** — see "Defects in the reference implementation itself" |
