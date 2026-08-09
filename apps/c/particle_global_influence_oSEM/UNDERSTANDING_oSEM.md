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
| randoms | `ops_fill_random_uniform` into int dats, per step | per-eddy LCG state carried as a particle dat |
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

## 3. Each eddy carries its own random stream

oSEM draws randoms into rank-indexed grid dats. Particles migrate between ranks,
so a rank-indexed random dat would hand a migrating eddy somebody else's stream.
Here the LCG state is a particle dat, so the stream belongs to the eddy and
travels with it — which is one of the three things that make the run
rank-invariant.

It also sidesteps a defect: `ops_fill_random_uniform` on an int dat never returns
a negative value, so oSEM's `(rng < 0) ? −1 : 1` sign draws are **always +1**.

**One trap this introduced.** Drawing an eddy's position and then its signs from
consecutive LCG states makes `ε_x` a deterministic function of `x` — and since
`compute_fluct` selects eddies by x, it selects a *biased* set of `ε_x`. That
produced rms 8.85 / 4.89 / 6.41 when all three must be equal, with the signs
individually unbiased so no test of the sign distribution would catch it. Fixed
with an output mixer (`lcg_mix`). Worth remembering for any per-particle RNG: a
running stream plus a position-dependent selection gives correlation that
checking the random values alone will not find.

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
