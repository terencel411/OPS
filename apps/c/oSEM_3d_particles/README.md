# oSEM_3d_particles

[`../oSEM_3d`](../oSEM_3d) with the synthetic eddies as **OPS particles**, and
the hand-rolled `MPI_Allgatherv` that fed `Kernel030` replaced by an OPS
reduction. Structured after
[`../oSEM_2d_particles`](../oSEM_2d_particles), which
does the same thing for the 2-D inlet-plane app.

The RK3 solver, the boundary conditions, the metrics, the averaging and the
HDF5 output are `../oSEM_3d`'s, untouched.

```bash
source ../../../configure_env_vars.sh

make opensbli_dev_seq && ./opensbli_dev_seq -niter 100        # start here
make opensbli_dev_mpi && OMP_NUM_THREADS=1 mpirun -np 4 ./opensbli_dev_mpi -niter 100
make opensbli_mpi                                             # via the translator
```

`OMP_NUM_THREADS=1` is not optional under MPI here — unset, every rank spawns a
thread per core and the run slows by two orders of magnitude.

| flag | default | |
|---|---|---|
| `-ngrid NX NY NZ` | `150 50 30` | `../oSEM_3d` hardwires `750 250 150`, which will not fit on a laptop |
| `-niter N` | 100 | |
| `-nout M` | 10 | HDF5 write interval |
| `-seed N` | 182383739 | selects the realisation |
| `-neddies N` | from the box | overrides the eddy count; **0 runs the solver with no synthetic turbulence** |
| `-degenerate-rng` | off | diagnostic: reproduces `../oSEM_3d`'s broken draw, see below |

## What was wrong, and what fixes it

`../oSEM_3d` holds the eddies in ten `ops_dat`s of shape `{eddies,1,1}` declared
on the fluid block (`defdec_data_set.h:625-715`), so MPI decomposes the eddy
list by the *fluid* partitioning — an eddy list sliced along x by a
decomposition that has nothing to do with eddies. `Kernel030` needs the whole
list on every rank, and gets it through seven `ops_arg_gbl` arrays filled by
`ops_dat_fetch_data` plus a hand-rolled `MPI_Allgatherv` over raw
`MPI_COMM_WORLD`, every timestep (`opensbli.cpp:416-491`).

Here an eddy is a particle. The gather is an **array-valued `ops_reduction`
declared `OPS_INC`, indexed by global eddy id**: every eddy adds itself into its
own slot, every other rank contributes `0.0` there, and the `MPI_Allreduce`
inside `ops_reduction_result` is therefore an allgather. It runs over
`OPS_MPI_GLOBAL` (`ops_mpi_rt_support.cpp:1285`), not the block communicator, so
every rank ends up with the complete list in id order — the array shape
`Kernel030` already wanted.

`Kernel030`'s body is unchanged apart from the indexing: seven arrays become one
buffer of `NCOMP = 7` doubles per eddy.

### The same bug bites the inlet profiles

`uinterp` and `a11..a33` are built by a kernel and then read back with
`ops_dat_fetch_data` (`../oSEM_3d/opensbli.cpp:257, 315-318`). That call returns
only **this rank's slice**, written at offset 0 — the identical defect. They are
gathered by the same `OPS_INC` reduction here.

## Results

267 eddies. The gather is the thing that had to become rank-invariant, so that
is what is measured — sums over the whole eddy list in id order, which cannot
depend on the decomposition if the gather is right.

```
eddy particles owned: 267 of 267
eddy gather [iter   0]: sum x -2.981333706498e+01 y 1.521454769194e+03 z 5.413193262120e+03  eps  +5  missing 0
eddy gather [iter  99]: sum x  2.115512114348e+01 y 1.501658060369e+03 z 5.258570050699e+03  eps -13  missing 0
eddy gather [iter 199]: sum x  3.536374305259e+00 y 1.470562355956e+03 z 5.293290030270e+03  eps -43  missing 0
eddy gather [iter 299]: sum x  1.773637430526e+01 y 1.582672131746e+03 z 5.567341910950e+03  eps -43  missing 0
eddy gather [iter 399]: sum x  3.736374305259e+00 y 1.703922939802e+03 z 5.411479673332e+03  eps -15  missing 0
```

**Bit-identical at np = 1, 2, 4 and 8**, over 400 steps at `-ngrid 60 20 16`,
with the population conserved throughout and no empty slot in the buffer. The
run is long enough for the recycle to fire — an eddy crosses the box in
`2*radius/(1.0*dt) = 187` steps, and the drifting `eps` sum is the signs being
re-drawn.

**Do not expect the flow field to match across rank counts.** `../oSEM_3d` is
not rank-invariant to begin with — 2x2x1 against 2x2x2 moves the fields by
O(0.2), which has nothing to do with the eddy path. That is why the probe above
is on the gather rather than on the HDF5 output.

## How OPS Particles is used here

A ground-up walkthrough, for anyone arriving from `../oSEM_3d` who has not used
the OPS particle API before.

### The ten dats that went away

`../oSEM_3d` held the eddy state in ten `ops_dat`s on the fluid block, each with
`size = {eddies, 1, 1}` (`defdec_data_set.h:625-715`):

| `../oSEM_3d` dat | dim | type | becomes, here |
|---|---|---|---|
| `eddy_x`, `eddy_y`, `eddy_z` | 1 each | double | `p_e` — one particle dat of dim 3 |
| `eddy_r` | 1 | double | `p_r` — particle dat, dim 1 |
| `eddy_eps_x/y/z` | 1 each | int | `p_eps` — one particle dat of dim 3, as doubles |
| `eddy_increment` | 1 | double | `ops_decl_const increment` — it is `1.0*dt` for every eddy, so no per-particle storage is needed |
| `eddy_x_rng`, `eddy_bulk_rng` | 1, 5 | int | `p_rnd` — one particle dat of dim 6, doubles in [0,1) |

Two dats here have no `../oSEM_3d` counterpart, because a distributed particle
set needs bookkeeping a static array does not: `p_pos` (dim 3, the position OPS
reads to decide ownership) and `p_id` (dim 1 int, the global eddy id that
survives any reordering).

**Why those dats were the bug.** `size = {eddies,1,1}` on the fluid block means
OPS decomposes the eddy list using the *fluid* partitioning — a 267-entry list
chopped along x by a decomposition designed for a 750-point grid. Nothing about
that slicing has anything to do with eddies, and every rank ends up holding an
arbitrary fragment. That is what the `MPI_Allgatherv` existed to paper over.

### Grid dat vs particle dat

| | `ops_decl_dat` (grid) | `ops_decl_particle_dat` |
|---|---|---|
| allocates | `dim` values at **every grid node** | `dim` values per **particle slot** |
| shape | 3-D, `size[]` plus halos `d_m`/`d_p` | 1-D array of length `Nmax` |
| indexed by | `(i,j,k)` — a fixed place in space | particle slot — **not stable**, it changes as the list is compacted |
| kernel accessor | `ACC<T>&` → `c(comp,i,j,k)` | `ACCP<T>&` → `p(comp)` |
| decomposed by | the block partitioning | particle *position* |

Because the slot is not stable, anything that must follow a specific eddy has to
key on something that is — hence `p_id`, and hence the random stream being keyed
on the global id rather than the slot.

### `d_coords`, and why it had to be created

`ops_create_bounding_box` and `ops_decl_mapping` each need **one** `ops_dat`
whose `dim` equals the block's dimensionality, holding the physical coordinate
of every node. `../oSEM_3d` stores coordinates as *three separate* dim-1 dats
(`x0_B0`, `x1_B0`, `x2_B0`), which does not fit — so `d_coords` is a dim-3 dat
on the same block and `KerPackCoords` copies the three into it.

Why coordinates rather than spacing? OPS has to turn a particle's physical
position into "which rank, which cell", and this grid is **sinh-stretched in y**.
Spacing varies by nearly a factor of seven from wall to freestream, so the only
reliable source is the node coordinates themselves.

### The bounding box

A `BoundingBox` is the physical region, in real coordinates, that the particle
set occupies. It exists in two forms at once: **global** (the whole domain) and
**local** (this rank's slice). The local box answers the central question of any
distributed particle code — *is this particle mine?*

```c
double dx_box[] = {0.0, 0.0, 0.0};
BoundingBox<Real> *box = ops_create_bounding_box(opensbliblock00, d_coords, 3, dx_box);
```

That call **computes nothing yet**. At declaration time the MPI decomposition
does not exist, so it only records the coordinate dat and validates that block,
dat and `dim` agree. The real work is deferred to
`ops_particle_setup_partition()`, which per rank:

1. reads the **first and last owned node** of the local coordinate dat → a raw
   local `[xmin, xmax]`;
2. does an `MPI_Sendrecv` so each rank adopts its neighbour's `xmin` as its own
   `xmax` — this is what makes the local boxes **tile with no gap**
   (`ops_particle_box_mpi_funcs.h:152`). Without it, a particle landing between
   one rank's last node and the next rank's first node would be owned by nobody;
3. `MPI_Allreduce` MIN/MAX to form the global box.

Accessors: `getLocalMin()`, `getLocalMax()`, `getGlobalMin()`, `getGlobalMax()`
return an `ops_point<T>` with `.x/.y/.z`; `getLocalMaxMin(T *xmin, T *xmax)`
fills two plain arrays. A second constructor,
`ops_create_bounding_box(block, dim, region)`, takes an explicit
`[xmin,ymin,zmin,xmax,ymax,zmax]` when the box should not come from a grid.

### The mapping (bins)

```c
ops_particle_mapping map = ops_decl_mapping(eddy_parts, d_coords, S3D_27pt,
                                            OPS_WITH_VIRTUAL, OPS_UNIFORM_STAG, 1);
```

A mapping bins particles into grid cells and keeps a cell-linked list —
`map->binhead` (first particle in each cell) and `map->parts_to_grid` (which
cell each particle sits in). It is the machinery for *short-range* work:
neighbour searches, or reading a grid dat at a particle's location.

**This app never uses it** — the coupling is all-to-all, not short-range. But
every particle loop takes `map` as an argument, so it has to exist.

### Reductions, and the trick that replaces `MPI_Allgatherv`

A parallel loop writes per-element. To combine into one answer across every
element, thread and rank, you use a reduction — three calls:

```c
/* 1. declare a handle. First argument is TOTAL BYTES, not element count. */
ops_reduction h_cnt = ops_decl_reduction_handle(sizeof(int), "int", "eddy_count");

/* 2. pass it into a loop. The kernel receives a bare T* of length dim. */
ops_particle_par_loop(KerCountEddies, ..., ops_arg_reduce(h_cnt, 1, "int", OPS_INC));

/* 3. collect. Triggers the combine and copies the answer out. */
int count = 0;
ops_reduction_result(h_cnt, &count);
```

`OPS_INC` sums, `OPS_MIN`/`OPS_MAX` take extrema, `OPS_WRITE` keeps the last
non-zero. Two conveniences: OPS **zeroes the handle for you** when
`ops_arg_reduce` is evaluated (`ops_lib_core.cpp:966-971`) — on every rank,
including ones owning no particles, which keeps the collective balanced — and
`ops_reduction_result` performs an `MPI_Allreduce`, so **every rank gets the
answer**.

**The trick.** A reduction handle need not be scalar; it can be an array of any
length. Declare it as `eddies x 7` doubles, have each eddy add itself into the
slot picked out by its **global id**, and every rank contributes `0.0` in the
slots it does not own. Then `sum over ranks = the complete array` — an allgather
expressed as an allreduce. It works because `ops_reduction_result` reduces over
`OPS_MPI_GLOBAL` (`ops_mpi_rt_support.cpp:1285`), the global communicator rather
than the block's, so the result reaches every rank however the particles are
spread. Cost is `eddies x 7` doubles per step regardless of the split — 1869
doubles here, measured at well under 1% of a timestep.

### The call sequence

```
  ops_create_bounding_box(block, d_coords, 3, dx)
  ops_decl_particle(block, "eddies", box)
  ops_decl_particle_pos_dat + ops_decl_particle_dat x5
  ops_decl_mapping(parts, d_coords, stencil, ...)
  ops_decl_reduction_handle
  ops_partition("")
  ops_par_loop  KerPackCoords            <- fill d_coords FIRST
  ops_particle_setup_partition()
  seed_eddies()                          <- host: draw, test ownership, set no_particles
  ops_particle_setup_maps_with_dats()
  fill p_rnd -> ops_particle_par_loop KerInitEddy
  |
  +-- per timestep -----------------------------------------------+
  |   fill p_rnd                            keyed on eddy_id      |
  |   ops_particle_par_loop KerConvectEddies                      |
  |   ops_particle_par_loop KerPublishEddy  ops_arg_reduce INC    |
  |   ops_reduction_result(h_eddy, eddy_all)  <- MPI_Allreduce    |
  |   ops_par_loop Kernel030                ops_arg_gbl(eddy_all) |
  +---------------------------------------------------------------+
```

The ordering is not stylistic. Several of those steps are hard constraints.

### Phase 1 — declare, before `ops_partition`

Metadata only; no memory is decomposed and no particle exists yet.

| call | what it does here |
|---|---|
| `ops_create_bounding_box(block, d_coords, 3, dx)` | records the region particles may occupy; validates only |
| `ops_decl_particle(block, "eddies", box)` | creates the set; gives `no_particles`, `Nmax`, `box_block` |
| `ops_decl_particle_pos_dat(parts, 3, base, NULL, "double", "position")` | the position dat OPS reads itself. Exactly one per set |
| `ops_decl_particle_dat(parts, dim, base, NULL, type, name)` | `p_e`(3), `p_r`(1), `p_eps`(3), `p_id`(1 int), `p_rnd`(6) |
| `ops_decl_mapping(...)` | the bins; required by the loop signature |
| `ops_decl_reduction_handle(nred*sizeof(double), "double", "eddy_all")` | the gather buffer, `NCOMP = 7` doubles per eddy |

**Constraint — position dimension.** `ops_decl_particle_pos_dat` throws unless
`data_size == block->dims` (`ops_particles_lib_core.h:762`). On a 3-D block the
position must be 3-D. The 2-D reference's arrangement — position `(y,z)`, `x` as
an ordinary dat — is therefore unavailable, and `ACC` is compiled for one
dimensionality per translation unit, so a 3-D app cannot host a 2-D block
either.

**Constraint — one block only.** Particles go on the **solver's own block**.
With more than one block OPS partitions the *ranks between blocks*:
`ops_mpi_partition.cpp:152-154` gives each block `nproc/nblocks` of them and
leaves `sb->comm == MPI_COMM_NULL` on the rest. A dedicated eddy block would
take half the ranks away from the solver, and it also crashes
`partitionBoundingBox`, which reaches for `sb->comm`
(`ops_particle_box_mpi_funcs.h:152`) without checking `sb->owned` — measured, at
np = 2, as `MPI_ERR_COMM` on rank 0 while rank 1 seeded all 267 eddies.

### Phase 2 — partition and populate

**Fill the coordinate dat first.** The box is derived from `d_coords`, so the
grid must be in it before the box is partitioned; get this order wrong and the
box is built from uninitialised memory.

```c
ops_par_loop(KerPackCoords, "KerPackCoords", opensbliblock00, 3, coord_range,
             ops_arg_dat(d_coords, 3, S3D_000, "double", OPS_WRITE),
             ops_arg_dat(x0_B0, 1, S3D_000, "double", OPS_READ),
             ops_arg_dat(x1_B0, 1, S3D_000, "double", OPS_READ),
             ops_arg_dat(x2_B0, 1, S3D_000, "double", OPS_READ));
```

**`ops_particle_setup_partition()`** — one call, four jobs, for every particle
set in the run: build the per-rank bounding box from the coordinate dat,
partition the walls, initialise the maps, set up intra-block communicators.

**`seed_eddies()`** is host code and has to be. A kernel iterates over particles
that already exist and cannot change the count — but which eddies a rank owns is
decided by *where they are*. So position must be settled before the set exists.

```c
Real u[3] = {ops_prandom_uniform(seed, i, 0u, 0), ...};   /* pure fn of GLOBAL id */
Real e[3];  e[0] = eddy_x_min + (eddy_x_max - eddy_x_min) * u[0];  ...

for (int d = 0; d < 3; d++) {                        /* clamp into the block   */
  q[d] = e[d] < glo[d] ? glo[d] : (e[d] > ghi[d] ? ghi[d] : e[d]);
  if (q[d] < lo[d] || q[d] >= hi[d]) outside = 1;    /* half-open: no double claim */
}
if (outside) continue;

xp[3*n+0] = q[0]; ...        /* p_pos — the ownership handle        */
ep[3*n+0] = e[0]; ...        /* p_e   — the eddy's real coordinates */
ip[n] = i;                   /* p_id  — the global eddy id          */
n++;
...
particle->no_particles = n;  /* THIS is what makes the particles exist */
```

The test is **half-open** at the top because the local boxes share a face after
the Sendrecv step; a closed test would let two ranks claim the same eddy.
`ops_prandom_uniform` is a pure function of the global id, so every rank
computes the identical position for eddy *i* — which is why the run is
bit-identical at any rank count. Call `ops_particle_realloc_data(particle,
neddy)` first if `neddy > particle->Nmax`.

*Gotcha:* `getLocalMaxMin` begins `if (!this->owned) return;` and leaves your
arrays **untouched** on a rank owning no part of the block. Initialise them to
an empty interval (`lo = +1e30, hi = -1e30`) first, or you test against stack
garbage.

**`ops_particle_setup_maps_with_dats(parts, dat_border, nborder)`** builds the
cell-linked list over the particles that now exist. `dat_border` lists the dats
that would travel with a particle during a border exchange.

#### Why the position dat is only an ownership handle

The eddy's coordinates ride in `p_e`, not in `p_pos`. They have to, because the
eddy box is not inside the fluid block: it overhangs on all three low faces — x,
y and z all start at `-radius` while the block spans `[0,375] x [0,100] x
[0,40]` — and it is wider than the domain in z, 44.68 against 40. The bounding
box is exactly `[first node .. last node]`, so an eddy at its true coordinates
would often fall outside every rank's box, and outside means owned by nobody.

`p_pos` is therefore the true position **clamped** into the block. That moves an
eddy by at most one radius and only in the overhang, and hands each eddy to the
rank holding the piece of inlet it actually influences. Since the eddy box is a
thin slab at `x ~ 0`, the eddies land on the ranks that own the inlet plane,
which is where `Kernel030` does its work. Ownership is settled once and never
migrates, so `p_pos` is read exactly once, by OPS, and nothing reads it after.

### Phase 3 — run

```c
ops_particle_par_loop(KerConvectEddies,    /* the kernel function          */
                      "convect_eddies",    /* name, for diagnostics        */
                      eddy_parts,          /* the particle set             */
                      3,                   /* dimensionality               */
                      OPS_PARTICLE_ITERATE_LOCAL,  /* which particles      */
                      eddy_region,         /* region (ITERATE_RANDOM only) */
                      map,                 /* the mapping                  */
                      ops_arg_dat_particle(p_e,   3, "double", eddy_parts, map, OPS_RW),
                      ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_RW),
                      ops_arg_dat_particle(p_rnd, 6, "double", eddy_parts, map, OPS_READ));
```

| iteration type | iterates over |
|---|---|
| `OPS_PARTICLE_ITERATE_LOCAL` | owned particles only; count is `particle->no_particles`. Every loop here uses it |
| `OPS_PARTICLE_ITERATE_ALL` | owned *and* virtual (ghost) particles |
| `OPS_PARTICLE_ITERATE_RANDOM` | particles inside the region argument — the only mode that reads it |

Because every loop is `ITERATE_LOCAL`, `eddy_region` is never read
(`ops_particle_seq.h:858-865`); it is passed because the signature demands it.

| argument constructor | kernel parameter | purpose |
|---|---|---|
| `ops_arg_dat_particle(dat, dim, type, parts, map, acc)` | `ACCP<T>&` | a per-particle field |
| `ops_arg_dat(dat, dim, stencil, type, acc)` | `ACC<T>&` | a grid field |
| `ops_arg_gbl(ptr, dim, type, acc)` | `const T*` | a whole host array, identical for every element |
| `ops_arg_reduce(handle, dim, type, acc)` | `T*` | reduction target — write into it, OPS combines |
| `ops_arg_idp()` | `const int*` | the particle's index. Unused here — the id lives in `p_id` |
| `ops_arg_idx()` | `const int*` | grid loops: the node's global `(i,j,k)` |

`ops_arg_gbl` and `ops_arg_reduce` look similar and are not: `gbl` broadcasts
host data *in*, `reduce` collects results *out*. The eddy buffer uses `reduce`
to fill it in the particle loop, then `gbl` to feed it to `Kernel030`.

**The gather.** Must be `ITERATE_LOCAL`: a ghost particle carries its owner's id
and under `ITERATE_ALL` would be added into the same slot twice.

```c
ops_particle_par_loop(KerPublishEddy, "publish_eddies", eddy_parts, 3,
                      OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
                      ops_arg_dat_particle(p_e,   3, "double", eddy_parts, map, OPS_READ),
                      ops_arg_dat_particle(p_r,   1, "double", eddy_parts, map, OPS_READ),
                      ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_READ),
                      ops_arg_dat_particle(p_id,  1, "int",    eddy_parts, map, OPS_READ),
                      ops_arg_reduce(h_eddy, nred, "double", OPS_INC));
ops_reduction_result(h_eddy, eddy_all);   /* MPI_Allreduce over OPS_MPI_GLOBAL */
```

### The kernels

`ACC<T>` is the grid accessor — `c(component, i, j, k)`. `ACCP<T>` is the
particle accessor — `p(component)` only. It is already positioned on the current
particle, because a particle has a position, not a grid index.

| kernel | kind | when |
|---|---|---|
| `KerPackCoords` | grid, `ACC<T>` | setup, once |
| `KerInitEddy` | particle | once, after seeding |
| `KerConvectEddies` | particle | every step |
| `KerPublishEddy` | particle + reduction | every step — **this is the gather** |
| `KerCountEddies` | particle + reduction | every step — population check |

```c
/* the three coordinate dats packed into the one dim-3 dat the box needs */
void KerPackCoords(ACC<double> &c, const ACC<double> &x0,
                   const ACC<double> &x1, const ACC<double> &x2) {
  c(0,0,0,0) = x0(0,0,0);  c(1,0,0,0) = x1(0,0,0);  c(2,0,0,0) = x2(0,0,0);
}

/* instantiate_eddies, minus the position seed_eddies has already set */
void KerInitEddy(ACCP<double> &r, ACCP<double> &eps, const ACCP<double> &rnd) {
  r(0) = radius;
  eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
}

/* convect_eddies, statement for statement */
void KerConvectEddies(ACCP<double> &e, ACCP<double> &eps, const ACCP<double> &rnd) {
  e(0) = e(0) + increment;
  if (e(0) > eddy_x_max) {
    e(0) = eddy_x_min;
    e(1) = eddy_y_min + rnd(1) * (eddy_y_max - eddy_y_min);
    e(2) = eddy_z_min + rnd(2) * (eddy_z_max - eddy_z_min);
    eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
    eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
    eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
  }
}

/* the gather. `all` is a plain double*, not an accessor — that is what an
   ops_arg_reduce argument looks like inside a kernel */
void KerPublishEddy(const ACCP<double> &e, const ACCP<double> &r,
                    const ACCP<double> &eps, const ACCP<int> &id, double *all) {
  const int s = NCOMP * id(0);
  all[s + E_X] += e(0);    all[s + E_Y]  += e(1);    all[s + E_Z]  += e(2);
  all[s + E_R] += r(0);
  all[s + E_SX] += eps(0); all[s + E_SY] += eps(1);  all[s + E_SZ] += eps(2);
}

void KerCountEddies(const ACCP<int> &id, int *count) { (void)id; *count += 1; }
```

The eddy's `e(0)` is a **phase, not a place**: it enters the shape function as
`exp(-0.5*(x/r)^2)`, so it modulates the eddy's strength. Crossing the box takes
`2*radius/increment = 187` steps, which is the eddy's lifetime and sets the time
correlation of the synthetic turbulence. That is also why refining the
streamwise grid does nothing for the eddies — nothing about them is resolved in
x.

### Randoms

There is no OPS generator a kernel can call — the whole public API fills a whole
`ops_dat` from the host. So the structure is fill-then-read, which is what
`../oSEM_3d` does too:

```c
ops_fill_random_uniform_particle(eddy_parts, p_rnd, p_id, seed_gbl,
                                 (unsigned int)(iter - start_iter) + 2u, OPS_PRNG_MINSTD);
```

This is *app* code (`ops_particle_random.h`), not library code.
`ops_fill_random_uniform` cannot be used: it throws on a particle dat, and its
stream is keyed on **storage slot**, which changes when a list is compacted. The
fill here is keyed on `(seed, global id, counter)`, so an eddy's randomness
belongs to the eddy rather than to whichever rank holds it. That is what makes
the field rank-invariant rather than merely complete.

### Function reference

| function | phase | one line |
|---|---|---|
| `ops_create_bounding_box` | declare | records the physical region; partitioning deferred |
| `ops_decl_particle` | declare | creates a particle set on a block |
| `ops_decl_particle_pos_dat` | declare | the position dat. Dim must equal block dims |
| `ops_decl_particle_dat` | declare | a per-particle field, `Nmax` slots x `dim` |
| `ops_decl_mapping` | declare | cell-linked list binning particles into grid cells |
| `ops_decl_reduction_handle` | declare | a reduction target. First arg is total *bytes* |
| `ops_particle_setup_partition` | setup | per-rank boxes, maps and comms. Needs the coordinate dat filled |
| `ops_particle_realloc_data` | setup | grows storage past `Nmax` |
| `ops_particle_setup_maps_with_dats` | setup | builds the bins over the particles that now exist |
| `ops_particle_par_loop` | run | runs a kernel over the particle set |
| `ops_arg_dat_particle` | run | passes a particle dat → `ACCP<T>&` |
| `ops_arg_reduce` | run | passes a reduction handle → `T*` |
| `ops_reduction_result` | run | combines across ranks (`MPI_Allreduce`) and copies out |
| `getLocalMaxMin` | host | this rank's box bounds. No-op if the rank owns nothing |
| `particle->no_particles` | host | owned count. Setting it is what makes particles exist |

### Constraints worth remembering

1. **Position dimension must equal block dimension**, or `ops_decl_particle_pos_dat` throws.
2. **Particles go on the solver's block.** Extra blocks divide the ranks, they do not share them.
3. **Fill the coordinate dat before `ops_particle_setup_partition`**, or the box is built from garbage.
4. **Set `no_particles` on the host.** A kernel cannot create particles.
5. **Reductions go over the global communicator**, so the gather reaches every rank even where the particle set is thin.
6. **`getLocalMaxMin` leaves its outputs untouched** on an idle rank. Initialise to an empty interval first.
7. **Particle slots are not stable.** Key anything eddy-specific on `p_id`, never the slot.
8. **The translator wants a bare integer literal** for an `ops_arg_reduce` dimension in a *grid* loop. Particle loops are not parsed by it and take any expression.

## Re-injection: the teleport is kept, ownership is by id

`convect_eddies` is `../oSEM_3d`'s statement for statement — advance x, and if
the eddy has left the box put it back at the inlet with a fresh y, z and signs.
That last part is a jump to an arbitrary point in the box, which OPS particle
migration cannot express: migration only hands a particle to a *neighbouring*
rank, and the 2-D reference measured exactly this failing from three ranks up.

The reference's answer was to change the model — give the eddy a transverse
velocity so it drifts instead of jumping. This app keeps the teleport and drops
the migration instead: **ownership is by `eddy_id`, fixed at seeding**. The
gather sums by global id, so it stays complete and exact whatever the
coordinates do, and the physics stays `../oSEM_3d`'s exactly.

What that costs: the eddy set stops being spatially meaningful after the first
recycle — the rank holding an eddy is no longer the rank whose subdomain it sits
in, and the bins built at startup go stale with it. Nothing here reads either.
`Kernel030` needs the whole list on every rank, not a neighbour search, and a
loop of particle dats, `ops_arg_gbl`, `ops_arg_idp` and `ops_arg_reduce` never
touches the mapping — `init_off_grids` and `compute_offsets`
(`ops_particle_seq.h:520,551`) dereference `map->parts_to_grid` only for
`OPS_ARG_DAT`. **If a later stage needs ownership to track position, that is the
point at which to add a re-cull, and it belongs there rather than here.**

## Two deliberate departures from `../oSEM_3d`

**The eddy signs.** There the test is `(rng < 0) ? -1 : 1` on an `int` dat
filled by `ops_fill_random_uniform`, which draws from
`uniform_int_distribution<int>(0, INT_MAX)` (`ops_lib_core.cpp:2604`) and is
never negative — so every sign comes out `+1`, and the position draws, built the
same way, fill an eighth of the box. This app draws doubles in `[0,1)` and
thresholds at `0.5`, which sidesteps the question. The randoms come from
`ops_particle_random.h`, taken unchanged from the 2-D reference: counter-based
and keyed on the **global eddy id**, so an eddy's stream belongs to the eddy and
not to whichever rank holds it. That is what makes the gather rank-invariant
rather than merely complete.

**`y_cutoff`.** Hardwired to 150 there, which is both the length of the
`a11..a33` profiles and the y extent of the `interp_RST` loop; at a reduced grid
that range runs off the block, so it is capped at `block0np1`. At the full
250-point grid it is 150 as before. The profiles are also declared with
`y_cutoff` entries rather than `ndata`, which is what they were allocated with.

## Why `../oSEM_3d` stays up at a grid where this app does not

At np = 1, 150x60x64, 1000 iterations, everything else equal:

| run | result |
|---|---|
| `../oSEM_3d` | 1000/1000 clean |
| this app, `-neddies 0` | 1000/1000 clean |
| this app, `-degenerate-rng` | 1000/1000 clean |
| this app, normal | **NaN at iteration 858**, at `(3, -2, 45)` |

That is not a defect in the port. `../oSEM_3d`'s own
`convect_eddies_rank0.txt` dump (10000 samples) shows its eddies are degenerate:
`y` in `[5.95, 14.02]` out of a box spanning `[-2.34, 14.04]`, `z` in
`[20.7, 42.2]` out of `[-2.34, 42.34]` — the upper half of each — and
`eps_x = eps_y = eps_z = +1` for **every** eddy. Cause is the
`ops_fill_random_uniform` int truncation described above: `rng >= 0` always, so
`(rng + 2^31)/(2^32-1)` lands in `[0.5, 1]` and `(rng < 0) ? -1 : 1` is always
`+1`. It therefore never forces the near-wall region, never forces half the
span, and its one-signed eddies add coherently into the `|u'| <= 0.2` clamp — a
smooth saturated offset rather than turbulence.

`-degenerate-rng` is the controlled test. It maps a correct uniform
`u -> 0.5 + 0.5u`, which is exactly what that arithmetic does, and changes
nothing else — same particles, same gather, same `Kernel030`. Under it this app
reproduces `../oSEM_3d`'s distribution (mean eddy `y` 9.87 and `z` 31.3 against
its measured 9.99 and 31.5, `eps` sum `+801` = 267 x 3 x (+1)) and survives the
full 1000 steps. So the blow-up tracks the eddy **distribution and sign
statistics** alone, not the particle, gather or coupling machinery.

What that leaves open: nobody has run this solver with working synthetic
turbulence at any resolution, so the blow-up is the first real test of the
solver and its wall boundary condition rather than a regression. Whether the
design grid 750x250x150 survives is untested here. One specific suspect before
blaming resolution: `../oSEM_3d`'s eddy box pads **below the wall**
(`eddy_y_min = 0 - radius`), so eddies sit at `y < 0` and force `v'` at a solid
surface; the 2-D `../oSEM` deliberately pads only the top.

## Ported verbatim, including one thing that looks wrong

`Kernel030`'s spanwise mirror computes `ztildesq_ghost` **without dividing by
`r^2`**, while the `ztildesq` it replaces in `rsq` is divided by it
(`opensbliblock00_kernels.h`, the `else` branch of the eddy loop). The two are
then mixed in `rsq = rsq - ztildesq + ztildesq_ghost` and `ztildesq_ghost` goes
into the shape function unnormalised.

It is ported unchanged. Fixing physics silently would destroy the only thing
that makes this app checkable against `../oSEM_3d` — that the two differ in the
eddy transport and the gather, and in nothing else. Worth deciding on
separately.

## Notes on the translator

Two things the newer translator (`Makefile.c_app`) will not accept that the
legacy one did:

- `ops_decl_const("uinterp", ny, "double", &uinterp[0])` — `&array[0]` fails to
  parse. Written as `uinterp`.
- an `ops_arg_reduce` dimension that is not a bare integer literal. `ny` and
  `4*y_cutoff` are runtime values, so the two profile reductions are declared at
  a fixed capacity (`UINTERP_CAP`, `RST_CAP` in `osem3d_common.h`) and only the
  first `ny` / `4*y_cutoff` slots are used. A macro does not work either:
  `parseIntLiteral` (`ops-translator/cpp/parser.py:188`) wants a single token,
  so the call sites spell the number out and a `static_assert` keeps it in step
  with the macro.

Grid kernel bodies are inlined by the translator into generated files that
include no app header, so `Kernel030` indexes the gather buffer with literals
(`e[0]`..`e[6]`) rather than the `E_*` names. The particle kernels are C++
templates compiled host-side and keep the names.

## Looking at it in ParaView

The solver writes `opensbli_output_%06d.h5` every `-nout` steps.
`make_xdmf.py` turns those into something ParaView opens directly:

```bash
mkdir run_paraview && cd run_paraview && cp ../opensbli_dev_mpi .
OMP_NUM_THREADS=1 mpirun -np 4 ./opensbli_dev_mpi -ngrid 150 60 64 -niter 2000 -nout 200
cd .. && python3 make_xdmf.py -i run_paraview
paraview run_paraview/paraview/osem3d.xmf
```

The raw frames are not directly plottable, for two reasons the script deals
with. Every dat is written **with its 5-cell halo on all six faces**, so a
dataset is `(np2+10, np1+10, np0+10)` and plotting it wraps the domain in a
shell of boundary-condition scratch; the halo is trimmed. And the solver stores
momentum, so `u = rhou/rho` is derived here and written as a vector, which is
what ParaView needs for glyphs and streamlines. Output carries `u0 u1 u2 umag
rho p T` plus a `velocity` vector, on the real curvilinear mesh — the grid is
sinh-stretched in y, so the geometry uses the stored `x0/x1/x2` rather than a
uniform box. `--delete-raw` drops each raw frame once converted, which roughly
halves the disk cost.

**What you will actually see.** The eddies drive the inlet boundary
(`Kernel030`, at x-index -2..0), so the synthetic turbulence is generated on the
inlet plane and then convects downstream. The domain is 375 long and the flow
covers `u0*dt` = 0.025 per step, so 2000 steps fills only the first ~50 units —
about a fifth of the box at `-ngrid 150 ...`. The thing worth looking at first
is a **Slice at x = 0** coloured by `u1` or `u2`: that is the SEM field itself,
and `u1`/`u2` are zero everywhere except where the eddies put them. For the
downstream development, expect to run O(10^4) steps.

## Files

| File | |
|---|---|
| `opensbli.cpp` | driver |
| `eddy_kernels.h` | the particle kernels: init, convect, publish, count |
| `osem3d_common.h` | gather buffer layout and the reduction capacities |
| `ops_particle_random.h` | gid-keyed RNG fill, unchanged from the 2-D reference |
| `make_xdmf.py` | HDF5 frames -> trimmed data + XDMF time series for ParaView |
| `opensbliblock00_kernels.h` | `../oSEM_3d`'s, with `Kernel030` reindexed and the two profile kernels given a reduction argument |
| `constants.h`, `defdec_data_set.h`, `stencils.h`, `bc_exchanges.h`, `io.h`, `Pir_data.h`, `TBL_data.h` | `../oSEM_3d`'s |

## What to check when it breaks

`eddy particles owned: N of 267` at startup is the seeding check — the local
boxes are supposed to tile the block, and anything less than 267 means an eddy
fell in a gap. `missing 0` on each gather line is the same question asked of the
buffer: a zero radius is a slot no rank wrote. The `sum x/y/z` values must agree
digit for digit between rank counts; if they do not, the gather is the suspect,
not the solver.

## Appendix — instantiating globally, then culling

This app does **not** use this approach. It is recorded only as a reference,
because it is the structure a reader coming from `../oSEM_3d` is likely to reach
for first. Instead of deciding ownership on the host, every rank instantiates
the whole eddy list in a kernel and then deletes the ones it does not own. Its
one real appeal is that the placement arithmetic then lives in a kernel, making
it a literal counterpart of `instantiate_eddies`.

```c
if (eddies > (int)eddy_parts->Nmax) ops_particle_realloc_data(eddy_parts, eddies);
eddy_parts->no_particles = eddies;      /* every rank, the whole list, in id order */

ops_fill_random_uniform_particle(eddy_parts, p_rnd, p_id, seed_gbl, 1u, OPS_PRNG_MINSTD);

/* 1. what an eddy IS -- runs on all N, writes position and everything else */
ops_particle_par_loop(KerInstantiateEddies, "instantiate_eddies", eddy_parts, 3,
                      OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
                      ops_arg_dat_particle(p_pos, 3, "double", eddy_parts, map, OPS_WRITE),
                      ops_arg_dat_particle(p_r,   1, "double", eddy_parts, map, OPS_WRITE),
                      ops_arg_dat_particle(p_eps, 3, "double", eddy_parts, map, OPS_WRITE),
                      ops_arg_dat_particle(p_id,  1, "int",    eddy_parts, map, OPS_WRITE),
                      ops_arg_dat_particle(p_rnd, 6, "double", eddy_parts, map, OPS_READ),
                      ops_arg_idp());                 /* <- the particle index */

/* 2. who KEEPS it -- the library's own bounds, so this cannot disagree with OPS */
double box_lo[OPS_MAX_DIM], box_hi[OPS_MAX_DIM];
box->getLocalMaxMin(box_lo, box_hi);
ops_particle_par_loop(KerMarkUnowned, "mark_unowned", eddy_parts, 3,
                      OPS_PARTICLE_ITERATE_LOCAL, eddy_region, map,
                      ops_arg_dat_particle(p_del, 1, "int",    eddy_parts, map, OPS_WRITE),
                      ops_arg_dat_particle(p_pos, 3, "double", eddy_parts, map, OPS_READ),
                      ops_arg_gbl(box_lo, 3, "double", OPS_READ),
                      ops_arg_gbl(box_hi, 3, "double", OPS_READ));

/* 3. hand the kernel's verdict to the library and compact */
memcpy(eddy_parts->mark_deletion, p_del->data,
       eddy_parts->no_particles * sizeof(int));
ops_particle_remove_particles(eddy_parts, true);

ops_particle_setup_maps_with_dats(eddy_parts, dat_border, nborder);
```

```c
void KerInstantiateEddies(ACCP<double> &pos, ACCP<double> &r, ACCP<double> &eps,
                          ACCP<int> &id, const ACCP<double> &rnd, const int *idp) {
  pos(0) = eddy_x_min + rnd(0) * (eddy_x_max - eddy_x_min);
  pos(1) = eddy_y_min + rnd(1) * (eddy_y_max - eddy_y_min);
  pos(2) = eddy_z_min + rnd(2) * (eddy_z_max - eddy_z_min);
  r(0)   = radius;
  eps(0) = (rnd(3) < 0.5) ? -1.0 : 1.0;
  eps(1) = (rnd(4) < 0.5) ? -1.0 : 1.0;
  eps(2) = (rnd(5) < 0.5) ? -1.0 : 1.0;
  id(0)  = idp[0];      /* capture the global id BEFORE the cull reorders slots */
}

void KerMarkUnowned(ACCP<int> &del, const ACCP<double> &pos,
                    const double *xmin, const double *xmax) {
  del(0) = 0;
  for (int d = 0; d < 3; d++)
    if (pos(d) < xmin[d] || pos(d) > xmax[d]) del(0) = 1;
}
```

**Why `idp[0]` is the global eddy id here.** `ops_arg_idp()` gives the kernel the
particle's *index*, normally just a local slot. But this loop runs **before the
cull**, at the one moment when every rank holds the whole list in id order — so
index and global id coincide. That window is also why `id(0) = idp[0]` must be
captured now: the moment `ops_particle_remove_particles` compacts the list,
slots shift and the correspondence is gone. A side effect worth knowing: during
that same window a slot-keyed random fill gives identical values on every rank,
because slot *is* the id. Per-step fills after the cull must still be id-keyed.

### Why host seeding was preferred

| | host seeding (used here) | global instantiate + cull |
|---|---|---|
| placement arithmetic lives in | host code | **a kernel** — matches `instantiate_eddies` one for one |
| memory and work at setup | proportional to the **owned** share | every rank allocates and writes **all N**, then throws most away |
| scales to large N? | yes | poorly — fine at 267 eddies, wrong at 10^6 |
| host code still required? | the whole seeding loop | yes — the `memcpy` into `mark_deletion` |
| ownership decision is | one host loop | split across a kernel and a host `memcpy` |
| extra state | none | a `p_del` particle dat |

Host seeding wins on every line that matters at scale: work and memory stay
proportional to what a rank actually owns, the ownership decision is one
readable loop rather than a kernel plus a host `memcpy`, and it needs no extra
particle dat. The cull approach only buys a cosmetic resemblance to
`instantiate_eddies`, and pays for it by having every rank build all N eddies
and discard most of them.

### Two things to get right, if you do use it

**Cull before the maps exist.** Use `ops_particle_remove_particles`, which
compacts the list and leaves the cell-linked list alone — correct here, because
`ops_particle_setup_maps_with_dats` has not run yet and builds the bins from
scratch afterwards. The other removal path,
`ops_particle_remove_delete_maps`, also repairs the maps and belongs in the
migration cycle, not here.

**Use the library's own bounds.** `getLocalMaxMin` is what the migration
machinery itself uses, so a cull written against it cannot disagree with OPS
about where a subdomain ends. Note it also returns without writing on a rank
that owns nothing — initialise the arrays first.

There is a call that would fold steps 2 and 3 together,
`ops_particle_user_delete` in `ops_particle_insert_del.h`, but that header does
not compile as shipped: line 108 declares
`BoundingBox *boxBlock = particle->box_block;` — `BoundingBox` is a class
template (`ops_bounding_box.h:67`) used without template arguments, and
`box_block` is a `char*`. Hence the explicit `memcpy`.
