# oSEM_3d eddy initialisation, as OPS particles

The `instantiate_eddies` kernel of [`../oSEM_3d`](../oSEM_3d), with each eddy an
**OPS particle** instead of a row in an `ops_dat` on the fluid block.

**Scope: initialisation only.** Instantiate the eddies in the domain, check the
field is what it should be, write it out and look at it. Convection, recycling,
the fluctuation field and the coupling to the inlet BC all come later.

## Why particles

In `../oSEM_3d` the ten eddy `ops_dat`s have shape `{eddies,1,1}` and are
declared on `opensbliblock00` (`defdec_data_set.h:625-715`), so MPI decomposes
them by the *fluid* partitioning — an eddy list sliced along x by a
decomposition that has nothing to do with eddies. `Kernel030` then needs the
whole list on every rank and receives it through seven
`ops_arg_gbl(..., OPS_READ)` arrays, which OPS does not communicate, so the app
carries a hand-rolled `MPI_Allgatherv` over raw `MPI_COMM_WORLD`
(`opensbli.cpp:456-531`) every timestep.

As particles, an eddy is owned by the rank whose subdomain it occupies.

## Status

| | dev_seq | seq | np=1 | np=2 | np=4 | np=8 |
|---|---|---|---|---|---|---|
| initialisation | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |

All four targets build. The eddy field is **bit-identical at every rank count**:
the HDF5 output from `mpirun -np 8`, sorted by `eddy_id`, compares bit-equal to
the serial output in both position and sign.

## Build and run

```bash
make oSEM_3d_particles_dev_seq && ./oSEM_3d_particles_dev_seq   # start here
make oSEM_3d_particles_dev_mpi && mpirun -np 4 ./oSEM_3d_particles_dev_mpi
make oSEM_3d_particles_seq                                      # via translator
make oSEM_3d_particles_mpi

python3 plot_osem3d_h5.py      # -> frames/*.png
```

Options: `-ngrid NX NY NZ`, `-seed N`, `-ops-rng`, `-noh5`.

## What the app does

```
delta  = 11.6973525411      radius = 0.2*delta = 2.33947   (delta == 5*radius)
eddy_x in [-radius,       radius]     a slab 2 radii thick across the inlet plane
eddy_y in [-radius, delta+radius]
eddy_z in [-radius, 40.0 +radius]
eddy_vol = 3423.5           eddies = trunc(vol/radius^3) = 267
```

`KerInstantiateEddies` in `eddy_kernels.h` is the direct analogue of
`../oSEM_3d/opensbliblock00_kernels.h:39-48` — a position in the box, three
random signs, the radius and the per-step convection increment. `oSEM_3d`'s
`eddy_x/y/z` become the particle position dat, `eddy_r`, `eddy_increment` and
`eddy_eps_x/y/z` become ordinary particle dats.

`../oSEM_3d` runs that kernel as an `ops_par_loop` over an iteration range of
`eddies` (`opensbli.cpp:313-323`); here it is an `ops_particle_par_loop` over
`eddies` particles. Nothing about the instantiation is host code.

## How one kernel comes to instantiate particles that do not exist yet

An `ops_particle_par_loop` iterates over particles that already exist, and a
kernel cannot change the particle count. So the sequence is create, write, cull:

```c
eddy_parts->no_particles = eddies;          /* every rank, the whole list */

ops_particle_par_loop(KerInstantiateEddies, ..., ops_arg_idp());

ops_particle_par_loop(KerMarkUnownedEddies, ..., box_lo, box_hi);
memcpy(eddy_parts->mark_deletion, p_del->data, ...);
ops_particle_remove_particles(eddy_parts, true);
```

**`ops_arg_idp()`** (`ops_particle_lib_core.cpp:2466`) hands the kernel the
particle index, and because the loop runs before the cull — when every rank
holds the whole list in order — that index *is* the global eddy index. It is
what the random stream is a function of, and `eddy_id` records it so it survives
the cull and every later migration.

**The random stream is inside the kernel.** `oSEM_3d` fills two `int` `ops_dat`s
with `ops_fill_random_uniform` and has the kernel read them; that is a stateful
generator run outside the kernel, and it is where its defects live (finding 1).
A counter-based hash of `(eddy index, draw index)` needs no dat and no state, so
it can live in the kernel and gives the same eddy the same draw on every rank.

`-ops-rng` runs the alternative — `ops_fill_random_uniform` into a `double`
particle dat, read by `KerInstantiateEddiesOpsRng` — to show why the hash is
there. It is a demonstration, not a supported mode: see "There is no RNG a
kernel can call" and "`ops_randomgen_init` cannot produce a rank-independent
stream" below. It passes in serial and diverges at every rank count above one.

**Ownership is a second kernel**, `KerMarkUnownedEddies`, applying
`BoundingBox::getLocalMaxMin()` — the library's own subdomain bounds, so this app
cannot disagree with the migration machinery about where a rank ends. That split
— one kernel for what an eddy *is*, one for who keeps it — is the same one OPS
makes between the insert kernel and the decide kernel of
`ops_particle_insert`.

The `memcpy` is the only host line in the sequence and it decides nothing; see
"Findings in the OPS particle API" for why `ops_particle_user_delete()` cannot
replace it.

No `ops_particle_par_loop` argument here touches the mapping: `init_off_grids`
and `compute_offsets` (`ops_particle_seq.h:520,551`) dereference
`map->parts_to_grid` only for `OPS_ARG_DAT` — grid dats — so a loop of particle
dats, `ops_arg_gbl` and `ops_arg_idp` runs correctly before any map is built.

## The eddies get their own block

Two properties of `oSEM_3d` force it.

**The eddy box overhangs the fluid domain on all three low faces** — x, y and z
all start at `-radius`, while the fluid block spans `[0,375] x [0,100] x [0,40]`.
The bounding box is derived from a coordinate `ops_dat` and is exactly
`[first node .. last node]`, so a fluid-derived box would put every eddy at
`x<0`, `y<0` or `z<0` outside it, and outside means deleted.

**The fluid wall-normal grid is sinh-stretched** (`opensbli.cpp:15`), which the
uniform mapping used here does not model. `OPS_NON_UNI_STAG` exists for that and
the coupling will have to use it.

So: a dedicated uniform 3-D block spanning exactly the eddy box, with a genuine
`(x, y, z)` position dat. Cells are half an eddy radius, giving 5 × 15 × 40
nodes. Nothing is solved on this grid; it defines the bounding box and the bins.

## Checking the initialisation

`report_eddy_field()` asks four questions, all properties of the eddy **list** —
no shape function, no Reynolds stresses, no physics:

- is the eddy count consistent with the box (`N = trunc(V/r³)`)?
- is every eddy inside the box?
- are the positions uniformly distributed over it?
- are the signs balanced and independent?

Two things about how, both learned the hard way:

**Uniformity is chi-square, not KS.** Chi-square needs only bin *counts*, which
reduce with one `MPI_SUM`. KS needs the globally sorted sample — a position
gather, and a different answer at every rank count. A diagnostic whose verdict
depends on the decomposition is worse than no diagnostic.

**The tests are two-sided, and there is a joint one.** A chi-square far *below*
its degrees of freedom is as damning as one far above: it means the points are
spread more evenly than chance allows, the signature of a lattice. And a lattice
has perfectly uniform *marginals*, so per-axis tests pass on one — only a joint
test in (y,z) sees it. An index-seeded LCG scores `chi2 = 5.9` against `df = 52`
per axis while its joint statistic is 370.

The sign tolerance is derived rather than picked: the +1 fraction of `n` fair
draws has `sigma = 0.5/sqrt(n)`, which is 3.06% at 267 eddies. The obvious
45–55% band is only ±1.63 sigma there and rejects **29% of correct seeds**
(measured over 4000). Four sigma leaves a per-component false-failure rate near
6e-5 and still rejects an all-one-sign field, which sits 16 sigma out.

---

## Findings in the OPS particle API

Turned up while moving the instantiation into a kernel. All are in the library,
not in this app.

### `ops_particle_insert_del.h` does not compile

Line 108, inside `ops_particle_insert_impl`, reads

```c
BoundingBox *boxBlock = particle->box_block;
```

`BoundingBox` is a class template (`ops_bounding_box.h:63`) and `box_block` is a
`char *` (`ops_particles_lib_core.h:145`), so the declaration is wrong twice
over and the header cannot be included at all. That takes
**`ops_particle_user_delete()`** — the kernel-driven deletion this app wants —
down with it, even though its own template is fine. Hence the `memcpy` into
`mark_deletion`. `LBM-PSM`, `LBM-PSM-Faxen` and `particle_dbg` all include this
header and so cannot currently build.

### `ops_arg_idp()` is inert inside `ops_particle_insert`

`find_offset` (`:70-88`) returns 0 for `OPS_ARG_IDP` and
`particle_param_handler::construct` zeroes it, so an insert kernel always sees
index 0. It carries the particle index only in `ops_particle_par_loop`, which is
the reason this app instantiates with a particle loop rather than with
`ops_particle_insert`.

### `ops_arg_gbl_particle` mis-tracks its offset in `ops_particle_insert`

`find_offset` computes the offset from `insert_elem` but then stores
`*iprev = elem`. The two agree only when every candidate is inserted, so a
per-particle global argument walks off its array as soon as a `kernel_decide`
rejects anything — i.e. under MPI. Combined with the previous item, an insert
kernel has no correct way to receive per-particle input.

### `ops_particle_rearrange_particles_for_removal` corrupts the bin chains

It swaps every registered particle dat down, then calls
`_ops_particle_copy_mapping_data_to` to repair the cell-linked list. But
`map->bin` and `map->parts_to_grid` are themselves registered particle dats
(`ops_particle_lib_core.cpp:1693-1695`, `assign = true`), so the generic swap has
already moved them and the chain walked no longer matches the indices searched
for. With 125 of 267 eddies removed under MPI the walk runs off the end of
`bins[]` and **hangs**. `ops_particle_remove_particles()` does the same
compaction without touching the maps, which is what this app uses; the maps are
rebuilt from scratch by `ops_particle_setup_maps_with_dats()` immediately after.

### `ops_particle_remove` is declared but never defined

`ops_particle_insert_del.h:421`. Calling it fails at link.

### There is no RNG a kernel can call, and none that fills a particle dat

The whole public API is four host functions (`ops_lib_core.h:1391-1397`):
`ops_randomgen_init`, `ops_fill_random_uniform`, `ops_fill_random_normal`,
`ops_randomgen_exit`. Each fills a whole `ops_dat`, so **none can be called from
inside a kernel** — curand/hiprand appear only in `ops_cuda_rt_support.h` /
`ops_hip_rt_support.h` as the device implementation of that same bulk fill, not
as a per-thread generator. That is why `eddy_kernels.h` carries its own hash.

`ops_fill_random_uniform` also **throws on a particle dat, at every rank
count**. Its last act is `ops_set_halo_dirtybit3`, for which it builds an
`ops_arg_dat` (`ops_lib_core.cpp:2618`), and `ops_arg_dat` rejects any dat with
`is_particle` set (`ops_lib_core.cpp:1229-1233`):

```
Error: ops_arg_dat_opt cannot be called for particle_ops_dat structure
```

The incompatibility is **only** that trailing bookkeeping. The fill loop
(`:2589-2612`) runs to completion first, so the numbers are already written and
`dirty_hd` is already set when the exception is raised — `-ops-rng` catches it
and carries on with values that are perfectly sound (267 eddies, every
distribution check passing in serial). A one-line guard in OPS would make the
call usable.

Sizing is *not* the problem, contrary to what this file said before:
`ops_dat_init_metadata_core` sets `size[n] = 1` for `n >= block->dims`, so a
particle dat is `{Nmax,1,1,1,1}` and the fill's `cumsize` comes out as exactly
`dim * Nmax` — the allocation.

### `ops_randomgen_init` cannot produce a rank-independent stream

It ignores its `options` argument and seeds
`seed + my_global_rank * 2654435761u` whenever comm size > 1
(`ops_lib_core.cpp:2570-2579`). Measured with `-ops-rng`, against a serial run
of the same mode:

| | eddies | unique `eddy_id` | duplicated | positions matching serial |
|---|---|---|---|---|
| np = 1 | 267 | 267 | 0 | 267 / 267 |
| np = 2 | 278 | 201 | 77 | 141 / 278 |
| np = 4 | 263 | 186 | 77 | 64 / 263 |
| np = 8 | 270 | 179 | 91 | 28 / 270 |

Every rank draws its own eddy 7 at its own position. Sometimes two ranks both
keep theirs (duplicated ids); sometimes each rank's draw lands in a *different*
rank's subdomain and nobody keeps it (unique < 267). The matching count tracks
rank 0's share of the box, because rank 0's offset is `0 * 2654435761 = 0` and
so rank 0 alone walks the serial stream.

Note the eddy **count** is a poor detector here — its expectation is 267 either
way, and 263 at np = 4 is within noise of passing. The field comparison is what
shows it.

### `ops_insert_random_particles.h` and `_v2` are dead code

OPS's own "particles from a distribution" API calls
`ops_find_intersection_region`, which is declared nowhere in the tree, and uses
bare `BoundingBox` like the insert/delete header. Only `ops_distributions.h`
compiles, and it is host-side too.

### Two smaller ones

- `ops_particle_realloc_data` throws "Number of requested allocated particles is
  smaller than maximum allocated particles" when asked for fewer than `Nmax`
  (`OPS_MAX_PART` = 1000), so it must be guarded rather than called
  unconditionally.
- `ops_particle_setup_maps_with_dats()` does **not** delete particles outside a
  rank's box. With every rank holding all 267 eddies it reports 534 at np = 2 and
  1068 at np = 4. The migration-cycle pair
  `ops_particle_update_map_lists_actual_hybrid()` +
  `ops_particle_remove_delete_maps()` does not fill the gap either: it returns
  `decide = 0` and deletes nothing in serial, and deadlocks at np = 2.

---

## Findings in oSEM_3d

Turned up while porting. All are in `../oSEM_3d`, not in this app.

### 1. The live instantiation is broken, in two ways

Both trace to one root cause: `instantiate_eddies`
(`opensbliblock00_kernels.h:39-48`) treats the RNG output as a full-range
**signed** int, while `ops_fill_random_uniform` fills `int` dats from
`std::uniform_int_distribution<int>(0, INT_MAX)` (`ops_lib_core.cpp:2604`) —
never negative. `eddy_x_rng` and `eddy_bulk_rng` are both `int` dats
(`defdec_data_set.h:705,714`). `convect_eddies` reuses the same arithmetic on
every respawn, so this is not confined to startup.

**Positions collapse into the upper half of every interval.**
`(rng + 2147483648.0)/4294967295.0` spans `[0.5, 1.0]`, not `[0,1)`:

| | actual | intended box |
|---|---|---|
| `eddy_x` | `[ 0.016,  2.331]` | `[−2.339,  2.339]` |
| `eddy_y` | `[ 5.869, 14.035]` | `[−2.339, 14.037]` |
| `eddy_z` | `[20.033, 42.331]` | `[−2.339, 42.339]` |

Eddies occupy an eighth of the box, and **nothing sits below `y = 5.87`** — the
whole lower boundary layer, including the near-wall region where the Reynolds
stresses peak, has no eddies.

**Every sign is +1.** `(rng < 0) ? -1 : 1` cannot take its first branch, so all
267 eddies are `(+1,+1,+1)`. The SEM builds a zero-mean fluctuation by
cancellation between `+1` and `-1` eddies; with no `-1` eddies it produces a
systematic offset instead.

`frames/osem3d_eddies_osemrng_000000.png` is this field, against
`frames/osem3d_eddies_000000.png` for the counter-based hash. It was produced by
a `-osem3d-rng` mode that has since been removed: reproducing it needs
`mt19937` and `std::uniform_int_distribution`, neither of which a kernel can
run, so keeping it meant keeping a host reimplementation of the instantiation —
exactly what this app no longer has.

### 2. The shape-function constant does not match the truncation

`Kernel030` uses `shape = 1/1.85 * exp(...)` and truncates on a **sphere**
(`rsq < 1.0`). But `1/1.85` is the **cube** normalisation: normalising each
direction independently over `|x̃| < 1` gives
`(∫₋₁¹ e^{-x²}dx)^{-3/2} = 1/1.8255`. The sphere's volume is `4π/3 = 4.19`
against the cube's 8, so the corners the constant paid for are never evaluated.
For the spherical truncation the constant should be `1/√2.38098 = 1/1.5430`.

Measured effect on its own: reconstructed stresses at **0.696** of prescribed.

### 3. The z-extension of the box double-counts against spanwise periodicity

The eddy box is extended by `r` at both ends in z **and** `Kernel030` wraps
spanwise with an explicit mirror. Those are alternatives, not complements: under
periodicity an eddy at `z = -2` already *is* an eddy at `z = 38`, so the margins
duplicate coverage that wrapping provides. The effective eddy density in z is
inflated by `(span + 2r)/span = 44.679/40 = 1.117`.

Measured: with the correct spherical constant the stresses come out at
**1.1175** of prescribed, against the predicted 1.11697 — a 0.05% match.

Combined with (2), the two errors partially cancel: `0.696 × 1.117 = 0.777`.

### 4. The spanwise mirror has a units error

```c
ztildesq       = (eddy_z - x2)*(eddy_z - x2) / (eddy_r*eddy_r);   // normalised
ztildesq_ghost = (eddy_z - x2 + 40.0)*(eddy_z - x2 + 40.0);       // NOT
rsq = rsq - ztildesq + ztildesq_ghost;
```

`r² = 5.47`, so the mirror triggers only within about one length unit of the
wrapped position instead of `r`, and the mirrored shape decays `r²` times too
fast. Spanwise periodicity is roughly 43% of its intended width.

### 5. Two smaller things

- `host_rng_sign()` (`opensbli.cpp:22-25`) takes signs from `s % 2` of
  `s = (5s+3) mod 2^29`. Since `5s+3 ≡ s+1 (mod 2)` that parity strictly
  alternates, giving `(a, -a, a)` per eddy — perfectly balanced and perfectly
  anti-correlated. It is dead code, but it documents the intended scheme, hence
  the sign-correlation check here.
- `ops_randomgen_init_host` (`ops_lib_core.cpp:2570-2579`) seeds with
  `seed + my_global_rank * 2654435761u` when comm size > 1, so anything built on
  `ops_fill_random_uniform` changes with the rank count.

Findings 2–4 were measured with a host-side reference evaluation that has since
been removed to keep this app to initialisation. They are recorded here because
they bear on how the eddy box and the eddy count should be defined — which *is*
an initialisation question — and they will need settling before the coupling.

---

## Files

| File | Contents |
|---|---|
| `sem3d.cpp` | declarations, the loop sequence, the checks |
| `sem3d_constants.h` | Geometry and eddy population, from `../oSEM_3d` |
| `eddy_kernels.h` | the RNG, `KerInstantiateEddies`, `KerInstantiateEddiesOpsRng`, `KerMarkUnownedEddies`, `KerInitEddyGrid` |
| `sem3d_stats.h` | chi-square and sign-statistics helpers |
| `sem3d_io.h` | Particle HDF5 writer |
| `plot_osem3d_h5.py` | 3-D cloud, (y,z) plane, side view, wall-normal spread |

## What to check when it breaks

- **Count below 267** — an eddy no rank claimed. `KerMarkUnownedEddies` and the
  binning have to agree on where a subdomain ends, which is why the kernel is
  given `getLocalMaxMin()` rather than bounds of this app's own construction.
- **Count a multiple of 267** — the cull did not happen. Every rank instantiates
  the whole list, so 534 at np = 2 and 1068 at np = 4 mean
  `ops_particle_remove_particles()` was skipped or `mark_deletion` never
  received the kernel's verdict.
- **A hang at np ≥ 2 in removal** — `ops_particle_rearrange_particles_for_removal`
  has crept back in; see the API findings above.
- **Degenerate z range** — `_ops_construct_local_box_from_dat()`
  (`ops_particle_box_host_funcs.h:45`) had a 3-D array-of-structs branch that
  never assigned `xmax[2]` until 2026-07-30. In serial that silently collapsed
  every particle onto `z = 0` *and still reported PASS*.
- **"Defined bounding box of non-positive volume"** — `KerInitEddyGrid` must run
  after `ops_partition()` and before `ops_particle_setup_partition()`.
  `ops_create_bounding_box()` stores only a pointer; the bounds are computed in
  the setup call.
- **"Introduce map do not project properly to block grid"** on `dev_seq` — a
  strided mapping has crept in. The two backends disagree by one on the
  divisibility rule (`ops_particle_host_single_node.cpp:3053` vs
  `ops_mpi_particle_core.cpp:441`), so no dat size satisfies both.
