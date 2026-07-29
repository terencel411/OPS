# Tutorial 1 — Constant-velocity drift

The smallest OPS Particles application that is still correct under MPI.

100 particles are seeded on a lattice and drift at a constant, prescribed
velocity. There is no fluid solver and no interpolation. After `N` steps every
particle must sit at exactly `x0 + v·N·dt`, and the program checks this itself —
so a correct run is unambiguous.

**Next:** [`../particle_tutorial_2_advect`](../particle_tutorial_2_advect) adds the
particle↔grid coupling loop.
**Reference:** `/home/terence411/proj5/installations/documentation/ops-particles.md`

---

## Build and run

```bash
source /home/terence411/proj5/installations/configure_env_vars.sh   # if not already set

make tutorial1_dev_seq          # serial, NO translator  <- start here
./tutorial1_dev_seq

make tutorial1_dev_mpi          # MPI, NO translator
mpirun -np 4 ./tutorial1_dev_mpi

make tutorial1_mpi              # MPI, via the OPS translator
mpirun -np 4 ./tutorial1_mpi
```

Note the target names are prefixed with the app name (`tutorial1_dev_seq`, not
`dev_seq`) — that is how OPS app Makefiles work.

**Use `dev_seq` / `dev_mpi` while learning.** They compile `drift.cpp` directly,
with no code generation step, so the edit→build→run loop is a couple of seconds
and compiler errors point at your own source. The translator targets (`seq`,
`mpi`) produce the same answers; switch to them once you are comfortable.

Verified working: `dev_seq`, `dev_mpi` (1/2/3/4 ranks), `mpi` (4 ranks) — all give
`max position error ≈ 2.2e-14`, `RESULT: PASS`.

---

## Expected output

```
xmin = [0.000000 0.000000] xmax = [1.000000 1.000000]
Box has bounding box =[0.000000 1.000000]x[0.000000 1.000000]
OPS Particles tutorial 1: constant-velocity drift
grid 41x41, 100 particles, v = (0.5, 0.25), dt = 0.001, 400 steps
step   100 / 400
...
---------------------------------------------
particles expected : 100
particles found    : 100
max position error : 2.209e-14  (tol 4.0e-10)
RESULT             : PASS
---------------------------------------------
```

> The first two lines are **debug `printf`s inside the OPS library itself**
> (`ops/c/include/ops_particle_box_host_funcs.h:132,140`), not something this app
> prints. Every OPS Particles app that builds its bounding box from a coordinate
> dat emits them. Ignore them.

Particle positions are dumped to `particles_step_*.txt` every 100 steps.

---

## What this tutorial teaches

### 1. Declaration order is not negotiable

```
ops_init
  → ops_decl_block → coordinate dat → stencils
  → ops_create_bounding_box          (needs the coordinate dat)
  → ops_decl_particle                (needs the box)
  → ops_decl_particle_pos_dat        (must precede any mapping)
  → other particle dats
  → ops_decl_mapping                 (needs the position dat)
  → ops_partition
  → fill the coordinate grid         ← easy to get wrong, see below
  → ops_particle_setup_partition
  → seed particles
  → ops_particle_setup_maps_with_dats
  → time loop
```

**The trap:** `ops_create_bounding_box` only stores a *pointer* to the coordinate
dat. The physical bounds are not computed until `ops_particle_setup_partition()`.
So the grid must contain real coordinates **before** that call. Fill it after
`ops_partition` and before `ops_particle_setup_partition`, or OPS throws:

```
what():  Error: Defined bounding box of non-positive volume.
```

because it derived the domain from an all-zeros coordinate array.

### 2. The grid here is scaffolding, not physics

`x_grid` exists only to give the bounding box and the mapping a coordinate
reference. It holds node coordinates and nothing else. Tutorial 2 puts real data
on the grid.

### 3. The seeding contract

Seeding is plain C++ writing into `dat->data` — no kernel, because no map exists
yet. You must:

1. call `ops_particle_realloc_data` if you need more than `Nmax` (**default 1000**)
2. write every per-particle field
3. **set `particle->no_particles` yourself** — nothing infers it
4. under MPI, seed only within this rank's *local* box bounds, and offset ids by
   an `MPI_Allgather` prefix sum so they are globally unique

Particle dats are AoS: `data[dim * particle_index + component]`.

### 4. `ACCP<T>&` kernels

```c
void KerUpdatePosition(ACCP<double>& xp, const ACCP<double>& up, const double *dt) {
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}
```

A particle accessor is already positioned on the current particle, so you index
only the component. Contrast `ACC<T>&` for grid data, which takes `(component, i, j)`.

Use `OPS_PARTICLE_ITERATE_LOCAL` for anything that advances state — `ITERATE_ALL`
would also advance ghost copies of particles another rank owns.

### 5. The migration cycle, and what the border list actually controls

`update_maps()` runs every step:

```c
int decide = ops_particle_update_map_lists_actual_hybrid(particle);
ops_particle_remove_delete_maps(particle, decide);
if (decide) ops_particle_intrablock_border_map_update (particle, dat_border,  nborder);
else        ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);
ops_particle_reset_flags(particle, decide);
```

Particles that barely moved are skipped; particles that changed cell are relinked
in place. Only when one crosses the halo band does the expensive rebuild fire —
and that decision is made collectively across ranks.

**Two different mechanisms are easy to confuse:**

| Mechanism | What it moves | Which dats |
|---|---|---|
| **Ownership migration** — a particle physically changes rank | the particle itself | **all** dats declared with `assign = true` (the default), automatically |
| **The border list** — building the ghost/virtual layer | copies of particles near a rank boundary | **only** the dats you pass in `dat_border` |

`_ops_particle_exchange_map_update` (the migration path) loops over
`particle->particle_dat[]` and moves everything registered there. It never looks
at your border list. So `p_x0` in this app travels with its particle whether or
not it appears in `dat_border` — and indeed removing it from that array still
passes on 4 ranks. (I checked; don't take it on faith.)

The border list decides what a **ghost** particle knows. A ghost is a read-only
copy, held by this rank, of a particle owned by a neighbour. If a kernel ever
reads a dat on ghost particles — via `OPS_PARTICLE_ITERATE_ALL`, an interaction
loop, or a grid loop that reaches ghosts — then that dat must be in the border
list, or the ghosts carry stale values. Tutorial 1 only ever iterates
`ITERATE_LOCAL`, so its border list is barely exercised; it is written out in
full here because that is the habit you want in a real application.

`nborder` is computed with `sizeof(arr)/sizeof(arr[0])`, never typed by hand.
The reference app `LBM-Particle_tracking` passes a literal `3` for a 4-element
array, silently dropping `particle_mass` from its ghost data.

---

## Things to try

| Experiment | What you should see |
|---|---|
| Run on 1, 2, 3, 4 ranks | Identical error every time. Migration is working. |
| Remove `p_x0` from `dat_border` | Still passes — proof that ownership migration moves all assigned dats regardless of the border list (see §5). |
| Increase `VEL` so particles leave the domain | Particle count drops — they are deleted on exit. `particles found` no longer matches. |
| Raise `NPX`/`NPY` above 1000 total | Shows why the `ops_particle_realloc_data` call is there. |
| Switch `ITERATE_LOCAL` → `ITERATE_ALL` | Serial is unchanged; multi-rank goes wrong, because ghosts get advanced too. |
| Print `particle->no_particles` and `no_virtual` each step | Watch the ghost layer appear and change as particles near rank boundaries. |

---

## Files

| File | Contents |
|---|---|
| `drift.cpp` | The whole application, commented step by step |
| `particle_kernels.h` | `KerSetVelocity`, `KerUpdatePosition` — `ACCP<T>&` kernels |
| `grid_kernels.h` | `KerInitGrid` — the only grid kernel |
| `user_types.h` | Empty; the translator looks for it by convention |
| `Makefile` | Targets: `dev_seq`, `dev_mpi`, `seq`, `mpi`, `openmp`, `mpi_openmp` |
