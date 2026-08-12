# oSEM_3d_particles_v1

[`../oSEM_3d`](../oSEM_3d) with the synthetic eddies as **OPS particles**, and
the hand-rolled `MPI_Allgatherv` that fed `Kernel030` replaced by an OPS
reduction. Structured after
[`../particle_global_influence_oSEM`](../particle_global_influence_oSEM), which
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

## The two structural decisions

### The eddies live on the fluid block

Not on a block of their own, which was the obvious first try. **OPS partitions
the ranks *between* blocks**: `ops_mpi_partition.cpp:152-154` gives each block
`nproc/nblocks` of them and leaves `sb->comm == MPI_COMM_NULL` on the rest. A
second block would take half the ranks away from the solver, and it also
crashes `partitionBoundingBox`, which reaches for `sb->comm`
(`ops_particle_box_mpi_funcs.h:152`) without checking `sb->owned` first —
measured, at np = 2, as `MPI_ERR_COMM` on rank 0 while rank 1 seeded all 267
eddies. Every particle app in the tree puts its particles on the solver's own
block.

The 2-D reference's arrangement is not available either: it makes the eddy
position `(y, z)` and carries `x` as an ordinary dat, but
`ops_decl_particle_pos_dat` throws unless the position dimension equals the
block's (`ops_particles_lib_core.h:762`), and `ACC` is compiled for one
dimensionality per translation unit, so a 3-D app cannot host a 2-D block.

### The position dat is the ownership handle, not the eddy's position

The eddy's coordinates ride in `p_e`, an ordinary particle dat. They have to,
because the eddy box is not inside the fluid block: it overhangs on all three
low faces — x, y and z all start at `-radius` while the block spans
`[0,375] x [0,100] x [0,40]` — and it is wider than the domain in z, 44.68
against 40. The bounding box OPS derives from a coordinate dat is exactly
`[first node .. last node]`, so an eddy at its true coordinates would often fall
outside every rank's box, and outside means owned by nobody.

`p_pos` is therefore the true position **clamped** into the block. That moves an
eddy by at most one radius and only in the overhang, and it hands each eddy to
the rank holding the piece of inlet it actually influences. Since the eddy box
is a thin slab at `x ≈ 0`, the eddies land on the ranks that own the inlet
plane, which is where `Kernel030` does its work.

Ownership is settled once, at seeding, and **never migrates** — see below. So
`p_pos` is read exactly once, by OPS, to distribute the eddies, and nothing
reads it afterwards.

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
