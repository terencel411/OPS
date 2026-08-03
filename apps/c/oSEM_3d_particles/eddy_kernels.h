/*
 * eddy_kernels.h -- oSEM_3d eddies as OPS particles
 *
 * ACC<T>&  is the grid accessor:     c(component, i, j, k)
 * ACCP<T>& is the particle accessor: p(component) -- already positioned on the
 *          current particle, so there is no (i,j,k); a particle has a position,
 *          not a grid index.
 */

#ifndef _EDDY_KERNELS_H_
#define _EDDY_KERNELS_H_

/* ------------------------------------------------------------------ *
 *  The eddy-box coordinate grid
 * ------------------------------------------------------------------ *
 * Nothing is solved on this grid. It exists so that
 * ops_create_bounding_box() has a coordinate dat to derive the particle
 * bounding box from, and so the mapping has cells to bin eddies into.
 *
 * The box comes out as exactly [first node .. last node] in each direction, so
 * main() sets the spacing to extent/(nodes-1) and the first and last nodes land
 * on the faces of the eddy box.
 *
 * Origin and spacing arrive as one 6-element ops_arg_gbl rather than through
 * ops_decl_const, so this kernel does not depend on what main() has registered.
 */
void KerInitEddyGrid(ACC<double> &c, const double *g, const int *idx) {
  c(0, 0, 0, 0) = g[0] + g[3] * (double)(idx[0]);
  c(1, 0, 0, 0) = g[1] + g[4] * (double)(idx[1]);
  c(2, 0, 0, 0) = g[2] + g[5] * (double)(idx[2]);
}

/* ------------------------------------------------------------------ *
 *  The random stream
 * ------------------------------------------------------------------ *
 * Three requirements, in order of how much they constrain the choice:
 *
 *   1. RANK-COUNT INDEPENDENCE. The eddy field must not depend on how the run
 *      is decomposed, or no two runs can be compared. This rules out OPS's own
 *      generator: ops_randomgen_init_host (ops_lib_core.cpp:2570-2579) seeds
 *      with `seed + my_global_rank * 2654435761u` whenever comm size > 1.
 *
 *   2. UNIFORM COVERAGE OF THE BOX, including the near-wall region.
 *
 *   3. BALANCED, INDEPENDENT SIGNS.
 *
 * Counter-based: draw k of eddy e is a pure function of (e, k) and the base
 * seed. Nothing is carried between draws, so requirement 1 holds by
 * construction -- there is no state whose evolution could depend on which rank
 * walked it. That is also what lets this live inside a kernel at all: a stateful
 * generator would need storage the kernel has no way to carry.
 *
 * The finalizer is SplitMix32's.
 *
 * These sit in the kernel header rather than the application because the
 * translator inlines this file with the kernels that use them. LBM-PSM does the
 * same with shape_func_interpol (../LBM-PSM/particle_kernels.h).
 */
static inline unsigned int sem_hash32(unsigned int x) {
  x ^= x >> 16;  x *= 0x21f0aaadu;
  x ^= x >> 15;  x *= 0xd35a2d97u;
  x ^= x >> 15;
  return x;
}

/* A private counter block per eddy, so adding a draw later does not renumber
   any other eddy's stream. */
#define DRAWS_PER_EDDY 8

static inline unsigned int sem_draw(unsigned int base, int counter) {
  return sem_hash32(base + 0x9e3779b9u * (unsigned int)counter);
}

/* [0, 1). 2^32 in the denominator, so 1.0 is unreachable -- OPS bins half-open
   [lo,hi), and a coordinate landing exactly on a box maximum reads as outside. */
static inline double sem_uniform(unsigned int base, int counter) {
  return (double)sem_draw(base, counter) / 4294967296.0;
}

static inline int sem_sign(unsigned int base, int counter) {
  return ((sem_draw(base, counter) >> 8) & 1u) ? 1 : -1;
}

/* ------------------------------------------------------------------ *
 *  instantiate_eddies
 * ------------------------------------------------------------------ *
 * The particle counterpart of ../oSEM_3d/opensbliblock00_kernels.h:39-48, which
 * ../oSEM_3d runs as an ops_par_loop over an iteration range of `eddies`
 * (opensbli.cpp:313-323). Here it is an ops_particle_par_loop over `eddies`
 * particles, and it writes the same fields: a position in the eddy box, the
 * radius, the per-step convection increment and three random signs.
 *
 * The geometry comes from the ops_decl_const names, exactly as it does there.
 *
 * WHERE THE RANDOMNESS COMES FROM. oSEM_3d fills two int ops_dats with
 * ops_fill_random_uniform and has the kernel read them; that is a stateful
 * generator run outside the kernel, and it is where its defects live (see the
 * README). Here the kernel hashes its own eddy index instead, which needs no
 * dat and no state, and gives the same eddy the same draw on every rank.
 *
 * idp[0] IS THE GLOBAL EDDY INDEX, not merely a local slot: this loop runs
 * before the ownership cull, when every rank holds the whole list in order. id
 * records it so that it survives the cull and every later migration.
 */
void KerInstantiateEddies(ACCP<double> &pos, ACCP<double> &r,
                          ACCP<double> &inc, ACCP<int> &eps,
                          ACCP<int> &ctr, ACCP<int> &id,
                          const int *idp) {

  const unsigned int base = (unsigned int)eddy_seed;
  const int c0 = DRAWS_PER_EDDY * idp[0];

  pos(0) = eddy_x_min + (eddy_x_max - eddy_x_min) * sem_uniform(base, c0 + 0);
  pos(1) = eddy_y_min + (eddy_y_max - eddy_y_min) * sem_uniform(base, c0 + 1);
  pos(2) = eddy_z_min + (eddy_z_max - eddy_z_min) * sem_uniform(base, c0 + 2);

  eps(0) = sem_sign(base, c0 + 3);
  eps(1) = sem_sign(base, c0 + 4);
  eps(2) = sem_sign(base, c0 + 5);

  r(0)   = radius;
  inc(0) = u0 * dt;

  /* The eddy's next unused counter, so the stream is a property of the eddy and
     migrates with it. A respawn, when there is one, should keep hashing
     counters from here rather than iterating a state. */
  ctr(0) = c0 + DRAWS_PER_EDDY;
  id(0)  = idp[0];
}

/* ------------------------------------------------------------------ *
 *  instantiate_eddies, reading a filled dat  (-ops-rng)
 * ------------------------------------------------------------------ *
 * The same instantiation, taking its randomness from a dat filled before the
 * loop instead of hashing the eddy index -- which is the structure ../oSEM_3d
 * uses (opensbli.cpp:305-311 fills eddy_x_rng and eddy_bulk_rng,
 * opensbliblock00_kernels.h:39-48 reads them).
 *
 * The fill is ops_fill_random_uniform_particle() from ops_particle_rng.h, the
 * particle counterpart of ops_fill_random_uniform() that OPS is missing. This
 * kernel exists to prove that function works end to end: filled by OPS-style
 * bulk call, consumed through ops_arg_dat_particle in an ops_particle_par_loop.
 *
 * The dat is double, so rng(k) is already in [0,1) and no rescaling of an int
 * range is needed -- which is also why this path cannot reproduce ../oSEM_3d's
 * finding 1.
 */
void KerInstantiateEddiesOpsRng(ACCP<double> &pos, ACCP<double> &r,
                                ACCP<double> &inc, ACCP<int> &eps,
                                ACCP<int> &ctr, ACCP<int> &id,
                                const ACCP<double> &rng, const int *idp) {

  pos(0) = eddy_x_min + (eddy_x_max - eddy_x_min) * rng(0);
  pos(1) = eddy_y_min + (eddy_y_max - eddy_y_min) * rng(1);
  pos(2) = eddy_z_min + (eddy_z_max - eddy_z_min) * rng(2);

  eps(0) = (rng(3) < 0.5) ? -1 : 1;
  eps(1) = (rng(4) < 0.5) ? -1 : 1;
  eps(2) = (rng(5) < 0.5) ? -1 : 1;

  r(0)   = radius;
  inc(0) = u0 * dt;

  /* No counter to carry. The hash path records where an eddy's stream has got
     to so a respawn can continue it; a bulk fill has no per-eddy position in
     the stream to record. */
  ctr(0) = 0;
  id(0)  = idp[0];
}

/* ------------------------------------------------------------------ *
 *  The ownership cull
 * ------------------------------------------------------------------ *
 * Writes 1 for an eddy this rank does not own, 0 otherwise.
 *
 * A kernel cannot change the particle count, so every rank instantiates the
 * whole list and ownership is settled afterwards. Splitting it out keeps the
 * two questions apart -- KerInstantiateEddies decides what an eddy IS, this
 * decides who keeps it -- which is the same split OPS makes between the insert
 * kernel and the decide kernel of ops_particle_insert.
 *
 * The test is the one ../LBM-PSM/particle_kernels.h:11-19 applies at insertion,
 * and the bounds it is given are BoundingBox::getLocalMaxMin(), so this app
 * cannot disagree with the migration machinery about where a subdomain ends.
 */
void KerMarkUnownedEddies(ACCP<int> &del, const ACCP<double> &pos,
                          const double *xmin, const double *xmax) {
  del(0) = 0;
  for (int d = 0; d < 3; d++)
    if (pos(d) < xmin[d] || pos(d) > xmax[d]) del(0) = 1;
}

#endif /* _EDDY_KERNELS_H_ */
