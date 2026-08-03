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
 * WHERE THE RANDOMNESS COMES FROM. The same place as ../oSEM_3d's: a dat filled
 * before the loop and read here. There is no OPS random generator a kernel can
 * call -- the whole public API (ops_lib_core.h:1391-1397) fills a whole ops_dat
 * from the host -- so fill-then-read is the only structure available.
 *
 * The fill is ops_fill_random_uniform_particle() from ops_particle_rng.h, which
 * is the particle counterpart of ops_fill_random_uniform() that OPS is missing:
 * the latter throws on a particle dat, and its stream is rank-seeded. See the
 * README.
 *
 * THE ARITHMETIC BELOW IS ../oSEM_3d's, UNCHANGED -- same +2147483648.0, same
 * /4294967295.0, same (rng < 0) ? -1 : 1, on an int dat exactly as there. Keeping
 * it identical is what makes this app a controlled comparison: the same kernel on
 * a CORRECTLY distributed int stream produces a correct eddy field, so the kernel
 * was never at fault and the whole of finding 1 belongs to
 * ops_fill_random_uniform drawing int dats from (0, INT_MAX) instead of the full
 * signed range. Only the fill differs between the two apps.
 *
 * ONE DETAIL THE ORIGINAL GLOSSES OVER: at rng = INT_MAX the expression
 * (2147483647 + 2147483648.0)/4294967295.0 is exactly 1.0, so this maps to a
 * CLOSED [0,1] and can place an eddy precisely on a box face, which a half-open
 * bin reads as outside. Probability 2^-32 per draw, about 2e-7 over the 801
 * position draws here. Left as-is so the arithmetic stays ../oSEM_3d's.
 *
 * idp[0] IS THE GLOBAL EDDY INDEX, not merely a local slot: this loop runs
 * before the ownership cull, when every rank holds the whole list in order. id
 * records it so that it survives the cull and every later migration.
 */
void KerInstantiateEddies(ACCP<double> &pos, ACCP<double> &r,
                          ACCP<double> &inc, ACCP<int> &eps, ACCP<int> &id,
                          const ACCP<int> &rng, const int *idp) {

  pos(0) = eddy_x_min + (rng(0) + 2147483648.0) / (4294967295.0) * (eddy_x_max - eddy_x_min);
  pos(1) = eddy_y_min + (rng(1) + 2147483648.0) / (4294967295.0) * (eddy_y_max - eddy_y_min);
  pos(2) = eddy_z_min + (rng(2) + 2147483648.0) / (4294967295.0) * (eddy_z_max - eddy_z_min);

  eps(0) = (rng(3) < 0) ? -1 : 1;
  eps(1) = (rng(4) < 0) ? -1 : 1;
  eps(2) = (rng(5) < 0) ? -1 : 1;

  r(0)   = radius;
  inc(0) = u0 * dt;
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
