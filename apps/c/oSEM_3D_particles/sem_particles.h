#ifndef _SEM_PARTICLES_H_
#define _SEM_PARTICLES_H_

/*
 * Eddies as OPS particles — scaffolding for oSEM_3D.
 *
 * Replaces the eddy ops_dats on opensbliblock00 (shape {eddies,1,1}, which ride
 * the flow block's decomposition and therefore need a gather every step) with
 * one OPS particle per eddy. See PORT_NOTES.md for the design and
 * ../oSEM_particles/README.md for the mapping constraints.
 *
 * Everything here is plain OPS API, so an OpenSBLI generator can emit it.
 */

/* ------------------------------------------------------------------ *
 *  Geometry of the particle mapping
 * ------------------------------------------------------------------ *
 * Derived from the flow parameters so it recomputes if the grid changes.
 *
 *   stride  : bin width in cells. Must be >= the interaction reach, so a +/-1
 *             stencil covers it and the ghost band stays one bin deep -- the
 *             only configuration that is correct under MPI (defect 1).
 *   nbin    : POWER OF TWO, so the stride divides every rank's subdomain for
 *             any power-of-two rank count (defect 4b).
 *   nodes   : nbin*stride + 1. Larger than the grid is fine; the dat may extend
 *             UPWARD freely. It may NOT start below the grid origin.
 */
struct sem_map_geom {
  int stride[3];
  int nbin[3];
  int nodes[3];
};

/* Smallest power of two >= v. */
inline int sem_next_pow2(int v) {
  int p = 1;
  while (p < v) p *= 2;
  return p;
}

/* reach[i] = cells an eddy influences in direction i; ngrid[i] = grid nodes. */
inline sem_map_geom sem_make_geom(const int reach[3], const int ngrid[3]) {
  sem_map_geom g;
  for (int i = 0; i < 3; i++) {
    g.stride[i] = (reach[i] < 1) ? 1 : reach[i];
    /* Enough bins to cover the grid, rounded up to a power of two. */
    const int need = (ngrid[i] + g.stride[i] - 1) / g.stride[i];
    g.nbin[i]  = sem_next_pow2(need);
    g.nodes[i] = g.nbin[i] * g.stride[i] + 1;
  }
  return g;
}

/* ------------------------------------------------------------------ *
 *  Seeding
 * ------------------------------------------------------------------ *
 * Every rank walks the same global eddy list and keeps the ones inside its own
 * subdomain, using the library's own ownership predicate.
 *
 * The stream is keyed on the GLOBAL eddy index, so the eddy field is identical
 * for any number of ranks. ops_fill_random_uniform cannot be used: it offsets
 * its seed by rank (ops_lib_core.cpp:2576) and returns non-negative ints only
 * (:2604).
 *
 * UNSIGNED arithmetic throughout -- signed overflow is undefined and at -O3 the
 * compiler folds the LCG recurrence, which collapses every draw to one value.
 *
 * CLAMPING: eddy centres are drawn inside [lo, hi], which the caller has
 * already intersected with the grid's coordinate range. Eddies centred in the
 * outer `radius` of the domain are therefore not represented -- a deliberate
 * modelling choice, because the bounding box cannot extend below the grid
 * origin (see PORT_NOTES.md).
 *
 * The position dat is (x_fixed, y, z): x is not a search dimension, so the
 * particle sits on a fixed plane inside the domain and its real x rides along
 * in p_x.
 */
inline void sem_seed_eddies(ops_particle particle,
                            ops_dat pos, ops_dat p_x, ops_dat p_r,
                            ops_dat p_inc, ops_dat p_eps, ops_dat p_seed,
                            int neddy, unsigned int base_seed,
                            double x_fixed, double increment, double radius,
                            const double lo[3], const double hi[3]) {

  BoundingBox<double> *box = (BoundingBox<double> *)particle->box_block;

  if (neddy > (int)particle->Nmax) ops_particle_realloc_data(particle, neddy);

  double *d_pos = (double *)pos->data;
  double *d_x   = (double *)p_x->data;
  double *d_r   = (double *)p_r->data;
  double *d_inc = (double *)p_inc->data;
  int    *d_eps = (int *)p_eps->data;
  int    *d_sd  = (int *)p_seed->data;

  int n = 0;
  for (int e = 0; e < neddy; e++) {
    /* Decorrelate consecutive indices before drawing. */
    unsigned int s = (base_seed + 2654435761u * (unsigned int)e) & 0x7fffffffu;

    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double xe = lo[0] + (hi[0] - lo[0]) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double ye = lo[1] + (hi[1] - lo[1]) * ((double)s / 2147483648.0);
    s = (1103515245u * s + 12345u) & 0x7fffffffu;
    const double ze = lo[2] + (hi[2] - lo[2]) * ((double)s / 2147483648.0);
    /* bit 8, not bit 0: the low bit of an LCG mod a power of two has period 2. */
    s = (1103515245u * s + 12345u) & 0x7fffffffu; const int ex = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu; const int ey = ((s >> 8) & 1u) ? 1 : -1;
    s = (1103515245u * s + 12345u) & 0x7fffffffu; const int ez = ((s >> 8) & 1u) ? 1 : -1;

    const double p3[3] = {x_fixed, ye, ze};
    if (!box->isCoordinateInBoundingBox(p3)) continue;

    d_pos[3 * n]     = x_fixed;
    d_pos[3 * n + 1] = ye;
    d_pos[3 * n + 2] = ze;
    d_x[n]           = xe;          /* the REAL streamwise position */
    d_r[n]           = radius;
    d_inc[n]         = increment;
    d_eps[3 * n]     = ex;
    d_eps[3 * n + 1] = ey;
    d_eps[3 * n + 2] = ez;
    d_sd[n]          = (int)s;
    n++;
  }

  particle->no_particles = n;
}

/* ------------------------------------------------------------------ *
 *  Per-step map / migration cycle
 * ------------------------------------------------------------------ */
inline void sem_update_maps(ops_particle particle,
                            ops_dat *dat_border, int nborder,
                            ops_dat *dat_forward, int nforward) {
  int decide = ops_particle_update_map_lists_actual_hybrid(particle);
  ops_particle_remove_delete_maps(particle, decide);
  if (decide)
    ops_particle_intrablock_border_map_update(particle, dat_border, nborder);
  else
    ops_particle_intrablock_forward_map_update(particle, dat_forward, nforward);
  ops_particle_reset_flags(particle, decide);
}

/* 3-D box stencil of half-width h: (2h+1)^3 points. */
inline int *sem_box_stencil(int h, int &npoints) {
  npoints = (2 * h + 1) * (2 * h + 1) * (2 * h + 1);
  int *s = (int *)malloc(sizeof(int) * 3 * npoints);
  int k = 0;
  for (int i = -h; i <= h; i++)
    for (int j = -h; j <= h; j++)
      for (int l = -h; l <= h; l++) {
        s[3 * k] = i; s[3 * k + 1] = j; s[3 * k + 2] = l; k++;
      }
  return s;
}

inline int sem_total_particles(ops_particle particle) {
  int n = (int)particle->no_particles;
#ifdef OPS_MPI
  int tot = 0;
  MPI_Allreduce(&n, &tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return tot;
#else
  return n;
#endif
}

#endif /* _SEM_PARTICLES_H_ */
