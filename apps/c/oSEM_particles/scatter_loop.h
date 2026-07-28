/*
 * scatter_loop.h  --  OPS Particles tutorial 3
 *
 * A thin rename of the grid-outer / particle-inner loop.
 *
 * WHY
 * ---
 * That loop is declared in ops_grid_part_seq_v2.h:562 as an overload of
 * ops_par_loop:
 *
 *     ops_par_loop(kernel, name, particle, map, map_stencil, dim, range, ...)
 *
 * C++ resolves it fine, but the OPS translator does not: it scans the
 * application source textually for "ops_par_loop(" and tries to code-generate
 * every match. Reaching this one it finds an ops_stencil where it expects the
 * `int dim`, and dies --
 *
 *     legacy : unknown access type for argument 11..14
 *     modern : Parse error ... Expected int expression
 *
 * Calling it through a differently-named wrapper keeps the translator's scan
 * from matching, so the ordinary grid loops in the same file still
 * code-generate normally while this one is left to the C++ templates (which is
 * all it ever needed -- particle loops are not code-generated in any case).
 */

#ifndef _SCATTER_LOOP_H_
#define _SCATTER_LOOP_H_

template <typename... ParamType, typename... OPSARG>
inline void ops_par_scatter_loop(void (*kernel)(ParamType...),
                                 char const *name, ops_particle particle,
                                 ops_particle_mapping map,
                                 ops_stencil map_stencil, int dim, int *range,
                                 OPSARG... arguments) {
  ops_par_loop(kernel, name, particle, map, map_stencil, dim, range,
               arguments...);
}

#endif /* _SCATTER_LOOP_H_ */
