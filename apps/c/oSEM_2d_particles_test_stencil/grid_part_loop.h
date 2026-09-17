#ifndef GRID_PART_LOOP_H
#define GRID_PART_LOOP_H

// The grid-outer / particle-inner ops_par_loop (ops_grid_part_seq_v2.h) under another
// name: ops-translator rejects that overload, and it only parses ops_par_loop by name.
template <typename... ParamType, typename... OPSARG>
inline void ops_par_grid_part_loop(void (*kernel)(ParamType...), char const *name,
                                   ops_particle particle, ops_particle_mapping map,
                                   ops_stencil map_stencil, int dim, int *range,
                                   OPSARG... arguments) {
  ops_par_loop(kernel, name, particle, map, map_stencil, dim, range, arguments...);
}

#endif /* GRID_PART_LOOP_H */
