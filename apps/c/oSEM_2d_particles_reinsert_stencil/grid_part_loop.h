#ifndef GRID_PART_LOOP_H
#define GRID_PART_LOOP_H

template <typename... ParamType, typename... OPSARG>
inline void ops_par_grid_part_loop(void (*kernel)(ParamType...), char const *name,
                                   ops_particle particle, ops_particle_mapping map,
                                   ops_stencil map_stencil, int dim, int *range,
                                   OPSARG... arguments) {
  ops_par_loop(kernel, name, particle, map, map_stencil, dim, range, arguments...);
}

#endif /* GRID_PART_LOOP_H */
