USE_HDF5=1
#
# oSEM_2d_particles_grid -- eddy search by the particle map instead of a gather
#
# Requires in the environment:
#   OPS_INSTALL_PATH   e.g. /home/terence411/proj5/installations/OPS/ops
#   OPS_COMPILER       e.g. gnu
#
# Targets:
#   make osem_2d_particles_grid_loop_dev_seq   serial, compiled directly (NO translator)
#   make osem_2d_particles_grid_loop_dev_mpi   MPI,    compiled directly (NO translator)
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=grid_kernels.h particle_kernels.h osem_common.h osem_constants.h osem_io.h ops_particle_random.h

OPS_FILES=osem_2d_particles_grid_loop.cpp

OPS_GENERATED_SUFFIX=_ops

APP=osem_2d_particles_grid_loop
MAIN_SRC=osem_2d_particles_grid_loop

# The grid-outer/particle-inner ops_par_loop is resolved by C++ templates and
# compiles host-side only, so only the direct (non-translator) targets are here.
TARGETS=dev_seq dev_mpi

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
