#
# oSEM_3d eddies as OPS particles
#
# Requires in the environment:
#   OPS_INSTALL_PATH, OPS_COMPILER, HDF5_INSTALL_PATH, MPI_INSTALL_PATH
#
# Targets (all prefixed with APP):
#   make oSEM_3d_particles_old_dev_seq   serial, compiled directly (NO translator)
#   make oSEM_3d_particles_old_dev_mpi   MPI,    compiled directly (NO translator)
#   make oSEM_3d_particles_old_seq       serial, via the OPS translator
#   make oSEM_3d_particles_old_mpi       MPI,    via the OPS translator
#
# Start with dev_seq: no code generation step, so the edit-build-run loop is a
# couple of seconds and compiler errors point at your own source.
#

USE_HDF5=1

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=eddy_kernels.h sem3d_constants.h sem3d_stats.h sem3d_io.h ops_particle_rng.h

OPS_FILES=sem3d.cpp

OPS_GENERATED_SUFFIX=_ops

APP=oSEM_3d_particles_old
MAIN_SRC=sem3d

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
