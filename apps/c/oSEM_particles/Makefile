#
# OPS Particles tutorial 3
#
# Requires in the environment:
#   OPS_INSTALL_PATH   e.g. /home/terence411/proj5/installations/OPS/ops
#   OPS_COMPILER       e.g. gnu
#
# Targets:
#   make dev_seq   serial, compiled directly (NO translator) -- start here
#   make dev_mpi   MPI,    compiled directly (NO translator)
#   make seq       serial, via the OPS translator
#   make mpi       MPI,    via the OPS translator
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi

HEADERS=grid_kernels.h particle_kernels.h scatter_loop.h sem_constants.h

OPS_FILES=sem.cpp

OPS_GENERATED_SUFFIX=_ops

APP=oSEM_particles
MAIN_SRC=sem

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
