#
# OPS Particles tutorial 1
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

USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=grid_kernels.h particle_kernels.h drift_io.h

OPS_FILES=drift.cpp

OPS_GENERATED_SUFFIX=_ops

APP=tutorial1
MAIN_SRC=drift

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
