USE_HDF5=1
#
# OPS Particles -- particle-mesh vs direct all-to-all
#
# Requires in the environment:
#   OPS_INSTALL_PATH   e.g. /home/terence411/proj5/installations/OPS/ops
#   OPS_COMPILER       e.g. gnu
#
# Targets:
#   make influence_osem_dev_seq   serial, compiled directly (NO translator)
#   make influence_osem_dev_mpi   MPI,    compiled directly (NO translator)
#   make influence_osem_mpi       MPI,    via the OPS translator
#



include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=grid_kernels.h particle_kernels.h osem_common.h osem_io.h

OPS_FILES=influence_osem.cpp

OPS_GENERATED_SUFFIX=_ops

APP=influence_osem
MAIN_SRC=influence_osem

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
