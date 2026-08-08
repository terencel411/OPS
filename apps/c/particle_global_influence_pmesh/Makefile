#
# OPS Particles -- particle-mesh vs direct all-to-all
#
# Requires in the environment:
#   OPS_INSTALL_PATH   e.g. /home/terence411/proj5/installations/OPS/ops
#   OPS_COMPILER       e.g. gnu
#
# Targets:
#   make influence_pm_dev_seq   serial, compiled directly (NO translator)
#   make influence_pm_dev_mpi   MPI,    compiled directly (NO translator)
#   make influence_pm_mpi       MPI,    via the OPS translator
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi

HEADERS=grid_kernels.h particle_kernels.h scatter_loop.h

OPS_FILES=influence_pm.cpp

OPS_GENERATED_SUFFIX=_ops

APP=influence_pm
MAIN_SRC=influence_pm

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
