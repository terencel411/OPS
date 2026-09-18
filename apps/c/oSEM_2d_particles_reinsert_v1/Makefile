USE_HDF5=1
#
# oSEM_2d_particles_reinsert_v1 -- an eddy leaving x_max is deleted and re-inserted
# at x_min on the rank that owns its new (y,z); no particle migration.
#
# Requires in the environment:
#   OPS_INSTALL_PATH   e.g. /home/terence411/proj5/installations/OPS/ops
#   OPS_COMPILER       e.g. gnu
#
# Targets:
#   make osem_2d_particles_reinsert_v1_dev_seq   serial, compiled directly (NO translator)
#   make osem_2d_particles_reinsert_v1_dev_mpi   MPI,    compiled directly (NO translator)
#   make osem_2d_particles_reinsert_v1_mpi       MPI,    via the OPS translator
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS=grid_kernels.h particle_kernels.h osem_common.h osem_constants.h osem_io.h ops_particle_random.h

OPS_FILES=osem_2d_particles_reinsert_v1.cpp

OPS_GENERATED_SUFFIX=_ops

APP=osem_2d_particles_reinsert_v1
MAIN_SRC=osem_2d_particles_reinsert_v1

# Particle kernels are resolved by C++ templates and compile host-side only,
# so GPU targets are deliberately not offered here.
TARGETS=dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
