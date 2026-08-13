#
# oSEM_3d_particles -- oSEM_3d with the eddies as OPS particles
#
# Requires in the environment (source ../../../configure_env_vars.sh):
#   OPS_INSTALL_PATH
#   OPS_COMPILER   (gnu, intel, ...)
#
# Targets:
#   make opensbli_dev_seq    serial, compiled directly (NO translator)
#   make opensbli_dev_mpi    MPI,    compiled directly (NO translator)
#   make opensbli_seq        serial, via the OPS translator
#   make opensbli_mpi        MPI,    via the OPS translator
#

USE_HDF5=1

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

HEADERS = opensbliblock00_kernels.h eddy_kernels.h defdec_data_set.h \
          constants.h stencils.h bc_exchanges.h io.h \
          osem3d_common.h ops_particle_random.h Pir_data.h TBL_data.h

OPS_FILES = opensbli.cpp

OPS_GENERATED_SUFFIX = _ops

APP = opensbli
MAIN_SRC = opensbli

# The particle kernels are resolved by C++ templates and compile host-side
# only, so the GPU targets are deliberately not offered here.
TARGETS = dev_seq dev_mpi seq openmp mpi mpi_openmp

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app
