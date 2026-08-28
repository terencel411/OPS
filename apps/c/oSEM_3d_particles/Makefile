#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

USE_HDF5=1
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

# Matched to oSEM_3d_particles so the two are compared under identical arithmetic.
CXXFLAGS := -O3 -march=native -fPIC -Wall -g -std=c++11 -fopenmp

HEADERS = opensbliblock00_kernels.h defdec_data_set.h

OTHER_FILES = 
OPS_GENERATED = opensbli_ops.cpp 
OPS_FILES = opensbli.cpp 


APP=opensbli
MAIN_SRC=opensbli

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
