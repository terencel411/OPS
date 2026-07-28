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



HEADERS = OPS_oSEM_kernels.h OPS_oSEM_eddy_functions.h OPS_oSEM_constants.h TBL_data.h

OTHER_FILES = 
OPS_GENERATED = OPS_oSEM_ops.cpp 
OPS_FILES = OPS_oSEM.cpp 


APP=OPS_oSEM
MAIN_SRC=OPS_oSEM

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.c_app_legacy
