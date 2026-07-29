// Auto-generated at 2026-07-29 00:37:16.454662 by ops-translator

// headers
#define OPS_2D
#define OPS_API 2

#include "ops_lib_core.h"

#include "user_types.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

typedef struct {
    int size;
    int *xdim, *ydim, *zdim;
    int *strideX, *strideY, *strideZ;
    int *mgridStrideX, *mgridStrideY, *mgridStrideZ;
} argDims_t;

//  global constants

void ops_init_backend(){}

void ops_decl_const_char(OPS_instance *instance, int dim, char const *type, int size, char *dat, char const *name) {
    ops_execute(instance);

}

// user kernel files
#include "KerInitGrid_kernel.cpp"
#include "KerZeroDensity_kernel.cpp"
#include "KerSumDensity_kernel.cpp"

