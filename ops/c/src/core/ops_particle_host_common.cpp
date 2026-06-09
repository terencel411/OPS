/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file
  * @brief OPS particles function that are common for MPI and sequential backened
  * @author Valantis Tsinginos
  * @details Implements functions that are common for MPI & sequential backend
  *          including function for mapping of particles into uniform and non-uniform
  *          structured grids
  */

#include "ops_lib_core.h"
#include <ops_exceptions.h>
#include <string>
#include <assert.h>
#include <array>
#include <limits>
#include <vector>

void _ops_build_uniform_dats(const int init, const int dim, const ops_dat grid,
                             const ops_dat xp, const size_t Np, const double *dx,
                             const ops_point xmin, const ops_point xmax,
                             ops_dat binhead, ops_dat bin) {

  int size[dim];
  size_t no_elems{1};
  for (int i = 0; i < dim; i++) {
    size[i] = grid->size[i];
    no_elems *= size[i];
  }

  /* Initialize elements */
  int* binhead_data = (int *)binhead->data;
  for (size_t i = 0; i < no_elems; i++)
    binhead_data[i] = -1;

  int* bin_data = (int *)bin->data;
  if (init) {
    for (size_t i = 0; i < Np; i++)
      bin_data[i] = -1;
  }

  /* Map particles to grid */
  double *xp_data = (double *)xp->data;

//#ifdef _OPENMP
//# pragma omp parallel for shared(xp_data)
//#endif
  for (long int i = Np - 1; i >= 0; i--) {
    int ibin = _ops_coord_to_bin(dim, xmin, xmax, dx, size, xp_data + xp->dim * i);
    //TODO: Add separation between local and not local elements
    if (ibin < 0) {
      printf("i = %d xp = [%f %f] ", i, xp_data[2 * i], xp_data[2 * i +1]);
      ops_printf("WARNING: Non-positiove value");
      continue;
    }
    bin_data[i] = binhead_data[ibin];
    binhead_data[ibin] = i;
  }
}


