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

/** @brief 3D heat diffusion PDE solved using ADI
 *  @author Endre Lazlo, converted to OPS by Gihan Mudalige
 *  @details PDE is solved with the ADI (Alternating Direction Implicit) method
 *  uses the Scalar tridiagonal solver for CPU and GPU written by Endre. Lazslo
**/

// standard headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

// OPS header file
#define OPS_3D
#include "ops_seq_v2.h"


#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#ifdef OPS_MPI
#include <mpi.h>
#endif
//#include "print_kernel.h"

#include "generate_kernel.h"
int main(int argc, char *argv[]) {
  // Set defaults options
  int base[3]={0, 0, 0};

  /**--- Initialisation----**/

  // OPS initialisation
  ops_init(argc, argv, 2);

  /* Insert block */
  ops_block block3D = ops_decl_block(3, "block3D");

  /* Generating box bound */
  ops_point xlow{0.0, 0.0, 0.0};
  ops_point xmax{10.0, 10.0, 10.0};

  BoundingBox *boundBlock = ops_create_bounding_box(3);

  /* Insert particle structure in the block */
  ops_particle particle = ops_decl_particle(block3D, boundBlock);
  double *temp = NULL;
  ops_dat particle_crd = ops_decl_particle_dat(particle, 3, base, temp, "double", "crd_x");

  ops_dat particle_act_crd = ops_decl_particle_pos_dat(particle, 3, base,
                                                       temp, "double","coords");

  int size[3] = {100, 100, 100};
  int d_m[3] ={-1, -1, -1};
  int d_p[3] ={1, 1, 1};

  ops_dat coords_data  = ops_decl_dat(block3D, 3, size, base, d_m, d_p, temp, "double", "coords_grid");

  /* declare stencil */
  int loca_dat[3]={0, 0, 0};
  ops_stencil local = ops_decl_stencil(3, 1, loca_dat, "local_Stencil");


  ops_partition("particle_partition");

  /* Init domain */
  double dx = (xmax.x - xlow.x) / static_cast<double>(size[0]-1);
  int range[6]{0, size[0], 0, size[1], 0, size[2]};
  ops_par_loop(KerGenerateKernel, "KerGenerateKernel", block3D, 3, range,
               ops_arg_dat(coords_data, 3, local, "double", OPS_WRITE),
               ops_arg_gbl(&dx, 1, "double", OPS_READ),
               ops_arg_idx());

  /* Verify partition */
  //TODO: Insert Bounding Box
  ops_set_bounding_box_from_dat(boundBlock, coords_data, 0.0, 3);


  ops_exit();
  return 0;
}
