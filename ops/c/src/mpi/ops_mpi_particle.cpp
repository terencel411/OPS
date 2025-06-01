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
  * @brief OPS mpi support functions for particle management
  * @author
  * @details Setup the fine details of domain decomposition for particle packages within
  *          OPS
  */

#include <math.h>
#include <mpi.h>
#include <ops_mpi_core.h>
#include <ops_exceptions.h>


/*-------------------------------------------------------------------------------------*
 *  Setting domain partition for particle data involving
 *  1. Particle domain partition
 *  2. Setting details of forward communication for particle exchange data
 */
void ops_particle_setup_partition() {

  OPS_instance *instance = OPS_instance::getOPSInstance();

  if (!ops_partitioned())
    throw OPSException(OPS_RUNTIME_ERROR, " Setting up particle partition must "
                                          "happen after domain partition.");

  // loop over all blocks
  for (int index = 0; index < instance->OPS_block_index; index++) {
    ops_block block= instance->OPS_block_list[index];

    sub_block *sb = OPS_sub_block_list[block->index];

    if (!sb->owned) return;

    /* Get access to particle data */
    sub_particle *spar = sb->sb_particle_list;
    int nparticles = instance->OPS_block_list[index].no_particle_structures;

    for (int ipartlist = 0; ipartlist < nparticles; ipartlist++) {
      ops_particle particle = spar[ipartlist].particle;
      sub_particle sub_part = spar[ipartlist];

      //Build bounding box for particle //
      ops_build_bounding_box(particle);

      //Set communication structures in terms of boxes

      //TODO:1. Find how far we go for data exchange
      //     2. Set number of loops in each  for rcv data
      ops_particle_setup_forward_comm(sub_particle);



    }
  }

}

