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
  * @brief Delete particle for sequential solvers
  * @author Gihan Mudalige
  * @details Implements common functions from the various GPU backends for the sequential
  * (no MPI) versions
  */

#include "ops_lib_core.h"
#include <string>
#include <assert.h>


/*----------------------------------------------------------------------------------------*/
/* Remove particles from particle based structures
 * Particles removed based on marking-Particles marked during halo exchange or via
 * direct checking when particle are shifted.
 *----------------------------------------------------------------------------------------*/

void ops_particle_remove(ops_particle particle) {

  const int dim = particle->block->dims;
  size_t Nlocal = particle->no_particles;
  size_t Ndel{0};
  double xmin[dim], xmax[dim];

  BoundingBox *box = particle->box_block;
  box->getLocalMaxMin(xmin, xmax);

  double *xpos = (double *)particle->particle_pos_dat[0]->data;

  int i = 0;
  while (i < Nlocal) {
    ops_point xlocal{xpos[dim * i], xpos[dim * i + 1], (dim == 3) ? xpos[dim * i + 2] : 0.0};

    int imark = particle->mark_deletion[i];

    if (imark == 0) {
      bool decide = box->isCoordinateInBoundingBox(xlocal);
      if (!decide) imark = 1;
    }

    if (imark == 1) {//Not-within block mark and shift for deletion
      _ops_particle_swap_data((char *)xpos, i, Nlocal-1, dim * sizeof(double));
      if (particle->particle_envelope != nullptr)
        _ops_particle_swap_data((char *)particle->particle_envelope->data, i, Nlocal-1,
                                particle->particle_envelope->elem_size);

      for (ops_dat &data : particle->particle_data) {
        _ops_particle_swap_data(data->data, i, Nlocal-1, data->elem_size);
      }
      Nlocal--;
    }
    else i++;
  }

  particle->no_particles = Nlocal;
  //TODO: For dynamic modeling vrf that this is within actual box.
}

/*----------------------------------------------------------------*/
/* Mark particles for deletion                                    */
/*----------------------------------------------------------------*/
void ops_particle_init_mark_deletion(ops_particle particle) {

  size_t Nlocal = particle->no_particles;

  for (int i = 0; i < Nlocal; i++)
    particle->mark_deletion[i] = 0; //Particle to remain
}

