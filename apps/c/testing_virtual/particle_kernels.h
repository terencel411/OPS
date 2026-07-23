/*
 * particle_kernels.h
 *
 *  Created on: Jan 27, 2026
 *      Author: valantis
 */

#ifndef _PARTICLE_KERNELS_H_
#define _PARTICLE_KERNELS_H_

void KerInitPartField(ACCP<double>& field) {

  field(0) = 0;
  field(1) = 0;
}


void KerUpdateParticleVel(ACCP<double>& up, const ACCP<double>& xp, const ACC<double>& u_x,
                          const ACC<double>& u_y, const ACC<double>& xgrid, const double *dx,
                          int *idx) {

  double xi = fabs((xp(0) - xgrid(0, 0, 0)) / (*dx));
  double nu = fabs((xp(1) - xgrid(1, 0, 0)) / (*dx));



  up(0) += (1. - xi) * (1. - nu) * u_x(0, 0, 0);
  up(1) += (1. - xi) * (1. - nu) * u_y(0, 0, 0);

//  printf("R %d: [%12.9e %12.9e] xf = [%12.9e %12.9e] uf = [%12.9e %12.9e] up = [%12.9e %12.9e]\n",
//         ops_get_proc(),xp(0), xp(1), xgrid(0, 0, 0), xgrid(1, 0, 0), u_x(0, 0, 0), u_y(0, 0, 0), up(0), up(1));

}

void KerUpdatePosition(ACCP<double>& xp, const ACCP<double>& up, const double *dt) {

//  printf("dt = %f\n", *dt);
  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

#endif /* APPS_C_OPS_LBM_CAVITY_MPI_PARTICLE_KERNELS_H_ */
