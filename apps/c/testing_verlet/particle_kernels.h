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

}

void KerUpdatePosition(ACCP<double>& xp, const ACCP<double>& up, const double *dt) {

  xp(0) += up(0) * (*dt);
  xp(1) += up(1) * (*dt);
}

void KernelGravity(ACCP<double>& Fd, const ACCP<double>& mass, const double *g) {


  Fd(0) = mass(0) * g[0];
  Fd(1) = mass(0) * g[1];
}

void KernelWallParticleInteraction(ACCPJ<double>& F, const ACCP<double>& xw,
                                   const ACCPJ<double>& xp, const ACCP<double>& nw,
                                   const ACCPJ<double>&up, const ACCPJ<int>& id,
                                   ACC_HIS<double>& his, const double *Kn) {

  double h[2] = {xp(0) - xw(0), xp(1) - xw(1)};


  double hnorm = h[0] * nw(0) + h[1] * nw(1);

  double delta = 0.1 - hnorm;

  if  (0.1 - hnorm > 0) {

    double Kn1, Kn2;

    if (id(0) == 1) {
      Kn1 = Kn[4];
      Kn2 = Kn[5];

    }
    else if (id(0) == 2) {
      Kn1 = Kn[2];
      Kn2 = Kn[3];
    }
    else {
      Kn1 = Kn[0];
      Kn2 = Kn[1];
    }

    double delta = 0.1 - hnorm;
    double un = up(0) * nw(0) + up(1) * nw(1);
    double fn;
    if (delta >= his(0)) {
      fn = Kn1 * delta;
      his(0) = delta;
    }
    else {
      double delta_p = his(0) * (1 - Kn1/Kn2);
      fn = Kn2 * (delta - delta_p) ;
    }

    F(0) += fn * nw(0);
    F(1) += fn * nw(1);

  }
  else {
    his(0) = 0.0;
    his(1) = 0.0;
  }
}

void KerUpdateInitialIntegrate(ACCP<double>& up, ACCP<double>& xpos, const ACCP<double>& F,
                               const ACCP<double>& mp, const double *dt) {
   double dtf = 0.5 * dt[0] / mp(0);
   up(0) += dtf * F(0);
   up(1) += dtf * F(1);



   xpos(0) += dt[0] * up(0);
   xpos(1) += dt[0] * up(1);
}

void KerUpdateFinalIntegrate(ACCP<double>& up, const ACCP<double>& F,
                             const ACCP<double>& mp, const double *dt) {
   double dtf = 0.5 * dt[0] / mp(0);
   up(0) += dtf * F(0);
   up(1) += dtf * F(1);
}


void KerSetHistory(double *data, const int *ndofs) {
  for (int i = 0; i < *ndofs; i++) data[i] = 0;
}
#endif /* APPS_C_OPS_LBM_CAVITY_MPI_PARTICLE_KERNELS_H_ */
