// Kernels for particle data structures

#include "kernel_functions.h"

void KerInsertData(ACCP<double>& up, ACCP<double>& Rp) {
  up(0, 0) = 0.0;
  up(1, 0) = 0.0;
  Rp(0) = 0.5;
}

int KerDecide(int dim, double xmin[], double xmax[], double *xlocal,
              double *shape) {

  for (int i = 0; i < dim; i++) {
    if (xlocal[i] < xmin[i] || xlocal[i] > xmax[i]) {
      return 0;
    }
  }

  return 1;
}

void KerUpdatePosition(ACCP<double> &x_p, const ACCP<double>& u_p,
                       const double *dt) {

 // printf("Prior update xp = [%12.9e %12.9e]\n", x_p(0, 0), x_p(1, 0));

  double xold = x_p(0, 0);
  x_p(0, 0) += u_p(0, 0) * (*dt);
  if (x_p(0, 0) <= 0.0)
    x_p(0, 0) = 0.5 * xold;
  else if (x_p(0, 0) >= 1.0)
    x_p(0, 0) = 0.5 * (xold + 1.0);

  xold = x_p(1, 0);

  x_p(1, 0) += u_p(1, 0) * (*dt);
  if (x_p(1, 0) <= 0.)
    x_p(1, 0) = 0.5 * xold;
  else if (x_p(1, 0) >= 1.0)
    x_p(1, 0) = 0.5 * (xold + 1.0);

  //printf("x_p = [%12.9e %12.9e]\n", x_p(0, 0), x_p(1, 0));
}


void KerUpdateParticleVelocity(ACCP<double>& u_p, const ACCP<double>& x_p,
                               const ACC<double>& u_f, const ACC<double>& v_f,
                               const ACC<double>& x_f,  const double *dx,
                               const int *idx) {

  //Get positions of fluid
  double x_fl[4], y_fl[4];
  double u_fl[4], v_fl[4];
  double xint[2];

  x_fl[0] = x_f(0, 0, 0);
  x_fl[1] = x_f(0, 1, 0);
  x_fl[2] = x_f(0, 1, 1);
  x_fl[3] = x_f(0, 0, 1);


  u_fl[0] = u_f(0, 0, 0);
  u_fl[1] = u_f(0, 1, 0);
  u_fl[2] = u_f(0, 1, 1);
  u_fl[3] = u_f(0, 0, 1);

  //Get data in y-direction
  y_fl[0] = x_f(1, 0, 0);
  y_fl[1] = x_f(1, 1, 0);
  y_fl[2] = x_f(1, 1, 1);
  y_fl[3] = x_f(1, 0, 1);

  v_fl[0] = v_f(0, 0, 0);
  v_fl[1] = v_f(0, 1, 0);
  v_fl[2] = v_f(0, 1, 1);
  v_fl[3] = v_f(0, 0, 1);

  xint[0] = x_p(0, 0);
  xint[1] = x_p(1, 0);



//  printf("Particle ( x =(%f %f) in [%d %d]: [0, 0] [%f %f] [1, 0] [%f %f] [1 1] [%f %f] [0 1] [%f %f]\n",
//         x_p(0), x_p(1), idx[0],idx[1], x_fl[0], y_fl[0], x_fl[1], y_fl[1], x_fl[2], y_fl[2], x_fl[3], y_fl[3]);
  // printf("Particle x = [%12.9e %12.9e] dx = %12.9e\n", xint[0], xint[1], *dx);

  u_p(0, 0) = interpolate_data_to_elem(u_fl, xint, x_fl, y_fl, *dx); //TODO
  u_p(1, 0) = interpolate_data_to_elem(v_fl, xint, x_fl, y_fl, *dx); //TODO

  if (xint[0] == 1.0)
    u_p(0, 0) = 0.0;

}

void KernelInitParticleVel(ACCP<double>& up, const int *idp) {

  up(0) = 0.;
  up(1) = 0.;
}

void KernelInitParticleField(ACCP<double>& field, const int *dim) {
  for (int idir = 0; idir < *dim; idir++)
    field(idir) = 0.0;
}

void KernelComputeParticleGridInter(ACCP<double> &up, const ACCP<double>& xp,
                                    const ACC<double>&xf, const ACC<double> &ux,
                                    const ACC<double>&uy,
                                    const double *dx, const int *id_map,
                                    const int *idx, const int *idp) {

  //Compute relative positions with respect to map point
  int ix = idx[0] - id_map[0];
  int iy = idx[1] - id_map[1];

  //With respect to current point
  double x_o[2]= { xf(0, -ix, -iy), xf(1, -ix, -iy)};
  double x_part[2] = {xp(0), xp(1)};
  double interpol = shape_func_interpol(ix, iy, x_part, x_o, *dx);
  up(0) += interpol * ux(0, 0, 0);
  up(1) += interpol  * uy(0, 0, 0);
}

void KerSolidFraction(ACC<double>& sf, ACC<double>& up_grid, const ACC<double>& x_grid,
                      const ACCP<double>& xp, const ACCP<double>& R_p,
                      const ACCP<double>& up,
                      const double *dx, const int *idx, const int *idp) {

  double d = (xp(0) - x_grid(0, 0, 0)) * (xp(0) - x_grid(0, 0, 0))
           + (xp(1) - x_grid(1, 0, 0)) * (xp(1) - x_grid(1, 0, 0));


  up_grid(0, 0, 0) = 0.0;
  up_grid(1, 0, 0) = 0.0;

  double PI = 3.141592653589793;
  double Rg = sqrt(0.5)* (*dx);
  double Rp = R_p(0);
  double sfp = 0.0;
  if (d < (Rp + Rg) * (Rp + Rg)) {

    up_grid(0, 0, 0) = up(0);
    up_grid(1, 0, 0) = up(1);

    if (d - (Rp-Rg) * (Rp - Rg) < 0.00000000001) {
      sfp = 1.0;
    }
    else {

      double dsqrt = sqrt(d);
      double d1 = (Rg * Rg - Rp * Rp + d) / (2. * dsqrt);
      double d2 = (Rp * Rp - Rg * Rg + d) / (2. * dsqrt);

      double vol = Rg * Rg * acos(d1/ Rg) - d1 * sqrt(Rg * Rg - d1 * d1)
                 + Rp * Rp * acos(d2/ Rp) - d2 * sqrt(Rp * Rp - d2 * d2);

      sfp = vol/(PI * Rg * Rg);
    }
  }

  sf(0, 0, 0) = sfp;

}

void KerComputeDragForce(ACCP<double>& Fd, const ACCP<double>& xp, const ACCP<double>& radp,
                         const ACC<double>& gFd,
                         const ACC<double>& xf, const double *dx,
                         const double *dt) {
  double Fx[] = {gFd(0, 0, 0), gFd(1, 0, 0)};

  double dx_part[] = {xp(0) - xf(0, 0, 0), xp(1) - xf(1, 0, 0)};

  double d = dx_part[0] * dx_part[0] + dx_part[1] * dx_part[1];
  double Rg = sqrt(0.5) * (*dx);
  if (d < (radp(0) + Rg) * (radp(0) + Rg)) {

    Fd(0) += Fx[0] * (*dx) * (*dx) / (*dt);
    Fd(1) += Fx[1] * (*dx) * (*dx) / (*dt);
    Fd(2) += (dx_part[0] * Fx[1] - dx_part[1] * Fx[0]) * (*dx) * (*dx) / (*dt);
  }

}



