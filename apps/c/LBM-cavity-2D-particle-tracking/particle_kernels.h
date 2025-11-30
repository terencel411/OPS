// Kernels for particle data structures

#include "kernel_functions.h"

void KerInsertData(ACCP<double>& up) {
  up(0, 0) = 0.0;
  up(1, 0) = 0.0;
}

int KerDecide(int dim, double xmin[], double xmax[], double *xlocal,
              double *shape) {
  for (int i = 0; i < dim; i++) {
    if (xlocal[i] < xmin[i] || xlocal[i] > xmax[i])
      return 0;
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
                               const ACC<double>& x_f,  const double *dx) {

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

  // printf("Particle x = [%12.9e %12.9e] dx = %12.9e\n", xint[0], xint[1], *dx);

  u_p(0, 0) = interpolate_data_to_elem(u_fl, xint, x_fl, y_fl, *dx); //TODO
  u_p(1, 0) = interpolate_data_to_elem(v_fl, xint, x_fl, y_fl, *dx); //TODO

  if (xint[0] == 1.0)
    u_p(0, 0) = 0.0;

/*
  double xcm =  0.5 * (x_fl[0] + x_fl[1]);
  double ycm =  0.5 * (y_fl[0] + y_fl[3]);
  printf("u_nodes = [%12.9e %12.9e %12.9e %12.9e]\n",
         u_fl[0], u_fl[1], u_fl[2], u_fl[3]);
  printf("v_nodes = [%12.9e %12.9e %12.9e %12.9e]\n", v_fl[0], v_fl[1], v_fl[2], v_fl[3]);
  printf("xfl = [%12.9e %12.9e %12.9e %12.9e]\n", x_fl[0], x_fl[1], x_fl[2], x_fl[3]);

  printf("xi= %12.9e nu  = %12.9e\n", (xint[0] - xcm) / (0.5 * (*dx)),
         (xint[1] - ycm) / (0.5 * (*dx)));
  printf("Computed velocity: u_p = [%12.9e %12.9e]\n", u_p(0, 0), u_p(1, 0));
*/

}
