//Kernels for lattice-boltzmann demo app
#include <math.h>
#include "kernel_functions.h"

void Ker_Diffuse_Refl_Norm_xneg(ACC<double> &f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f3 = f(3, 0, 0);
  double f7 = f(7, 0, 0);
  double f6 = f(6, 0, 0);

  double rho_w =  6 * (f3 + f6 + f7) / (u * u - sqrt3 * u + 1);//node

  f(5, 0, 0) = f7 + rho_w * ( u + v) / (6. * sqrt3);
  f(1, 0, 0) = f3 + 2. * rho_w * u / (3. * sqrt3);
  f(8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2. - u * u - v * v) / 9.;
  f(2, 0, 0) = -(u * u - 2. * (1. + sqrt3 * v + v * v) * rho_w) / 18.;
  f(4, 0, 0) = -((-2. + u * u + 2. * sqrt3 * v - 2. * v * v) * rho_w) / 18;
}

void Ker_Diffuse_Refl_Norm_Poix_xneg(ACC<double>& f, const ACC<double>& xf,
                                     const double *uw, const double *H) {

  double yc = (xf(1, 0, 0) - 0.5 * (*H));
  double u = uw[0] * (1.0 - yc * yc / (0.25 * (*H) * (*H)));

  double v = uw[1];

  double sqrt3 = sqrt(3);

  double f3 = f(3, 0, 0);
  double f7 = f(7, 0, 0);
  double f6 = f(6, 0, 0);

  double rho_w =  6 * (f3 + f6 + f7) / (u * u - sqrt3 * u + 1);//node

  f(5, 0, 0) = f7 + rho_w * ( u + v) / (6. * sqrt3);
  f(1, 0, 0) = f3 + 2. * rho_w * u / (3. * sqrt3);
  f(8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2. - u * u - v * v) / 9.;
  f(2, 0, 0) = -(u * u - 2. * (1. + sqrt3 * v + v * v) * rho_w) / 18.;
  f(4, 0, 0) = -((-2. + u * u + 2. * sqrt3 * v - 2. * v * v) * rho_w) / 18;


}


void Ker_Diffuse_Refl_Norm_xpos(ACC<double> &f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f5 = f(5, 0, 0);
  double f1 = f(1, 0, 0);
  double f8 = f(8, 0, 0);

  double rho_w = 6 * (f1 + f5 + f8) / (u * u + sqrt3 * u + 1);//node

  //Update populations
  f(7 ,0, 0) = f5 - rho_w * (u + v) / (6. * sqrt3);
  f(3, 0, 0) = f1 - 2. * rho_w * u / (3. * sqrt3);
  f(6, 0, 0) = f8 + rho_w * (v - u) / (6. * sqrt3);
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
  f(2, 0, 0) = -(u * u - 2. * (1. + sqrt3 * v + v * v) * rho_w) / 18.;
  f(4, 0, 0) = -((-2. + u * u + 2. * sqrt3 * v - 2. * v * v) * rho_w) / 18;
}

void Ker_Diffuse_Refl_Norm_Pois_xpos(ACC<double>& f, ACC<double>& xf,
                                     double *uw, double *H) {

  double yc = (xf(1, 0, 0) - 0.5 * (*H));
  double u = uw[0] * (1.0 - yc * yc / (0.25 * (*H) * (*H)));

  double v = uw[1];

  double f5 = f(5, 0, 0);
  double f1 = f(1, 0, 0);
  double f8 = f(8, 0, 0);

  double sqrt3 = sqrt(3);


  double rho_w = 6 * (f1 + f5 + f8) / (u * u + sqrt3 * u + 1);//node

  //Update populations
  f(7 ,0, 0) = f5 - rho_w * (u + v) / (6. * sqrt3);
  f(3, 0, 0) = f1 - 2. * rho_w * u / (3. * sqrt3);
  f(6, 0, 0) = f8 + rho_w * (v - u) / (6. * sqrt3);
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
  f(2, 0, 0) = -(u * u - 2. * (1. + sqrt3 * v + v * v) * rho_w) / 18.;
  f(4, 0, 0) = -((-2. + u * u + 2. * sqrt3 * v - 2. * v * v) * rho_w) / 18;
}

void Ker_Diffuse_Refl_Norm_xpos_yneg(ACC<double>& f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f1 = f(1, 0, 0);
  double f4 = f(4, 0, 0);
  double f8 = f(8, 0, 0);
  double rho_w = (36 * (f1 + f4 + f8)) /
                    (9. + 5. * sqrt3 * u + 3. * u * u -
                     5. * sqrt3 * v - 3. * u * v + 3. * v * v);//node

  f(6, 0, 0) = f8 + rho_w * (v - u) / (6 * sqrt3);
  f(3, 0, 0) = f1 - 2 * rho_w * u / (3 * sqrt3);
  f(2, 0, 0) = f4 + 2 * rho_w * v / (3 * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2 - u * u - v * v) / 9;
  f(5, 0, 0) = (1 + u * u + sqrt3 * v + v * v +
                             u * (sqrt3 + 3 * v))* rho_w /
                            36;
  f(7, 0, 0) =
      (1 - sqrt3 * u + u * u - sqrt3 * v + 3 * u * v + v * v) *
      rho_w / 36;
}

void Ker_Diffuse_Refl_Norm_yneg(ACC<double> &f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f4 = f(4, 0, 0);
  double f8 = f(8, 0, 0);
  double f7 = f(7, 0, 0);

  double rho_w = 6. * (f4 + f8 + f7) / (v * v - sqrt3 * v + 1.);
  f(2, 0, 0) = f4 + 2. * rho_w * v / (3. * sqrt3);
  f(6, 0, 0) = f8 + rho_w * (v - u) / (6. * sqrt3);
  f(5, 0, 0) = f7 + rho_w * (u + v) / (6. * sqrt3);
  f(1, 0, 0) = ((2. + 2. * sqrt3 * u + 2. * u * u - v * v) * rho_w) / 18.;
  f(3, 0, 0) = -((-2. + 2. * sqrt3 * u - 2. * u * u + v * v) * rho_w) / 18;
  f(0, 0, 0) =  2 * rho_w * (2. - u * u - v * v) / 9;
}

void Ker_Diffuse_Refl_Norm_ypos(ACC<double> &f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f2 = f(2, 0, 0);
  double f5 = f(5, 0, 0);
  double f6 = f(6, 0, 0);

  double rho_w = 6. * (f2 + f5 + f6) / (v * v + sqrt3 * v + 1);

  f(4, 0, 0) = f2 - 2. * rho_w * v / (3. * sqrt3);
  f(8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  f(7, 0, 0) = f5 - rho_w * (u + v) / (6. * sqrt3);
  f(1, 0, 0) = ((2. + 2. * sqrt3 * u + 2. * u * u - v * v) * rho_w) / 18.;
  f(3, 0, 0) = -((-2 + 2 * sqrt3 * u - 2. * u * u + v * v) * rho_w) / 18;
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
}

void Ker_Diffuse_Refl_Norm_xneg_yneg(ACC<double>&f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f3 = f(3, 0, 0);
  double f7 = f(7, 0, 0);
  double f4 = f(4, 0, 0);

  double rho_w = (-36. * (f3 + f4 + f7)) /
                 (-9. + 5. * sqrt3 * u - 3. * u * u + 5. * sqrt3 * v
                  -3. * u * v - 3. * v * v);

  f(1, 0, 0) = f3 + 2. * rho_w * u / (3. * sqrt3);
  f(5, 0, 0) = f7 + rho_w * (u + v) / (6. * sqrt3);
  f(2, 0, 0) = f4 + 2. * rho_w * v / (3. * sqrt3);
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
  f(6, 0, 0) = (1. + u * u + sqrt3 * v + v * v - u * (sqrt3 + 3. * v)) * rho_w / 36;
  f(8, 0, 0) = (1. + u * u + u * (sqrt3 - 3. * v) - sqrt3 * v + v * v) *rho_w / 36;
}

void Ker_Diffuse_Refl_Norm_xneg_ypos(ACC<double> &f, const double *u_w) {
  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f2 = f(2, 0, 0);
  double f3 = f(3, 0, 0);
  double f6 = f(6, 0, 0);

  double rho_w = (-36 * (f2 + f3 + f6)) /
                  (-9. + 5. * sqrt3 * u - 3. * u * u -
                    5. * sqrt3 * v + 3. * u * v - 3. * v * v);
  f(8, 0, 0) = f6 + rho_w * (u - v) / (6. * sqrt3);
  f(1, 0, 0) = f3 + 2 * rho_w * u / (3. * sqrt3);
  f(4, 0, 0) = f2 - 2 * rho_w * v / (3. * sqrt3);
  f(0, 0, 0) = 2. * rho_w * (2. - u * u - v * v) / 9.;
  f(5, 0, 0) = (1. + u * u + sqrt3 * v + v * v +
                u * (sqrt3 + 3 * v)) * rho_w / 36.;
  f(7, 0, 0) = (1. - sqrt3 * u + u * u - sqrt3 * v + 3 * u * v + v * v) * rho_w / 36.;
}

void Ker_Diffuse_Refl_xpos_ypos(ACC<double>&f, const double *u_w) {

  double u = u_w[0];
  double v = u_w[1];

  double sqrt3 = sqrt(3);

  double f1 = f(1, 0, 0);
  double f5 = f(5, 0, 0);
  double f2 = f(2, 0, 0);
  double rho_w = (36 * (f1 + f2 + f5)) /
                    (9 + 5 * sqrt3 * u + 3 * u * u +
                     5 * sqrt3 * v + 3 * u * v + 3 * v * v);//Node

  f(3, 0, 0) = f1 - 2 * rho_w * u / (3 * sqrt3);
  f(4, 0, 0) = f2 - 2 * rho_w * v / (3 * sqrt3);
  f(7, 0, 0) = f5 - rho_w * (u + v) / (6 * sqrt3);
  f(0, 0, 0) = 2 * rho_w * (2 - u * u - v * v) / 9;
  f(6, 0, 0) =
      (1 + u * u + sqrt3 * v + v * v - u * (sqrt3 + 3 * v)) *
      rho_w / 36;
  f(8, 0, 0) =
      (1 + u * u + u * (sqrt3 - 3 * v) - sqrt3 * v + v * v) *
      rho_w / 36;
}

void KerComputeMacros(ACC<double>& rho, ACC<double>& u, ACC<double>& v,
                      const ACC<double>& f,
                      const int *cx, const int *cy, const double *cs) {
  rho(0, 0) = 0.0;
  u(0, 0) = 0.0;
  v(0, 0) = 0.0;

  for (int i = 0; i < 9; i++) {
    rho(0, 0) += f(i, 0, 0);
    u(0, 0) += (*cs) * static_cast<double>(cx[i]) * f(i, 0, 0);
    v(0, 0) += (*cs) * cy[i] * f(i, 0, 0);
  }

  u(0, 0) /= rho(0, 0);
  v(0, 0) /= rho(0, 0);
}

void KerStream(ACC<double>& f, const ACC<double>& f_work,
               const int *cx, const int *cy) {
  for (int i = 0; i < 9; i++)
    f(i, 0, 0) = f_work(i, -cx[i], -cy[i]); //df(0, 0) = (f(0,1, 0) - f(0, -1, 0))/(2 * dx); -
}

void KerCopy(ACC<double>& f_dest, const ACC<double>& f_src) {

  for (int i = 0; i < 9; i++)
    f_dest(i, 0, 0) = f_src(i, 0, 0);
}

void KerCollision(ACC<double>& f_work, const ACC<double>& f, const ACC<double>& rho,
                  const ACC<double>& u, const ACC<double>& v, const double *omega,
                  const int *cx, const int *cy, const double *wi, const double *cs) {


  double rhoF = rho(0, 0);
  double uF = u(0, 0);
  double vF = v(0, 0);
  for (int i = 0; i < 9; i++) {
    double feq = calculate_feq2D(rhoF, uF, vF, cx[i], cy[i], wi[i], *cs);
    f_work(i, 0, 0) = (1. - (*omega)) * f(i, 0, 0) + (*omega) * feq;
  }

}

void KerInitMacros(ACC<double>& rho, ACC<double>& u, ACC<double>& v) {

  rho(0, 0) = 1.0;
  u(0, 0) = 0.0;
  v(0, 0) = 0.0;
}

void KerInitMacrosPois(ACC<double>& rho, ACC<double>& u, ACC<double>& v,
                       const ACC<double>& xf, const double *u0,
                       const double *H) {
  rho(0, 0 ) = 1.0;
  v(0, 0) = 0.0;

  double yc = (xf(1, 0, 0) - 0.5 * (*H));
  u(0, 0) = (*u0) * (1.0 - yc * yc / (0.25 * (*H) * (*H)));
}

void KerInitF(ACC<double>& f, const ACC<double>& rho, const ACC<double>& u,
              const ACC<double>& v, const int *cx, const int *cy,
              const double *wi, const double *cs) {
  double rhoF = rho(0, 0);
  double uF = u(0, 0);
  double vF = v(0, 0);
  for (int i = 0; i < 9; i++)
    f(i, 0, 0) = calculate_feq2D(rhoF, uF, vF, cx[i], cy[i], wi[i], *cs);
}

void KerInitGrid(ACC<double>& xf, const double *dx, const int *idx)  {
  xf(0, 0, 0) = dx[0] * static_cast<double>(idx[0]);
  xf(1, 0, 0) = dx[1] * static_cast<double>(idx[1]);
}

void KerInitField(ACC<double>& field, const int *ndim) {

  for (int idir = 0; idir < *ndim; idir++)
    field(idir, 0, 0) = 0.0;
}

void KerCollisionPSM(ACC<double>& fWork, ACC<double>& Fd, const ACC<double>& f,
                     const ACC<double>& rho, const ACC<double>& u_fluid,
                     const ACC<double>&v_fluid,
                     const ACC<double>& sf, const ACC<double>& u_part, const double *omega,
                     const int *cx, const int *cy, const double *wi, const double *oppos,
                     const double *cs)
{
  double feq[9];
  //Compute feq

  double rhoF = rho(0, 0);
  double uF = u_fluid(0, 0);
  double vF = v_fluid(0, 0);

  double sfp = sf(0, 0);
  double poros = 1.0 - sfp;

  double Beta = sfp * (1/(*omega) - 0.5) / (1 - sfp + 1./(*omega) - 0.5);

  //Beta = sfp;

  double fd_loc[2] = {0.0, 0.0};
  for (int ivel = 0; ivel < 9; ivel++)
    feq[ivel] = calculate_feq2D(rhoF, uF, vF, cx[ivel], cy[ivel], wi[ivel], *cs);

  for (int ivel = 0; ivel < 9; ivel++) {
    fWork(ivel, 0, 0) = f(ivel, 0, 0) * (1.0 - (1.0 - Beta) * (*omega))
                      + (1. - Beta) * (*omega) * feq[ivel];

    double OmegaS = 0.0;
    if (sf(0, 0) > 0.0) {
      int opp = oppos[ivel];
      double feqS = calculate_feq2D(rhoF, u_part(0, 0, 0),  u_part(1, 0, 0),
                                    cx[ivel], cy[ivel], wi[ivel], *cs);
      OmegaS = (-f(ivel, 0, 0) + feqS) * Beta;

      fd_loc[0] -= (*cs) * cx[ivel] * OmegaS;
      fd_loc[1] -= (*cs) * cy[ivel] * OmegaS;
    }

    fWork(ivel, 0, 0) +=  OmegaS;
  }

  Fd(0, 0, 0) = fd_loc[0];
  Fd(1, 0, 0) = fd_loc[1];


}
