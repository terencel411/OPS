/*
 * kernel_functions.h
 *
 *  Created on: Sep 3, 2025
 *      Author: valantis
 */

#ifndef KERNEL_FUNCTIONS_H_
#define KERNEL_FUNCTIONS_H_

inline double calculate_feq2D(const double rho,const double u, const double v,
                              const int cx,const int cy, const double wi,
                              double cs) {

  double c_x = static_cast<double>(cx);
  double c_y = static_cast<double>(cy);

  double cu = cs * c_x * u + cs * c_y * v;
  double c2 = cs * c_x * cs * c_x
            + cs * c_y * cs * c_y;
  double cu2 = cu * cu;
  double u2 = u * u + v * v;
  double res = 1.0 + cu + 0.5 * (cu2 - u2);

  return wi * rho * res;
}

inline double interpolate_data_to_elem(double data[], double x_int[],
                                       double xf[], double yf[], double  dx) {

  double xcm = (xf[0] + xf[1]) * 0.5;
  double ycm = (yf[0] + yf[2]) * 0.5;

  double xi = (x_int[0] - xcm) / (0.5 * dx);
  double nu = (x_int[1] - ycm) / (0.5 * dx);

  double N1, N2, N3, N4;

  N1 = 0.25 * (1. - xi) * (1. - nu);
  N2 = 0.25 * (1. + xi) * (1. - nu);
  N3 = 0.25 * (1. + xi) * (1. + nu);
  N4 = 0.25 * (1. - xi) * (1. + nu);
/*
  printf("1-xi 1 + nu = [%12.9e %12.9e]\n", 1-xi, 1 + nu);
  printf("xi = %12.9e nu = %12.9e\n", xi, nu);
  printf("N1 = %12.9e N2 = %12.9e N3 = %12.9e N4 = %12.9e\n",
         N1, N2, N3, N4);
*/
  return data[0] * N1 + data[1] * N2 + data[2] * N3 + data[3] * N4;



}
#endif /* APPS_C_OPS_LBM_CAVITY_2D_KERNEL_FUNCTIONS_H_ */
