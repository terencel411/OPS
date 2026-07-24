/*
 * kernel_functions.h
 *
 *  Created on: Jan 27, 2026
 *      Author: valantis
 */

#ifndef _KERNEL_FUNCTIONS_H_
#define _KERNEL_FUNCTIONS_H_

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

inline double calc_body_force(const double uF, const double vF, const double *force, double cx,
                              double cy, const double wi,const double cs) {

  double sum = 0.0;
  double ub[2] = {uF, vF};

  double c_x = static_cast<double>(cx) * cs;
  double c_y = static_cast<double>(cy) * cs;
  double c1[] = {c_x, c_y};
  double sum1;
  for (int i = 0; i < 2; i++)  {
    sum1 = c1[i];
    for (int j = 0; j < 2; j++) {
      sum1 += c1[i] * c1[j] * ub[j];
      if (i == j)
        sum1 -= 1. * ub[j];
    }
    sum += sum1 * force[i];
  }

  return sum * wi;
}


#endif /* APPS_C_OPS_LBM_CAVITY_MPI_KERNEL_FUNCTIONS_H_ */
