#ifndef GENERATE_KERNEL_H
#define GENERATE_KERNEL_H

void KerGenerateKernel(ACC<double>& coords, const double *dx,
                      int* idx) {

  coords(0, 0, 0, 0) = static_cast<double>(idx[0]) * (*dx);
  coords(1, 0, 0, 0) = static_cast<double>(idx[1]) * (*dx);
  coords(2, 0, 0, 0) = static_cast<double>(idx[2]) * (*dx);

}

#endif /* GENERATE_KERNEL_H */
