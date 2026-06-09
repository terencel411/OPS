#ifndef GENERATE_KERNEL_H
#define GENERATE_KERNEL_H

void KerGenerateKernel(ACC<double>& coords, ACC<double>& test_field,
                       const double *dx, int* idx) {
  coords(0, 0, 0, 0) = static_cast<double>(idx[0]) * (dx[0]);
  coords(1, 0, 0, 0) = static_cast<double>(idx[1]) * (dx[0]);
  coords(2, 0, 0, 0) = static_cast<double>(idx[2]) * (dx[0]);

  test_field(0, 0, 0, 0) =  static_cast<double>(idx[0]);
  test_field(1, 0, 0, 0) = static_cast<double>(idx[1]);
  test_field(2, 0, 0, 0) = static_cast<double>(idx[2]);

}

int KerDecide(int dim, double xmin[], double xmax[], double *xlocal,
               double *shape) {

  for (int i = 0; i < dim; i++) {
    if (xlocal[i] < xmin[i] || xlocal[i] > xmax[i])
      return 0;
  }

  return 1;
}

void KerInsertData(const double *dats, ACCP<double>& rad,const double *rad_ins,
                   ACCP<double>& adds, const double *inse) {

  rad(0) = *rad_ins;

  adds(0, 0) = inse[0];
  adds(1, 0) = inse[1];
  adds(2, 0) = inse[2];
}

void KerComputeVel(const double *dt, ACC<double>& field, ACC<double>& coords, ACCP<double>& xpos,
                   ACCP<double>& u) {
  u(0, 0) += (*dt) * 0.02;
  u(1, 0) += (*dt) * 0.02;
  u(2, 0) += (*dt) * 0.01;

//  printf("Grid points are [%f %f %f] and particle crds are [%f %f %f]\n",coords(0, 0, 0, 0), coords(1, 0, 0, 0),
//         coords(2, 0, 0, 0), xpos(0, 0), xpos(1, 0), xpos(2, 0));
//  printf("Grid points of adjacent cells [%f %f %f]\n", coords(0, 0, 1, 0), coords(1, 0, 1, 0), coords(2, 0, 1, 0));

}

void KernelGrid(ACC<double>& field_grid, const ACCP<double>& field_prt) {
   printf("ACC field [%f %f %f] particle field =[%f %f %f]\n",
          field_grid(0, 0, 0, 0), field_grid(1, 0, 0, 0),
          field_grid(2, 0, 0, 0), field_prt(0, 0),
          field_prt(1, 0), field_prt(2, 0));
}
#endif /* GENERATE_KERNEL_H */
