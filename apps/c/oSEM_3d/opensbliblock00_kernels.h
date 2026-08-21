#ifndef OPENSBLIBLOCK00_KERNEL_H
#define OPENSBLIBLOCK00_KERNEL_H

void interp_RST(const ACC<double>& x1_B0, ACC<double>& a11, ACC<double>& a21, ACC<double>& a22, ACC<double>& a33, const double* ydata, const double* uudata, const double* uvdata, const double* vvdata, const double* wwdata){
  // assumes zero uw and vw terms

  double y = 0.08548942989301017 * x1_B0(0,0,0);
  double uu;
  double uv;
  double vv;
  double ww;
 
  if (y >= 1.0 || y <= 0.0){
    a11(0,0,0) = 0.0;
    a21(0,0,0) = 0.0;
    a22(0,0,0) = 0.0;
    a33(0,0,0) = 0.0;
  }
  else{
    for (int i{1}; i < ndata; i++){
      if(x1_B0(0,0,0) < ydata[i]){
        // evaluate RST 
        uu = uudata[i-1] + (uudata[i] - uudata[i-1]) / (ydata[i] - ydata[i-1]) * (x1_B0(0,0,0) - ydata[i-1]);
        uv = uvdata[i-1] + (uvdata[i] - uvdata[i-1]) / (ydata[i] - ydata[i-1]) * (x1_B0(0,0,0) - ydata[i-1]);
        vv = vvdata[i-1] + (vvdata[i] - vvdata[i-1]) / (ydata[i] - ydata[i-1]) * (x1_B0(0,0,0) - ydata[i-1]);
        ww = wwdata[i-1] + (wwdata[i] - wwdata[i-1]) / (ydata[i] - ydata[i-1]) * (x1_B0(0,0,0) - ydata[i-1]);
        
        a11(0,0,0) = sqrt(uu);
        a21(0,0,0) = uv / a11(0,0,0);
        a22(0,0,0) = sqrt(vv - a21(0,0,0)*a21(0,0,0));
        a33(0,0,0) = sqrt(ww);
        
        break;
      }
    }
  }
}

void instantiate_eddies(ACC<double>& eddy_x, ACC<double>& eddy_y, ACC<double>& eddy_z, ACC<double>& eddy_r, ACC<double>& eddy_increment, ACC<int>& eddy_eps_x, ACC<int>& eddy_eps_y, ACC<int>& eddy_eps_z, const ACC<int>& eddy_x_rng, const ACC<int>& eddy_bulk_rng){
  eddy_x(0,0,0) = eddy_x_min + (eddy_x_rng(0,0,0) + 2147483648.0) / (4294967295.0) * (eddy_x_max - eddy_x_min);
  eddy_y(0,0,0) = eddy_y_min + (eddy_bulk_rng(0,0,0,0) + 2147483648.0) / (4294967295.0) * (eddy_y_max - eddy_y_min);
  eddy_z(0,0,0) = eddy_z_min + (eddy_bulk_rng(1,0,0,0) + 2147483648.0) / (4294967295.0) * (eddy_z_max - eddy_z_min);
  eddy_eps_x(0,0,0) = (eddy_bulk_rng(2,0,0,0) < 0) ? -1 : 1;
  eddy_eps_y(0,0,0) = (eddy_bulk_rng(3,0,0,0) < 0) ? -1 : 1;
  eddy_eps_z(0,0,0) = (eddy_bulk_rng(4,0,0,0) < 0) ? -1 : 1;
  eddy_r(0,0,0) = radius;
  eddy_increment(0,0,0) = 1.0 * dt;
}

void convect_eddies(ACC<double>& eddy_x, ACC<double>& eddy_y, ACC<double>& eddy_z, ACC<double>& eddy_r, const ACC<double>& eddy_increment, ACC<int>& eddy_eps_x, ACC<int>& eddy_eps_y, ACC<int>& eddy_eps_z, const ACC<int>& eddy_bulk_rng){
  eddy_x(0,0,0) = eddy_x(0,0,0) + eddy_increment(0,0,0);
  
  if(eddy_x(0,0,0) > eddy_x_max){
    eddy_x(0,0,0) = eddy_x_min;
    eddy_y(0,0,0) = eddy_y_min + (eddy_bulk_rng(0,0,0,0) + 2147483648.0) / (4294967295.0) * (eddy_y_max - eddy_y_min);
    eddy_z(0,0,0) = eddy_z_min + (eddy_bulk_rng(1,0,0,0) + 2147483648.0) / (4294967295.0) * (eddy_z_max - eddy_z_min);
    eddy_eps_x(0,0,0) = (eddy_bulk_rng(2,0,0,0) < 0) ? -1 : 1;
    eddy_eps_y(0,0,0) = (eddy_bulk_rng(3,0,0,0) < 0) ? -1 : 1;
    eddy_eps_z(0,0,0) = (eddy_bulk_rng(4,0,0,0) < 0) ? -1 : 1;
  }
}

void uinterp_kernel(ACC<double>& d_uinterp, const ACC<double>& x1_B0, const int* idx){
  double w1;
  double w2;
  if(x1_B0(0,0,0) >= yprofdata[sizeof(yprofdata)/sizeof(double)-1]){
    d_uinterp(0,0,0) = 1.0;
  }
  else{
    for (int i{1}; i < sizeof(yprofdata)/sizeof(double); i++){
      if(x1_B0(0,0,0) < yprofdata[i]){
        w1 = 1 - (x1_B0(0,0,0) - yprofdata[i-1]) / (yprofdata[i]-yprofdata[i-1]);
        w2 = 1 - w1;
        d_uinterp(0,0,0) = w1 * uprofdata[i-1] + w2 * uprofdata[i];
        break;
      }
    }
  }
}

 void opensbliblock00Kernel036(ACC<double> &rhoE_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0, ACC<double>
&rhou2_B0, ACC<double> &x0_B0, ACC<double> &x2_B0, ACC<double> &rho_B0, ACC<double> &x1_B0, const int *idx)
{
   double T = 0.0;
   double p = 0.0;
   double psifn = 0.0;
   double u0 = 0.0;
   double u1 = 0.0;
   double u2 = 0.0;
   double x1d = 0.0;
   x0_B0(0,0,0) = Delta0block0*idx[0];

   x1_B0(0,0,0) = Lx1*sinh(by*invLx1*Delta1block0*idx[1])/sinh(by);

   x2_B0(0,0,0) = Delta2block0*idx[2];

   x1d = 0.08548942989301017*x1_B0(0,0,0);

    u0 = ((x1_B0(0,0,0) < 0) ? (
   0.0
)
: ((x1_B0(0,0,0) > 0 && 0.08548942989301017*x1_B0(0,0,0) < 1.0) ? (
  
      pow(x1d, (1.0/7.0))
)
: (
   1.0
)));


   u1 = 0.0;

   u2 = 0.0;

   psifn = pow(u0, 2.0);

    T = ((x1_B0(0,0,0) <= 0) ? (
   Twall
)
: ((x1_B0(0,0,0) > 0 && 0.08548942989301017*x1_B0(0,0,0) < 1.0) ? (
   Twall
      + psifn*(1 - Twall)
)
: (
   1.0
)));

   p = 1/((Minf*Minf)*gama);

   rho_B0(0,0,0) = gama*p*(Minf*Minf)/T;

   rhou0_B0(0,0,0) = gama*p*u0*(Minf*Minf)/T;

   rhou1_B0(0,0,0) = gama*p*u1*(Minf*Minf)/T;

   rhou2_B0(0,0,0) = gama*p*u2*(Minf*Minf)/T;

   rhoE_B0(0,0,0) = p/(-1 + gama) + 0.5*(u0*u0)*rho_B0(0,0,0);

}

 void opensbliblock00Kernel038(const ACC<double> &x1_B0, ACC<double> &D11_B0, ACC<double> &detJ_B0, ACC<double> &wk4_B0,
const int *idx)
{
   double d1_x1_dy = 0.0;
    d1_x1_dy = invDelta1block0*((idx[1] == 0) ? (
   -3*x1_B0(0,2,0) + 4*x1_B0(0,1,0) - (25.0/12.0)*x1_B0(0,0,0) -
      (1.0/4.0)*x1_B0(0,4,0) + ((4.0/3.0))*x1_B0(0,3,0)
)
: ((idx[1] == 1) ? (
   -(5.0/6.0)*x1_B0(0,0,0) -
      (1.0/2.0)*x1_B0(0,2,0) - (1.0/4.0)*x1_B0(0,-1,0) + ((1.0/12.0))*x1_B0(0,3,0) + ((3.0/2.0))*x1_B0(0,1,0)
)
:
      ((idx[1] == -1 + block0np1) ? (
   -4*x1_B0(0,-1,0) + 3*x1_B0(0,-2,0) - (4.0/3.0)*x1_B0(0,-3,0) +
      ((1.0/4.0))*x1_B0(0,-4,0) + ((25.0/12.0))*x1_B0(0,0,0)
)
: ((idx[1] == -2 + block0np1) ? (
  
      ((1.0/2.0))*x1_B0(0,-2,0) - (3.0/2.0)*x1_B0(0,-1,0) - (1.0/12.0)*x1_B0(0,-3,0) + ((1.0/4.0))*x1_B0(0,1,0) +
      ((5.0/6.0))*x1_B0(0,0,0)
)
: (
   -(2.0/3.0)*x1_B0(0,-1,0) - (1.0/12.0)*x1_B0(0,2,0) + ((1.0/12.0))*x1_B0(0,-2,0)
      + ((2.0/3.0))*x1_B0(0,1,0)
)))));

   wk4_B0(0,0,0) = d1_x1_dy;

   detJ_B0(0,0,0) = d1_x1_dy;

   D11_B0(0,0,0) = 1.0/(d1_x1_dy);

}

void opensbliblock00Kernel039(ACC<double> &D11_B0, ACC<double> &detJ_B0)
{
   D11_B0(-1,0,0) = D11_B0(1,0,0);

   detJ_B0(-1,0,0) = detJ_B0(1,0,0);

   D11_B0(-2,0,0) = D11_B0(2,0,0);

   detJ_B0(-2,0,0) = detJ_B0(2,0,0);

}

void opensbliblock00Kernel040(ACC<double> &D11_B0, ACC<double> &detJ_B0)
{
   D11_B0(1,0,0) = D11_B0(-1,0,0);

   detJ_B0(1,0,0) = detJ_B0(-1,0,0);

   D11_B0(2,0,0) = D11_B0(-2,0,0);

   detJ_B0(2,0,0) = detJ_B0(-2,0,0);

}

void opensbliblock00Kernel041(ACC<double> &D11_B0, ACC<double> &detJ_B0)
{
   D11_B0(0,-1,0) = D11_B0(0,1,0);

   detJ_B0(0,-1,0) = detJ_B0(0,1,0);

   D11_B0(0,-2,0) = D11_B0(0,2,0);

   detJ_B0(0,-2,0) = detJ_B0(0,2,0);

}

void opensbliblock00Kernel042(ACC<double> &D11_B0, ACC<double> &detJ_B0)
{
   D11_B0(0,1,0) = D11_B0(0,-1,0);

   detJ_B0(0,1,0) = detJ_B0(0,-1,0);

   D11_B0(0,2,0) = D11_B0(0,-2,0);

   detJ_B0(0,2,0) = detJ_B0(0,-2,0);

}

void opensbliblock00Kernel046(const ACC<double> &D11_B0, ACC<double> &SD111_B0, const int *idx)
{
   double d1_D11_dy = 0.0;
    d1_D11_dy = invDelta1block0*((idx[1] == 0) ? (
   -3*D11_B0(0,2,0) + 4*D11_B0(0,1,0) - (25.0/12.0)*D11_B0(0,0,0) -
      (1.0/4.0)*D11_B0(0,4,0) + ((4.0/3.0))*D11_B0(0,3,0)
)
: ((idx[1] == 1) ? (
   -(5.0/6.0)*D11_B0(0,0,0) -
      (1.0/2.0)*D11_B0(0,2,0) - (1.0/4.0)*D11_B0(0,-1,0) + ((1.0/12.0))*D11_B0(0,3,0) + ((3.0/2.0))*D11_B0(0,1,0)
)
:
      ((idx[1] == -1 + block0np1) ? (
   -4*D11_B0(0,-1,0) + 3*D11_B0(0,-2,0) - (4.0/3.0)*D11_B0(0,-3,0) +
      ((1.0/4.0))*D11_B0(0,-4,0) + ((25.0/12.0))*D11_B0(0,0,0)
)
: ((idx[1] == -2 + block0np1) ? (
  
      ((1.0/2.0))*D11_B0(0,-2,0) - (3.0/2.0)*D11_B0(0,-1,0) - (1.0/12.0)*D11_B0(0,-3,0) + ((1.0/4.0))*D11_B0(0,1,0) +
      ((5.0/6.0))*D11_B0(0,0,0)
)
: (
   -(2.0/3.0)*D11_B0(0,-1,0) - (1.0/12.0)*D11_B0(0,2,0) +
      ((1.0/12.0))*D11_B0(0,-2,0) + ((2.0/3.0))*D11_B0(0,1,0)
)))));

   SD111_B0(0,0,0) = d1_D11_dy;

}

 void opensbliblock00Kernel030(const ACC<double> &x1_B0, ACC<double> &rhoE_B0, ACC<double> &rhou0_B0, ACC<double>
&rhou1_B0, ACC<double> &rhou2_B0, ACC<double> &rho_B0, const ACC<double>& x2_B0, const double* eddy_x_gbl, const double* eddy_y_gbl, const double* eddy_z_gbl, const double* eddy_r_gbl,
const int* eddy_eps_x_gbl, const int* eddy_eps_y_gbl, const int* eddy_eps_z_gbl, const int* idx)
{
   double T = 0.0;
   double p = 0.0;
   double psifn = 0.0;
   double u0 = 0.0;
   double u1 = 0.0;
   double u2 = 0.0;
   double x1d = 0.0;
   double up = 0.0;
   double vp = 0.0;
   double wp = 0.0;
   double xtildesq = 0.0;
   double ytildesq = 0.0;
   double ztildesq = 0.0;
   double ztildesq_ghost = 0.0;
   double rho_temp = 0.0;
   double shape = 0.0;
   double rsq = 0.0;

   
   x1d = 0.08548942989301017*x1_B0(0,0,0);

   u0 = (x1_B0(0,0,0) < 0) ? 0.0 : 
   (x1_B0(0,0,0) > 0.0 && x1d < 1.0) ? uinterp[idx[1]]: 1.0;

   u1 = 0.0;

   u2 = 0.0;

   psifn = pow(u0, 2.0);

    T = ((x1_B0(0,0,0) <= 0) ? (
   Twall
)
: ((x1_B0(0,0,0) > 0 && 0.08548942989301017*x1_B0(0,0,0) < 1.0) ? (
   Twall
      + psifn*(1 - Twall)
)
: (
   1.0
)));

   p = 1/((Minf*Minf)*gama);
   
   rho_temp = gama*p*(Minf*Minf)/T;

   if(x1d < 1.0 && x1d > 0.0){
     for (int i{0}; i < eddies; i++){
       xtildesq = (eddy_x_gbl[i])*(eddy_x_gbl[i])/(eddy_r_gbl[i]*eddy_r_gbl[i]);
       ytildesq = (eddy_y_gbl[i]-x1_B0(0,0,0))*(eddy_y_gbl[i]-x1_B0(0,0,0))/(eddy_r_gbl[i]*eddy_r_gbl[i]);
       ztildesq = (eddy_z_gbl[i]-x2_B0(0,0,0))*(eddy_z_gbl[i]-x2_B0(0,0,0))/(eddy_r_gbl[i]*eddy_r_gbl[i]);
       rsq = xtildesq + ytildesq + ztildesq;
       if(rsq < 1.0){
         shape = 1.0/1.85;
         shape *= exp(-0.5*xtildesq);
         shape *= exp(-0.5*ytildesq);
         shape *= exp(-0.5*ztildesq);
         
         up += a11[idx[1]] * eddy_eps_x_gbl[i] * shape;
         vp += (a21[idx[1]] * eddy_eps_x_gbl[i] + a22[idx[1]] * eddy_eps_y_gbl[i]) * shape;
         wp += (a33[idx[1]] * eddy_eps_z_gbl[i]) * shape; // a31, a32 are zero
         
         
       }
       else{ // ghost boundary mirror
         if(eddy_z_gbl[i] < 20.0){ // half domain width
           ztildesq_ghost = (eddy_z_gbl[i] - x2_B0(0,0,0) + 40.0)*(eddy_z_gbl[i] - x2_B0(0,0,0) + 40.0);
         }
         else{
           ztildesq_ghost = (eddy_z_gbl[i] - x2_B0(0,0,0) - 40.0)*(eddy_z_gbl[i] - x2_B0(0,0,0) - 40.0);
         }
         rsq = rsq - ztildesq + ztildesq_ghost;
         if(rsq < 1.0){
           shape = 1.0/1.85;
           shape *= exp(-0.5*xtildesq);
           shape *= exp(-0.5*ytildesq);
           shape *= exp(-0.5*ztildesq_ghost);
           
           up += (a11[idx[1]] * eddy_eps_x_gbl[i]) * shape;
           vp += (a21[idx[1]] * eddy_eps_x_gbl[i] + a22[idx[1]] * eddy_eps_y_gbl[i]) * shape;
           wp += (a33[idx[1]] * eddy_eps_z_gbl[i]) * shape;
         }
       }
     }
   }
   
   up = (std::abs(up) < 0.2) ? up : std::abs(up)/up * 0.2;
   vp = (std::abs(vp) < 0.2) ? vp : std::abs(vp)/vp * 0.2;
   wp = (std::abs(wp) < 0.2) ? wp : std::abs(wp)/wp * 0.2;

   rho_B0(0,0,0) = rho_temp;

   rhou0_B0(0,0,0) = rho_temp * (u0+up);

   rhou1_B0(0,0,0) = rho_temp * (u1+vp);

   rhou2_B0(0,0,0) = rho_temp * (u2+wp);

   rhoE_B0(0,0,0) = p/(-1 + gama) + 0.5*((u0+up)*(u0+up))*rho_B0(0,0,0);

}

 void opensbliblock00Kernel031(ACC<double> &rhoE_B0, ACC<double> &rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0,
ACC<double> &rhou2_B0)
{
   rho_B0(0,0,0) = rho_B0(-1,0,0);

   rhou0_B0(0,0,0) = rhou0_B0(-1,0,0);

   rhou1_B0(0,0,0) = rhou1_B0(-1,0,0);

   rhou2_B0(0,0,0) = rhou2_B0(-1,0,0);

   rhoE_B0(0,0,0) = rhoE_B0(-1,0,0);

   rho_B0(1,0,0) = rho_B0(-1,0,0);

   rhou0_B0(1,0,0) = rhou0_B0(-1,0,0);

   rhou1_B0(1,0,0) = rhou1_B0(-1,0,0);

   rhou2_B0(1,0,0) = rhou2_B0(-1,0,0);

   rhoE_B0(1,0,0) = rhoE_B0(-1,0,0);

   rho_B0(2,0,0) = rho_B0(-1,0,0);

   rhou0_B0(2,0,0) = rhou0_B0(-1,0,0);

   rhou1_B0(2,0,0) = rhou1_B0(-1,0,0);

   rhou2_B0(2,0,0) = rhou2_B0(-1,0,0);

   rhoE_B0(2,0,0) = rhoE_B0(-1,0,0);

}

 void opensbliblock00Kernel032(ACC<double> &rhoE_B0, ACC<double> &rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0,
ACC<double> &rhou2_B0)
{
   double Pwall = 0.0;
   double T1 = 0.0;
   double T2 = 0.0;
   double T_above = 0.0;
   double rho_halo_1 = 0.0;
   double rho_halo_2 = 0.0;
   double u01 = 0.0;
   double u02 = 0.0;
   double u11 = 0.0;
   double u12 = 0.0;
   double u21 = 0.0;
   double u22 = 0.0;
   rhou0_B0(0,0,0) = 0.0;

   rhou1_B0(0,0,0) = 0.0;

   rhou2_B0(0,0,0) = 0.0;

   rhoE_B0(0,0,0) = Twall*1.0/(Minf*Minf)*rho_B0(0,0,0)/(gama*(-1.0 + gama));

    Pwall = (-1 + gama)*(-(((1.0/2.0))*(rhou0_B0(0,0,0)*rhou0_B0(0,0,0)) + ((1.0/2.0))*(rhou1_B0(0,0,0)*rhou1_B0(0,0,0))
      + ((1.0/2.0))*(rhou2_B0(0,0,0)*rhou2_B0(0,0,0)))/rho_B0(0,0,0) + rhoE_B0(0,0,0));

   u01 = rhou0_B0(0,1,0)/rho_B0(0,1,0);

   u02 = rhou0_B0(0,2,0)/rho_B0(0,2,0);

   u11 = rhou1_B0(0,1,0)/rho_B0(0,1,0);

   u12 = rhou1_B0(0,2,0)/rho_B0(0,2,0);

   u21 = rhou2_B0(0,1,0)/rho_B0(0,1,0);

   u22 = rhou2_B0(0,2,0)/rho_B0(0,2,0);

    T_above = (Minf*Minf)*(-1 + gama)*(-(((1.0/2.0))*(rhou0_B0(0,1,0)*rhou0_B0(0,1,0)) +
      ((1.0/2.0))*(rhou1_B0(0,1,0)*rhou1_B0(0,1,0)) + ((1.0/2.0))*(rhou2_B0(0,1,0)*rhou2_B0(0,1,0)))/rho_B0(0,1,0) +
      rhoE_B0(0,1,0))*gama/rho_B0(0,1,0);

   T1 = -T_above + 2*Twall;

   rho_halo_1 = (Minf*Minf)*gama*Pwall/T1;

   rho_B0(0,-1,0) = rho_halo_1;

   rhou0_B0(0,-1,0) = -rho_halo_1*u01;

   rhou1_B0(0,-1,0) = -rho_halo_1*u11;

   rhou2_B0(0,-1,0) = -rho_halo_1*u21;

   rhoE_B0(0,-1,0) = inv_gamma_m1*Pwall + ((1.0/2.0))*((u01*u01) + (u11*u11) + (u21*u21))*rho_halo_1;

   T2 = -2*T_above + 3*Twall;

   rho_halo_1 = (Minf*Minf)*gama*Pwall/T1;

   rho_B0(0,-1,0) = rho_halo_1;

   rho_halo_2 = (Minf*Minf)*gama*Pwall/T2;

   rho_B0(0,-2,0) = rho_halo_2;

   rhou0_B0(0,-1,0) = -rho_halo_1*u01;

   rhou1_B0(0,-1,0) = -rho_halo_1*u11;

   rhou2_B0(0,-1,0) = -rho_halo_1*u21;

   rhou0_B0(0,-2,0) = -rho_halo_2*u02;

   rhou1_B0(0,-2,0) = -rho_halo_2*u12;

   rhou2_B0(0,-2,0) = -rho_halo_2*u22;

   rhoE_B0(0,-1,0) = inv_gamma_m1*Pwall + ((1.0/2.0))*((u01*u01) + (u11*u11) + (u21*u21))*rho_halo_1;

   rhoE_B0(0,-2,0) = inv_gamma_m1*Pwall + ((1.0/2.0))*((u02*u02) + (u12*u12) + (u22*u22))*rho_halo_2;

}

 void opensbliblock00Kernel033(ACC<double> &rhoE_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0, ACC<double> &rho_B0,
ACC<double> &rhou2_B0)
{
   rho_B0(0,0,0) = 1.000000000000000;

   rhou0_B0(0,0,0) = 1.000000000000000;

   rhou1_B0(0,0,0) = 0.0;

   rhou2_B0(0,0,0) = 0.0;

   rhoE_B0(0,0,0) = 0.946428571428572 + 0.5*(rhou2_B0(0,0,0)*rhou2_B0(0,0,0))/rho_B0(0,0,0);

}

void opensbliblock00Kernel001(const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, ACC<double> &u0_B0)
{
   u0_B0(0,0,0) = rhou0_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel003(const ACC<double> &rho_B0, const ACC<double> &rhou1_B0, ACC<double> &u1_B0)
{
   u1_B0(0,0,0) = rhou1_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel005(const ACC<double> &rho_B0, const ACC<double> &rhou2_B0, ACC<double> &u2_B0)
{
   u2_B0(0,0,0) = rhou2_B0(0,0,0)/rho_B0(0,0,0);

}

 void opensbliblock00Kernel016(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &u0_B0, const
ACC<double> &u1_B0, const ACC<double> &u2_B0, ACC<double> &p_B0)
{
    p_B0(0,0,0) = (-1 + gama)*(-(1.0/2.0)*(u0_B0(0,0,0)*u0_B0(0,0,0))*rho_B0(0,0,0) -
      (1.0/2.0)*(u1_B0(0,0,0)*u1_B0(0,0,0))*rho_B0(0,0,0) - (1.0/2.0)*(u2_B0(0,0,0)*u2_B0(0,0,0))*rho_B0(0,0,0) +
      rhoE_B0(0,0,0));

}

void opensbliblock00Kernel007(const ACC<double> &p_B0, const ACC<double> &rho_B0, ACC<double> &T_B0)
{
   T_B0(0,0,0) = (Minf*Minf)*gama*p_B0(0,0,0)/rho_B0(0,0,0);

}

void opensbliblock00Kernel022(const ACC<double> &T_B0, ACC<double> &mu_B0)
{
   mu_B0(0,0,0) = T_B0(0,0,0)*sqrt(T_B0(0,0,0))*(1.0 + SuthT*invRefT)/(SuthT*invRefT + T_B0(0,0,0));

}

void opensbliblock00Kernel000(const ACC<double> &u0_B0, ACC<double> &wk0_B0, const int *idx)
{
   if (idx[0] == 0){

       wk0_B0(0,0,0) = (3.0*u0_B0(1,0,0) + 0.333333333333333*u0_B0(3,0,0) - 1.5*u0_B0(2,0,0) -
            1.83333333333333*u0_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == 1){

       wk0_B0(0,0,0) = (0.0394168524399447*u0_B0(2,0,0) + 0.00571369039775442*u0_B0(4,0,0) +
            0.719443173328855*u0_B0(1,0,0) - 0.322484932882161*u0_B0(0,0,0) - 0.0658051057710389*u0_B0(3,0,0) -
            0.376283677513354*u0_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 2){

       wk0_B0(0,0,0) = (0.197184333887745*u0_B0(0,0,0) + 0.521455851089587*u0_B0(1,0,0) +
            0.113446470384241*u0_B0(-2,0,0) - 0.00412637789557492*u0_B0(3,0,0) - 0.0367146847001261*u0_B0(2,0,0) -
            0.791245592765872*u0_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 3){

       wk0_B0(0,0,0) = (0.0451033223343881*u0_B0(0,0,0) + 0.652141084861241*u0_B0(1,0,0) +
            0.121937153224065*u0_B0(-2,0,0) - 0.00932597985049999*u0_B0(-3,0,0) - 0.727822147724592*u0_B0(-1,0,0) -
            0.082033432844602*u0_B0(2,0,0))*invDelta0block0;

   }

   else if (idx[0] == -1 + block0np0){

       wk0_B0(0,0,0) = (-4*u0_B0(-1,0,0) + 3*u0_B0(-2,0,0) - (4.0/3.0)*u0_B0(-3,0,0) + ((1.0/4.0))*u0_B0(-4,0,0) +
            ((25.0/12.0))*u0_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == -2 + block0np0){

       wk0_B0(0,0,0) = (((1.0/2.0))*u0_B0(-2,0,0) - (3.0/2.0)*u0_B0(-1,0,0) - (1.0/12.0)*u0_B0(-3,0,0) +
            ((1.0/4.0))*u0_B0(1,0,0) + ((5.0/6.0))*u0_B0(0,0,0))*invDelta0block0;

   }

   else{

       wk0_B0(0,0,0) = (-(2.0/3.0)*u0_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0) +
            ((2.0/3.0))*u0_B0(1,0,0))*invDelta0block0;

   }

}

void opensbliblock00Kernel002(const ACC<double> &u1_B0, ACC<double> &wk1_B0, const int *idx)
{
   if (idx[0] == 0){

       wk1_B0(0,0,0) = (3.0*u1_B0(1,0,0) + 0.333333333333333*u1_B0(3,0,0) - 1.5*u1_B0(2,0,0) -
            1.83333333333333*u1_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == 1){

       wk1_B0(0,0,0) = (0.0394168524399447*u1_B0(2,0,0) + 0.00571369039775442*u1_B0(4,0,0) +
            0.719443173328855*u1_B0(1,0,0) - 0.322484932882161*u1_B0(0,0,0) - 0.0658051057710389*u1_B0(3,0,0) -
            0.376283677513354*u1_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 2){

       wk1_B0(0,0,0) = (0.197184333887745*u1_B0(0,0,0) + 0.521455851089587*u1_B0(1,0,0) +
            0.113446470384241*u1_B0(-2,0,0) - 0.00412637789557492*u1_B0(3,0,0) - 0.0367146847001261*u1_B0(2,0,0) -
            0.791245592765872*u1_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 3){

       wk1_B0(0,0,0) = (0.0451033223343881*u1_B0(0,0,0) + 0.652141084861241*u1_B0(1,0,0) +
            0.121937153224065*u1_B0(-2,0,0) - 0.00932597985049999*u1_B0(-3,0,0) - 0.727822147724592*u1_B0(-1,0,0) -
            0.082033432844602*u1_B0(2,0,0))*invDelta0block0;

   }

   else if (idx[0] == -1 + block0np0){

       wk1_B0(0,0,0) = (-4*u1_B0(-1,0,0) + 3*u1_B0(-2,0,0) - (4.0/3.0)*u1_B0(-3,0,0) + ((1.0/4.0))*u1_B0(-4,0,0) +
            ((25.0/12.0))*u1_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == -2 + block0np0){

       wk1_B0(0,0,0) = (((1.0/2.0))*u1_B0(-2,0,0) - (3.0/2.0)*u1_B0(-1,0,0) - (1.0/12.0)*u1_B0(-3,0,0) +
            ((1.0/4.0))*u1_B0(1,0,0) + ((5.0/6.0))*u1_B0(0,0,0))*invDelta0block0;

   }

   else{

       wk1_B0(0,0,0) = (-(2.0/3.0)*u1_B0(-1,0,0) - (1.0/12.0)*u1_B0(2,0,0) + ((1.0/12.0))*u1_B0(-2,0,0) +
            ((2.0/3.0))*u1_B0(1,0,0))*invDelta0block0;

   }

}

void opensbliblock00Kernel004(const ACC<double> &u2_B0, ACC<double> &wk2_B0, const int *idx)
{
   if (idx[0] == 0){

       wk2_B0(0,0,0) = (3.0*u2_B0(1,0,0) + 0.333333333333333*u2_B0(3,0,0) - 1.5*u2_B0(2,0,0) -
            1.83333333333333*u2_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == 1){

       wk2_B0(0,0,0) = (0.0394168524399447*u2_B0(2,0,0) + 0.00571369039775442*u2_B0(4,0,0) +
            0.719443173328855*u2_B0(1,0,0) - 0.322484932882161*u2_B0(0,0,0) - 0.0658051057710389*u2_B0(3,0,0) -
            0.376283677513354*u2_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 2){

       wk2_B0(0,0,0) = (0.197184333887745*u2_B0(0,0,0) + 0.521455851089587*u2_B0(1,0,0) +
            0.113446470384241*u2_B0(-2,0,0) - 0.00412637789557492*u2_B0(3,0,0) - 0.0367146847001261*u2_B0(2,0,0) -
            0.791245592765872*u2_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 3){

       wk2_B0(0,0,0) = (0.0451033223343881*u2_B0(0,0,0) + 0.652141084861241*u2_B0(1,0,0) +
            0.121937153224065*u2_B0(-2,0,0) - 0.00932597985049999*u2_B0(-3,0,0) - 0.727822147724592*u2_B0(-1,0,0) -
            0.082033432844602*u2_B0(2,0,0))*invDelta0block0;

   }

   else if (idx[0] == -1 + block0np0){

       wk2_B0(0,0,0) = (-4*u2_B0(-1,0,0) + 3*u2_B0(-2,0,0) - (4.0/3.0)*u2_B0(-3,0,0) + ((1.0/4.0))*u2_B0(-4,0,0) +
            ((25.0/12.0))*u2_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == -2 + block0np0){

       wk2_B0(0,0,0) = (((1.0/2.0))*u2_B0(-2,0,0) - (3.0/2.0)*u2_B0(-1,0,0) - (1.0/12.0)*u2_B0(-3,0,0) +
            ((1.0/4.0))*u2_B0(1,0,0) + ((5.0/6.0))*u2_B0(0,0,0))*invDelta0block0;

   }

   else{

       wk2_B0(0,0,0) = (-(2.0/3.0)*u2_B0(-1,0,0) - (1.0/12.0)*u2_B0(2,0,0) + ((1.0/12.0))*u2_B0(-2,0,0) +
            ((2.0/3.0))*u2_B0(1,0,0))*invDelta0block0;

   }

}

void opensbliblock00Kernel006(const ACC<double> &T_B0, ACC<double> &wk3_B0, const int *idx)
{
   if (idx[0] == 0){

       wk3_B0(0,0,0) = (3.0*T_B0(1,0,0) + 0.333333333333333*T_B0(3,0,0) - 1.5*T_B0(2,0,0) -
            1.83333333333333*T_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == 1){

       wk3_B0(0,0,0) = (0.0394168524399447*T_B0(2,0,0) + 0.00571369039775442*T_B0(4,0,0) + 0.719443173328855*T_B0(1,0,0)
            - 0.322484932882161*T_B0(0,0,0) - 0.0658051057710389*T_B0(3,0,0) -
            0.376283677513354*T_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 2){

       wk3_B0(0,0,0) = (0.197184333887745*T_B0(0,0,0) + 0.521455851089587*T_B0(1,0,0) + 0.113446470384241*T_B0(-2,0,0) -
            0.00412637789557492*T_B0(3,0,0) - 0.0367146847001261*T_B0(2,0,0) -
            0.791245592765872*T_B0(-1,0,0))*invDelta0block0;

   }

   else if (idx[0] == 3){

       wk3_B0(0,0,0) = (0.0451033223343881*T_B0(0,0,0) + 0.652141084861241*T_B0(1,0,0) + 0.121937153224065*T_B0(-2,0,0)
            - 0.00932597985049999*T_B0(-3,0,0) - 0.727822147724592*T_B0(-1,0,0) -
            0.082033432844602*T_B0(2,0,0))*invDelta0block0;

   }

   else if (idx[0] == -1 + block0np0){

       wk3_B0(0,0,0) = (-4*T_B0(-1,0,0) + 3*T_B0(-2,0,0) - (4.0/3.0)*T_B0(-3,0,0) + ((1.0/4.0))*T_B0(-4,0,0) +
            ((25.0/12.0))*T_B0(0,0,0))*invDelta0block0;

   }

   else if (idx[0] == -2 + block0np0){

       wk3_B0(0,0,0) = (((1.0/2.0))*T_B0(-2,0,0) - (3.0/2.0)*T_B0(-1,0,0) - (1.0/12.0)*T_B0(-3,0,0) +
            ((1.0/4.0))*T_B0(1,0,0) + ((5.0/6.0))*T_B0(0,0,0))*invDelta0block0;

   }

   else{

       wk3_B0(0,0,0) = (-(2.0/3.0)*T_B0(-1,0,0) - (1.0/12.0)*T_B0(2,0,0) + ((1.0/12.0))*T_B0(-2,0,0) +
            ((2.0/3.0))*T_B0(1,0,0))*invDelta0block0;

   }

}

void opensbliblock00Kernel008(const ACC<double> &u0_B0, ACC<double> &wk4_B0, const int *idx)
{
   if (idx[1] == 0){

       wk4_B0(0,0,0) = (-3*u0_B0(0,2,0) + 4*u0_B0(0,1,0) - (25.0/12.0)*u0_B0(0,0,0) - (1.0/4.0)*u0_B0(0,4,0) +
            ((4.0/3.0))*u0_B0(0,3,0))*invDelta1block0;

   }

   else if (idx[1] == 1){

       wk4_B0(0,0,0) = (-(5.0/6.0)*u0_B0(0,0,0) - (1.0/2.0)*u0_B0(0,2,0) - (1.0/4.0)*u0_B0(0,-1,0) +
            ((1.0/12.0))*u0_B0(0,3,0) + ((3.0/2.0))*u0_B0(0,1,0))*invDelta1block0;

   }

   else if (idx[1] == -1 + block0np1){

       wk4_B0(0,0,0) = (-4*u0_B0(0,-1,0) + 3*u0_B0(0,-2,0) - (4.0/3.0)*u0_B0(0,-3,0) + ((1.0/4.0))*u0_B0(0,-4,0) +
            ((25.0/12.0))*u0_B0(0,0,0))*invDelta1block0;

   }

   else if (idx[1] == -2 + block0np1){

       wk4_B0(0,0,0) = (((1.0/2.0))*u0_B0(0,-2,0) - (3.0/2.0)*u0_B0(0,-1,0) - (1.0/12.0)*u0_B0(0,-3,0) +
            ((1.0/4.0))*u0_B0(0,1,0) + ((5.0/6.0))*u0_B0(0,0,0))*invDelta1block0;

   }

   else{

       wk4_B0(0,0,0) = (-(2.0/3.0)*u0_B0(0,-1,0) - (1.0/12.0)*u0_B0(0,2,0) + ((1.0/12.0))*u0_B0(0,-2,0) +
            ((2.0/3.0))*u0_B0(0,1,0))*invDelta1block0;

   }

}

void opensbliblock00Kernel009(const ACC<double> &u1_B0, ACC<double> &wk5_B0, const int *idx)
{
   if (idx[1] == 0){

       wk5_B0(0,0,0) = (-3*u1_B0(0,2,0) + 4*u1_B0(0,1,0) - (25.0/12.0)*u1_B0(0,0,0) - (1.0/4.0)*u1_B0(0,4,0) +
            ((4.0/3.0))*u1_B0(0,3,0))*invDelta1block0;

   }

   else if (idx[1] == 1){

       wk5_B0(0,0,0) = (-(5.0/6.0)*u1_B0(0,0,0) - (1.0/2.0)*u1_B0(0,2,0) - (1.0/4.0)*u1_B0(0,-1,0) +
            ((1.0/12.0))*u1_B0(0,3,0) + ((3.0/2.0))*u1_B0(0,1,0))*invDelta1block0;

   }

   else if (idx[1] == -1 + block0np1){

       wk5_B0(0,0,0) = (-4*u1_B0(0,-1,0) + 3*u1_B0(0,-2,0) - (4.0/3.0)*u1_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,-4,0) +
            ((25.0/12.0))*u1_B0(0,0,0))*invDelta1block0;

   }

   else if (idx[1] == -2 + block0np1){

       wk5_B0(0,0,0) = (((1.0/2.0))*u1_B0(0,-2,0) - (3.0/2.0)*u1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,-3,0) +
            ((1.0/4.0))*u1_B0(0,1,0) + ((5.0/6.0))*u1_B0(0,0,0))*invDelta1block0;

   }

   else{

       wk5_B0(0,0,0) = (-(2.0/3.0)*u1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0) +
            ((2.0/3.0))*u1_B0(0,1,0))*invDelta1block0;

   }

}

void opensbliblock00Kernel010(const ACC<double> &u2_B0, ACC<double> &wk6_B0, const int *idx)
{
   if (idx[1] == 0){

       wk6_B0(0,0,0) = (-3*u2_B0(0,2,0) + 4*u2_B0(0,1,0) - (25.0/12.0)*u2_B0(0,0,0) - (1.0/4.0)*u2_B0(0,4,0) +
            ((4.0/3.0))*u2_B0(0,3,0))*invDelta1block0;

   }

   else if (idx[1] == 1){

       wk6_B0(0,0,0) = (-(5.0/6.0)*u2_B0(0,0,0) - (1.0/2.0)*u2_B0(0,2,0) - (1.0/4.0)*u2_B0(0,-1,0) +
            ((1.0/12.0))*u2_B0(0,3,0) + ((3.0/2.0))*u2_B0(0,1,0))*invDelta1block0;

   }

   else if (idx[1] == -1 + block0np1){

       wk6_B0(0,0,0) = (-4*u2_B0(0,-1,0) + 3*u2_B0(0,-2,0) - (4.0/3.0)*u2_B0(0,-3,0) + ((1.0/4.0))*u2_B0(0,-4,0) +
            ((25.0/12.0))*u2_B0(0,0,0))*invDelta1block0;

   }

   else if (idx[1] == -2 + block0np1){

       wk6_B0(0,0,0) = (((1.0/2.0))*u2_B0(0,-2,0) - (3.0/2.0)*u2_B0(0,-1,0) - (1.0/12.0)*u2_B0(0,-3,0) +
            ((1.0/4.0))*u2_B0(0,1,0) + ((5.0/6.0))*u2_B0(0,0,0))*invDelta1block0;

   }

   else{

       wk6_B0(0,0,0) = (-(2.0/3.0)*u2_B0(0,-1,0) - (1.0/12.0)*u2_B0(0,2,0) + ((1.0/12.0))*u2_B0(0,-2,0) +
            ((2.0/3.0))*u2_B0(0,1,0))*invDelta1block0;

   }

}

void opensbliblock00Kernel011(const ACC<double> &T_B0, ACC<double> &wk7_B0, const int *idx)
{
   if (idx[1] == 0){

       wk7_B0(0,0,0) = (-3*T_B0(0,2,0) + 4*T_B0(0,1,0) - (25.0/12.0)*T_B0(0,0,0) - (1.0/4.0)*T_B0(0,4,0) +
            ((4.0/3.0))*T_B0(0,3,0))*invDelta1block0;

   }

   else if (idx[1] == 1){

       wk7_B0(0,0,0) = (-(5.0/6.0)*T_B0(0,0,0) - (1.0/2.0)*T_B0(0,2,0) - (1.0/4.0)*T_B0(0,-1,0) +
            ((1.0/12.0))*T_B0(0,3,0) + ((3.0/2.0))*T_B0(0,1,0))*invDelta1block0;

   }

   else if (idx[1] == -1 + block0np1){

       wk7_B0(0,0,0) = (-4*T_B0(0,-1,0) + 3*T_B0(0,-2,0) - (4.0/3.0)*T_B0(0,-3,0) + ((1.0/4.0))*T_B0(0,-4,0) +
            ((25.0/12.0))*T_B0(0,0,0))*invDelta1block0;

   }

   else if (idx[1] == -2 + block0np1){

       wk7_B0(0,0,0) = (((1.0/2.0))*T_B0(0,-2,0) - (3.0/2.0)*T_B0(0,-1,0) - (1.0/12.0)*T_B0(0,-3,0) +
            ((1.0/4.0))*T_B0(0,1,0) + ((5.0/6.0))*T_B0(0,0,0))*invDelta1block0;

   }

   else{

       wk7_B0(0,0,0) = (-(2.0/3.0)*T_B0(0,-1,0) - (1.0/12.0)*T_B0(0,2,0) + ((1.0/12.0))*T_B0(0,-2,0) +
            ((2.0/3.0))*T_B0(0,1,0))*invDelta1block0;

   }

}

void opensbliblock00Kernel012(const ACC<double> &u0_B0, ACC<double> &wk8_B0)
{
    wk8_B0(0,0,0) = (-(2.0/3.0)*u0_B0(0,0,-1) - (1.0/12.0)*u0_B0(0,0,2) + ((1.0/12.0))*u0_B0(0,0,-2) +
      ((2.0/3.0))*u0_B0(0,0,1))*invDelta2block0;

}

void opensbliblock00Kernel013(const ACC<double> &u1_B0, ACC<double> &wk9_B0)
{
    wk9_B0(0,0,0) = (-(2.0/3.0)*u1_B0(0,0,-1) - (1.0/12.0)*u1_B0(0,0,2) + ((1.0/12.0))*u1_B0(0,0,-2) +
      ((2.0/3.0))*u1_B0(0,0,1))*invDelta2block0;

}

void opensbliblock00Kernel014(const ACC<double> &u2_B0, ACC<double> &wk10_B0)
{
    wk10_B0(0,0,0) = (-(2.0/3.0)*u2_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2) +
      ((2.0/3.0))*u2_B0(0,0,1))*invDelta2block0;

}

void opensbliblock00Kernel015(const ACC<double> &T_B0, ACC<double> &wk11_B0)
{
    wk11_B0(0,0,0) = (-(2.0/3.0)*T_B0(0,0,-1) - (1.0/12.0)*T_B0(0,0,2) + ((1.0/12.0))*T_B0(0,0,-2) +
      ((2.0/3.0))*T_B0(0,0,1))*invDelta2block0;

}


 void opensbliblock00Kernel028(const ACC<double> &D11_B0, const ACC<double> &p_B0, const ACC<double> &rhoE_B0, const
ACC<double> &rho_B0, const ACC<double> &rhou0_B0, const ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, const
ACC<double> &u0_B0, const ACC<double> &u1_B0, const ACC<double> &u2_B0, const ACC<double> &wk0_B0, const ACC<double>
&wk10_B0, const ACC<double> &wk1_B0, const ACC<double> &wk2_B0, const ACC<double> &wk4_B0, const ACC<double> &wk5_B0,
const ACC<double> &wk6_B0, const ACC<double> &wk8_B0, const ACC<double> &wk9_B0, ACC<double> &Residual0_B0, ACC<double>
&Residual1_B0, ACC<double> &Residual2_B0, ACC<double> &Residual3_B0, ACC<double> &Residual4_B0, const int *idx)
{
   double d1_inv_rhoErho_dx = 0.0;
   double d1_inv_rhoErho_dy = 0.0;
   double d1_inv_rhoErho_dz = 0.0;
   double d1_p_dx = 0.0;
   double d1_p_dy = 0.0;
   double d1_p_dz = 0.0;
   double d1_pu0_dx = 0.0;
   double d1_pu1_dy = 0.0;
   double d1_pu2_dz = 0.0;
   double d1_rhoEu0_dx = 0.0;
   double d1_rhoEu1_dy = 0.0;
   double d1_rhoEu2_dz = 0.0;
   double d1_rhou0_dx = 0.0;
   double d1_rhou0u0_dx = 0.0;
   double d1_rhou0u1_dy = 0.0;
   double d1_rhou0u2_dz = 0.0;
   double d1_rhou1_dy = 0.0;
   double d1_rhou1u0_dx = 0.0;
   double d1_rhou1u1_dy = 0.0;
   double d1_rhou1u2_dz = 0.0;
   double d1_rhou2_dz = 0.0;
   double d1_rhou2u0_dx = 0.0;
   double d1_rhou2u1_dy = 0.0;
   double d1_rhou2u2_dz = 0.0;
   if (idx[0] == 0){
       d1_p_dx = (3.0*p_B0(1,0,0) + 0.333333333333333*p_B0(3,0,0) - 1.5*p_B0(2,0,0) - 1.83333333333333*p_B0(0,0,0))*invDelta0block0;
       d1_pu0_dx = (3.0*p_B0(1,0,0)*u0_B0(1,0,0) + 0.333333333333333*p_B0(3,0,0)*u0_B0(3,0,0) - 1.5*p_B0(2,0,0)*u0_B0(2,0,0) - 1.83333333333333*p_B0(0,0,0)*u0_B0(0,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (3.0*u0_B0(1,0,0)*rhoE_B0(1,0,0) + 0.333333333333333*u0_B0(3,0,0)*rhoE_B0(3,0,0) - 1.5*u0_B0(2,0,0)*rhoE_B0(2,0,0) - 1.83333333333333*u0_B0(0,0,0)*rhoE_B0(0,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (3.0*rhoE_B0(1,0,0)/rho_B0(1,0,0) + 0.333333333333333*rhoE_B0(3,0,0)/rho_B0(3,0,0) - 1.5*rhoE_B0(2,0,0)/rho_B0(2,0,0) - 1.83333333333333*rhoE_B0(0,0,0)/rho_B0(0,0,0))*invDelta0block0;
       d1_rhou0_dx = (3.0*rhou0_B0(1,0,0) + 0.333333333333333*rhou0_B0(3,0,0) - 1.5*rhou0_B0(2,0,0) - 1.83333333333333*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (3.0*u0_B0(1,0,0)*rhou0_B0(1,0,0) + 0.333333333333333*u0_B0(3,0,0)*rhou0_B0(3,0,0) - 1.5*u0_B0(2,0,0)*rhou0_B0(2,0,0) - 1.83333333333333*u0_B0(0,0,0)*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (3.0*u0_B0(1,0,0)*rhou1_B0(1,0,0) + 0.333333333333333*u0_B0(3,0,0)*rhou1_B0(3,0,0) - 1.5*u0_B0(2,0,0)*rhou1_B0(2,0,0) - 1.83333333333333*u0_B0(0,0,0)*rhou1_B0(0,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (3.0*u0_B0(1,0,0)*rhou2_B0(1,0,0) + 0.333333333333333*u0_B0(3,0,0)*rhou2_B0(3,0,0) - 1.5*u0_B0(2,0,0)*rhou2_B0(2,0,0) - 1.83333333333333*u0_B0(0,0,0)*rhou2_B0(0,0,0))*invDelta0block0;
   }
   else if (idx[0] == 1){
       d1_p_dx = (0.0394168524399447*p_B0(2,0,0) + 0.00571369039775442*p_B0(4,0,0) + 0.719443173328855*p_B0(1,0,0) - 0.322484932882161*p_B0(0,0,0) - 0.0658051057710389*p_B0(3,0,0) - 0.376283677513354*p_B0(-1,0,0))*invDelta0block0;
       d1_pu0_dx = (0.0394168524399447*p_B0(2,0,0)*u0_B0(2,0,0) + 0.00571369039775442*p_B0(4,0,0)*u0_B0(4,0,0) + 0.719443173328855*p_B0(1,0,0)*u0_B0(1,0,0) - 0.322484932882161*p_B0(0,0,0)*u0_B0(0,0,0) - 0.0658051057710389*p_B0(3,0,0)*u0_B0(3,0,0) - 0.376283677513354*p_B0(-1,0,0)*u0_B0(-1,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (0.0394168524399447*u0_B0(2,0,0)*rhoE_B0(2,0,0) + 0.00571369039775442*u0_B0(4,0,0)*rhoE_B0(4,0,0) + 0.719443173328855*u0_B0(1,0,0)*rhoE_B0(1,0,0) - 0.322484932882161*u0_B0(0,0,0)*rhoE_B0(0,0,0) - 0.0658051057710389*u0_B0(3,0,0)*rhoE_B0(3,0,0) - 0.376283677513354*u0_B0(-1,0,0)*rhoE_B0(-1,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (0.0394168524399447*rhoE_B0(2,0,0)/rho_B0(2,0,0) + 0.00571369039775442*rhoE_B0(4,0,0)/rho_B0(4,0,0) + 0.719443173328855*rhoE_B0(1,0,0)/rho_B0(1,0,0) - 0.322484932882161*rhoE_B0(0,0,0)/rho_B0(0,0,0) - 0.0658051057710389*rhoE_B0(3,0,0)/rho_B0(3,0,0) - 0.376283677513354*rhoE_B0(-1,0,0)/rho_B0(-1,0,0))*invDelta0block0;
       d1_rhou0_dx = (0.0394168524399447*rhou0_B0(2,0,0) + 0.00571369039775442*rhou0_B0(4,0,0) + 0.719443173328855*rhou0_B0(1,0,0) - 0.322484932882161*rhou0_B0(0,0,0) - 0.0658051057710389*rhou0_B0(3,0,0) - 0.376283677513354*rhou0_B0(-1,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (0.0394168524399447*u0_B0(2,0,0)*rhou0_B0(2,0,0) + 0.00571369039775442*u0_B0(4,0,0)*rhou0_B0(4,0,0) + 0.719443173328855*u0_B0(1,0,0)*rhou0_B0(1,0,0) - 0.322484932882161*u0_B0(0,0,0)*rhou0_B0(0,0,0) - 0.0658051057710389*u0_B0(3,0,0)*rhou0_B0(3,0,0) - 0.376283677513354*u0_B0(-1,0,0)*rhou0_B0(-1,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (0.0394168524399447*u0_B0(2,0,0)*rhou1_B0(2,0,0) + 0.00571369039775442*u0_B0(4,0,0)*rhou1_B0(4,0,0) + 0.719443173328855*u0_B0(1,0,0)*rhou1_B0(1,0,0) - 0.322484932882161*u0_B0(0,0,0)*rhou1_B0(0,0,0) - 0.0658051057710389*u0_B0(3,0,0)*rhou1_B0(3,0,0) - 0.376283677513354*u0_B0(-1,0,0)*rhou1_B0(-1,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (0.0394168524399447*u0_B0(2,0,0)*rhou2_B0(2,0,0) + 0.00571369039775442*u0_B0(4,0,0)*rhou2_B0(4,0,0) + 0.719443173328855*u0_B0(1,0,0)*rhou2_B0(1,0,0) - 0.322484932882161*u0_B0(0,0,0)*rhou2_B0(0,0,0) - 0.0658051057710389*u0_B0(3,0,0)*rhou2_B0(3,0,0) - 0.376283677513354*u0_B0(-1,0,0)*rhou2_B0(-1,0,0))*invDelta0block0;
   }
   else if (idx[0] == 2){
       d1_p_dx = (0.197184333887745*p_B0(0,0,0) + 0.521455851089587*p_B0(1,0,0) + 0.113446470384241*p_B0(-2,0,0) - 0.00412637789557492*p_B0(3,0,0) - 0.0367146847001261*p_B0(2,0,0) - 0.791245592765872*p_B0(-1,0,0))*invDelta0block0;
       d1_pu0_dx = (0.197184333887745*p_B0(0,0,0)*u0_B0(0,0,0) + 0.521455851089587*p_B0(1,0,0)*u0_B0(1,0,0) + 0.113446470384241*p_B0(-2,0,0)*u0_B0(-2,0,0) - 0.00412637789557492*p_B0(3,0,0)*u0_B0(3,0,0) - 0.0367146847001261*p_B0(2,0,0)*u0_B0(2,0,0) - 0.791245592765872*p_B0(-1,0,0)*u0_B0(-1,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (0.197184333887745*u0_B0(0,0,0)*rhoE_B0(0,0,0) + 0.521455851089587*u0_B0(1,0,0)*rhoE_B0(1,0,0) + 0.113446470384241*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) - 0.00412637789557492*u0_B0(3,0,0)*rhoE_B0(3,0,0) - 0.0367146847001261*u0_B0(2,0,0)*rhoE_B0(2,0,0) - 0.791245592765872*u0_B0(-1,0,0)*rhoE_B0(-1,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (0.197184333887745*rhoE_B0(0,0,0)/rho_B0(0,0,0) + 0.521455851089587*rhoE_B0(1,0,0)/rho_B0(1,0,0) + 0.113446470384241*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) - 0.00412637789557492*rhoE_B0(3,0,0)/rho_B0(3,0,0) - 0.0367146847001261*rhoE_B0(2,0,0)/rho_B0(2,0,0) - 0.791245592765872*rhoE_B0(-1,0,0)/rho_B0(-1,0,0))*invDelta0block0;
       d1_rhou0_dx = (0.197184333887745*rhou0_B0(0,0,0) + 0.521455851089587*rhou0_B0(1,0,0) + 0.113446470384241*rhou0_B0(-2,0,0) - 0.00412637789557492*rhou0_B0(3,0,0) - 0.0367146847001261*rhou0_B0(2,0,0) - 0.791245592765872*rhou0_B0(-1,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (0.197184333887745*u0_B0(0,0,0)*rhou0_B0(0,0,0) + 0.521455851089587*u0_B0(1,0,0)*rhou0_B0(1,0,0) + 0.113446470384241*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) - 0.00412637789557492*u0_B0(3,0,0)*rhou0_B0(3,0,0) - 0.0367146847001261*u0_B0(2,0,0)*rhou0_B0(2,0,0) - 0.791245592765872*u0_B0(-1,0,0)*rhou0_B0(-1,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (0.197184333887745*u0_B0(0,0,0)*rhou1_B0(0,0,0) + 0.521455851089587*u0_B0(1,0,0)*rhou1_B0(1,0,0) + 0.113446470384241*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) - 0.00412637789557492*u0_B0(3,0,0)*rhou1_B0(3,0,0) - 0.0367146847001261*u0_B0(2,0,0)*rhou1_B0(2,0,0) - 0.791245592765872*u0_B0(-1,0,0)*rhou1_B0(-1,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (0.197184333887745*u0_B0(0,0,0)*rhou2_B0(0,0,0) + 0.521455851089587*u0_B0(1,0,0)*rhou2_B0(1,0,0) + 0.113446470384241*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) - 0.00412637789557492*u0_B0(3,0,0)*rhou2_B0(3,0,0) - 0.0367146847001261*u0_B0(2,0,0)*rhou2_B0(2,0,0) - 0.791245592765872*u0_B0(-1,0,0)*rhou2_B0(-1,0,0))*invDelta0block0;
   }
   else if (idx[0] == 3){
       d1_p_dx = (0.0451033223343881*p_B0(0,0,0) + 0.652141084861241*p_B0(1,0,0) + 0.121937153224065*p_B0(-2,0,0) - 0.00932597985049999*p_B0(-3,0,0) - 0.727822147724592*p_B0(-1,0,0) - 0.082033432844602*p_B0(2,0,0))*invDelta0block0;
       d1_pu0_dx = (0.0451033223343881*p_B0(0,0,0)*u0_B0(0,0,0) + 0.652141084861241*p_B0(1,0,0)*u0_B0(1,0,0) + 0.121937153224065*p_B0(-2,0,0)*u0_B0(-2,0,0) - 0.00932597985049999*p_B0(-3,0,0)*u0_B0(-3,0,0) - 0.727822147724592*p_B0(-1,0,0)*u0_B0(-1,0,0) - 0.082033432844602*p_B0(2,0,0)*u0_B0(2,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (0.0451033223343881*u0_B0(0,0,0)*rhoE_B0(0,0,0) + 0.652141084861241*u0_B0(1,0,0)*rhoE_B0(1,0,0) + 0.121937153224065*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) - 0.00932597985049999*u0_B0(-3,0,0)*rhoE_B0(-3,0,0) - 0.727822147724592*u0_B0(-1,0,0)*rhoE_B0(-1,0,0) - 0.082033432844602*u0_B0(2,0,0)*rhoE_B0(2,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (0.0451033223343881*rhoE_B0(0,0,0)/rho_B0(0,0,0) + 0.652141084861241*rhoE_B0(1,0,0)/rho_B0(1,0,0) + 0.121937153224065*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) - 0.00932597985049999*rhoE_B0(-3,0,0)/rho_B0(-3,0,0) - 0.727822147724592*rhoE_B0(-1,0,0)/rho_B0(-1,0,0) - 0.082033432844602*rhoE_B0(2,0,0)/rho_B0(2,0,0))*invDelta0block0;
       d1_rhou0_dx = (0.0451033223343881*rhou0_B0(0,0,0) + 0.652141084861241*rhou0_B0(1,0,0) + 0.121937153224065*rhou0_B0(-2,0,0) - 0.00932597985049999*rhou0_B0(-3,0,0) - 0.727822147724592*rhou0_B0(-1,0,0) - 0.082033432844602*rhou0_B0(2,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (0.0451033223343881*u0_B0(0,0,0)*rhou0_B0(0,0,0) + 0.652141084861241*u0_B0(1,0,0)*rhou0_B0(1,0,0) + 0.121937153224065*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) - 0.00932597985049999*u0_B0(-3,0,0)*rhou0_B0(-3,0,0) - 0.727822147724592*u0_B0(-1,0,0)*rhou0_B0(-1,0,0) - 0.082033432844602*u0_B0(2,0,0)*rhou0_B0(2,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (0.0451033223343881*u0_B0(0,0,0)*rhou1_B0(0,0,0) + 0.652141084861241*u0_B0(1,0,0)*rhou1_B0(1,0,0) + 0.121937153224065*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) - 0.00932597985049999*u0_B0(-3,0,0)*rhou1_B0(-3,0,0) - 0.727822147724592*u0_B0(-1,0,0)*rhou1_B0(-1,0,0) - 0.082033432844602*u0_B0(2,0,0)*rhou1_B0(2,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (0.0451033223343881*u0_B0(0,0,0)*rhou2_B0(0,0,0) + 0.652141084861241*u0_B0(1,0,0)*rhou2_B0(1,0,0) + 0.121937153224065*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) - 0.00932597985049999*u0_B0(-3,0,0)*rhou2_B0(-3,0,0) - 0.727822147724592*u0_B0(-1,0,0)*rhou2_B0(-1,0,0) - 0.082033432844602*u0_B0(2,0,0)*rhou2_B0(2,0,0))*invDelta0block0;
   }
   else if (idx[0] == -1 + block0np0){
       d1_p_dx = (-4*p_B0(-1,0,0) + 3*p_B0(-2,0,0) - (4.0/3.0)*p_B0(-3,0,0) + ((1.0/4.0))*p_B0(-4,0,0) + ((25.0/12.0))*p_B0(0,0,0))*invDelta0block0;
       d1_pu0_dx = (-4*p_B0(-1,0,0)*u0_B0(-1,0,0) + 3*p_B0(-2,0,0)*u0_B0(-2,0,0) - (4.0/3.0)*p_B0(-3,0,0)*u0_B0(-3,0,0) + ((1.0/4.0))*p_B0(-4,0,0)*u0_B0(-4,0,0) + ((25.0/12.0))*p_B0(0,0,0)*u0_B0(0,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (-4*u0_B0(-1,0,0)*rhoE_B0(-1,0,0) + 3*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) - (4.0/3.0)*u0_B0(-3,0,0)*rhoE_B0(-3,0,0) + ((1.0/4.0))*u0_B0(-4,0,0)*rhoE_B0(-4,0,0) + ((25.0/12.0))*u0_B0(0,0,0)*rhoE_B0(0,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (-4*rhoE_B0(-1,0,0)/rho_B0(-1,0,0) + 3*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) - (4.0/3.0)*rhoE_B0(-3,0,0)/rho_B0(-3,0,0) + ((1.0/4.0))*rhoE_B0(-4,0,0)/rho_B0(-4,0,0) + ((25.0/12.0))*rhoE_B0(0,0,0)/rho_B0(0,0,0))*invDelta0block0;
       d1_rhou0_dx = (-4*rhou0_B0(-1,0,0) + 3*rhou0_B0(-2,0,0) - (4.0/3.0)*rhou0_B0(-3,0,0) + ((1.0/4.0))*rhou0_B0(-4,0,0) + ((25.0/12.0))*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (-4*u0_B0(-1,0,0)*rhou0_B0(-1,0,0) + 3*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) - (4.0/3.0)*u0_B0(-3,0,0)*rhou0_B0(-3,0,0) + ((1.0/4.0))*u0_B0(-4,0,0)*rhou0_B0(-4,0,0) + ((25.0/12.0))*u0_B0(0,0,0)*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (-4*u0_B0(-1,0,0)*rhou1_B0(-1,0,0) + 3*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) - (4.0/3.0)*u0_B0(-3,0,0)*rhou1_B0(-3,0,0) + ((1.0/4.0))*u0_B0(-4,0,0)*rhou1_B0(-4,0,0) + ((25.0/12.0))*u0_B0(0,0,0)*rhou1_B0(0,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (-4*u0_B0(-1,0,0)*rhou2_B0(-1,0,0) + 3*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) - (4.0/3.0)*u0_B0(-3,0,0)*rhou2_B0(-3,0,0) + ((1.0/4.0))*u0_B0(-4,0,0)*rhou2_B0(-4,0,0) + ((25.0/12.0))*u0_B0(0,0,0)*rhou2_B0(0,0,0))*invDelta0block0;
   }
   else if (idx[0] == -2 + block0np0){
       d1_p_dx = (((1.0/2.0))*p_B0(-2,0,0) - (3.0/2.0)*p_B0(-1,0,0) - (1.0/12.0)*p_B0(-3,0,0) + ((1.0/4.0))*p_B0(1,0,0) + ((5.0/6.0))*p_B0(0,0,0))*invDelta0block0;
       d1_pu0_dx = (((1.0/2.0))*p_B0(-2,0,0)*u0_B0(-2,0,0) - (3.0/2.0)*p_B0(-1,0,0)*u0_B0(-1,0,0) - (1.0/12.0)*p_B0(-3,0,0)*u0_B0(-3,0,0) + ((1.0/4.0))*p_B0(1,0,0)*u0_B0(1,0,0) + ((5.0/6.0))*p_B0(0,0,0)*u0_B0(0,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (((1.0/2.0))*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) - (3.0/2.0)*u0_B0(-1,0,0)*rhoE_B0(-1,0,0) - (1.0/12.0)*u0_B0(-3,0,0)*rhoE_B0(-3,0,0) + ((1.0/4.0))*u0_B0(1,0,0)*rhoE_B0(1,0,0) + ((5.0/6.0))*u0_B0(0,0,0)*rhoE_B0(0,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (((1.0/2.0))*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) - (3.0/2.0)*rhoE_B0(-1,0,0)/rho_B0(-1,0,0) - (1.0/12.0)*rhoE_B0(-3,0,0)/rho_B0(-3,0,0) + ((1.0/4.0))*rhoE_B0(1,0,0)/rho_B0(1,0,0) + ((5.0/6.0))*rhoE_B0(0,0,0)/rho_B0(0,0,0))*invDelta0block0;
       d1_rhou0_dx = (((1.0/2.0))*rhou0_B0(-2,0,0) - (3.0/2.0)*rhou0_B0(-1,0,0) - (1.0/12.0)*rhou0_B0(-3,0,0) + ((1.0/4.0))*rhou0_B0(1,0,0) + ((5.0/6.0))*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (((1.0/2.0))*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) - (3.0/2.0)*u0_B0(-1,0,0)*rhou0_B0(-1,0,0) - (1.0/12.0)*u0_B0(-3,0,0)*rhou0_B0(-3,0,0) + ((1.0/4.0))*u0_B0(1,0,0)*rhou0_B0(1,0,0) + ((5.0/6.0))*u0_B0(0,0,0)*rhou0_B0(0,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (((1.0/2.0))*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) - (3.0/2.0)*u0_B0(-1,0,0)*rhou1_B0(-1,0,0) - (1.0/12.0)*u0_B0(-3,0,0)*rhou1_B0(-3,0,0) + ((1.0/4.0))*u0_B0(1,0,0)*rhou1_B0(1,0,0) + ((5.0/6.0))*u0_B0(0,0,0)*rhou1_B0(0,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (((1.0/2.0))*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) - (3.0/2.0)*u0_B0(-1,0,0)*rhou2_B0(-1,0,0) - (1.0/12.0)*u0_B0(-3,0,0)*rhou2_B0(-3,0,0) + ((1.0/4.0))*u0_B0(1,0,0)*rhou2_B0(1,0,0) + ((5.0/6.0))*u0_B0(0,0,0)*rhou2_B0(0,0,0))*invDelta0block0;
   }
   else{
       d1_p_dx = (-(2.0/3.0)*p_B0(-1,0,0) - (1.0/12.0)*p_B0(2,0,0) + ((1.0/12.0))*p_B0(-2,0,0) + ((2.0/3.0))*p_B0(1,0,0))*invDelta0block0;
       d1_pu0_dx = (-(2.0/3.0)*p_B0(-1,0,0)*u0_B0(-1,0,0) - (1.0/12.0)*p_B0(2,0,0)*u0_B0(2,0,0) + ((1.0/12.0))*p_B0(-2,0,0)*u0_B0(-2,0,0) + ((2.0/3.0))*p_B0(1,0,0)*u0_B0(1,0,0))*invDelta0block0;
       d1_rhoEu0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhoE_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhoE_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0)*rhoE_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhoE_B0(1,0,0))*invDelta0block0;
       d1_inv_rhoErho_dx = (-(2.0/3.0)*rhoE_B0(-1,0,0)/rho_B0(-1,0,0) - (1.0/12.0)*rhoE_B0(2,0,0)/rho_B0(2,0,0) + ((1.0/12.0))*rhoE_B0(-2,0,0)/rho_B0(-2,0,0) + ((2.0/3.0))*rhoE_B0(1,0,0)/rho_B0(1,0,0))*invDelta0block0;
       d1_rhou0_dx = (-(2.0/3.0)*rhou0_B0(-1,0,0) - (1.0/12.0)*rhou0_B0(2,0,0) + ((1.0/12.0))*rhou0_B0(-2,0,0) + ((2.0/3.0))*rhou0_B0(1,0,0))*invDelta0block0;
       d1_rhou0u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou0_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou0_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0)*rhou0_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou0_B0(1,0,0))*invDelta0block0;
       d1_rhou1u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou1_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou1_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0)*rhou1_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou1_B0(1,0,0))*invDelta0block0;
       d1_rhou2u0_dx = (-(2.0/3.0)*u0_B0(-1,0,0)*rhou2_B0(-1,0,0) - (1.0/12.0)*u0_B0(2,0,0)*rhou2_B0(2,0,0) + ((1.0/12.0))*u0_B0(-2,0,0)*rhou2_B0(-2,0,0) + ((2.0/3.0))*u0_B0(1,0,0)*rhou2_B0(1,0,0))*invDelta0block0;
   }
   if (idx[1] == 0){
       d1_p_dy = (-3*p_B0(0,2,0) + 4*p_B0(0,1,0) - (25.0/12.0)*p_B0(0,0,0) - (1.0/4.0)*p_B0(0,4,0) + ((4.0/3.0))*p_B0(0,3,0))*invDelta1block0;
       d1_pu1_dy = (-3*p_B0(0,2,0)*u1_B0(0,2,0) + 4*p_B0(0,1,0)*u1_B0(0,1,0) - (25.0/12.0)*p_B0(0,0,0)*u1_B0(0,0,0) - (1.0/4.0)*p_B0(0,4,0)*u1_B0(0,4,0) + ((4.0/3.0))*p_B0(0,3,0)*u1_B0(0,3,0))*invDelta1block0;
       d1_rhoEu1_dy = (-3*u1_B0(0,2,0)*rhoE_B0(0,2,0) + 4*u1_B0(0,1,0)*rhoE_B0(0,1,0) - (25.0/12.0)*u1_B0(0,0,0)*rhoE_B0(0,0,0) - (1.0/4.0)*u1_B0(0,4,0)*rhoE_B0(0,4,0) + ((4.0/3.0))*u1_B0(0,3,0)*rhoE_B0(0,3,0))*invDelta1block0;
       d1_inv_rhoErho_dy = (-3*rhoE_B0(0,2,0)/rho_B0(0,2,0) + 4*rhoE_B0(0,1,0)/rho_B0(0,1,0) - (25.0/12.0)*rhoE_B0(0,0,0)/rho_B0(0,0,0) - (1.0/4.0)*rhoE_B0(0,4,0)/rho_B0(0,4,0) + ((4.0/3.0))*rhoE_B0(0,3,0)/rho_B0(0,3,0))*invDelta1block0;
       d1_rhou0u1_dy = (-3*u1_B0(0,2,0)*rhou0_B0(0,2,0) + 4*u1_B0(0,1,0)*rhou0_B0(0,1,0) - (25.0/12.0)*u1_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/4.0)*u1_B0(0,4,0)*rhou0_B0(0,4,0) + ((4.0/3.0))*u1_B0(0,3,0)*rhou0_B0(0,3,0))*invDelta1block0;
       d1_rhou1_dy = (-3*rhou1_B0(0,2,0) + 4*rhou1_B0(0,1,0) - (25.0/12.0)*rhou1_B0(0,0,0) - (1.0/4.0)*rhou1_B0(0,4,0) + ((4.0/3.0))*rhou1_B0(0,3,0))*invDelta1block0;
       d1_rhou1u1_dy = (-3*u1_B0(0,2,0)*rhou1_B0(0,2,0) + 4*u1_B0(0,1,0)*rhou1_B0(0,1,0) - (25.0/12.0)*u1_B0(0,0,0)*rhou1_B0(0,0,0) - (1.0/4.0)*u1_B0(0,4,0)*rhou1_B0(0,4,0) + ((4.0/3.0))*u1_B0(0,3,0)*rhou1_B0(0,3,0))*invDelta1block0;
       d1_rhou2u1_dy = (-3*u1_B0(0,2,0)*rhou2_B0(0,2,0) + 4*u1_B0(0,1,0)*rhou2_B0(0,1,0) - (25.0/12.0)*u1_B0(0,0,0)*rhou2_B0(0,0,0) - (1.0/4.0)*u1_B0(0,4,0)*rhou2_B0(0,4,0) + ((4.0/3.0))*u1_B0(0,3,0)*rhou2_B0(0,3,0))*invDelta1block0;
   }
   else if (idx[1] == 1){
       d1_p_dy = (-(5.0/6.0)*p_B0(0,0,0) - (1.0/2.0)*p_B0(0,2,0) - (1.0/4.0)*p_B0(0,-1,0) + ((1.0/12.0))*p_B0(0,3,0) + ((3.0/2.0))*p_B0(0,1,0))*invDelta1block0;
       d1_pu1_dy = (-(5.0/6.0)*p_B0(0,0,0)*u1_B0(0,0,0) - (1.0/2.0)*p_B0(0,2,0)*u1_B0(0,2,0) - (1.0/4.0)*p_B0(0,-1,0)*u1_B0(0,-1,0) + ((1.0/12.0))*p_B0(0,3,0)*u1_B0(0,3,0) + ((3.0/2.0))*p_B0(0,1,0)*u1_B0(0,1,0))*invDelta1block0;
       d1_rhoEu1_dy = (-(5.0/6.0)*u1_B0(0,0,0)*rhoE_B0(0,0,0) - (1.0/2.0)*u1_B0(0,2,0)*rhoE_B0(0,2,0) - (1.0/4.0)*u1_B0(0,-1,0)*rhoE_B0(0,-1,0) + ((1.0/12.0))*u1_B0(0,3,0)*rhoE_B0(0,3,0) + ((3.0/2.0))*u1_B0(0,1,0)*rhoE_B0(0,1,0))*invDelta1block0;
       d1_inv_rhoErho_dy = (-(5.0/6.0)*rhoE_B0(0,0,0)/rho_B0(0,0,0) - (1.0/2.0)*rhoE_B0(0,2,0)/rho_B0(0,2,0) - (1.0/4.0)*rhoE_B0(0,-1,0)/rho_B0(0,-1,0) + ((1.0/12.0))*rhoE_B0(0,3,0)/rho_B0(0,3,0) + ((3.0/2.0))*rhoE_B0(0,1,0)/rho_B0(0,1,0))*invDelta1block0;
       d1_rhou0u1_dy = (-(5.0/6.0)*u1_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/2.0)*u1_B0(0,2,0)*rhou0_B0(0,2,0) - (1.0/4.0)*u1_B0(0,-1,0)*rhou0_B0(0,-1,0) + ((1.0/12.0))*u1_B0(0,3,0)*rhou0_B0(0,3,0) + ((3.0/2.0))*u1_B0(0,1,0)*rhou0_B0(0,1,0))*invDelta1block0;
       d1_rhou1_dy = (-(5.0/6.0)*rhou1_B0(0,0,0) - (1.0/2.0)*rhou1_B0(0,2,0) - (1.0/4.0)*rhou1_B0(0,-1,0) + ((1.0/12.0))*rhou1_B0(0,3,0) + ((3.0/2.0))*rhou1_B0(0,1,0))*invDelta1block0;
       d1_rhou1u1_dy = (-(5.0/6.0)*u1_B0(0,0,0)*rhou1_B0(0,0,0) - (1.0/2.0)*u1_B0(0,2,0)*rhou1_B0(0,2,0) - (1.0/4.0)*u1_B0(0,-1,0)*rhou1_B0(0,-1,0) + ((1.0/12.0))*u1_B0(0,3,0)*rhou1_B0(0,3,0) + ((3.0/2.0))*u1_B0(0,1,0)*rhou1_B0(0,1,0))*invDelta1block0;
       d1_rhou2u1_dy = (-(5.0/6.0)*u1_B0(0,0,0)*rhou2_B0(0,0,0) - (1.0/2.0)*u1_B0(0,2,0)*rhou2_B0(0,2,0) - (1.0/4.0)*u1_B0(0,-1,0)*rhou2_B0(0,-1,0) + ((1.0/12.0))*u1_B0(0,3,0)*rhou2_B0(0,3,0) + ((3.0/2.0))*u1_B0(0,1,0)*rhou2_B0(0,1,0))*invDelta1block0;
   }
   else if (idx[1] == -1 + block0np1){
       d1_p_dy = (-4*p_B0(0,-1,0) + 3*p_B0(0,-2,0) - (4.0/3.0)*p_B0(0,-3,0) + ((1.0/4.0))*p_B0(0,-4,0) + ((25.0/12.0))*p_B0(0,0,0))*invDelta1block0;
       d1_pu1_dy = (-4*p_B0(0,-1,0)*u1_B0(0,-1,0) + 3*p_B0(0,-2,0)*u1_B0(0,-2,0) - (4.0/3.0)*p_B0(0,-3,0)*u1_B0(0,-3,0) + ((1.0/4.0))*p_B0(0,-4,0)*u1_B0(0,-4,0) + ((25.0/12.0))*p_B0(0,0,0)*u1_B0(0,0,0))*invDelta1block0;
       d1_rhoEu1_dy = (-4*u1_B0(0,-1,0)*rhoE_B0(0,-1,0) + 3*u1_B0(0,-2,0)*rhoE_B0(0,-2,0) - (4.0/3.0)*u1_B0(0,-3,0)*rhoE_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,-4,0)*rhoE_B0(0,-4,0) + ((25.0/12.0))*u1_B0(0,0,0)*rhoE_B0(0,0,0))*invDelta1block0;
       d1_inv_rhoErho_dy = (-4*rhoE_B0(0,-1,0)/rho_B0(0,-1,0) + 3*rhoE_B0(0,-2,0)/rho_B0(0,-2,0) - (4.0/3.0)*rhoE_B0(0,-3,0)/rho_B0(0,-3,0) + ((1.0/4.0))*rhoE_B0(0,-4,0)/rho_B0(0,-4,0) + ((25.0/12.0))*rhoE_B0(0,0,0)/rho_B0(0,0,0))*invDelta1block0;
       d1_rhou0u1_dy = (-4*u1_B0(0,-1,0)*rhou0_B0(0,-1,0) + 3*u1_B0(0,-2,0)*rhou0_B0(0,-2,0) - (4.0/3.0)*u1_B0(0,-3,0)*rhou0_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,-4,0)*rhou0_B0(0,-4,0) + ((25.0/12.0))*u1_B0(0,0,0)*rhou0_B0(0,0,0))*invDelta1block0;
       d1_rhou1_dy = (-4*rhou1_B0(0,-1,0) + 3*rhou1_B0(0,-2,0) - (4.0/3.0)*rhou1_B0(0,-3,0) + ((1.0/4.0))*rhou1_B0(0,-4,0) + ((25.0/12.0))*rhou1_B0(0,0,0))*invDelta1block0;
       d1_rhou1u1_dy = (-4*u1_B0(0,-1,0)*rhou1_B0(0,-1,0) + 3*u1_B0(0,-2,0)*rhou1_B0(0,-2,0) - (4.0/3.0)*u1_B0(0,-3,0)*rhou1_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,-4,0)*rhou1_B0(0,-4,0) + ((25.0/12.0))*u1_B0(0,0,0)*rhou1_B0(0,0,0))*invDelta1block0;
       d1_rhou2u1_dy = (-4*u1_B0(0,-1,0)*rhou2_B0(0,-1,0) + 3*u1_B0(0,-2,0)*rhou2_B0(0,-2,0) - (4.0/3.0)*u1_B0(0,-3,0)*rhou2_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,-4,0)*rhou2_B0(0,-4,0) + ((25.0/12.0))*u1_B0(0,0,0)*rhou2_B0(0,0,0))*invDelta1block0;
   }
   else if (idx[1] == -2 + block0np1){
       d1_p_dy = (((1.0/2.0))*p_B0(0,-2,0) - (3.0/2.0)*p_B0(0,-1,0) - (1.0/12.0)*p_B0(0,-3,0) + ((1.0/4.0))*p_B0(0,1,0) + ((5.0/6.0))*p_B0(0,0,0))*invDelta1block0;
       d1_pu1_dy = (((1.0/2.0))*p_B0(0,-2,0)*u1_B0(0,-2,0) - (3.0/2.0)*p_B0(0,-1,0)*u1_B0(0,-1,0) - (1.0/12.0)*p_B0(0,-3,0)*u1_B0(0,-3,0) + ((1.0/4.0))*p_B0(0,1,0)*u1_B0(0,1,0) + ((5.0/6.0))*p_B0(0,0,0)*u1_B0(0,0,0))*invDelta1block0;
       d1_rhoEu1_dy = (((1.0/2.0))*u1_B0(0,-2,0)*rhoE_B0(0,-2,0) - (3.0/2.0)*u1_B0(0,-1,0)*rhoE_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,-3,0)*rhoE_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,1,0)*rhoE_B0(0,1,0) + ((5.0/6.0))*u1_B0(0,0,0)*rhoE_B0(0,0,0))*invDelta1block0;
       d1_inv_rhoErho_dy = (((1.0/2.0))*rhoE_B0(0,-2,0)/rho_B0(0,-2,0) - (3.0/2.0)*rhoE_B0(0,-1,0)/rho_B0(0,-1,0) - (1.0/12.0)*rhoE_B0(0,-3,0)/rho_B0(0,-3,0) + ((1.0/4.0))*rhoE_B0(0,1,0)/rho_B0(0,1,0) + ((5.0/6.0))*rhoE_B0(0,0,0)/rho_B0(0,0,0))*invDelta1block0;
       d1_rhou0u1_dy = (((1.0/2.0))*u1_B0(0,-2,0)*rhou0_B0(0,-2,0) - (3.0/2.0)*u1_B0(0,-1,0)*rhou0_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,-3,0)*rhou0_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,1,0)*rhou0_B0(0,1,0) + ((5.0/6.0))*u1_B0(0,0,0)*rhou0_B0(0,0,0))*invDelta1block0;
       d1_rhou1_dy = (((1.0/2.0))*rhou1_B0(0,-2,0) - (3.0/2.0)*rhou1_B0(0,-1,0) - (1.0/12.0)*rhou1_B0(0,-3,0) + ((1.0/4.0))*rhou1_B0(0,1,0) + ((5.0/6.0))*rhou1_B0(0,0,0))*invDelta1block0;
       d1_rhou1u1_dy = (((1.0/2.0))*u1_B0(0,-2,0)*rhou1_B0(0,-2,0) - (3.0/2.0)*u1_B0(0,-1,0)*rhou1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,-3,0)*rhou1_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,1,0)*rhou1_B0(0,1,0) + ((5.0/6.0))*u1_B0(0,0,0)*rhou1_B0(0,0,0))*invDelta1block0;
       d1_rhou2u1_dy = (((1.0/2.0))*u1_B0(0,-2,0)*rhou2_B0(0,-2,0) - (3.0/2.0)*u1_B0(0,-1,0)*rhou2_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,-3,0)*rhou2_B0(0,-3,0) + ((1.0/4.0))*u1_B0(0,1,0)*rhou2_B0(0,1,0) + ((5.0/6.0))*u1_B0(0,0,0)*rhou2_B0(0,0,0))*invDelta1block0;
   }
   else{
       d1_p_dy = (-(2.0/3.0)*p_B0(0,-1,0) - (1.0/12.0)*p_B0(0,2,0) + ((1.0/12.0))*p_B0(0,-2,0) + ((2.0/3.0))*p_B0(0,1,0))*invDelta1block0;
       d1_pu1_dy = (-(2.0/3.0)*p_B0(0,-1,0)*u1_B0(0,-1,0) - (1.0/12.0)*p_B0(0,2,0)*u1_B0(0,2,0) + ((1.0/12.0))*p_B0(0,-2,0)*u1_B0(0,-2,0) + ((2.0/3.0))*p_B0(0,1,0)*u1_B0(0,1,0))*invDelta1block0;
       d1_rhoEu1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhoE_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhoE_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0)*rhoE_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhoE_B0(0,1,0))*invDelta1block0;
       d1_inv_rhoErho_dy = (-(2.0/3.0)*rhoE_B0(0,-1,0)/rho_B0(0,-1,0) - (1.0/12.0)*rhoE_B0(0,2,0)/rho_B0(0,2,0) + ((1.0/12.0))*rhoE_B0(0,-2,0)/rho_B0(0,-2,0) + ((2.0/3.0))*rhoE_B0(0,1,0)/rho_B0(0,1,0))*invDelta1block0;
       d1_rhou0u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou0_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou0_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0)*rhou0_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou0_B0(0,1,0))*invDelta1block0;
       d1_rhou1_dy = (-(2.0/3.0)*rhou1_B0(0,-1,0) - (1.0/12.0)*rhou1_B0(0,2,0) + ((1.0/12.0))*rhou1_B0(0,-2,0) + ((2.0/3.0))*rhou1_B0(0,1,0))*invDelta1block0;
       d1_rhou1u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou1_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou1_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0)*rhou1_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou1_B0(0,1,0))*invDelta1block0;
       d1_rhou2u1_dy = (-(2.0/3.0)*u1_B0(0,-1,0)*rhou2_B0(0,-1,0) - (1.0/12.0)*u1_B0(0,2,0)*rhou2_B0(0,2,0) + ((1.0/12.0))*u1_B0(0,-2,0)*rhou2_B0(0,-2,0) + ((2.0/3.0))*u1_B0(0,1,0)*rhou2_B0(0,1,0))*invDelta1block0;
   }
   d1_p_dz = (-(2.0/3.0)*p_B0(0,0,-1) - (1.0/12.0)*p_B0(0,0,2) + ((1.0/12.0))*p_B0(0,0,-2) + ((2.0/3.0))*p_B0(0,0,1))*invDelta2block0;
   d1_pu2_dz = (-(2.0/3.0)*p_B0(0,0,-1)*u2_B0(0,0,-1) - (1.0/12.0)*p_B0(0,0,2)*u2_B0(0,0,2) + ((1.0/12.0))*p_B0(0,0,-2)*u2_B0(0,0,-2) + ((2.0/3.0))*p_B0(0,0,1)*u2_B0(0,0,1))*invDelta2block0;
   d1_rhoEu2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhoE_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhoE_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2)*rhoE_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhoE_B0(0,0,1))*invDelta2block0;
   d1_inv_rhoErho_dz = (-(2.0/3.0)*rhoE_B0(0,0,-1)/rho_B0(0,0,-1) - (1.0/12.0)*rhoE_B0(0,0,2)/rho_B0(0,0,2) + ((1.0/12.0))*rhoE_B0(0,0,-2)/rho_B0(0,0,-2) + ((2.0/3.0))*rhoE_B0(0,0,1)/rho_B0(0,0,1))*invDelta2block0;
   d1_rhou0u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou0_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou0_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2)*rhou0_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou0_B0(0,0,1))*invDelta2block0;
   d1_rhou1u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou1_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou1_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2)*rhou1_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou1_B0(0,0,1))*invDelta2block0;
   d1_rhou2_dz = (-(2.0/3.0)*rhou2_B0(0,0,-1) - (1.0/12.0)*rhou2_B0(0,0,2) + ((1.0/12.0))*rhou2_B0(0,0,-2) + ((2.0/3.0))*rhou2_B0(0,0,1))*invDelta2block0;
   d1_rhou2u2_dz = (-(2.0/3.0)*u2_B0(0,0,-1)*rhou2_B0(0,0,-1) - (1.0/12.0)*u2_B0(0,0,2)*rhou2_B0(0,0,2) + ((1.0/12.0))*u2_B0(0,0,-2)*rhou2_B0(0,0,-2) + ((2.0/3.0))*u2_B0(0,0,1)*rhou2_B0(0,0,1))*invDelta2block0;
   Residual0_B0(0,0,0) = -d1_rhou0_dx - d1_rhou2_dz - D11_B0(0,0,0)*d1_rhou1_dy;
   Residual1_B0(0,0,0) = -d1_p_dx - (1.0/2.0)*d1_rhou0u0_dx - (1.0/2.0)*d1_rhou0u2_dz - (1.0/2.0)*u0_B0(0,0,0)*d1_rhou0_dx - (1.0/2.0)*u0_B0(0,0,0)*d1_rhou2_dz - (1.0/2.0)*D11_B0(0,0,0)*d1_rhou0u1_dy - (1.0/2.0)*wk0_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/2.0)*wk8_B0(0,0,0)*rhou2_B0(0,0,0) - (1.0/2.0)*u0_B0(0,0,0)*D11_B0(0,0,0)*d1_rhou1_dy - (1.0/2.0)*D11_B0(0,0,0)*wk4_B0(0,0,0)*rhou1_B0(0,0,0);
   Residual2_B0(0,0,0) = -(1.0/2.0)*d1_rhou1u0_dx - (1.0/2.0)*d1_rhou1u2_dz - D11_B0(0,0,0)*d1_p_dy - (1.0/2.0)*(D11_B0(0,0,0)*d1_rhou1_dy + d1_rhou0_dx + d1_rhou2_dz)*u1_B0(0,0,0) - (1.0/2.0)*D11_B0(0,0,0)*d1_rhou1u1_dy - (1.0/2.0)*wk1_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/2.0)*wk9_B0(0,0,0)*rhou2_B0(0,0,0) - (1.0/2.0)*D11_B0(0,0,0)*wk5_B0(0,0,0)*rhou1_B0(0,0,0);
   Residual3_B0(0,0,0) = -d1_p_dz - (1.0/2.0)*d1_rhou2u0_dx - (1.0/2.0)*d1_rhou2u2_dz - (1.0/2.0)*(D11_B0(0,0,0)*d1_rhou1_dy + d1_rhou0_dx + d1_rhou2_dz)*u2_B0(0,0,0) - (1.0/2.0)*D11_B0(0,0,0)*d1_rhou2u1_dy - (1.0/2.0)*wk2_B0(0,0,0)*rhou0_B0(0,0,0) - (1.0/2.0)*wk10_B0(0,0,0)*rhou2_B0(0,0,0) - (1.0/2.0)*D11_B0(0,0,0)*wk6_B0(0,0,0)*rhou1_B0(0,0,0);
   Residual4_B0(0,0,0) = -d1_pu0_dx - d1_pu2_dz - (1.0/2.0)*d1_rhoEu0_dx - (1.0/2.0)*d1_rhoEu2_dz - D11_B0(0,0,0)*d1_pu1_dy - (1.0/2.0)*D11_B0(0,0,0)*d1_rhoEu1_dy - (1.0/2.0)*rhou0_B0(0,0,0)*d1_inv_rhoErho_dx - (1.0/2.0)*rhou2_B0(0,0,0)*d1_inv_rhoErho_dz - (1.0/2.0)*(D11_B0(0,0,0)*d1_rhou1_dy + d1_rhou0_dx + d1_rhou2_dz)*rhoE_B0(0,0,0)/rho_B0(0,0,0) - (1.0/2.0)*D11_B0(0,0,0)*rhou1_B0(0,0,0)*d1_inv_rhoErho_dy;
}

 void opensbliblock00Kernel029(const ACC<double> &D11_B0, const ACC<double> &SD111_B0, const ACC<double> &T_B0, const ACC<double> &mu_B0, const ACC<double> &u0_B0, const ACC<double> &u1_B0, const ACC<double> &u2_B0, const ACC<double> &wk0_B0, const ACC<double> &wk10_B0, const ACC<double> &wk11_B0, const ACC<double> &wk1_B0, const ACC<double> &wk2_B0, const ACC<double> &wk3_B0, const ACC<double> &wk4_B0, const ACC<double> &wk5_B0, const ACC<double> &wk6_B0, const ACC<double> &wk7_B0, const ACC<double> &wk8_B0, const ACC<double> &wk9_B0, ACC<double> &Residual1_B0, ACC<double> &Residual2_B0, ACC<double> &Residual3_B0, ACC<double> &Residual4_B0, const int *idx)
{
   double d1_mu_dx = 0.0; double d1_mu_dy = 0.0; double d1_mu_dz = 0.0;
   double d1_wk0_dy = 0.0; double d1_wk0_dz = 0.0; double d1_wk1_dy = 0.0;
   double d1_wk2_dz = 0.0; double d1_wk5_dz = 0.0; double d1_wk6_dz = 0.0;
   double d2_T_dx = 0.0; double d2_T_dy = 0.0; double d2_T_dz = 0.0;
   double d2_u0_dx = 0.0; double d2_u0_dy = 0.0; double d2_u0_dz = 0.0;
   double d2_u1_dx = 0.0; double d2_u1_dy = 0.0; double d2_u1_dz = 0.0;
   double d2_u2_dx = 0.0; double d2_u2_dy = 0.0; double d2_u2_dz = 0.0;
   if (idx[0] == 0){
       d1_mu_dx = (3.0*mu_B0(1,0,0) + 0.333333333333333*mu_B0(3,0,0) - 1.5*mu_B0(2,0,0) - 1.83333333333333*mu_B0(0,0,0))*invDelta0block0;
   }
   else if (idx[0] == 1){
       d1_mu_dx = (0.0394168524399447*mu_B0(2,0,0) + 0.00571369039775442*mu_B0(4,0,0) + 0.719443173328855*mu_B0(1,0,0) - 0.322484932882161*mu_B0(0,0,0) - 0.0658051057710389*mu_B0(3,0,0) - 0.376283677513354*mu_B0(-1,0,0))*invDelta0block0;
   }
   else if (idx[0] == 2){
       d1_mu_dx = (0.197184333887745*mu_B0(0,0,0) + 0.521455851089587*mu_B0(1,0,0) + 0.113446470384241*mu_B0(-2,0,0) - 0.00412637789557492*mu_B0(3,0,0) - 0.0367146847001261*mu_B0(2,0,0) - 0.791245592765872*mu_B0(-1,0,0))*invDelta0block0;
   }
   else if (idx[0] == 3){
       d1_mu_dx = (0.0451033223343881*mu_B0(0,0,0) + 0.652141084861241*mu_B0(1,0,0) + 0.121937153224065*mu_B0(-2,0,0) - 0.00932597985049999*mu_B0(-3,0,0) - 0.727822147724592*mu_B0(-1,0,0) - 0.082033432844602*mu_B0(2,0,0))*invDelta0block0;
   }
   else if (idx[0] == -1 + block0np0){
       d1_mu_dx = (-4*mu_B0(-1,0,0) + 3*mu_B0(-2,0,0) - (4.0/3.0)*mu_B0(-3,0,0) + ((1.0/4.0))*mu_B0(-4,0,0) + ((25.0/12.0))*mu_B0(0,0,0))*invDelta0block0;
   }
   else if (idx[0] == -2 + block0np0){
       d1_mu_dx = (((1.0/2.0))*mu_B0(-2,0,0) - (3.0/2.0)*mu_B0(-1,0,0) - (1.0/12.0)*mu_B0(-3,0,0) + ((1.0/4.0))*mu_B0(1,0,0) + ((5.0/6.0))*mu_B0(0,0,0))*invDelta0block0;
   }
   else{
       d1_mu_dx = (-(2.0/3.0)*mu_B0(-1,0,0) - (1.0/12.0)*mu_B0(2,0,0) + ((1.0/12.0))*mu_B0(-2,0,0) + ((2.0/3.0))*mu_B0(1,0,0))*invDelta0block0;
   }
   if (idx[0] == 0){
       d2_T_dx = (-(26.0/3.0)*T_B0(1,0,0) - (14.0/3.0)*T_B0(3,0,0) + ((11.0/12.0))*T_B0(4,0,0) + ((19.0/2.0))*T_B0(2,0,0) + ((35.0/12.0))*T_B0(0,0,0))*inv2Delta0block0;
       d2_u0_dx = (-(26.0/3.0)*u0_B0(1,0,0) - (14.0/3.0)*u0_B0(3,0,0) + ((11.0/12.0))*u0_B0(4,0,0) + ((19.0/2.0))*u0_B0(2,0,0) + ((35.0/12.0))*u0_B0(0,0,0))*inv2Delta0block0;
       d2_u1_dx = (-(26.0/3.0)*u1_B0(1,0,0) - (14.0/3.0)*u1_B0(3,0,0) + ((11.0/12.0))*u1_B0(4,0,0) + ((19.0/2.0))*u1_B0(2,0,0) + ((35.0/12.0))*u1_B0(0,0,0))*inv2Delta0block0;
       d2_u2_dx = (-(26.0/3.0)*u2_B0(1,0,0) - (14.0/3.0)*u2_B0(3,0,0) + ((11.0/12.0))*u2_B0(4,0,0) + ((19.0/2.0))*u2_B0(2,0,0) + ((35.0/12.0))*u2_B0(0,0,0))*inv2Delta0block0;
   }
   else if (idx[0] == 1){
       d2_T_dx = (((1.0/2.0))*T_B0(1,0,0) - (5.0/3.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(3,0,0) + ((1.0/3.0))*T_B0(2,0,0) + ((11.0/12.0))*T_B0(-1,0,0))*inv2Delta0block0;
       d2_u0_dx = (((1.0/2.0))*u0_B0(1,0,0) - (5.0/3.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(3,0,0) + ((1.0/3.0))*u0_B0(2,0,0) + ((11.0/12.0))*u0_B0(-1,0,0))*inv2Delta0block0;
       d2_u1_dx = (((1.0/2.0))*u1_B0(1,0,0) - (5.0/3.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(3,0,0) + ((1.0/3.0))*u1_B0(2,0,0) + ((11.0/12.0))*u1_B0(-1,0,0))*inv2Delta0block0;
       d2_u2_dx = (((1.0/2.0))*u2_B0(1,0,0) - (5.0/3.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(3,0,0) + ((1.0/3.0))*u2_B0(2,0,0) + ((11.0/12.0))*u2_B0(-1,0,0))*inv2Delta0block0;
   }
   else if (idx[0] == -1 + block0np0){
       d2_T_dx = (-(26.0/3.0)*T_B0(-1,0,0) - (14.0/3.0)*T_B0(-3,0,0) + ((11.0/12.0))*T_B0(-4,0,0) + ((19.0/2.0))*T_B0(-2,0,0) + ((35.0/12.0))*T_B0(0,0,0))*inv2Delta0block0;
       d2_u0_dx = (-(26.0/3.0)*u0_B0(-1,0,0) - (14.0/3.0)*u0_B0(-3,0,0) + ((11.0/12.0))*u0_B0(-4,0,0) + ((19.0/2.0))*u0_B0(-2,0,0) + ((35.0/12.0))*u0_B0(0,0,0))*inv2Delta0block0;
       d2_u1_dx = (-(26.0/3.0)*u1_B0(-1,0,0) - (14.0/3.0)*u1_B0(-3,0,0) + ((11.0/12.0))*u1_B0(-4,0,0) + ((19.0/2.0))*u1_B0(-2,0,0) + ((35.0/12.0))*u1_B0(0,0,0))*inv2Delta0block0;
       d2_u2_dx = (-(26.0/3.0)*u2_B0(-1,0,0) - (14.0/3.0)*u2_B0(-3,0,0) + ((11.0/12.0))*u2_B0(-4,0,0) + ((19.0/2.0))*u2_B0(-2,0,0) + ((35.0/12.0))*u2_B0(0,0,0))*inv2Delta0block0;
   }
   else if (idx[0] == -2 + block0np0){
       d2_T_dx = (((1.0/2.0))*T_B0(-1,0,0) - (5.0/3.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(-3,0,0) + ((1.0/3.0))*T_B0(-2,0,0) + ((11.0/12.0))*T_B0(1,0,0))*inv2Delta0block0;
       d2_u0_dx = (((1.0/2.0))*u0_B0(-1,0,0) - (5.0/3.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(-3,0,0) + ((1.0/3.0))*u0_B0(-2,0,0) + ((11.0/12.0))*u0_B0(1,0,0))*inv2Delta0block0;
       d2_u1_dx = (((1.0/2.0))*u1_B0(-1,0,0) - (5.0/3.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(-3,0,0) + ((1.0/3.0))*u1_B0(-2,0,0) + ((11.0/12.0))*u1_B0(1,0,0))*inv2Delta0block0;
       d2_u2_dx = (((1.0/2.0))*u2_B0(-1,0,0) - (5.0/3.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(-3,0,0) + ((1.0/3.0))*u2_B0(-2,0,0) + ((11.0/12.0))*u2_B0(1,0,0))*inv2Delta0block0;
   }
   else{
       d2_T_dx = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(-2,0,0) - (1.0/12.0)*T_B0(2,0,0) + ((4.0/3.0))*T_B0(1,0,0) + ((4.0/3.0))*T_B0(-1,0,0))*inv2Delta0block0;
       d2_u0_dx = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(-2,0,0) - (1.0/12.0)*u0_B0(2,0,0) + ((4.0/3.0))*u0_B0(1,0,0) + ((4.0/3.0))*u0_B0(-1,0,0))*inv2Delta0block0;
       d2_u1_dx = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(-2,0,0) - (1.0/12.0)*u1_B0(2,0,0) + ((4.0/3.0))*u1_B0(1,0,0) + ((4.0/3.0))*u1_B0(-1,0,0))*inv2Delta0block0;
       d2_u2_dx = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(-2,0,0) - (1.0/12.0)*u2_B0(2,0,0) + ((4.0/3.0))*u2_B0(1,0,0) + ((4.0/3.0))*u2_B0(-1,0,0))*inv2Delta0block0;
   }
   if (idx[1] == 0){
       d1_mu_dy = (-3*mu_B0(0,2,0) + 4*mu_B0(0,1,0) - (25.0/12.0)*mu_B0(0,0,0) - (1.0/4.0)*mu_B0(0,4,0) + ((4.0/3.0))*mu_B0(0,3,0))*invDelta1block0;
       d1_wk0_dy = (-3*wk0_B0(0,2,0) + 4*wk0_B0(0,1,0) - (25.0/12.0)*wk0_B0(0,0,0) - (1.0/4.0)*wk0_B0(0,4,0) + ((4.0/3.0))*wk0_B0(0,3,0))*invDelta1block0;
       d1_wk1_dy = (-3*wk1_B0(0,2,0) + 4*wk1_B0(0,1,0) - (25.0/12.0)*wk1_B0(0,0,0) - (1.0/4.0)*wk1_B0(0,4,0) + ((4.0/3.0))*wk1_B0(0,3,0))*invDelta1block0;
   }
   else if (idx[1] == 1){
       d1_mu_dy = (-(5.0/6.0)*mu_B0(0,0,0) - (1.0/2.0)*mu_B0(0,2,0) - (1.0/4.0)*mu_B0(0,-1,0) + ((1.0/12.0))*mu_B0(0,3,0) + ((3.0/2.0))*mu_B0(0,1,0))*invDelta1block0;
       d1_wk0_dy = (-(5.0/6.0)*wk0_B0(0,0,0) - (1.0/2.0)*wk0_B0(0,2,0) - (1.0/4.0)*wk0_B0(0,-1,0) + ((1.0/12.0))*wk0_B0(0,3,0) + ((3.0/2.0))*wk0_B0(0,1,0))*invDelta1block0;
       d1_wk1_dy = (-(5.0/6.0)*wk1_B0(0,0,0) - (1.0/2.0)*wk1_B0(0,2,0) - (1.0/4.0)*wk1_B0(0,-1,0) + ((1.0/12.0))*wk1_B0(0,3,0) + ((3.0/2.0))*wk1_B0(0,1,0))*invDelta1block0;
   }
   else if (idx[1] == -1 + block0np1){
       d1_mu_dy = (-4*mu_B0(0,-1,0) + 3*mu_B0(0,-2,0) - (4.0/3.0)*mu_B0(0,-3,0) + ((1.0/4.0))*mu_B0(0,-4,0) + ((25.0/12.0))*mu_B0(0,0,0))*invDelta1block0;
       d1_wk0_dy = (-4*wk0_B0(0,-1,0) + 3*wk0_B0(0,-2,0) - (4.0/3.0)*wk0_B0(0,-3,0) + ((1.0/4.0))*wk0_B0(0,-4,0) + ((25.0/12.0))*wk0_B0(0,0,0))*invDelta1block0;
       d1_wk1_dy = (-4*wk1_B0(0,-1,0) + 3*wk1_B0(0,-2,0) - (4.0/3.0)*wk1_B0(0,-3,0) + ((1.0/4.0))*wk1_B0(0,-4,0) + ((25.0/12.0))*wk1_B0(0,0,0))*invDelta1block0;
   }
   else if (idx[1] == -2 + block0np1){
       d1_mu_dy = (((1.0/2.0))*mu_B0(0,-2,0) - (3.0/2.0)*mu_B0(0,-1,0) - (1.0/12.0)*mu_B0(0,-3,0) + ((1.0/4.0))*mu_B0(0,1,0) + ((5.0/6.0))*mu_B0(0,0,0))*invDelta1block0;
       d1_wk0_dy = (((1.0/2.0))*wk0_B0(0,-2,0) - (3.0/2.0)*wk0_B0(0,-1,0) - (1.0/12.0)*wk0_B0(0,-3,0) + ((1.0/4.0))*wk0_B0(0,1,0) + ((5.0/6.0))*wk0_B0(0,0,0))*invDelta1block0;
       d1_wk1_dy = (((1.0/2.0))*wk1_B0(0,-2,0) - (3.0/2.0)*wk1_B0(0,-1,0) - (1.0/12.0)*wk1_B0(0,-3,0) + ((1.0/4.0))*wk1_B0(0,1,0) + ((5.0/6.0))*wk1_B0(0,0,0))*invDelta1block0;
   }
   else{
       d1_mu_dy = (-(2.0/3.0)*mu_B0(0,-1,0) - (1.0/12.0)*mu_B0(0,2,0) + ((1.0/12.0))*mu_B0(0,-2,0) + ((2.0/3.0))*mu_B0(0,1,0))*invDelta1block0;
       d1_wk0_dy = (-(2.0/3.0)*wk0_B0(0,-1,0) - (1.0/12.0)*wk0_B0(0,2,0) + ((1.0/12.0))*wk0_B0(0,-2,0) + ((2.0/3.0))*wk0_B0(0,1,0))*invDelta1block0;
       d1_wk1_dy = (-(2.0/3.0)*wk1_B0(0,-1,0) - (1.0/12.0)*wk1_B0(0,2,0) + ((1.0/12.0))*wk1_B0(0,-2,0) + ((2.0/3.0))*wk1_B0(0,1,0))*invDelta1block0;
   }
   if (idx[1] == 0){
       d2_T_dy = (-(26.0/3.0)*T_B0(0,1,0) - (14.0/3.0)*T_B0(0,3,0) + ((11.0/12.0))*T_B0(0,4,0) + ((19.0/2.0))*T_B0(0,2,0) + ((35.0/12.0))*T_B0(0,0,0))*inv2Delta1block0;
       d2_u0_dy = (-(26.0/3.0)*u0_B0(0,1,0) - (14.0/3.0)*u0_B0(0,3,0) + ((11.0/12.0))*u0_B0(0,4,0) + ((19.0/2.0))*u0_B0(0,2,0) + ((35.0/12.0))*u0_B0(0,0,0))*inv2Delta1block0;
       d2_u1_dy = (-(26.0/3.0)*u1_B0(0,1,0) - (14.0/3.0)*u1_B0(0,3,0) + ((11.0/12.0))*u1_B0(0,4,0) + ((19.0/2.0))*u1_B0(0,2,0) + ((35.0/12.0))*u1_B0(0,0,0))*inv2Delta1block0;
       d2_u2_dy = (-(26.0/3.0)*u2_B0(0,1,0) - (14.0/3.0)*u2_B0(0,3,0) + ((11.0/12.0))*u2_B0(0,4,0) + ((19.0/2.0))*u2_B0(0,2,0) + ((35.0/12.0))*u2_B0(0,0,0))*inv2Delta1block0;
   }
   else if (idx[1] == 1){
       d2_T_dy = (((1.0/2.0))*T_B0(0,1,0) - (5.0/3.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,3,0) + ((1.0/3.0))*T_B0(0,2,0) + ((11.0/12.0))*T_B0(0,-1,0))*inv2Delta1block0;
       d2_u0_dy = (((1.0/2.0))*u0_B0(0,1,0) - (5.0/3.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,3,0) + ((1.0/3.0))*u0_B0(0,2,0) + ((11.0/12.0))*u0_B0(0,-1,0))*inv2Delta1block0;
       d2_u1_dy = (((1.0/2.0))*u1_B0(0,1,0) - (5.0/3.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,3,0) + ((1.0/3.0))*u1_B0(0,2,0) + ((11.0/12.0))*u1_B0(0,-1,0))*inv2Delta1block0;
       d2_u2_dy = (((1.0/2.0))*u2_B0(0,1,0) - (5.0/3.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,3,0) + ((1.0/3.0))*u2_B0(0,2,0) + ((11.0/12.0))*u2_B0(0,-1,0))*inv2Delta1block0;
   }
   else if (idx[1] == -1 + block0np1){
       d2_T_dy = (-(26.0/3.0)*T_B0(0,-1,0) - (14.0/3.0)*T_B0(0,-3,0) + ((11.0/12.0))*T_B0(0,-4,0) + ((19.0/2.0))*T_B0(0,-2,0) + ((35.0/12.0))*T_B0(0,0,0))*inv2Delta1block0;
       d2_u0_dy = (-(26.0/3.0)*u0_B0(0,-1,0) - (14.0/3.0)*u0_B0(0,-3,0) + ((11.0/12.0))*u0_B0(0,-4,0) + ((19.0/2.0))*u0_B0(0,-2,0) + ((35.0/12.0))*u0_B0(0,0,0))*inv2Delta1block0;
       d2_u1_dy = (-(26.0/3.0)*u1_B0(0,-1,0) - (14.0/3.0)*u1_B0(0,-3,0) + ((11.0/12.0))*u1_B0(0,-4,0) + ((19.0/2.0))*u1_B0(0,-2,0) + ((35.0/12.0))*u1_B0(0,0,0))*inv2Delta1block0;
       d2_u2_dy = (-(26.0/3.0)*u2_B0(0,-1,0) - (14.0/3.0)*u2_B0(0,-3,0) + ((11.0/12.0))*u2_B0(0,-4,0) + ((19.0/2.0))*u2_B0(0,-2,0) + ((35.0/12.0))*u2_B0(0,0,0))*inv2Delta1block0;
   }
   else if (idx[1] == -2 + block0np1){
       d2_T_dy = (((1.0/2.0))*T_B0(0,-1,0) - (5.0/3.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,-3,0) + ((1.0/3.0))*T_B0(0,-2,0) + ((11.0/12.0))*T_B0(0,1,0))*inv2Delta1block0;
       d2_u0_dy = (((1.0/2.0))*u0_B0(0,-1,0) - (5.0/3.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,-3,0) + ((1.0/3.0))*u0_B0(0,-2,0) + ((11.0/12.0))*u0_B0(0,1,0))*inv2Delta1block0;
       d2_u1_dy = (((1.0/2.0))*u1_B0(0,-1,0) - (5.0/3.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,-3,0) + ((1.0/3.0))*u1_B0(0,-2,0) + ((11.0/12.0))*u1_B0(0,1,0))*inv2Delta1block0;
       d2_u2_dy = (((1.0/2.0))*u2_B0(0,-1,0) - (5.0/3.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,-3,0) + ((1.0/3.0))*u2_B0(0,-2,0) + ((11.0/12.0))*u2_B0(0,1,0))*inv2Delta1block0;
   }
   else{
       d2_T_dy = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,-2,0) - (1.0/12.0)*T_B0(0,2,0) + ((4.0/3.0))*T_B0(0,1,0) + ((4.0/3.0))*T_B0(0,-1,0))*inv2Delta1block0;
       d2_u0_dy = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,-2,0) - (1.0/12.0)*u0_B0(0,2,0) + ((4.0/3.0))*u0_B0(0,1,0) + ((4.0/3.0))*u0_B0(0,-1,0))*inv2Delta1block0;
       d2_u1_dy = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,-2,0) - (1.0/12.0)*u1_B0(0,2,0) + ((4.0/3.0))*u1_B0(0,1,0) + ((4.0/3.0))*u1_B0(0,-1,0))*inv2Delta1block0;
       d2_u2_dy = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,-2,0) - (1.0/12.0)*u2_B0(0,2,0) + ((4.0/3.0))*u2_B0(0,1,0) + ((4.0/3.0))*u2_B0(0,-1,0))*inv2Delta1block0;
   }
   d2_T_dz = (-(5.0/2.0)*T_B0(0,0,0) - (1.0/12.0)*T_B0(0,0,-2) - (1.0/12.0)*T_B0(0,0,2) + ((4.0/3.0))*T_B0(0,0,1) + ((4.0/3.0))*T_B0(0,0,-1))*inv2Delta2block0;
   d1_mu_dz = (-(2.0/3.0)*mu_B0(0,0,-1) - (1.0/12.0)*mu_B0(0,0,2) + ((1.0/12.0))*mu_B0(0,0,-2) + ((2.0/3.0))*mu_B0(0,0,1))*invDelta2block0;
   d2_u0_dz = (-(5.0/2.0)*u0_B0(0,0,0) - (1.0/12.0)*u0_B0(0,0,-2) - (1.0/12.0)*u0_B0(0,0,2) + ((4.0/3.0))*u0_B0(0,0,1) + ((4.0/3.0))*u0_B0(0,0,-1))*inv2Delta2block0;
   d2_u1_dz = (-(5.0/2.0)*u1_B0(0,0,0) - (1.0/12.0)*u1_B0(0,0,-2) - (1.0/12.0)*u1_B0(0,0,2) + ((4.0/3.0))*u1_B0(0,0,1) + ((4.0/3.0))*u1_B0(0,0,-1))*inv2Delta2block0;
   d2_u2_dz = (-(5.0/2.0)*u2_B0(0,0,0) - (1.0/12.0)*u2_B0(0,0,-2) - (1.0/12.0)*u2_B0(0,0,2) + ((4.0/3.0))*u2_B0(0,0,1) + ((4.0/3.0))*u2_B0(0,0,-1))*inv2Delta2block0;
   d1_wk0_dz = (-(2.0/3.0)*wk0_B0(0,0,-1) - (1.0/12.0)*wk0_B0(0,0,2) + ((1.0/12.0))*wk0_B0(0,0,-2) + ((2.0/3.0))*wk0_B0(0,0,1))*invDelta2block0;
   d1_wk2_dz = (-(2.0/3.0)*wk2_B0(0,0,-1) - (1.0/12.0)*wk2_B0(0,0,2) + ((1.0/12.0))*wk2_B0(0,0,-2) + ((2.0/3.0))*wk2_B0(0,0,1))*invDelta2block0;
   d1_wk5_dz = (-(2.0/3.0)*wk5_B0(0,0,-1) - (1.0/12.0)*wk5_B0(0,0,2) + ((1.0/12.0))*wk5_B0(0,0,-2) + ((2.0/3.0))*wk5_B0(0,0,1))*invDelta2block0;
   d1_wk6_dz = (-(2.0/3.0)*wk6_B0(0,0,-1) - (1.0/12.0)*wk6_B0(0,0,2) + ((1.0/12.0))*wk6_B0(0,0,-2) + ((2.0/3.0))*wk6_B0(0,0,1))*invDelta2block0;
   Residual1_B0(0,0,0) = (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*d1_mu_dz + (-(2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*wk0_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*d1_mu_dx + (((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + (D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u0_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk1_dy + D11_B0(0,0,0)*wk4_B0(0,0,0)*SD111_B0(0,0,0) + d2_u0_dz)*invRe*mu_B0(0,0,0) + (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*D11_B0(0,0,0)*d1_mu_dy + Residual1_B0(0,0,0);
   Residual2_B0(0,0,0) = (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*d1_mu_dx + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*d1_mu_dz + (((1.0/3.0))*D11_B0(0,0,0)*d1_wk0_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk6_dz + ((4.0/3.0))*(D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u1_dy + ((4.0/3.0))*D11_B0(0,0,0)*wk5_B0(0,0,0)*SD111_B0(0,0,0) + d2_u1_dx + d2_u1_dz)*invRe*mu_B0(0,0,0) + (-(2.0/3.0)*wk0_B0(0,0,0) - (2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*D11_B0(0,0,0)*d1_mu_dy + Residual2_B0(0,0,0);
   Residual3_B0(0,0,0) = (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*d1_mu_dx + (-(2.0/3.0)*wk0_B0(0,0,0) + ((4.0/3.0))*wk10_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*d1_mu_dz + (((1.0/3.0))*d1_wk0_dz + ((4.0/3.0))*d2_u2_dz + (D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u2_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk5_dz + D11_B0(0,0,0)*wk6_B0(0,0,0)*SD111_B0(0,0,0) + d2_u2_dx)*invRe*mu_B0(0,0,0) + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*D11_B0(0,0,0)*d1_mu_dy + Residual3_B0(0,0,0);
   Residual4_B0(0,0,0) = (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk1_B0(0,0,0) + (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*u1_B0(0,0,0)*d1_mu_dx + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk9_B0(0,0,0) + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*u1_B0(0,0,0)*d1_mu_dz + (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk2_B0(0,0,0) + (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk8_B0(0,0,0) + (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*u0_B0(0,0,0)*d1_mu_dz + (wk2_B0(0,0,0) + wk8_B0(0,0,0))*invRe*u2_B0(0,0,0)*d1_mu_dx + (-(2.0/3.0)*wk0_B0(0,0,0) + ((4.0/3.0))*wk10_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk10_B0(0,0,0) + (-(2.0/3.0)*wk0_B0(0,0,0) + ((4.0/3.0))*wk10_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*u2_B0(0,0,0)*d1_mu_dz + (-(2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*wk0_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*mu_B0(0,0,0)*wk0_B0(0,0,0) + (-(2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*wk0_B0(0,0,0) - (2.0/3.0)*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*u0_B0(0,0,0)*d1_mu_dx + (((1.0/3.0))*d1_wk0_dz + ((4.0/3.0))*d2_u2_dz + (D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u2_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk5_dz + D11_B0(0,0,0)*wk6_B0(0,0,0)*SD111_B0(0,0,0) + d2_u2_dx)*invRe*mu_B0(0,0,0)*u2_B0(0,0,0) + (((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + (D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u0_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk1_dy + D11_B0(0,0,0)*wk4_B0(0,0,0)*SD111_B0(0,0,0) + d2_u0_dz)*invRe*mu_B0(0,0,0)*u0_B0(0,0,0) + (((1.0/3.0))*D11_B0(0,0,0)*d1_wk0_dy + ((1.0/3.0))*D11_B0(0,0,0)*d1_wk6_dz + ((4.0/3.0))*(D11_B0(0,0,0)*D11_B0(0,0,0))*d2_u1_dy + ((4.0/3.0))*D11_B0(0,0,0)*wk5_B0(0,0,0)*SD111_B0(0,0,0) + d2_u1_dx + d2_u1_dz)*invRe*mu_B0(0,0,0)*u1_B0(0,0,0) + (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*mu_B0(0,0,0)*D11_B0(0,0,0)*wk4_B0(0,0,0) + (D11_B0(0,0,0)*wk4_B0(0,0,0) + wk1_B0(0,0,0))*invRe*u0_B0(0,0,0)*D11_B0(0,0,0)*d1_mu_dy + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*mu_B0(0,0,0)*D11_B0(0,0,0)*wk6_B0(0,0,0) + (D11_B0(0,0,0)*wk6_B0(0,0,0) + wk9_B0(0,0,0))*invRe*u2_B0(0,0,0)*D11_B0(0,0,0)*d1_mu_dy + (-(2.0/3.0)*wk0_B0(0,0,0) - (2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*mu_B0(0,0,0)*D11_B0(0,0,0)*wk5_B0(0,0,0) + (-(2.0/3.0)*wk0_B0(0,0,0) - (2.0/3.0)*wk10_B0(0,0,0) + ((4.0/3.0))*D11_B0(0,0,0)*wk5_B0(0,0,0))*invRe*u1_B0(0,0,0)*D11_B0(0,0,0)*d1_mu_dy + ((D11_B0(0,0,0)*D11_B0(0,0,0))*d2_T_dy + D11_B0(0,0,0)*wk7_B0(0,0,0)*SD111_B0(0,0,0) + d2_T_dx + d2_T_dz)*invPr*invRe*inv2Minf*inv_gamma_m1*mu_B0(0,0,0) + invPr*invRe*inv2Minf*inv_gamma_m1*wk3_B0(0,0,0)*d1_mu_dx + invPr*invRe*inv2Minf*inv_gamma_m1*wk11_B0(0,0,0)*d1_mu_dz + (D11_B0(0,0,0)*D11_B0(0,0,0))*invPr*invRe*inv2Minf*inv_gamma_m1*wk7_B0(0,0,0)*d1_mu_dy + Residual4_B0(0,0,0);
}

 void opensbliblock00Kernel049(const ACC<double> &Residual0_B0, const ACC<double> &Residual1_B0, const ACC<double> &Residual2_B0, const ACC<double> &Residual3_B0, const ACC<double> &Residual4_B0, ACC<double> &rhoE_B0, ACC<double> &rhoE_RKold_B0, ACC<double> &rho_B0, ACC<double> &rho_RKold_B0, ACC<double> &rhou0_B0, ACC<double> &rhou0_RKold_B0, ACC<double> &rhou1_B0, ACC<double> &rhou1_RKold_B0, ACC<double> &rhou2_B0, ACC<double> &rhou2_RKold_B0, const double *rkA, const double *rkB)
{
   rho_RKold_B0(0,0,0) = rkA[0]*rho_RKold_B0(0,0,0) + dt*Residual0_B0(0,0,0);
   rho_B0(0,0,0) = rkB[0]*rho_RKold_B0(0,0,0) + rho_B0(0,0,0);
   rhou0_RKold_B0(0,0,0) = rkA[0]*rhou0_RKold_B0(0,0,0) + dt*Residual1_B0(0,0,0);
   rhou0_B0(0,0,0) = rkB[0]*rhou0_RKold_B0(0,0,0) + rhou0_B0(0,0,0);
   rhou1_RKold_B0(0,0,0) = rkA[0]*rhou1_RKold_B0(0,0,0) + dt*Residual2_B0(0,0,0);
   rhou1_B0(0,0,0) = rkB[0]*rhou1_RKold_B0(0,0,0) + rhou1_B0(0,0,0);
   rhou2_RKold_B0(0,0,0) = rkA[0]*rhou2_RKold_B0(0,0,0) + dt*Residual3_B0(0,0,0);
   rhou2_B0(0,0,0) = rkB[0]*rhou2_RKold_B0(0,0,0) + rhou2_B0(0,0,0);
   rhoE_RKold_B0(0,0,0) = rkA[0]*rhoE_RKold_B0(0,0,0) + dt*Residual4_B0(0,0,0);
   rhoE_B0(0,0,0) = rkB[0]*rhoE_RKold_B0(0,0,0) + rhoE_B0(0,0,0);
}
 void opensbliblock00Kernel047(const ACC<double> &rhoE_B0, const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, const ACC<double> &rhou1_B0, const ACC<double> &rhou2_B0, ACC<double> &rhoE_mean_B0, ACC<double> &rho_mean_B0, ACC<double> &rhou0_mean_B0, ACC<double> &rhou0u0_mean_B0, ACC<double> &rhou0u1_mean_B0, ACC<double> &rhou0u2_mean_B0, ACC<double> &rhou1_mean_B0, ACC<double> &rhou1u1_mean_B0, ACC<double> &rhou1u2_mean_B0, ACC<double> &rhou2_mean_B0, ACC<double> &rhou2u2_mean_B0, ACC<double>& rhorhou0u0_mean_B0)
{
   double rho_inv = 0.0;
   rho_mean_B0(0,0,0) = rho_B0(0,0,0) + rho_mean_B0(0,0,0);
   rhou0_mean_B0(0,0,0) = rhou0_B0(0,0,0) + rhou0_mean_B0(0,0,0);
   rhou1_mean_B0(0,0,0) = rhou1_B0(0,0,0) + rhou1_mean_B0(0,0,0);
   rhou2_mean_B0(0,0,0) = rhou2_B0(0,0,0) + rhou2_mean_B0(0,0,0);
   rhoE_mean_B0(0,0,0) = rhoE_B0(0,0,0) + rhoE_mean_B0(0,0,0);
   rho_inv = 1.0/rho_B0(0,0,0);
   rhou0u0_mean_B0(0,0,0) = (rhou0_B0(0,0,0)*rhou0_B0(0,0,0))*rho_inv + rhou0u0_mean_B0(0,0,0);
   rhou1u1_mean_B0(0,0,0) = (rhou1_B0(0,0,0)*rhou1_B0(0,0,0))*rho_inv + rhou1u1_mean_B0(0,0,0);
   rhou2u2_mean_B0(0,0,0) = (rhou2_B0(0,0,0)*rhou2_B0(0,0,0))*rho_inv + rhou2u2_mean_B0(0,0,0);
   rhou0u1_mean_B0(0,0,0) = rhou0_B0(0,0,0)*rhou1_B0(0,0,0)*rho_inv + rhou0u1_mean_B0(0,0,0);
   rhou1u2_mean_B0(0,0,0) = rhou1_B0(0,0,0)*rhou2_B0(0,0,0)*rho_inv + rhou1u2_mean_B0(0,0,0);
   rhou0u2_mean_B0(0,0,0) = rhou0_B0(0,0,0)*rhou2_B0(0,0,0)*rho_inv + rhou0u2_mean_B0(0,0,0);
   rhorhou0u0_mean_B0(0, 0, 0) = rhou0_B0(0,0,0) * rhou0_B0(0,0,0) + rhorhou0u0_mean_B0(0,0,0);
}
 void opensbliblock00Kernel048(ACC<double> &rhoE_mean_B0, ACC<double> &rho_mean_B0, ACC<double> &rhou0_mean_B0, ACC<double> &rhou0u0_mean_B0, ACC<double> &rhou0u1_mean_B0, ACC<double> &rhou0u2_mean_B0, ACC<double> &rhou1_mean_B0, ACC<double> &rhou1u1_mean_B0, ACC<double> &rhou1u2_mean_B0, ACC<double> &rhou2_mean_B0, ACC<double> &rhou2u2_mean_B0, ACC<double>& rhorhou0u0_mean_B0, ACC<double>& taux0x1_mean_B0, ACC<double>& l_mean_B0, ACC<double>& du0dx1_mean_B0, ACC<double>& mu_mean_B0, ACC<double>& u0_mean_B0, ACC<double>& u1_mean_B0, ACC<double>& u2_mean_B0, ACC<double>& u0u0_mean_B0, ACC<double>& u1u1_mean_B0, ACC<double>& u2u2_mean_B0, ACC<double>& u0u1_mean_B0, ACC<double>& utau_mean_B0)
{
   rho_mean_B0(0,0,0) = invniter*rho_mean_B0(0,0,0);
   rhou0_mean_B0(0,0,0) = invniter*rhou0_mean_B0(0,0,0);
   rhou1_mean_B0(0,0,0) = invniter*rhou1_mean_B0(0,0,0);
   rhou2_mean_B0(0,0,0) = invniter*rhou2_mean_B0(0,0,0);
   rhoE_mean_B0(0,0,0) = invniter*rhoE_mean_B0(0,0,0);
   rhou0u0_mean_B0(0,0,0) = invniter*rhou0u0_mean_B0(0,0,0);
   rhou1u1_mean_B0(0,0,0) = invniter*rhou1u1_mean_B0(0,0,0);
   rhou2u2_mean_B0(0,0,0) = invniter*rhou2u2_mean_B0(0,0,0);
   rhou0u1_mean_B0(0,0,0) = invniter*rhou0u1_mean_B0(0,0,0);
   rhou1u2_mean_B0(0,0,0) = invniter*rhou1u2_mean_B0(0,0,0);
   rhou0u2_mean_B0(0,0,0) = invniter*rhou0u2_mean_B0(0,0,0);
   rhorhou0u0_mean_B0(0,0,0) = invniter*rhorhou0u0_mean_B0(0,0,0);
   taux0x1_mean_B0(0,0,0) = invniter*taux0x1_mean_B0(0,0,0);
   l_mean_B0(0,0,0) = invniter*l_mean_B0(0,0,0);
   du0dx1_mean_B0(0,0,0) = invniter*du0dx1_mean_B0(0,0,0);
   mu_mean_B0(0,0,0) = invniter*mu_mean_B0(0,0,0);
   u0_mean_B0(0,0,0) = invniter*u0_mean_B0(0,0,0);
   u1_mean_B0(0,0,0) = invniter*u1_mean_B0(0,0,0);
   u2_mean_B0(0,0,0) = invniter*u2_mean_B0(0,0,0);
   u0u0_mean_B0(0,0,0) = invniter*u0u0_mean_B0(0,0,0);
   u1u1_mean_B0(0,0,0) = invniter*u1u1_mean_B0(0,0,0);
   u2u2_mean_B0(0,0,0) = invniter*u2u2_mean_B0(0,0,0);
   u0u1_mean_B0(0,0,0) = invniter*u0u1_mean_B0(0,0,0);
   utau_mean_B0(0,0,0) = invniter*utau_mean_B0(0,0,0);
}
// new kernel: eval taux0_mean_B0. rhorhou0u0 is evaluated in kernel047, with the other averaging operations.
void opensbliblock00Kernel_new01(const ACC<double>& mu_B0, const ACC<double>& rho_B0, const ACC<double>& rhou0_B0, const ACC<double>& rhou1_B0, const ACC<double>& rhou2_B0, const ACC<double>& D11_B0, ACC<double>& taux0x1_mean_B0, ACC<double>& l_mean_B0, ACC<double>& mu_mean_B0, ACC<double>& du0dx1_mean_B0, ACC<double>& u0_mean_B0, ACC<double>& u1_mean_B0, ACC<double>& u2_mean_B0, ACC<double>& u0u0_mean_B0, ACC<double>& u1u1_mean_B0, ACC<double>& u2u2_mean_B0, ACC<double>& u0u1_mean_B0, ACC<double>& utau_mean_B0, const int* idx){
   double d1_u_dy = 0.0;
   if (idx[1] == 0){
       d1_u_dy = (-3*rhou0_B0(0,2,0)/rho_B0(0,2,0) + 4*rhou0_B0(0,1,0)/rho_B0(0,1,0) - (25.0/12.0)*rhou0_B0(0,0,0)/rho_B0(0,0,0) - (1.0/4.0)*rhou0_B0(0,4,0)/rho_B0(0,4,0) + ((4.0/3.0))*rhou0_B0(0,3,0)/rho_B0(0,3,0))*invDelta1block0;
   }
   else if (idx[1] == 1){
       d1_u_dy = (-(5.0/6.0)*rhou0_B0(0,0,0)/rho_B0(0,0,0) - (1.0/2.0)*rhou0_B0(0,2,0)/rho_B0(0,2,0) - (1.0/4.0)*rhou0_B0(0,-1,0)/rho_B0(0,-1,0) + ((1.0/12.0))*rhou0_B0(0,3,0)/rho_B0(0,3,0) + ((3.0/2.0))*rhou0_B0(0,1,0)/rho_B0(0,1,0))*invDelta1block0;
   }
   else if (idx[1] == -1 + block0np1){
       d1_u_dy = (-4*rhou0_B0(0,-1,0)/rho_B0(0,-1,0) + 3*rhou0_B0(0,-2,0)/rho_B0(0,-2,0) - (4.0/3.0)*rhou0_B0(0,-3,0)/rho_B0(0,-3,0) + ((1.0/4.0))*rhou0_B0(0,-4,0)/rho_B0(0,-4,0) + ((25.0/12.0))*rhou0_B0(0,0,0)/rho_B0(0,0,0))*invDelta1block0;
   }
   else if (idx[1] == -2 + block0np1){
       d1_u_dy = (((1.0/2.0))*rhou0_B0(0,-2,0)/rho_B0(0,-2,0) - (3.0/2.0)*rhou0_B0(0,-1,0)/rho_B0(0,-1,0) - (1.0/12.0)*rhou0_B0(0,-3,0)/rho_B0(0,-3,0) + ((1.0/4.0))*rhou0_B0(0,1,0)/rho_B0(0,1,0) + ((5.0/6.0))*rhou0_B0(0,0,0)/rho_B0(0,0,0))*invDelta1block0;
   }
   else{
       d1_u_dy = (-(2.0/3.0)*rhou0_B0(0,-1,0)/rho_B0(0,-1,0) - (1.0/12.0)*rhou0_B0(0,2,0)/rho_B0(0,2,0) + ((1.0/12.0))*rhou0_B0(0,-2,0)/rho_B0(0,-2,0) + ((2.0/3.0))*rhou0_B0(0,1,0)/rho_B0(0,1,0))*invDelta1block0;
   }
   taux0x1_mean_B0(0,0,0) = mu_B0(0,0,0) * D11_B0(0,0,0) * d1_u_dy + taux0x1_mean_B0(0,0,0);
   double d1_u_dy_inv = (std::abs(d1_u_dy) < 0.000001) ? 100000 : 1/d1_u_dy;
   l_mean_B0(0,0,0) = pow(1.0/Re, 0.5) * pow(mu_B0(0,0,0) * d1_u_dy_inv / (rho_B0(0,0,0)), 0.5) + l_mean_B0(0,0,0);
   du0dx1_mean_B0(0,0,0) = D11_B0(0,0,0) * d1_u_dy + du0dx1_mean_B0(0,0,0);
   mu_mean_B0(0,0,0) = mu_B0(0,0,0) + mu_mean_B0(0,0,0);
   u0_mean_B0(0,0,0) = u0_mean_B0(0,0,0) + rhou0_B0(0,0,0)/rho_B0(0,0,0);
   u1_mean_B0(0,0,0) = u1_mean_B0(0,0,0) + rhou1_B0(0,0,0)/rho_B0(0,0,0);
   u2_mean_B0(0,0,0) = u2_mean_B0(0,0,0) + rhou2_B0(0,0,0)/rho_B0(0,0,0);
   u0u0_mean_B0(0,0,0) = u0u0_mean_B0(0,0,0) + rhou0_B0(0,0,0)/rho_B0(0,0,0)*rhou0_B0(0,0,0)/rho_B0(0,0,0);
   u1u1_mean_B0(0,0,0) = u1u1_mean_B0(0,0,0) + rhou1_B0(0,0,0)/rho_B0(0,0,0)*rhou1_B0(0,0,0)/rho_B0(0,0,0);
   u2u2_mean_B0(0,0,0) = u2u2_mean_B0(0,0,0) + rhou2_B0(0,0,0)/rho_B0(0,0,0)*rhou2_B0(0,0,0)/rho_B0(0,0,0);
   u0u1_mean_B0(0,0,0) = u0u1_mean_B0(0,0,0) + rhou0_B0(0,0,0)/rho_B0(0,0,0)*rhou1_B0(0,0,0)/rho_B0(0,0,0);
   utau_mean_B0(0,0,0) = utau_mean_B0(0,0,0) + sqrt(std::abs(mu_B0(0,0,0)*D11_B0(0,0,0)*d1_u_dy/rho_B0(0,0,0)));
}

#endif