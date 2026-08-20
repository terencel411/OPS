const double velocity_fn(double& y, double& z){ // cannot convert non-const to const? verify this
    return u0;
}

const double set_increment(double& y, double& z){
    return velocity_fn(y, z) * dt;
}

const double radius_fn(double& y,  double& z){
    return 0.41 * delta;
}

void reset2(double& x, double& y, double& z, double& increment, double& radius, int& eps_x, int& eps_y, int& eps_z,
            double& y_min, double& y_max, double& z_min, double& z_max){
    x = x_min;
    y = y_rand(urng);
    z = z_rand(urng);
    increment = set_increment(y, z);
    radius = radius_fn(y, z);
    
    eps_x = eps_map[eps_rand(urng)];
    eps_y = eps_map[eps_rand(urng)];
    eps_z = eps_map[eps_rand(urng)];

    y_min = y - radius;
    y_max = y + radius;
    z_min = z - radius;
    z_max = z + radius;
}

void reset(double& x, double& y, double& z, double& increment, double& radius, int& eps_x, int& eps_y, int& eps_z){
    x = x_min;
    y = y_rand(urng);
    z = z_rand(urng);
    increment = set_increment(y, z);
    radius = radius_fn(y, z);
    
    eps_x = eps_map[eps_rand(urng)];
    eps_y = eps_map[eps_rand(urng)];
    eps_z = eps_map[eps_rand(urng)];
}


void convect(double& x, double& y, double& z, double& increment, double& radius, int& eps_x, int& eps_y, int& eps_z){
    x += increment;
    if (x > x_max){
        reset(x, y, z, increment, radius, eps_x, eps_y, eps_z);
    }
}

void convect2(double& x, double& y, double& z, double& increment, double& radius, int& eps_x, int& eps_y, int& eps_z,
            double& y_min, double& y_max, double& z_min, double& z_max){

    x += increment;
    if(x > x_max){
        reset2(x, y, z, increment, radius, eps_x, eps_y, eps_z, y_min, y_max, z_min, z_max);
    }

}


void convect_eddies2(double* x, double* y, double* z, double* increment, double* radius, int* eps_x, int* eps_y, int* eps_z,
                    double* y_min, double* y_max, double* z_min, double* z_max){
    for(int i{0}; i < eddies; i++){
        convect2(x[i], y[i], z[i], increment[i] ,radius[i], eps_x[i], eps_y[i], eps_z[i], y_min[i], y_max[i], z_min[i], z_max[i]);
    }
}


void instantiate_eddies2(double* x, double* y, double* z, double* increment, double* radius, int* eps_x, int* eps_y, int* eps_z,
                        double* y_min, double* y_max, double* z_min, double* z_max, int* seeds){
    std::uniform_real_distribution<double> x_rand = std::uniform_real_distribution<double>(x_min, x_max);
    std::uniform_int_distribution<int> seed_rand = std::uniform_int_distribution<int>(0, m-1);
    for(int i{0}; i< eddies; i++){
        x[i] = x_rand(urng);
        y[i] = y_rand(urng);
        z[i] = z_rand(urng);
        increment[i] = set_increment(y[i], z[i]);
        radius[i] = radius_fn(y[i], z[i]);
    
        eps_x[i] = eps_map[eps_rand(urng)];
        eps_y[i] = eps_map[eps_rand(urng)];
        eps_z[i] = eps_map[eps_rand(urng)];

        y_min[i] = y[i] - radius[i];
        y_max[i] = y[i] + radius[i];
        z_min[i] = z[i] - radius[i];
        z_max[i] = z[i] + radius[i];

        seeds[i] = seed_rand(urng);

    }
}

void print_eddy_pos(double* x, double* y, double* z){
    for(int i{0}; i < eddies; i++){
        printf("%f, %f, %f \n", x[i], y[i], z[i]);
    }
}

void print_eddy_info(double* x, double* y, double* z, double* r){
    for(int i{0}; i < eddies; i++){
        printf("%f, %f, %f, %f \n", x[i], y[i], z[i], r[i]);
    }
}

void print_norm_x(double* x, double* r){
    for(int i{0}; i < eddies; i++){
        printf("%f \n", x[i] / r[i]);
    }
}

void calc_eddies(int& eddies, const double& vol, const double& rep_radius){
    eddies = trunc(vol / pow(rep_radius, 3));
}