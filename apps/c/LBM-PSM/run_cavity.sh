#!/bin/bash

cd $OPS_INSTALL_PATH/../apps/c/LBM-PSM-Fax_II

rm rho_timestep_*.txt
rm u_timestep_*.txt
rm v_timestep_*.txt
rm particle_data_*.txt
rm sf_timestep_*.txt

./lattboltz2d_dev_seq

