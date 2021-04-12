#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=u_nn_m5
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module load julia/1.5.3
julia comm_mcdts_L_uni_mean_Tw_1_K_1.jl
