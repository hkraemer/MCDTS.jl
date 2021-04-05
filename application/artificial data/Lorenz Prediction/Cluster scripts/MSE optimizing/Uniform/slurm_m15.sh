#!/bin/bash

#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --job-name=m_m3
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module load julia/1.5.3
julia comm_mcdts_U_multi_mean_Tw_5_K_1_n.jl
