#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=63_n_FNN2
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module load julia/1.5.3
julia comm_Lorenz63_prediction_mcdts_n_FNN2.jl
