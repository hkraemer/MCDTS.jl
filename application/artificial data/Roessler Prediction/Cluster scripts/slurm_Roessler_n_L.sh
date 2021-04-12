#!/bin/bash

#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --job-name=63_n_L
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module load julia/1.5.3
julia comm_Roessler_prediction_mcdts_n_L.jl
