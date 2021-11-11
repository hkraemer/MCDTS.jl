#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=63_pec2
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module load julia/1.5.3
julia comm_Lorenz63_prediction_pecuzal.jl