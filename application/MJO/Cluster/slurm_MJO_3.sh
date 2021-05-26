#!/bin/bash

#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --job-name=MJO_3
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=6G

module load julia/1.5.3
julia comm_Predict_time_series3.jl
