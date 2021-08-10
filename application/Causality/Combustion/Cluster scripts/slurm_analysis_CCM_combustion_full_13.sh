#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=CCM_13
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=6G

module load julia/1.5.3

julia comm_CCM_full_13.jl
