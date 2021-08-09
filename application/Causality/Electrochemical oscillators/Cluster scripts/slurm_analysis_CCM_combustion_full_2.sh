#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=CCM_E_2
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=6G

module load julia/1.5.3

julia comm_CCM_full_2.jl
