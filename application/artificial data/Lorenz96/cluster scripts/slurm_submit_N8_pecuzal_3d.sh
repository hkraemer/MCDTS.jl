#!/bin/bash

#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --job-name=8pec_3
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

module load julia/1.5.3
module load hpc
julia comm_lorenz96_N8_cluster_pecuzal_3d.jl $SLURM_NTASKS
