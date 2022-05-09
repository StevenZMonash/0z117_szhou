#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --mail-user=jzho0060@student.monash.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out/%j_slurm_log.log
#SBATCH --error=out/%j_slurm_err.err

module --force purge

module load git/2.18.0
module load git-lfs/2.4.0
module load git-lfs/2.4.0
module load gcc/9.2.0
module load openmpi/4.0.2
module load matplotlib/3.2.1-python-3.7.4
module load pandas/1.0.5-python-3.7.4
module load mpi4py/3.0.3-python-3.7.4
module load tensorflow/2.1.0-python-3.7.4

source /home/jizhou/test_env/bin/activate
python Training.py