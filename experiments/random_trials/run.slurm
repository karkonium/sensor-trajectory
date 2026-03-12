#!/bin/bash
#SBATCH --job-name=random-trials
#SBATCH --partition=cpu_pr3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd /mnt/hpc/work/a5karki/sensor-trajectory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate SensorMovement

export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHON_THREADS=1

python experiments/random_trials/driver.py
