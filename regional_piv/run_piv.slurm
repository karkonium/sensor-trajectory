#!/bin/bash
#SBATCH --job-name=regional_piv
#SBATCH --partition=cpu_pr3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=<user_id>@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

cd "$SLURM_SUBMIT_DIR"

module load python/3.10   # or whatever your cluster uses

# thread settings: BLAS 1 thread, Python N threads
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    export PYTHON_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    export PYTHON_THREADS=1
fi

python driver.py
