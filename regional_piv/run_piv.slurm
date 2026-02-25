#!/bin/bash
#SBATCH --job-name=regional_piv
#SBATCH --partition=cpu_pr3
#SBATCH --time=102:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


# Go to project root (one level above regional_piv)
cd /mnt/hpc/work/a5karki/sensor-trajectory

# activate our conda env 
# adjust this path to whatever you use in our ~/.bashrc
# common options:
#   ~/miniconda3/etc/profile.d/conda.sh
#   ~/anaconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate SensorMovement

# Make sure Python can see both data_generation/ and regional_piv/
export PYTHONPATH="$PWD:$PYTHONPATH"

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

# Run driver
python regional_piv/driver.py
# python data_generation/kolmogorov_flow.py
# python regional_piv/kolmo_scales.py