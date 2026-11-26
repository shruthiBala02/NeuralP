#!/bin/bash
# Run the timing benchmark as a SLURM job:   sbatch scripts/benchmark.sbatch
# Submit from the repository folder. Fill in the partition (and module line) if your cluster needs them.
#SBATCH --job-name=neuralP-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm-benchmark-%j.out
##SBATCH --partition=YOUR_GPU_PARTITION

# module load python/3.11
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
nvidia-smi
python -m neuralp.benchmark
