#!/bin/bash
#SBATCH --job-name=thesis-metrics
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate thesis-metrics
python thesis_metrics/run_privacy_metrics.py
