#!/bin/bash
#SBATCH --job-name=cluster
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate synth-kg
python create_clusterize_dataset.py -k 6000 -s 30000 -o alpacare_k-6000_s-30000.parquet
python create_clusterize_dataset.py -k 6001 -s 10001 -o alpacare_k-6000_s-10000.parquet
python create_clusterize_dataset.py -k 4000 -s 10000 -o alpacare_k-4000_s-10000.parquet
