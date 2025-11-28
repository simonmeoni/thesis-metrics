#!/bin/bash
#SBATCH --job-name=mimic-icd-eval
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=log/mimic_icd_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --error=log/mimic_icd_%A_%a.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr
#SBATCH --array=0-15  # 16 combinations: 4 datasets × 4 top-k values

# MIMIC-III ICD-9 Classification Utility Evaluation
# Evaluates keyword-replaced datasets (v1) with different top-k configurations
#
# Configuration matrix:
# - 4 datasets: 4% keyword, 4% keyword→rephrased, 6% keyword, 6% keyword→rephrased
# - 4 top-k values: 20, 50, 100, 400
# = 16 total combinations

# Dataset paths
DATASETS=(
    "data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet"
    "data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced_rephrased.parquet"
    "data/mimic-iii/6%-dpo_2-8l6avevp_keyword_replaced.parquet"
    "data/mimic-iii/6%-dpo_2-8l6avevp_keyword_replaced_rephrased.parquet"
)

# Dataset names for output/logging
DATASET_NAMES=(
    "4pct_keyword"
    "4pct_keyword_rephrased"
    "6pct_keyword"
    "6pct_keyword_rephrased"
)

# Top-k values for ICD classification
TOPKS=(20 50 100 400)

# Calculate which dataset and top-k to use for this array task
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 4))
TOPK_IDX=$((SLURM_ARRAY_TASK_ID % 4))

DATASET_PATH=${DATASETS[$DATASET_IDX]}
DATASET_NAME=${DATASET_NAMES[$DATASET_IDX]}
TOPK=${TOPKS[$TOPK_IDX]}

echo "=========================================="
echo "MIMIC-III ICD-9 Classification Evaluation"
echo "=========================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET_NAME"
echo "Dataset Path: $DATASET_PATH"
echo "Top-K: $TOPK"
echo "Seed: 42"
echo "=========================================="

# Load environment
module purge
module load miniforge/24.9.0
conda activate synth-kg
module load cuda

# Create log directory if it doesn't exist
mkdir -p log

# Run ICD classification evaluation
echo "Running ICD classification with top-$TOPK on $DATASET_NAME..."

# Use Python to run with proper Hydra overrides
python -m thesis_metrics.cli.downstream_eval \
    --config-path=../../configs/ds_stream/icd \
    --config-name=$TOPK \
    task.type=icd_classification \
    task.model_name=microsoft/deberta-v3-base \
    task.train_data_path=$DATASET_PATH \
    task.test_data_path=data/mimic-iii/privacy_dataset_filtered.parquet \
    task.text_column=generation_rephrased \
    task.top_k_labels=$TOPK \
    task.precision_mode=true \
    task.percentile_threshold=10 \
    task.threshold=0.5 \
    wandb_project=style-transfer-icd-seed \
    seed=42

echo "✓ Completed: $DATASET_NAME with top-$TOPK"
