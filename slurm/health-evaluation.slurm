#!/bin/bash
#SBATCH --job-name=health-eval
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

# initialize variables with default values
OUTPUT_PATH="default_path"
DOWNSTREAM_DS_PATH=""
HYDRA_CONFIG="health"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
  --HYDRA_CONFIG)
    HYDRA_CONFIG="$2"
    shift
    ;;
  --DOWNSTREAM_DS_PATH)
    DOWNSTREAM_DS_PATH="$2"
    shift
    ;;
  --GROUP_ID)
    GROUP_ID="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

module purge
module load arch/h100
module load miniforge/24.9.0
module load cuda
conda activate synth-kg

# Set Python path for imports
export PYTHONPATH="/lustre/fswork/projects/rech/lch/ufw96he/thesis-metrics:$PYTHONPATH"

# Set project root for absolute paths
PROJECT_ROOT="/lustre/fswork/projects/rech/lch/ufw96he/thesis-metrics"
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p datasets/health/eval/model_outputs
mkdir -p log
mkdir -p lora

# Disable vLLM compilation to prevent freezing
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MERGE_ID=$(printf "%06d" $((RANDOM % 1000000)))
MERGE_OUTPUT_PATH=./lora/merge-${MERGE_ID}

# Step 1: Generate AlpaCare baseline responses
python thesis_metrics/core/alpacare/alpacare_generate.py --output_path "$OUTPUT_PATH" --dataset "$DOWNSTREAM_DS_PATH" --tp "$SLURM_GPUS_ON_NODE"

# Step 2: Train model on AlpaCare outputs using Hydra
python thesis_metrics/core/alpacare/sft_train.py --config-name="$HYDRA_CONFIG" dataset=./evaluation_alpacare_sft.parquet merge_output_path="$MERGE_OUTPUT_PATH" group_id="$GROUP_ID"

# Step 3: Generate responses with fine-tuned model
python thesis_metrics/core/alpacare/model_generate.py --model "$MERGE_OUTPUT_PATH" --output_path "$OUTPUT_PATH" --tp "$SLURM_GPUS_ON_NODE"
