#!/bin/bash
# =============================================================================
# Flexible Evaluation Runner for Jean Zay
# =============================================================================
# Usage:
#   ./run_evaluations.sh [OPTIONS]
#
# Options:
#   --mimic              Run MIMIC ICD evaluations
#   --alpacare           Run AlpaCare evaluations
#   --all                Run both MIMIC and AlpaCare (default if no option)
#   --topk 20,50,100,400 Comma-separated list of top-k values for MIMIC
#                        (default: 20,50,100,400 - all values)
#
# Examples:
#   ./run_evaluations.sh --alpacare              # Only AlpaCare
#   ./run_evaluations.sh --mimic --topk 20,50   # MIMIC with top-20 and top-50
#   ./run_evaluations.sh --all                   # Everything
# =============================================================================

set -e  # Exit on error

# Parse command-line arguments
RUN_MIMIC=false
RUN_ALPACARE=false
TOPK_VALUES="20,50,100,400"  # Default: all values

while [[ $# -gt 0 ]]; do
    case $1 in
        --mimic)
            RUN_MIMIC=true
            shift
            ;;
        --alpacare)
            RUN_ALPACARE=true
            shift
            ;;
        --all)
            RUN_MIMIC=true
            RUN_ALPACARE=true
            shift
            ;;
        --topk)
            TOPK_VALUES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mimic] [--alpacare] [--all] [--topk VALUES]"
            exit 1
            ;;
    esac
done

# If no options specified, run everything
if [ "$RUN_MIMIC" = false ] && [ "$RUN_ALPACARE" = false ]; then
    echo "No options specified. Running everything by default."
    RUN_MIMIC=true
    RUN_ALPACARE=true
fi

# Navigate to project directory
cd $WORK/thesis-metrics

echo "=========================================="
echo "Starting Evaluations"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "User: $USER"
echo "Date: $(date)"
echo "MIMIC: $RUN_MIMIC"
echo "AlpaCare: $RUN_ALPACARE"
echo "Top-K values: $TOPK_VALUES"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating synth-kg environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate synth-kg

# Set Python path for imports
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Install thesis-metrics package in editable mode if not already installed
echo "Ensuring thesis-metrics package is installed..."
if ! python -c "import thesis_metrics" 2>/dev/null; then
    echo "Installing thesis-metrics package in editable mode..."
    pip install -e .
    echo "✓ thesis-metrics package installed"
else
    echo "✓ thesis-metrics package already installed"
fi
echo ""

# Create log directory
mkdir -p log

# Check current job queue
echo "Checking current job queue..."
CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
echo "Current jobs in queue: $CURRENT_JOBS"
echo ""

# =============================================================================
# MIMIC-III ICD-9 Classification Evaluation
# =============================================================================
if [ "$RUN_MIMIC" = true ]; then
    echo "=========================================="
    echo "MIMIC-III ICD-9 Classification"
    echo "=========================================="

    # Parse top-k values and calculate array indices
    IFS=',' read -ra TOPKS <<< "$TOPK_VALUES"
    NUM_TOPKS=${#TOPKS[@]}
    NUM_DATASETS=4
    TOTAL_TASKS=$((NUM_DATASETS * NUM_TOPKS))

    echo "Datasets: 4 (4% keyword, 4% keyword→rephrased, 6% keyword, 6% keyword→rephrased)"
    echo "Top-K values: ${TOPKS[*]}"
    echo "Total tasks: $TOTAL_TASKS ($NUM_DATASETS datasets × $NUM_TOPKS top-k values)"
    echo ""

    # Create a custom SLURM script for selected top-k values
    CUSTOM_SLURM="slurm/mimic_icd_eval_custom.slurm"

    # Copy the original and modify the array size and top-k selection
    cp slurm/mimic_icd_eval.slurm "$CUSTOM_SLURM"

    # Update array size
    sed -i "s/#SBATCH --array=0-15/#SBATCH --array=0-$((TOTAL_TASKS - 1))/" "$CUSTOM_SLURM"

    # Update top-k values in the script
    TOPK_ARRAY="TOPKS=(${TOPKS[*]})"
    sed -i "s/TOPKS=(20 50 100 400)/$TOPK_ARRAY/" "$CUSTOM_SLURM"

    # Update calculation to use NUM_TOPKS
    sed -i "s/TOPK_IDX=\$((SLURM_ARRAY_TASK_ID % 4))/TOPK_IDX=\$((SLURM_ARRAY_TASK_ID % $NUM_TOPKS))/" "$CUSTOM_SLURM"
    sed -i "s/DATASET_IDX=\$((SLURM_ARRAY_TASK_ID / 4))/DATASET_IDX=\$((SLURM_ARRAY_TASK_ID / $NUM_TOPKS))/" "$CUSTOM_SLURM"

    echo "Submitting job array ($TOTAL_TASKS tasks)..."
    echo ""

    # Submit with error handling
    if MIMIC_SUBMIT_OUTPUT=$(sbatch "$CUSTOM_SLURM" 2>&1); then
        MIMIC_JOB_ID=$(echo "$MIMIC_SUBMIT_OUTPUT" | awk '{print $4}')
        echo "✓ MIMIC ICD job submitted: $MIMIC_JOB_ID"
        echo "  Job array: 0-$((TOTAL_TASKS - 1)) ($TOTAL_TASKS tasks)"
        echo "  Custom script: $CUSTOM_SLURM"
    else
        echo "✗ MIMIC ICD job submission failed:"
        echo "$MIMIC_SUBMIT_OUTPUT"
        echo ""
        echo "Possible reasons:"
        echo "  - Job quota exceeded (current jobs: $CURRENT_JOBS)"
        echo "  - QOS limit reached"
        echo ""
        echo "Try reducing the number of top-k values with --topk"
        MIMIC_JOB_ID="FAILED"
    fi
    echo ""
fi

# =============================================================================
# AlpaCare Health Dataset Evaluation
# =============================================================================
if [ "$RUN_ALPACARE" = true ]; then
    echo "=========================================="
    echo "AlpaCare Health Dataset Evaluation"
    echo "=========================================="
    echo "Submitting 4 evaluation pipelines (8 jobs total)..."
    echo ""

    # Track successful submissions
    ALPACARE_SUCCESS=0
    ALPACARE_TOTAL=4

    # 1. DPO Keyword Replaced Only
    echo "[1/4] DPO Keyword Replaced Only..."
    if python thesis_metrics/cli/alpacare_eval.py \
        --model_id dpo_keyword \
        --downstream_ds_path data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced.parquet 2>&1; then
        echo "✓ DPO keyword jobs submitted"
        ALPACARE_SUCCESS=$((ALPACARE_SUCCESS + 1))
    else
        echo "✗ DPO keyword submission failed"
    fi
    echo ""

    # 2. DPO Keyword → Rephrased (Chained)
    echo "[2/4] DPO Keyword → Rephrased (Chained)..."
    if python thesis_metrics/cli/alpacare_eval.py \
        --model_id dpo_keyword_rephrased \
        --downstream_ds_path data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced_rephrased.parquet 2>&1; then
        echo "✓ DPO keyword→rephrased jobs submitted"
        ALPACARE_SUCCESS=$((ALPACARE_SUCCESS + 1))
    else
        echo "✗ DPO keyword→rephrased submission failed"
    fi
    echo ""

    # 3. DP-SFT Keyword Replaced Only
    echo "[3/4] DP-SFT Keyword Replaced Only..."
    if python thesis_metrics/cli/alpacare_eval.py \
        --model_id dpsft_keyword \
        --downstream_ds_path data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced.parquet 2>&1; then
        echo "✓ DP-SFT keyword jobs submitted"
        ALPACARE_SUCCESS=$((ALPACARE_SUCCESS + 1))
    else
        echo "✗ DP-SFT keyword submission failed"
    fi
    echo ""

    # 4. DP-SFT Keyword → Rephrased (Chained)
    echo "[4/4] DP-SFT Keyword → Rephrased (Chained)..."
    if python thesis_metrics/cli/alpacare_eval.py \
        --model_id dpsft_keyword_rephrased \
        --downstream_ds_path data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced_rephrased.parquet 2>&1; then
        echo "✓ DP-SFT keyword→rephrased jobs submitted"
        ALPACARE_SUCCESS=$((ALPACARE_SUCCESS + 1))
    else
        echo "✗ DP-SFT keyword→rephrased submission failed"
    fi
    echo ""

    echo "AlpaCare submissions: $ALPACARE_SUCCESS/$ALPACARE_TOTAL successful"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Evaluation Submission Summary"
echo "=========================================="
echo ""

if [ "$RUN_MIMIC" = true ]; then
    if [ "$MIMIC_JOB_ID" = "FAILED" ]; then
        echo "MIMIC ICD: ✗ FAILED"
    else
        echo "MIMIC ICD: ✓ Job $MIMIC_JOB_ID"
    fi
fi

if [ "$RUN_ALPACARE" = true ]; then
    echo "AlpaCare: $ALPACARE_SUCCESS/$ALPACARE_TOTAL pipelines submitted"
fi

echo ""
echo "Monitor jobs:"
echo "  squeue -u $USER"
echo ""
echo "View logs:"
echo "  ls -lt log/ | head -20"
echo ""
echo "Results tracked in Weights & Biases:"
echo "  - MIMIC: Project 'style-transfer-icd-seed'"
echo "  - AlpaCare: Project 'synth-kg'"
echo ""
echo "=========================================="
