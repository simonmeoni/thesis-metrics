# AlpaCare Data Utility Evaluation - Commands

**Date**: 2025-11-28
**Purpose**: Evaluate data utility of keyword-replaced AlpaCare datasets using AlpaCare baseline comparison
**Datasets**: DPO and DP-SFT with keyword replacement and chained rephrasing

## Overview

Testing the downstream task performance (data utility) of rephrased AlpaCare datasets by running the full AlpaCare evaluation pipeline:
1. AlpaCare baseline generation
2. SFT training on baseline outputs
3. Model generation from fine-tuned model
4. Preference evaluation against reference models (GPT-3.5, GPT-4, Claude-2, text-davinci-003)

## Datasets to Evaluate

We are testing **4 keyword-based variants** to compare:
- Keyword replacement alone vs. keyword + standard rephrasing (chained)
- Across both DPO and DP-SFT training methods

| Model ID | Dataset | Processing Method |
|----------|---------|-------------------|
| `dpo_keyword` | `model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced.parquet` | Keyword replacement only |
| `dpo_keyword_rephrased` | `model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced_rephrased.parquet` | Keyword → Standard rephrasing |
| `dpsft_keyword` | `model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced.parquet` | Keyword replacement only |
| `dpsft_keyword_rephrased` | `model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced_rephrased.parquet` | Keyword → Standard rephrasing |

## Commands to Run on Jean Zay

### 1. DPO Keyword Replaced Only

```bash
uv run alpacare-eval \
    --model_id dpo_keyword \
    --downstream_ds_path data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced.parquet \
    --output_path outputs/alpacare_eval/dpo_keyword
```

### 2. DPO Keyword → Rephrased (Chained)

```bash
uv run alpacare-eval \
    --model_id dpo_keyword_rephrased \
    --downstream_ds_path data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_keyword_replaced_rephrased.parquet \
    --output_path outputs/alpacare_eval/dpo_keyword_rephrased
```

### 3. DP-SFT Keyword Replaced Only

```bash
uv run alpacare-eval \
    --model_id dpsft_keyword \
    --downstream_ds_path data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced.parquet \
    --output_path outputs/alpacare_eval/dpsft_keyword
```

### 4. DP-SFT Keyword → Rephrased (Chained)

```bash
uv run alpacare-eval \
    --model_id dpsft_keyword_rephrased \
    --downstream_ds_path data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_keyword_replaced_rephrased.parquet \
    --output_path outputs/alpacare_eval/dpsft_keyword_rephrased
```

## Job Details

Each command submits **2 SLURM jobs**:
1. **health-evaluation.slurm**: AlpaCare generation + SFT training + model generation (GPU required, ~2 hours)
2. **health-preference-eval.slurm**: Preference evaluation (CPU only, ~1 hour)

**Total**: 4 commands × 2 jobs = 8 SLURM jobs

### Resource Requirements per Job

**Health Evaluation:**
- Partition: `gpu_p6`
- GPU: 1x H100
- CPUs: 40
- Time: 2 hours
- QoS: `qos_gpu_h100-dev`

**Preference Evaluation:**
- Partition: `compil`
- CPUs: 10
- Time: 1 hour

## Expected Outputs

For each variant, outputs will be saved to `outputs/alpacare_eval/{model_id}/`:

```
outputs/alpacare_eval/dpo_keyword/
├── evaluation_alpacare_sft.parquet   # AlpaCare baseline responses
├── evaluation.parquet                 # Fine-tuned model responses
└── gpt_results/                       # Pairwise comparison results
    ├── {reference_model}_reference_first.jsonl
    └── {reference_model}_reference_last.jsonl
```

## Results Tracking

Results will be logged to **Weights & Biases**:
- **Project**: synth-kg
- **Run names**:
  - `eval_dpo_keyword_eval`
  - `eval_dpo_keyword_rephrased_eval`
  - `eval_dpsft_keyword_eval`
  - `eval_dpsft_keyword_rephrased_eval`
- **Metrics**: Preference scores against each reference model (0.0 to 1.0)

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View logs
ls -lt log/slurm-*.out | head

# Follow specific job
tail -f log/slurm-JOBID.out
```

## Analysis Questions

After completion, compare:
1. **Keyword alone vs. chained**: Does adding standard rephrasing after keyword replacement improve or degrade utility?
2. **DPO vs. DP-SFT**: Does the training method affect how rephrasing impacts utility?
3. **Privacy-utility tradeoff**: Cross-reference with privacy metrics to understand tradeoff

## Notes

- Model IDs use **underscores** (not hyphens) to avoid path construction issues in SLURM scripts
- The `alpacare-eval` CLI automatically handles job submission and dependency management
- Jobs run sequentially: preference evaluation waits for health evaluation to complete
- Reference model outputs are pre-computed in `alpacare-evaluation/reference_outputs/`

## Next Steps

1. Submit all 4 commands on Jean Zay
2. Monitor job completion via `squeue` and W&B
3. Analyze preference scores across variants
4. Document findings in comparison report
5. Update DATASET_CATALOG.md with results
