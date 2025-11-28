# MIMIC-III ICD Classification Utility Evaluation - Commands

**Date**: 2025-11-28
**Purpose**: Evaluate data utility of keyword-replaced MIMIC-III datasets (v1) using ICD-9 multi-label classification
**Task**: ICD-9 code prediction from clinical notes
**Top-K Values**: 20, 50, 100, 400 (most common ICD-9 codes)

## Overview

Testing the downstream task performance (data utility) of rephrased MIMIC-III datasets by training ICD-9 classification models. This measures whether the rephrased synthetic data preserves enough medical information for diagnostic coding tasks.

## Datasets to Evaluate

Testing **4 keyword-based variants (v1 only)**:

### 4% Epsilon (DPO-2)
| Variant | Dataset Path | Processing Method |
|---------|--------------|-------------------|
| Keyword | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced.parquet` | Keyword replacement only |
| Keyword → Rephrased | `data/mimic-iii/4%-dpo_2-yv5v4516_keyword_replaced_rephrased.parquet` | Keyword → Standard rephrasing |

### 6% Epsilon (DPO-2)
| Variant | Dataset Path | Processing Method |
|---------|--------------|-------------------|
| Keyword | `data/mimic-iii/6%-dpo_2-8l6avevp_keyword_replaced.parquet` | Keyword replacement only |
| Keyword → Rephrased | `data/mimic-iii/6%-dpo_2-8l6avevp_keyword_replaced_rephrased.parquet` | Keyword → Standard rephrasing |

## Job Configuration

**Total evaluations**: 16 (4 datasets × 4 top-k values)

**SLURM Script**: `slurm/mimic_icd_eval.slurm`

**Job Array**: 0-15
- Array tasks 0-3: 4% keyword (top-20, 50, 100, 400)
- Array tasks 4-7: 4% keyword→rephrased (top-20, 50, 100, 400)
- Array tasks 8-11: 6% keyword (top-20, 50, 100, 400)
- Array tasks 12-15: 6% keyword→rephrased (top-20, 50, 100, 400)

## Command to Submit on Jean Zay

```bash
# Navigate to project directory
cd $WORK/thesis-metrics

# Create log directory
mkdir -p log

# Submit the job array
sbatch slurm/mimic_icd_eval.slurm
```

This single command will submit all 16 evaluations as a job array!

## Resource Requirements

Per job (each array task):
- **Partition**: `gpu_p6`
- **GPU**: 1x H100
- **CPUs**: 16
- **Time**: 4 hours
- **QoS**: `qos_gpu_h100-dev`

## Expected Outputs

For each configuration, models will be saved to:
```
models/icd/{run_name}/
├── config.json
├── model.safetensors
├── training_args.bin
└── results/
    ├── eval_results.json
    └── predictions.npz
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View specific array task log
tail -f log/mimic_icd_JOBID_TASKID.out

# Example: View task 0 (4% keyword, top-20)
tail -f log/mimic_icd_12345_0.out

# View all logs
ls -lt log/mimic_icd_*.out | head
```

## Results Tracking

Results will be logged to **Weights & Biases**:
- **Project**: `style-transfer-icd-seed`
- **Metrics**:
  - `eval/f1_micro`
  - `eval/f1_macro`
  - `eval/precision`
  - `eval/recall`
  - `eval/accuracy`

## Array Task ID Mapping

| Array Task | Dataset | Top-K |
|------------|---------|-------|
| 0 | 4% keyword | 20 |
| 1 | 4% keyword | 50 |
| 2 | 4% keyword | 100 |
| 3 | 4% keyword | 400 |
| 4 | 4% keyword→rephrased | 20 |
| 5 | 4% keyword→rephrased | 50 |
| 6 | 4% keyword→rephrased | 100 |
| 7 | 4% keyword→rephrased | 400 |
| 8 | 6% keyword | 20 |
| 9 | 6% keyword | 50 |
| 10 | 6% keyword | 100 |
| 11 | 6% keyword | 400 |
| 12 | 6% keyword→rephrased | 20 |
| 13 | 6% keyword→rephrased | 50 |
| 14 | 6% keyword→rephrased | 100 |
| 15 | 6% keyword→rephrased | 400 |

## Analysis Questions

After completion, compare:
1. **Keyword vs. Chained**: Does adding standard rephrasing after keyword replacement improve or degrade utility?
2. **Across Top-K**: Does the effect differ for easier (top-20) vs harder (top-400) tasks?
3. **Epsilon Effect**: Does privacy budget (4% vs 6%) interact with rephrasing methods?
4. **Privacy-Utility Tradeoff**: Cross-reference with privacy metrics to understand tradeoff

## Next Steps

1. Submit the job: `sbatch slurm/mimic_icd_eval.slurm`
2. Monitor progress via `squeue` and logs
3. Collect F1 scores from W&B for each configuration
4. Create comparison visualization (bar charts)
5. Document findings in analysis report
6. Update DATASET_CATALOG.md with utility scores

## Notes

- Using `seed=42` for reproducibility
- DeBERTa-v3-base provides strong baseline for medical text
- Training takes ~1-2 hours per configuration
- Test data uses the filtered privacy dataset
- All jobs run in parallel (subject to cluster availability)
