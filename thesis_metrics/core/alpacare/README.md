# AlpaCare Evaluation Module

This module contains the AlpaCare evaluation pipeline, integrated from `alpacare-evaluation/` into the main thesis-metrics codebase.

## Overview

The AlpaCare evaluation pipeline evaluates synthetic health datasets using:
1. **AlpaCare Baseline Generation**: Generate responses using AlpaCare-llama2-13b
2. **SFT Training**: Fine-tune a model on AlpaCare outputs
3. **Model Generation**: Generate responses from the fine-tuned model
4. **Preference Evaluation**: Compare against reference models (GPT-3.5, GPT-4, Claude-2, text-davinci-003)

## Module Structure

```
thesis_metrics/core/alpacare/
├── __init__.py
├── alpacare_generate.py      # Generate AlpaCare baseline responses
├── model_generate.py          # Generate responses from fine-tuned model
├── sft_train.py               # Supervised fine-tuning on AlpaCare outputs
├── gpt_eval.py                # GPT-based pairwise evaluation
├── preference_eval.py         # Calculate preference scores
├── alpaca_eval_chat_gpt.txt   # Evaluation prompt template
└── configs/
    ├── default.yaml           # Default training configuration
    └── health.yaml            # Health domain-specific config
```

## Installation

The AlpaCare evaluation requires additional dependencies. Install them with:

```bash
uv pip install -e ".[alpacare]"
```

This installs:
- PyTorch and Transformers (model inference)
- TRL and PEFT (training)
- vLLM (efficient inference)
- OpenAI API (evaluation)

## Usage

### Via CLI

The main entry point is the `alpacare-eval` command:

```bash
uv run alpacare-eval \
    --model_id my_model_v1 \
    --downstream_ds_path /path/to/scored_eval.parquet \
    --output_path /path/to/outputs
```

**Full options:**
```bash
uv run alpacare-eval \
    --model_id my_model_v1 \
    --downstream_ds_path /path/to/scored_eval.parquet \
    --output_path /path/to/outputs \
    --group_id exp001 \
    --step dpo-1 \
    --size 60000 \
    --suffix_run_name custom_suffix \
    --sts_model FremyCompany/BioLORD-2023 \
    --slurm_dir /path/to/slurm/scripts
```

### Required Parameters

- `--model_id`: Model identifier for tracking and naming
- `--downstream_ds_path`: Path to scored evaluation dataset (parquet)
- `--output_path`: Base path for saving outputs

### Optional Parameters

- `--group_id`: Group ID for tracking experiments (default: auto-generated)
- `--step`: Training step identifier (default: 'eval')
- `--size`: Dataset size (default: 60000)
- `--suffix_run_name`: Suffix for run name (default: step value)
- `--sts_model`: STS model for scoring (default: FremyCompany/BioLORD-2023)
- `--slurm_dir`: Path to SLURM scripts (default: alpacare-evaluation/slurm/)

## SLURM Scripts

The CLI submits SLURM jobs using scripts in `alpacare-evaluation/slurm/`:

1. **health-evaluation.slurm**: AlpaCare gen → SFT training → Model gen
   - Resources: 1x H100 GPU, 40 CPUs, 2 hours

2. **health-preference-eval.slurm**: Preference evaluation
   - Resources: 10 CPUs, 1 hour (no GPU)

These scripts remain in the original `alpacare-evaluation/` directory to maintain compatibility with existing workflows.

## Reference Data

Pre-computed reference model outputs remain in:
```
alpacare-evaluation/reference_outputs/
├── claude-2/iCliniq_output.jsonl
├── gpt-3.5-turbo/iCliniq_output.jsonl
├── gpt-4/iCliniq_output.jsonl
└── text-davinci-003/iCliniq_output.jsonl
```

## Output Structure

```
{output_path}/
├── evaluation_alpacare_sft.parquet   # AlpaCare baseline responses
├── evaluation.parquet                 # Fine-tuned model responses
└── gpt_results/                       # Pairwise comparison results
    ├── {model}_gpt-3.5-turbo_..._reference_first.jsonl
    └── {model}_gpt-3.5-turbo_..._reference_last.jsonl
```

## Integration Notes

### Changes from Original

1. **CLI Integration**: New CLI entry point at `thesis_metrics.cli.alpacare_eval`
2. **SLURM Utilities**: Job management extracted to `thesis_metrics.utils.slurm.SlurmJobManager`
3. **Module Organization**: Evaluation scripts moved to `thesis_metrics.core.alpacare/`
4. **Dependency Management**: AlpaCare dependencies are optional (`pip install -e ".[alpacare]"`)

### Original Location

The original standalone pipeline remains at `alpacare-evaluation/` for reference and compatibility.

### Migration Path

To fully migrate from the standalone pipeline:
1. Use `alpacare-eval` CLI instead of `run_evaluation.py`
2. SLURM scripts remain in `alpacare-evaluation/slurm/`
3. Reference outputs remain in `alpacare-evaluation/reference_outputs/`

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
ls -lt log/slurm-*.out | head
tail -f log/slurm-JOBID.out
```

## Weights & Biases

Results are logged to W&B:
- **Project**: synth-kg
- **Run name**: `eval_{model_id}_{suffix_run_name}`
- **Group**: Your specified group_id
- **Metrics**: Preference scores for each reference model

## See Also

- Original implementation: `alpacare-evaluation/README.md`
- Privacy metrics: `thesis_metrics/core/evaluation.py`
- Rephrasing: `thesis_metrics/core/rephrasing.py`
