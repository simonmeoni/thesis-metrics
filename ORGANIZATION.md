# Project Organization Guide

## Directory Structure

```
thesis-metrics/
├── data/                          # All dataset files
│   ├── mimic-iii/                # MIMIC-III datasets
│   │   ├── privacy_dataset_filtered.parquet         # Source private data
│   │   ├── 4%-dpo_2-yv5v4516.parquet               # Original synthetic
│   │   ├── 4%-dpo_2-yv5v4516_rephrased_*.parquet   # Standard rephrased
│   │   └── 4%-dpo_2-yv5v4516_keyword_replaced.parquet  # Keyword replaced
│   └── alpacare/                 # AlpaCare datasets
│       ├── alpacare_from_private_k-4000_full.parquet  # Source private data
│       ├── model=*_dpo-1_*.parquet                     # Original synthetic
│       ├── model=*_dpo-1_*_rephrased_*.parquet        # Standard rephrased
│       └── model=*_dpo-1_*_keyword_replaced.parquet   # Keyword replaced
│
├── configs/                       # Hydra configuration files
│   ├── config.yaml               # Main config
│   ├── dataset/                  # Dataset-specific configs
│   │   ├── mimic_4pct_dpo2.yaml                    # Original
│   │   ├── mimic_4pct_dpo2_rephrased.yaml         # Rephrased
│   │   └── mimic_4pct_dpo2_keyword_replaced.yaml  # Keyword replaced
│   └── rephrasing/               # Rephrasing method configs
│       ├── standard.yaml
│       └── keyword_replace.yaml
│
├── outputs/                       # Evaluation outputs
│   └── YYYY-MM-DD/               # Organized by date
│       └── HH-MM-SS/             # Then by time
│           ├── results.csv       # Evaluation metrics
│           └── config.yaml       # Run configuration
│
├── md/                           # Analysis reports
│   ├── YYYY-MM-DD_<topic>.md    # Date-prefixed reports
│   └── ...
│
├── thesis_metrics/               # Source code
│   ├── __init__.py
│   ├── cli.py                    # Main evaluation CLI
│   ├── rephrase_cli.py          # Rephrasing CLI
│   ├── evaluation.py            # Privacy evaluators
│   ├── rephrasing.py            # Rephrasing logic
│   ├── privacy_attacks.py       # Attack implementations
│   └── utils.py                 # Utilities
│
├── DATASET_CATALOG.md            # Master catalog of all datasets
├── ORGANIZATION.md               # This file
└── README.md                     # Project README
```

---

## File Naming Conventions

### 1. Dataset Files

**Pattern**: `<base_name>_<processing>_<model>.parquet`

**Examples**:
```
# Original (no suffix)
4%-dpo_2-yv5v4516.parquet

# Standard rephrased
4%-dpo_2-yv5v4516_rephrased_llama-3.3-70b-versatile.parquet

# Keyword replaced
4%-dpo_2-yv5v4516_keyword_replaced.parquet
```

### 2. Configuration Files

**Pattern**: `<dataset>_<variant>.yaml`

**Examples**:
```
mimic_4pct_dpo2.yaml                    # Original
mimic_4pct_dpo2_rephrased.yaml         # Standard rephrased
mimic_4pct_dpo2_keyword_replaced.yaml  # Keyword replaced
```

### 3. Reports

**Pattern**: `YYYY-MM-DD_<descriptive_name>.md`

**Examples**:
```
2025-11-21_keyword_replacement_vs_rephrasing_comparison.md
2025-11-20_mimic_rephrasing_analysis.md
2025-11-20_alpacare_rephrasing_analysis.md
```

---

## Workflow Guide

### 1. Finding Existing Datasets

Check `DATASET_CATALOG.md` for:
- All available datasets
- Processing history
- Evaluation results
- File paths

### 2. Running Evaluations

**For existing datasets**:
```bash
# Check configs/dataset/ for available dataset configs
ls configs/dataset/

# Run evaluation
uv run privacy-metrics dataset=mimic_4pct_dpo2_keyword_replaced
```

**Results location**: `outputs/YYYY-MM-DD/HH-MM-SS/results.csv`

### 3. Creating New Enhanced Datasets

**Standard rephrasing**:
```bash
uv run rephrase \
  rephrasing=standard \
  rephrasing.input_path="data/mimic-iii/DATASET.parquet" \
  rephrasing.output_path="data/mimic-iii/DATASET_rephrased_MODEL.parquet"
```

**Keyword replacement**:
```bash
uv run rephrase \
  rephrasing=keyword_replace \
  rephrasing.input_path="data/mimic-iii/DATASET.parquet" \
  rephrasing.output_path="data/mimic-iii/DATASET_keyword_replaced.parquet"
```

### 4. Tracking New Datasets

After creating a new dataset or running an evaluation:

1. Update `DATASET_CATALOG.md` with:
   - File path
   - Processing method
   - Evaluation results
   - Link to report

2. Create dataset config in `configs/dataset/` if evaluating

3. Write analysis report in `md/YYYY-MM-DD_<topic>.md`

---

## Quick Reference Commands

### List All Datasets
```bash
# Original datasets
ls data/mimic-iii/*.parquet | grep -v "rephrased\|keyword"
ls data/alpacare/*.parquet | grep -v "rephrased\|keyword"

# Rephrased datasets
ls data/**/*rephrased*.parquet

# Keyword-replaced datasets
ls data/**/*keyword_replaced*.parquet
```

### List All Configs
```bash
ls configs/dataset/*.yaml
```

### List Recent Evaluations
```bash
ls -lt outputs/ | head -10
```

### Find Evaluation by Dataset
```bash
grep -r "mimic_4pct_dpo2_keyword_replaced" outputs/
```

---

## Dataset Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Original Synthetic Data                   │
│              (Generated with DP-SFT or DPO)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────┴────────┐
                │                 │
                v                 v
    ┌──────────────────┐  ┌──────────────────┐
    │ Standard         │  │ Keyword          │
    │ Rephrasing       │  │ Replacement      │
    │                  │  │                  │
    │ Paraphrase full  │  │ Replace specific │
    │ text             │  │ keywords         │
    └────────┬─────────┘  └────────┬─────────┘
             │                     │
             v                     v
    ┌──────────────────┐  ┌──────────────────┐
    │ Privacy          │  │ Privacy          │
    │ Evaluation       │  │ Evaluation       │
    └────────┬─────────┘  └────────┬─────────┘
             │                     │
             v                     v
    ┌──────────────────┐  ┌──────────────────┐
    │ Results &        │  │ Results &        │
    │ Report           │  │ Report           │
    └──────────────────┘  └──────────────────┘
```

---

## Best Practices

### 1. Always Check DATASET_CATALOG.md First
Before creating new datasets or running evaluations, check if the work has already been done.

### 2. Use Descriptive Names
When creating new reports or configs, use clear, descriptive names that explain the content.

### 3. Update Catalog Immediately
After creating a dataset or running an evaluation, immediately update `DATASET_CATALOG.md`.

### 4. Keep Reports in md/
All analysis reports go in `md/` with date prefix `YYYY-MM-DD_`.

### 5. Version Control Everything Except Data
Git tracks:
- Code (thesis_metrics/)
- Configs (configs/)
- Reports (md/)
- Catalogs (DATASET_CATALOG.md)

Git ignores:
- Data files (data/**/*.parquet)
- Evaluation outputs (outputs/)

---

## Current Dataset Summary

| Base Dataset | Versions Available | Best Privacy |
|--------------|-------------------|--------------|
| **MIMIC 4% DPO-2** | Original, Rephrased, Keyword Replaced | 28.91% linkage (keyword) |
| **MIMIC 6% DPO-2** | Original, Rephrased, Keyword Replaced | 30.68% linkage (keyword) |
| **AlpaCare DPO-1** | Original, Rephrased, Keyword Replaced | 14.41% linkage (keyword) |
| **AlpaCare DP-SFT** | Original, Rephrased, Keyword Replaced | 5.28% linkage (keyword) ⭐ |

**Legend**:
- ⭐ = Best overall privacy protection

---

## Troubleshooting

### "Can't find dataset config"
- Check `configs/dataset/` for available configs
- Verify the config name matches the dataset parameter
- Create new config if needed

### "Results file not found"
- Check `outputs/YYYY-MM-DD/` for recent runs
- Verify evaluation completed successfully
- Check Hydra output directory setting

### "Lost track of which dataset is which"
- Consult `DATASET_CATALOG.md`
- Check file modification dates: `ls -lt data/mimic-iii/`
- Read dataset metadata in config files

---

**Last Updated**: 2025-11-21
