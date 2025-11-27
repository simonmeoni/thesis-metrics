# Downstream Tasks (dstream_tasks)

This folder contains the ICD and NER downstream task experiments that can be run independently from the main style-transfer project.

## Project Structure

```
thesis-metrics/
├── dstream_tasks/          # Downstream task implementations
│   ├── icd/               # ICD classification task
│   │   ├── train.py       # Training script for ICD
│   │   └── bs_train.py    # Baseline training
│   ├── ner/               # Named Entity Recognition task
│   │   └── train.py       # Training script for NER
│   └── baseline/          # Baseline models
├── configs/               # Configuration files
│   └── ds_stream/         # Downstream task configs
│       ├── icd/           # ICD configurations
│       ├── ner/           # NER configurations
│       └── training_args.yaml
├── hf_datasets/           # Data directory (you need to add your datasets here)
│   ├── mimic_iii_icd/
│   │   └── data/
│   │       ├── train_ds.csv
│   │       ├── test_ds.csv
│   │       └── ICD9_descriptions
│   └── mimic_iii_ner/
│       └── data/
│           ├── ner-*-train.parquet
│           └── ner-gold-test.parquet
└── pyproject.toml         # Project dependencies
```

## Setup

1. Install dependencies:
```bash
cd ~/Code/thesis-metrics
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

2. Add your datasets:
   - Place ICD datasets in `hf_datasets/mimic_iii_icd/data/`
   - Place NER datasets in `hf_datasets/mimic_iii_ner/data/`

## Running Experiments

### ICD Classification

```bash
# Run with a specific config
python dstream_tasks/icd/train.py

# The default config is configs/ds_stream/20.yaml
# You can override parameters:
python dstream_tasks/icd/train.py dataset.name=gold seed=42
```

Available ICD configs:
- `20.yaml` - Top 20 ICD codes
- `default.yaml` - Base configuration
- Check `configs/ds_stream/icd/` for more options

### NER (Named Entity Recognition)

```bash
# Run with default config
python dstream_tasks/ner/train.py

# The default config is configs/ds_stream/ner_train.yaml
# Override parameters:
python dstream_tasks/ner/train.py dataset.name=0.06-2-ofzh3aqu seed=42
```

Available NER configs:
- `gold.yaml` - Gold standard data
- `gen-0.yaml`, `gen-1.yaml`, `gen-2.yaml` - Generated data variants
- Check `configs/ds_stream/ner/` for more options

## Configuration

The experiments use Hydra for configuration management. Key configuration parameters:

### ICD Task
- `model`: Model to use (default: microsoft/deberta-v3-base)
- `dataset.name`: Dataset variant to use
- `dataset.topk`: Number of top ICD codes to predict
- `dataset.percentile`: Percentile threshold for filtering
- `dataset.random_sampling`: Whether to use random sampling
- `seed`: Random seed for reproducibility
- `wandb_project`: W&B project name

### NER Task
- `model`: Model to use (default: microsoft/deberta-v3-base)
- `dataset.name`: Dataset variant to use
- `dataset.percentile`: Percentile threshold for filtering
- `dataset.random_sampling`: Whether to use random sampling
- `seed`: Random seed for reproducibility
- `wandb_project`: W&B project name

## Using Different Datasets

To use your own datasets:

1. **For ICD**: Place your data in `hf_datasets/mimic_iii_icd/data/` with these files:
   - `train_ds.csv`: Training data with columns for text and ICD labels
   - `test_ds.csv`: Test data
   - `ICD9_descriptions`: ICD code descriptions (tab-separated)

2. **For NER**: Place your data in `hf_datasets/mimic_iii_ner/data/` as parquet files:
   - `ner-{dataset_name}-train.parquet`: Training data
   - `ner-gold-test.parquet`: Test data

3. Update the config files in `configs/ds_stream/` to reference your dataset names

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- transformers
- datasets
- torch
- hydra-core
- wandb
- evaluate
- nltk
- seqeval (for NER)

## Notes

- The code expects data paths relative to the project root
- Training outputs are saved to `models/icd/` or `models/ner/`
- Experiment tracking uses Weights & Biases (wandb)
- Make sure to set up your wandb credentials before running
