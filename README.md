# Thesis Metrics

Privacy metrics evaluation for synthetic medical data.

## Installation

Using `uv`:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv pip install -e .

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

## Usage

### Basic usage with default config (AlpaCare DPO dataset):

```bash
privacy-metrics
```

### Evaluate DP-SFT dataset:

```bash
privacy-metrics dataset=alpacare_dpsft
```

### Enable WandB logging:

```bash
privacy-metrics wandb.enabled=true wandb.project=my-project
```

### Customize attacks:

```bash
privacy-metrics attacks.enabled=[linkage_tfidf,proximity_tfidf]
```

### View all configuration options:

```bash
privacy-metrics --help
privacy-metrics --cfg job
```

## Configuration

All configuration is managed through Hydra. See `thesis_metrics/configs/` for available options:

- `config.yaml` - Main configuration
- `dataset/` - Dataset configurations
- `attacks/` - Privacy attack configurations

## Output

Results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:
- `results.csv` - Privacy metrics results
- `config.yaml` - Configuration used for the run
- `.hydra/` - Hydra runtime configuration

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run ruff for linting
ruff check thesis_metrics/

# Format code
ruff format thesis_metrics/
```