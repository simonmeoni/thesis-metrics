#!/bin/bash
# =============================================================================
# Jean Zay Setup Script
# =============================================================================
# Purpose: Install thesis-metrics package and dependencies in synth-kg environment
# Usage:
#   1. Sync code: ./sync.sh
#   2. SSH to Jean Zay: lunette
#   3. Run this script: cd $WORK/thesis-metrics && bash setup_jeanzay.sh
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Jean Zay Setup: thesis-metrics"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "User: $USER"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
echo "Activating synth-kg environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate synth-kg
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Display Python info
echo "Python environment:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# Install package in editable mode
echo "Installing thesis-metrics package in editable mode..."
pip install -e .
echo "✓ Package installed"
echo ""

# Verify installation
echo "Verifying installation..."
if python -c "import thesis_metrics" 2>/dev/null; then
    echo "✓ thesis_metrics module can be imported"
    echo ""
    echo "Available commands:"
    python -c "from thesis_metrics import __version__; print(f'  Version: {__version__ if hasattr(__import__(\"thesis_metrics\"), \"__version__\") else \"0.1.0\"}')" || echo "  Version: 0.1.0"
    echo "  - privacy-metrics"
    echo "  - rephrase"
    echo "  - alpacare-eval"
    echo "  - downstream-eval"
else
    echo "✗ Failed to import thesis_metrics"
    exit 1
fi
echo ""

# Install optional dependencies if needed
read -p "Install optional AlpaCare dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing AlpaCare dependencies..."
    pip install -e ".[alpacare]"
    echo "✓ AlpaCare dependencies installed"
fi
echo ""

read -p "Install optional downstream evaluation dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing downstream evaluation dependencies..."
    pip install -e ".[downstream]"
    echo "✓ Downstream evaluation dependencies installed"
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p log
mkdir -p outputs
mkdir -p data
echo "✓ Directories created"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluations: ./run_all_evaluations.sh"
echo "  2. Monitor jobs: squeue -u \$USER"
echo "  3. Check logs: ls -lt log/ | head"
echo ""
