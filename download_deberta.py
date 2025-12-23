#!/usr/bin/env python3
"""
Download DeBERTa model and evaluation metrics for offline use on Jean Zay compute nodes.
Run this script on the login node where internet access is available.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch

def download_model():
    model_name = "microsoft/deberta-v3-base"

    print(f"Downloading {model_name}...")
    print("This will cache the model and metrics for offline use on compute nodes.\n")

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer downloaded and cached")

    # Download model
    print("\nDownloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=20,  # Match your top_k_labels configuration
        problem_type="multi_label_classification"
    )
    print(f"✓ Model downloaded and cached")

    # Download evaluation metrics
    print("\nDownloading evaluation metrics...")
    metrics_to_download = ["accuracy", "f1", "precision", "recall"]
    for metric_name in metrics_to_download:
        print(f"  - Downloading {metric_name}...")
        metric = evaluate.load(metric_name)
        print(f"    ✓ {metric_name} cached")

    # Show cache location
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    print(f"\n✓ All resources cached in: {cache_dir}")
    print("\nYou can now run your SLURM jobs - they will use the cached resources.")

if __name__ == "__main__":
    download_model()
