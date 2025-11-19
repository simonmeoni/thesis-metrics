#!/usr/bin/env python3
"""
Script to create a clustered dataset from private_seed.parquet
"""

import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os

# Force CPU usage to avoid MPS segmentation fault on Apple Silicon
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"


def extract_keywords(instruction_text):
    """Extract keywords from instruction text by splitting on ###Keywords and then on \n\n -"""
    if "### Keywords:" in instruction_text:
        keywords = (
            instruction_text.split("### Keywords:")[1]
            .split("### Knowledge Base")[0]
            .strip()
            .split(", ")
        )
        return keywords
    return instruction_text


def main():
    parser = argparse.ArgumentParser(
        description="Create clustered dataset from private_seed.parquet"
    )
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=50,
        help="Number of clusters (default: 50)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=None,
        help="Size of dataset to use (default: all)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input parquet file (default: private_seed.parquet)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="clustered_dataset.parquet",
        help="Output parquet file (default: clustered_dataset.parquet)",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)

    if args.size is not None:
        df = df.head(args.size)
        print(f"Using first {args.size} rows")

    print(f"Dataset shape: {df.shape}")

    # Use response column for clustering
    response_texts = df["response"].tolist()

    print("Loading sentence transformer model...")
    model = SentenceTransformer("FremyCompany/BioLORD-2023-M")

    print("Encoding texts...")
    embeddings = []
    batch_size = 64

    for i in tqdm(range(0, len(response_texts), batch_size), desc="Encoding"):
        batch_texts = response_texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"Clustering into {args.clusters} clusters...")

    # Normalize embeddings for better clustering
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]

    # Force CPU usage for kmeans to avoid MPS issues
    kmeans = faiss.Kmeans(dimension, args.clusters, niter=20, verbose=True, gpu=False)
    embeddings = embeddings.astype(np.float32)
    kmeans.train(embeddings)

    # Get cluster assignments
    _, cluster_ids = kmeans.index.search(embeddings.astype(np.float32), 1)
    cluster_ids = cluster_ids.flatten()

    print("Processing instructions to extract keywords...")
    processed_instructions = []
    for instruction in tqdm(df["instruction"], desc="Processing instructions"):
        keywords = extract_keywords(instruction)
        processed_instructions.append(keywords)

    print("Creating output dataset...")
    output_df = pd.DataFrame(
        {
            "keyword": processed_instructions,
            "instruction": df["instruction"],
            "response": df["response"],
            "cluster_id": cluster_ids,
        }
    )

    print("Filtering clusters with less than 5 samples...")
    cluster_counts = output_df["cluster_id"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 3].index
    output_df = output_df[output_df["cluster_id"].isin(valid_clusters)]

    print(f"Filtered dataset shape: {output_df.shape}")
    print(f"Remaining clusters: {len(valid_clusters)}")

    print("Creating train/dev/test splits...")
    # Initialize split column
    output_df["split"] = ""

    # For each cluster, assign samples to splits ensuring each cluster appears in all splits
    for cluster_id in valid_clusters:
        cluster_mask = output_df["cluster_id"] == cluster_id
        cluster_indices = output_df[cluster_mask].index.tolist()

        # Shuffle indices for random assignment
        np.random.shuffle(cluster_indices)

        # Calculate split sizes (at least 1 sample per split)
        n_samples = len(cluster_indices)
        n_train = max(1, int(0.6 * n_samples))
        n_dev = max(1, int(0.2 * n_samples))
        n_test = max(1, n_samples - n_train - n_dev)

        # Adjust if we don't have enough samples
        if n_train + n_dev + n_test > n_samples:
            if n_samples >= 3:
                n_train = n_samples - 2
                n_dev = 1
                n_test = 1
            else:
                # For clusters with exactly 5 samples, distribute as evenly as possible
                n_train = n_samples - 2
                n_dev = 1
                n_test = 1

        # Assign splits
        output_df.loc[cluster_indices[:n_train], "split"] = "train"
        output_df.loc[cluster_indices[n_train : n_train + n_dev], "split"] = "dev"
        output_df.loc[cluster_indices[n_train + n_dev :], "split"] = "test"

    # Print split statistics
    split_counts = output_df["split"].value_counts()
    print(f"Split distribution: {dict(split_counts)}")

    # Verify each cluster appears in all splits
    cluster_split_check = output_df.groupby("cluster_id")["split"].nunique()
    clusters_in_all_splits = (cluster_split_check == 3).sum()
    print(
        f"Clusters appearing in all 3 splits: {clusters_in_all_splits}/{len(valid_clusters)}"
    )

    print(f"Saving to {args.output}...")
    output_df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
