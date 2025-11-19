import pandas as pd
import numpy as np
from pathlib import Path


def filter_by_patient_id_count(df, min_samples=3):
    """
    Filter the dataset to only include patient_ids with at least min_samples occurrences.
    """
    patient_counts = df["patient_id"].value_counts()
    valid_patients = patient_counts[patient_counts >= min_samples].index
    filtered_df = df[df["patient_id"].isin(valid_patients)]

    print(
        f"Original dataset: {len(df)} samples, {df['patient_id'].nunique()} unique patients"
    )
    print(
        f"Filtered dataset: {len(filtered_df)} samples, {filtered_df['patient_id'].nunique()} unique patients"
    )
    print(
        f"Removed {len(df) - len(filtered_df)} samples from {df['patient_id'].nunique() - filtered_df['patient_id'].nunique()} patients"
    )

    return filtered_df


def create_patient_aware_split(df, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2):
    """
    Create train/dev/test splits ensuring each patient_id appears in all splits.
    For each patient, distribute their samples across splits according to the ratios.
    """
    df_split = df.copy()
    df_split["split"] = ""

    for patient_id in df["patient_id"].unique():
        patient_data = df[df["patient_id"] == patient_id]
        n_samples = len(patient_data)

        # Calculate split sizes for this patient
        n_train = max(1, int(n_samples * train_ratio))
        n_dev = max(1, int(n_samples * dev_ratio))
        n_test = max(1, n_samples - n_train - n_dev)

        # Adjust if total exceeds available samples
        total = n_train + n_dev + n_test
        if total > n_samples:
            # Reduce the largest split first
            if n_train >= n_dev and n_train >= n_test:
                n_train = n_samples - n_dev - n_test
            elif n_dev >= n_test:
                n_dev = n_samples - n_train - n_test
            else:
                n_test = n_samples - n_train - n_dev

        # Get indices for this patient
        patient_indices = patient_data.index.tolist()
        np.random.shuffle(patient_indices)

        # Assign splits
        train_indices = patient_indices[:n_train]
        dev_indices = patient_indices[n_train : n_train + n_dev]
        test_indices = patient_indices[n_train + n_dev :]

        df_split.loc[train_indices, "split"] = "train"
        df_split.loc[dev_indices, "split"] = "dev"
        df_split.loc[test_indices, "split"] = "test"

    # Print split statistics
    split_stats = df_split["split"].value_counts()
    print(f"\nSplit distribution:")
    for split_name in ["train", "dev", "test"]:
        count = split_stats.get(split_name, 0)
        percentage = (count / len(df_split)) * 100
        unique_patients = df_split[df_split["split"] == split_name][
            "patient_id"
        ].nunique()
        print(
            f"{split_name}: {count} samples ({percentage:.1f}%), {unique_patients} unique patients"
        )

    # Verify each patient appears in all splits
    for split_name in ["train", "dev", "test"]:
        patients_in_split = set(
            df_split[df_split["split"] == split_name]["patient_id"].unique()
        )
        all_patients = set(df_split["patient_id"].unique())
        missing_patients = all_patients - patients_in_split
        if missing_patients:
            print(
                f"WARNING: {len(missing_patients)} patients missing from {split_name} split"
            )
        else:
            print(f"✓ All patients present in {split_name} split")

    return df_split


def main():
    # Default values
    input_file = "./4%-sft-mru97w7c.parquet"
    output_file = "./4%-sft-mru97w7c-splitted.parquet"
    min_samples = 3
    train_ratio = 0.6
    dev_ratio = 0.1
    test_ratio = 0.3
    seed = 42

    # Set random seed
    np.random.seed(seed)

    # Load the dataset
    print(f"Loading dataset from: {input_file}")
    df = pd.read_parquet(input_file)

    # Apply patient_id filter
    filtered_df = filter_by_patient_id_count(df, min_samples)

    # Create train/dev/test splits
    print(
        f"\nCreating splits with ratios - train: {train_ratio}, dev: {dev_ratio}, test: {test_ratio}"
    )
    split_df = create_patient_aware_split(
        filtered_df, train_ratio, dev_ratio, test_ratio
    )

    # Save the dataset with splits
    output_path = Path(output_file)
    print(f"\nSaving dataset with splits to: {output_path}")
    split_df.to_parquet(output_path, index=False)

    print("Dataset creation completed successfully!")


if __name__ == "__main__":
    main()
