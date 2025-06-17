import pandas as pd
import json
from pathlib import Path


def load_all_jsonl_files():
    """Load and concatenate all JSONL files from with-proxy-tokens folder"""
    jsonl_dir = Path("pre-processing/with-proxy-tokens")
    all_data = {}

    for jsonl_file in jsonl_dir.glob("*.jsonl"):
        print(f"Loading {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # Each line contains patient_id as key with list of records
                for patient_id, records in data.items():
                    if patient_id not in all_data:
                        all_data[patient_id] = []
                    all_data[patient_id].extend(records)

    return all_data


def create_keywords_mapping(jsonl_data):
    """Create mapping from keywords to patient_id"""
    keywords_to_patient = {}

    for patient_id, records in jsonl_data.items():
        for record in records:
            keywords = record.get("keywords", "")
            if keywords:
                # Convert keywords string to list, split by comma and clean
                keywords_list = [
                    kw.strip().lower() for kw in keywords.split(",") if kw.strip()
                ]
                # Use a tuple as key since lists aren't hashable
                keywords_key = tuple(sorted(keywords_list))
                keywords_to_patient[keywords_key] = {
                    "patient_id": patient_id,
                    "keywords": keywords_list,
                }

    return keywords_to_patient


def extract_keywords_from_prompt(prompt):
    """Extract keywords from prompt by splitting on 'Keywords:'"""
    keywords_part = prompt.split("Keywords:")[-1][:-8].strip()
    # Convert to list, split by comma and clean
    keywords_list = [
        kw.strip().lower() for kw in keywords_part.split(",") if kw.strip()
    ]
    return keywords_list


def merge_parquet_with_jsonl(parquet_file, keywords_mapping):
    """Merge parquet file with JSONL data based on keywords matching"""
    print(f"Processing {parquet_file}")

    # Load parquet file
    df = pd.read_parquet(parquet_file)

    # Initialize new columns
    df["patient_id"] = None
    df["keywords"] = None

    matched_count = 0
    total_count = len(df)

    # Process each row
    for idx, row in df.iterrows():

        prompt = row["prompts"]
        extracted_keywords = extract_keywords_from_prompt(prompt)

        # Check if extracted keywords list is contained in any keywords_mapping list
        matched_key = None
        if extracted_keywords:
            for keywords_key, mapping_data in keywords_mapping.items():
                mapping_keywords = mapping_data["keywords"]
                # Check if all extracted keywords are contained in the mapping keywords
                if all(kw in mapping_keywords for kw in extracted_keywords):
                    matched_key = keywords_key
                    break

        if matched_key:
            df.at[idx, "patient_id"] = keywords_mapping[matched_key]["patient_id"]
            df.at[idx, "keywords"] = ", ".join(
                keywords_mapping[matched_key]["keywords"]
            )
            matched_count += 1

    # Remove eval_prompt columns
    eval_prompt_cols = [col for col in df.columns if col.startswith("eval_prompt_")]
    df = df.drop(columns=eval_prompt_cols)

    print(f"  Matched {matched_count}/{total_count} records with patient_id")
    return df


def main():
    # Load all JSONL data
    print("Loading JSONL files...")
    jsonl_data = load_all_jsonl_files()
    print(f"Loaded data for {len(jsonl_data)} patients")

    # Create keywords mapping
    print("Creating keywords mapping...")
    keywords_mapping = create_keywords_mapping(jsonl_data)
    print(f"Created mapping for {len(keywords_mapping)} keyword sets")

    # Process each parquet file
    generated_dir = Path("pre-proccessing/generated")
    for parquet_file in generated_dir.glob("*.parquet"):
        # Merge with JSONL data
        merged_df = merge_parquet_with_jsonl(parquet_file, keywords_mapping)

        # Save merged file
        output_file = f"{parquet_file.name}"
        merged_df.to_parquet(output_file, index=False)
        print(f"Saved merged file: {output_file}")

    print("Merging complete!")


if __name__ == "__main__":
    main()
