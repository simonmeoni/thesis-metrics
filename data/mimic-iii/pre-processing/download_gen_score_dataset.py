import os
import json
import re
import wandb
import pandas as pd
from pathlib import Path


def parse_runs_file(file_path):
    """Parse runs.txt file to extract run information"""
    runs = []

    with open(file_path, "r") as f:
        content = f.read()

    # Extract ratio sections
    ratio_sections = re.findall(
        r"ratio (\d+)%:(.*?)(?=ratio \d+%:|$)", content, re.DOTALL
    )

    for ratio, section in ratio_sections:
        # Extract run information from each section
        sft_match = re.search(
            r"sft: clinical-dream-team/score-style-transfer/([a-zA-Z0-9]+)", section
        )
        dpo1_match = re.search(
            r"dpo-1: clinical-dream-team/score-style-transfer/([a-zA-Z0-9]+)", section
        )
        dpo2_match = re.search(
            r"dpo-2: clinical-dream-team/score-style-transfer/([a-zA-Z0-9]+)", section
        )

        if sft_match:
            runs.append(
                {
                    "ratio": f"{ratio}%",
                    "step": "sft",
                    "run_id": sft_match.group(1),
                    "full_path": f"clinical-dream-team/score-style-transfer/{sft_match.group(1)}",
                }
            )

        if dpo1_match:
            runs.append(
                {
                    "ratio": f"{ratio}%",
                    "step": "dpo-1",
                    "run_id": dpo1_match.group(1),
                    "full_path": f"clinical-dream-team/score-style-transfer/{dpo1_match.group(1)}",
                }
            )

        if dpo2_match:
            runs.append(
                {
                    "ratio": f"{ratio}%",
                    "step": "dpo-2",
                    "run_id": dpo2_match.group(1),
                    "full_path": f"clinical-dream-team/score-style-transfer/{dpo2_match.group(1)}",
                }
            )

    return runs


def download_gen_score_dataset(run_info, output_dir):
    """Download gen_score_dataset table from a wandb run"""
    try:
        # Initialize wandb API
        api = wandb.Api()

        # Get the run
        run = api.run(run_info["full_path"])

        # Create filename
        filename = f"{run_info['ratio']}-{run_info['step'].replace('-','_')}-{run_info['run_id']}.parquet"

        # Create generated directory if it doesn't exist
        generated_dir = Path(output_dir) / "generated"
        generated_dir.mkdir(exist_ok=True)

        filepath = generated_dir / filename

        # Check if gen_score_dataset table exists
        tables = [
            artifact
            for artifact in run.logged_artifacts()
            if "gen_score_dataset" in artifact.name
        ]

        if not tables:
            print(f"No gen_score_dataset table found for run {run_info['run_id']}")
            return False

        # Download the most recent gen_score_dataset table
        table_artifact = tables[-1]  # Get the latest one
        table_artifact.download(root=str(generated_dir))

        # Find the downloaded table file and rename it
        downloaded_files = list(generated_dir.glob("**/gen_score_dataset*"))
        if downloaded_files:
            # Move and rename the file
            downloaded_file = downloaded_files[0]

            with open(downloaded_file, "r") as f:
                wandb_table = json.load(f)

            # Extract columns and data from WandB table format
            columns = wandb_table.get("columns", [])
            data = wandb_table.get("data", [])

            # Create DataFrame from columns and data
            df = pd.DataFrame(data, columns=columns)

            df.to_parquet(filepath, index=False)
            downloaded_file.unlink()  # Remove original file

        print(f"Downloaded: {filename}")
        return True

    except Exception as e:
        print(f"Error downloading {run_info['run_id']}: {str(e)}")
        return False


def main():
    """Main function to orchestrate the download process"""
    runs_file = "runs.txt"
    output_dir = "."

    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found")
        return

    # Parse runs file
    runs = parse_runs_file(runs_file)
    print(f"Found {len(runs)} runs to process")

    # Download each run's gen_score_dataset
    successful_downloads = 0
    for run_info in runs:
        print(
            f"Processing {run_info['ratio']} {run_info['step']} {run_info['run_id']}..."
        )
        if download_gen_score_dataset(run_info, output_dir):
            successful_downloads += 1

    print(f"\nCompleted: {successful_downloads}/{len(runs)} downloads successful")


if __name__ == "__main__":
    main()
