import wandb
import measure
import pandas as pd


def main():
    wandb.init(
        project="thesis-metrics",
    )
    run_alpacare_analysis()
    run_mimic_iii_analysis()
    wandb.finish()


def run_alpacare_analysis():
    runs = {
        "dp-sft": "./data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft.parquet",
        "dpo-1": "./data/alpacare/model=0fe1620_size=60000_step=dpo-1.parquet",
    }
    ground_truth = "./data/alpacare/private.parquet"

    for step in runs.keys():
        run = runs[step]
        references = pd.read_parquet(ground_truth)["response"].tolist()
        generated_df = pd.read_parquet(run)
        predictions = generated_df["response"].tolist()
        print(f"Running analysis for {step} with run {run}")
        translation_results = measure.translation_metrics(
            predictions=predictions, references=references
        )
        semantic_results = measure.semantic_metrics(
            predictions=predictions, references=references
        )


def run_mimic_iii_analysis():
    pass


if __name__ == "__main__":
    main()
