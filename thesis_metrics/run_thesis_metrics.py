from measure import semantic
import wandb
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("Starting thesis metrics analysis")
    wandb.init(
        project="thesis-metrics",
    )
    logging.info("Starting AlpaCare analysis")
    run_alpacare_analysis()
    logging.info("Starting MIMIC-III analysis")
    run_mimic_iii_analysis()
    logging.info("Analysis complete, finishing WandB run")
    wandb.finish()


def clean_data(predictions, references):
    """Clean data by removing examples with no tokens or NaN values."""
    logging.info(f"Original dataset size: {len(predictions)} examples")

    clean_predictions = []
    clean_references = []

    for pred, ref in zip(predictions, references):
        # Skip if either prediction or reference is NaN or None
        if pd.isna(pred) or pd.isna(ref) or pred is None or ref is None:
            continue

        # Skip if either is empty string or contains only whitespace
        if not str(pred).strip() or not str(ref).strip():
            continue

        clean_predictions.append(pred)
        clean_references.append(ref)

    removed_count = len(predictions) - len(clean_predictions)
    logging.info(f"Removed {removed_count} examples with NaN/empty values")
    logging.info(f"Clean dataset size: {len(clean_predictions)} examples")

    return clean_predictions, clean_references


def run_alpacare_analysis():
    logging.info("Processing AlpaCare dataset")
    runs = {
        "dp-sft": "./data/alpacare/model=fc850cd_size=60000_step=dp-sft_sort=mes.parquet",
        "dpo-1": "./data/alpacare/model=718c609_size=60000_step=dpo-1_sort=mes.parquet",
    }
    ground_truth = "./data/alpacare/private.parquet"

    for step in runs.keys():
        logging.info(f"Processing step: {step}")
        run = runs[step]
        df = pd.read_parquet(run)
        # df = pd.merge(df, pd.read_parquet(ground_truth), on="instruction")
        gdf = pd.read_parquet(ground_truth)
        references = gdf["response"].tolist()[:100]
        results_mean = {}
        predictions = (
            df["chosen"].tolist()[:100]
            if step != "dp-sft"
            else df["response"].tolist()[:100]
        )

        # Clean the data
        clean_predictions, clean_references = clean_data(predictions, references)

        if len(clean_predictions) == 0:
            logging.warning(f"No valid examples found for {step}, skipping")
            continue

        logging.info(f"Running metrics calculation for {step}")
        # translation_results = semantic.translation_metrics(
        #     predictions=clean_predictions, references=clean_references
        # )
        semantic_results = semantic.similarity_metrics(
            predictions=clean_predictions, references=clean_references
        )
        # for metric_name, score in (translation_results | semantic_results).items():
        for metric_name, score in semantic_results.items():
            if metric_name not in results_mean:
                results_mean[metric_name] = []
            results_mean[metric_name].append(score)
        logging.info(f"Results for {step}:")
        for metric_name, scores in results_mean.items():
            mean_score = sum(scores) / len(scores)
            logging.info(f"{metric_name}: {mean_score:.4f}")
        semantic_results = {
            f"alpacare/{step}/{metric}": score
            for metric, score in semantic_results.items()
        }
        # translation_results = {
        #     f"alpacare/{step}/{metric}": score
        #     for metric, score in translation_results.items()
        # }
        all_results = {
            **semantic_results,
            # **translation_results,
        }
        logging.info(f"Logging metrics to WandB for {step}")
        wandb.log(all_results)


def run_mimic_iii_analysis():
    logging.info("Processing MIMIC-III dataset")
    data_files = [
        "./data/mimic-iii/4%-dpo_1-mh67il8t.parquet",
        "./data/mimic-iii/4%-dpo_2-yv5v4516.parquet",
        "./data/mimic-iii/4%-sft-mru97w7c.parquet",
        "./data/mimic-iii/6%-dpo_1-tg4jsyrw.parquet",
        "./data/mimic-iii/6%-dpo_2-8l6avevp.parquet",
        "./data/mimic-iii/6%-sft-ofzh3aqu.parquet",
    ]
    logging.info(f"Found {len(data_files)} MIMIC-III files to process")

    for data in data_files:
        logging.info(f"Processing file: {data}")
        generated_df = pd.read_parquet(data)
        references = generated_df["ground_texts"].tolist()
        predictions = generated_df["generation_1"].tolist()

        # Clean the data
        clean_predictions, clean_references = clean_data(predictions, references)

        if len(clean_predictions) == 0:
            logging.warning(f"No valid examples found in {data}, skipping")
            continue

        logging.info(f"Running metrics calculation for {data}")
        translation_results = semantic.translation_metrics(
            predictions=clean_predictions, references=clean_references
        )
        semantic_results = semantic.similarity_metrics(
            predictions=clean_predictions, references=clean_references
        )
        semantic_results = {
            f"mimic-iii/{data.split('.parquet')[0].split('/')[-1]}/{metric}": score
            for metric, score in semantic_results.items()
        }
        translation_results = {
            f"mimic-iii/{data.split('.parquet')[0].split('/')[-1]}/{metric}": score
            for metric, score in translation_results.items()
        }
        all_results = {**semantic_results, **translation_results}
        logging.info(f"Logging metrics to WandB for {data}")
        wandb.log(all_results)


if __name__ == "__main__":
    main()
