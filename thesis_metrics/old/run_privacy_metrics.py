import wandb
import pandas as pd
import logging
from thesis_metrics.privacy_attacks import random_leakage_attack, proximity_attack_tfidf, proximity_attack_random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



def main():
    logging.info("Starting thesis metrics analysis")
    wandb.init(
        project="thesis-metrics",
    )
    run_privacy_analysis()
    wandb.finish()


def run_privacy_analysis():
    logging.info("Processing AlpaCare dataset")
    runs = {
        "dpo-1_rephrased": {
            "rephrased": "data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes_rephrased_llama-3.3-70b-versatile.parquet",
            "original": "data/alpacare/model=0fe1620_size=60000_step=dpo-1_sort=mes.parquet"
        },
        "dp-sft_rephrased": {
            "rephrased": "data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes_rephrased_llama-3.3-70b-versatile.parquet",
            "original": "data/alpacare/model=ovs3z1ey_size=60000_step=dp-sft_sort=mes.parquet"
        },
    }

    def extract_keywords_from_instruction(instruction_text):
        """Extract keywords from instruction text by splitting on ###Keywords and then on \n\n -"""
        if "### Keywords:" in instruction_text:
            keywords = (
                instruction_text.split("### Keywords:")[1]
                .split("### Knowledge Base")[0]
                .strip()
            )
            return keywords
        return instruction_text

    result_dict = {}
    for run in runs.keys():
        logging.info(f"Processing step: {run}")
        result_dict = {}
        logging.info(f"Running metrics calculation for {run}")

        # Load rephrased synthetic dataset
        df_rephrased = pd.read_parquet(runs[run]["original"])
        print(df_rephrased)

        logging.info(f"Loaded rephrased dataset: {len(df_rephrased)} rows")

        # Load real private source dataset (the base dataset used to create model outputs)
        df_source = pd.read_parquet('data/alpacare/alpacare_k-4000_s-10000.parquet')
        logging.info(f"Loaded source dataset: {len(df_source)} rows")

        # Extract keywords from both datasets for matching
        # Source has keywords embedded in instruction text (same as model dataset)
        df_source['keywords_str'] = df_source['instruction'].apply(extract_keywords_from_instruction)
        # Model dataset also has keywords embedded in instruction text
        df_rephrased['keywords_str'] = df_rephrased['instruction'].apply(extract_keywords_from_instruction)

        # DEBUG: Check keyword extraction
        logging.info(f"Source keywords - Null count: {df_source['keywords_str'].isna().sum()}")
        logging.info(f"Rephrased keywords - Null count: {df_rephrased['keywords_str'].isna().sum()}")
        logging.info(f"Source keywords - Sample: {df_source['keywords_str'].head(3).tolist()}")
        logging.info(f"Rephrased keywords - Sample: {df_rephrased['keywords_str'].head(3).tolist()}")

        # Check for common keywords
        source_keywords = set(df_source['keywords_str'].dropna())
        rephrased_keywords = set(df_rephrased['keywords_str'].dropna())
        common_keywords = source_keywords & rephrased_keywords
        logging.info(f"Common keywords count: {len(common_keywords)}")
        if len(common_keywords) > 0:
            logging.info(f"Sample common keywords: {list(common_keywords)[:3]}")
        else:
            logging.warning("NO COMMON KEYWORDS FOUND!")
            logging.info(f"Sample source keywords: {list(source_keywords)[:3]}")
            logging.info(f"Sample rephrased keywords: {list(rephrased_keywords)[:3]}")

        # Merge on keywords to get real private data
        df = df_rephrased.merge(
            df_source[['keywords_str', 'response']],
            on='keywords_str',
            how='inner',
            suffixes=('_synthetic', '_private')
        )
        logging.info(f"Merged dataset: {len(df)} rows (joined on keywords)")

        # Factorize cluster_id to get numeric private_id
        df['private_id'], _ = pd.factorize(df['cluster_id'])
        df.sort_values(by=["split"], inplace=True, ascending=True)

        train_ds = df[df["split"] == "train"]
        dev_ds = df[df["split"] == "dev"]
        test_ds = df[df["split"] == "test"]
        concat_df = pd.concat([train_ds, dev_ds, test_ds])

        logging.info(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}, Test: {len(test_ds)}")
        # Prepare public df (rephrased synthetic text) and private df (real private text)
        public_df = pd.DataFrame({
            "text": df["chosen"].tolist(),  # rephrased synthetic text
            "private_id": df["private_id"].tolist(),
        })

        private_df = pd.concat([dev_ds, test_ds])
        private_df = pd.DataFrame({
            "text": private_df["response"].tolist(),  # real private patient text
            "private_id": private_df["private_id"].tolist(),
        })
        tfidf_result = measure.privacy.linkage_attack_tfidf(
            public_df=public_df,
            private_df=private_df,
            text_col="text",
            id_col="private_id",
        )
        random_result = random_leakage_attack(
            public_df=public_df,
            private_df=private_df,
            id_col="private_id",
        )
        logging.info(f"Random leakage attack results for {run}: {random_result}")
        proximity_result = proximity_attack_tfidf(
            synthetic_df=public_df,
            private_df=private_df,
            text_col="text",
            id_col="private_id",
        )
        logging.info(f"Proximity attack TF-IDF results for {run}: {proximity_result}")
        # proximity_embeddings_result = proximity_attack_embeddings(
        #     synthetic_df=public_df,
        #     private_df=private_df,
        #     text_col="text",
        #     id_col="private_id",
        # )
        # logging.info(f"Proximity attack embeddings results for {run}: {proximity_embeddings_result}")
        proximity_random_result = proximity_attack_random(
            synthetic_df=public_df,
            private_df=private_df,
            text_col="text",
            id_col="private_id",
        )
        logging.info(f"Proximity attack random results for {run}: {proximity_random_result}")
        # membership_result = measure.privacy.membership_attack(
        #     model_name_or_path="gpt2",
        #     device="cuda",
        #     public_df=df,
        #     private_df=private_df,
        #     text_col="text",
        # )
        result = {
            # **author_result,
            # **anonymity_result,
            # **embedding_result,
            **tfidf_result,
            **random_result,
            **proximity_result,
            # **proximity_embeddings_result,
            **proximity_random_result,
            # **membership_result,
        }

        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {run}")
        print(f"{'='*80}")
        for metric, value in result.items():
            print(f"{metric}: {value}")
        print(f"{'='*80}\n")

        logging.info(f"Logging metrics to WandB for {run}")
        for metric, value in result.items():
            result_dict[f"{run}/{metric}"] = value
        wandb.log(result_dict)


if __name__ == "__main__":
    main()
