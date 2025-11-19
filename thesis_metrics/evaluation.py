"""Privacy evaluation module"""

import logging

import pandas as pd
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, TextColumn

from thesis_metrics.privacy_attacks import (
    linkage_attack_tfidf,
    proximity_attack_random,
    proximity_attack_tfidf,
    random_leakage_attack,
)
from thesis_metrics.utils import extract_keywords_from_instruction

logger = logging.getLogger(__name__)


class PrivacyEvaluator:
    """Privacy metrics evaluator"""

    def __init__(self, cfg: DictConfig):
        """
        Initialize evaluator

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg

    def load_datasets(self):
        """Load synthetic and source datasets"""
        logger.info("Loading datasets...")

        df_synthetic = pd.read_parquet(self.cfg.dataset.synthetic_path)
        logger.info(f"Loaded synthetic dataset: {len(df_synthetic)} rows")

        df_source = pd.read_parquet(self.cfg.dataset.source_path)
        logger.info(f"Loaded source dataset: {len(df_source)} rows")

        return df_synthetic, df_source

    def extract_keywords(self, df_synthetic, df_source):
        """Extract keywords from both datasets"""
        logger.info("Extracting keywords...")

        df_source["keywords_str"] = df_source[self.cfg.dataset.columns.instruction].apply(
            extract_keywords_from_instruction
        )
        df_synthetic["keywords_str"] = df_synthetic[self.cfg.dataset.columns.instruction].apply(
            extract_keywords_from_instruction
        )

        # Check for common keywords
        source_keywords = set(df_source["keywords_str"].dropna())
        synthetic_keywords = set(df_synthetic["keywords_str"].dropna())
        common_keywords = source_keywords & synthetic_keywords

        match_rate = 100 * len(common_keywords) / len(synthetic_keywords)
        logger.info(
            f"Common keywords: {len(common_keywords)}/{len(synthetic_keywords)} ({match_rate:.1f}%)"
        )

        if len(common_keywords) == 0:
            raise ValueError("No common keywords found between source and synthetic datasets!")

        return df_synthetic, df_source

    def merge_datasets(self, df_synthetic, df_source):
        """Merge synthetic and source datasets on keywords"""
        logger.info("Merging datasets...")

        df = df_synthetic.merge(
            df_source[
                [
                    "keywords_str",
                    self.cfg.dataset.columns.response,
                    self.cfg.dataset.columns.cluster_id,
                ]
            ],
            on="keywords_str",
            how="inner",
            suffixes=("_synthetic", "_private"),
        )

        merge_rate = 100 * len(df) / len(df_synthetic)
        logger.info(f"Merged dataset: {len(df)}/{len(df_synthetic)} rows ({merge_rate:.1f}%)")

        return df

    def prepare_attack_dataframes(self, df):
        """Prepare public and private dataframes for privacy attacks"""
        logger.info("Preparing attack dataframes...")

        # Create private_id from cluster_id
        df["private_id"], _ = pd.factorize(df[self.cfg.dataset.columns.cluster_id])

        # Public dataset (synthetic text)
        public_df = pd.DataFrame(
            {
                "text": df[self.cfg.dataset.columns.synthetic_text].tolist(),
                "private_id": df["private_id"].tolist(),
            }
        )

        # Private dataset (real private text)
        private_df = pd.DataFrame(
            {
                "text": df[self.cfg.dataset.columns.response].tolist(),
                "private_id": df["private_id"].tolist(),
            }
        )

        logger.info(
            f"Public: {len(public_df)} rows, {public_df['private_id'].nunique()} unique IDs"
        )
        logger.info(
            f"Private: {len(private_df)} rows, {private_df['private_id'].nunique()} unique IDs"
        )

        return public_df, private_df

    def run_attacks(self, public_df, private_df):
        """Run enabled privacy attacks"""
        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:

            for attack_name in self.cfg.attacks.enabled:
                attack_config = self.cfg.attacks[attack_name]
                task = progress.add_task(f"Running {attack_config.name}...", total=None)

                if attack_name == "linkage_tfidf":
                    result = linkage_attack_tfidf(
                        public_df=public_df,
                        private_df=private_df,
                        text_col="text",
                        id_col="private_id",
                    )
                elif attack_name == "random_baseline":
                    result = random_leakage_attack(
                        public_df=public_df,
                        private_df=private_df,
                        id_col="private_id",
                    )
                elif attack_name == "proximity_tfidf":
                    result = proximity_attack_tfidf(
                        synthetic_df=public_df,
                        private_df=private_df,
                        text_col="text",
                        id_col="private_id",
                    )
                elif attack_name == "proximity_random":
                    result = proximity_attack_random(
                        synthetic_df=public_df,
                        private_df=private_df,
                        text_col="text",
                        id_col="private_id",
                    )

                results.update(result)
                progress.update(task, completed=True)
                logger.info(f"{attack_config.name}: {result}")

        return results

    def run(self) -> pd.DataFrame:
        """
        Run privacy evaluation

        Returns:
            DataFrame with results
        """

        # Load datasets
        df_synthetic, df_source = self.load_datasets()

        # Extract keywords
        df_synthetic, df_source = self.extract_keywords(df_synthetic, df_source)

        # Merge datasets
        df = self.merge_datasets(df_synthetic, df_source)

        # Prepare attack dataframes
        public_df, private_df = self.prepare_attack_dataframes(df)

        # Run attacks
        results = self.run_attacks(public_df, private_df)

        # Convert to DataFrame
        results_df = pd.DataFrame([results])
        results_df.insert(0, "dataset", self.cfg.dataset.name)

        return results_df
