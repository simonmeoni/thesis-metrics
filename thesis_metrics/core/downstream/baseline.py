"""Semantic similarity evaluation for synthetic text quality assessment"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class SemanticSimilarityEvaluator:
    """Evaluates semantic similarity between synthetic and original medical texts.

    This evaluator uses sentence transformers to measure how well synthetic text
    preserves the semantic meaning of original clinical notes. Higher similarity
    scores indicate better utility preservation.

    Args:
        model_name: Name of the sentence transformer model to use
        model_checkpoint: Optional path to a fine-tuned model checkpoint
        batch_size: Batch size for encoding texts
        device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_checkpoint: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading semantic model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Load checkpoint if provided
        if model_checkpoint is not None:
            checkpoint_path = Path(model_checkpoint)
            if not checkpoint_path.exists():
                raise ValueError(f"Model checkpoint not found: {checkpoint_path}")
            logger.info(f"Loading model checkpoint from {checkpoint_path}")
            self.model = self.model.load(str(checkpoint_path))

    def compute_similarity_scores(
        self,
        original_texts: List[str],
        synthetic_texts: List[str],
    ) -> List[float]:
        """Compute pairwise cosine similarity between original and synthetic texts.

        Args:
            original_texts: List of original/ground truth texts
            synthetic_texts: List of corresponding synthetic texts

        Returns:
            List of similarity scores (0-1 range)

        Raises:
            ValueError: If input lists have different lengths
        """
        if len(original_texts) != len(synthetic_texts):
            raise ValueError(
                f"Length mismatch: {len(original_texts)} original texts "
                f"vs {len(synthetic_texts)} synthetic texts"
            )

        logger.info(f"Encoding {len(original_texts)} text pairs")

        # Encode texts
        original_embeddings = self.model.encode(
            original_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        synthetic_embeddings = self.model.encode(
            synthetic_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        # Compute pairwise cosine similarity
        scores = [
            util.cos_sim(orig_emb, synth_emb)[0][0].item()
            for orig_emb, synth_emb in zip(original_embeddings, synthetic_embeddings)
        ]

        return scores

    def evaluate(
        self,
        dataset: pd.DataFrame,
        original_column: str = "ground_texts",
        synthetic_column: str = "synthetic_texts",
    ) -> dict:
        """Evaluate semantic similarity on a dataset.

        Args:
            dataset: DataFrame containing original and synthetic texts
            original_column: Column name for original texts
            synthetic_column: Column name for synthetic texts

        Returns:
            Dictionary with evaluation metrics:
                - mean_similarity: Average similarity score
                - median_similarity: Median similarity score
                - std_similarity: Standard deviation of scores
                - min_similarity: Minimum score
                - max_similarity: Maximum score
                - scores: List of all individual scores
        """
        if original_column not in dataset.columns:
            raise ValueError(f"Column '{original_column}' not found in dataset")
        if synthetic_column not in dataset.columns:
            raise ValueError(f"Column '{synthetic_column}' not found in dataset")

        original_texts = dataset[original_column].tolist()
        synthetic_texts = dataset[synthetic_column].tolist()

        scores = self.compute_similarity_scores(original_texts, synthetic_texts)
        scores_tensor = torch.tensor(scores)

        results = {
            "mean_similarity": float(torch.mean(scores_tensor)),
            "median_similarity": float(torch.median(scores_tensor)),
            "std_similarity": float(torch.std(scores_tensor)),
            "min_similarity": float(torch.min(scores_tensor)),
            "max_similarity": float(torch.max(scores_tensor)),
            "num_samples": len(scores),
            "scores": scores,
        }

        logger.info(f"Semantic similarity evaluation complete:")
        logger.info(f"  Mean: {results['mean_similarity']:.4f}")
        logger.info(f"  Median: {results['median_similarity']:.4f}")
        logger.info(f"  Std: {results['std_similarity']:.4f}")

        return results

    def evaluate_keywords(
        self,
        dataset: pd.DataFrame,
        original_column: str = "ground_texts",
        prompt_column: str = "prompts",
        keyword_separator: str = "Keywords: ",
        keyword_suffix: str = "[/INST]\n",
    ) -> dict:
        """Evaluate semantic similarity between original texts and extracted keywords.

        This method is useful for evaluating baseline models that only use keyword
        extraction instead of full text generation.

        Args:
            dataset: DataFrame containing texts and prompts
            original_column: Column name for original texts
            prompt_column: Column name for prompts containing keywords
            keyword_separator: String that precedes keywords in prompt
            keyword_suffix: String that follows keywords in prompt

        Returns:
            Dictionary with evaluation metrics (same as evaluate())
        """
        if original_column not in dataset.columns:
            raise ValueError(f"Column '{original_column}' not found in dataset")
        if prompt_column not in dataset.columns:
            raise ValueError(f"Column '{prompt_column}' not found in dataset")

        # Extract keywords from prompts
        def extract_keywords(prompt: str) -> str:
            try:
                keywords = prompt.split(keyword_separator)[1]
                keywords = keywords.removesuffix(keyword_suffix)
                return keywords
            except (IndexError, AttributeError):
                logger.warning(f"Failed to extract keywords from prompt: {prompt[:100]}...")
                return ""

        dataset = dataset.copy()
        dataset["keywords"] = dataset[prompt_column].apply(extract_keywords)

        # Filter out failed extractions
        original_count = len(dataset)
        dataset = dataset[dataset["keywords"] != ""]
        if len(dataset) < original_count:
            logger.warning(
                f"Dropped {original_count - len(dataset)} samples due to keyword extraction failures"
            )

        return self.evaluate(dataset, original_column=original_column, synthetic_column="keywords")
