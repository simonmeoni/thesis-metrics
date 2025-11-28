"""ICD-9 code classification for medical utility evaluation"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class ICDClassifier:
    """Multi-label ICD-9 code classification for evaluating synthetic medical data utility.

    This classifier trains and evaluates models that predict ICD-9 diagnosis codes
    from clinical notes. It can be used to assess whether synthetic data preserves
    enough medical information for downstream diagnostic coding tasks.

    Args:
        model_name: HuggingFace model name for sequence classification
        icd9_descriptions_path: Path to ICD9 descriptions file (code\\tdescription format)
        top_k_labels: Number of most common ICD-9 codes to use (or -1 for all)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        icd9_descriptions_path: Optional[Union[str, Path]] = None,
        top_k_labels: int = 20,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.top_k_labels = top_k_labels
        self.seed = seed

        # Load ICD-9 descriptions
        if icd9_descriptions_path:
            self.icd9_descriptions = self._load_icd9_descriptions(icd9_descriptions_path)
        else:
            self.icd9_descriptions = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        self.class2id: Dict[str, int] = {}
        self.id2class: Dict[int, str] = {}
        self.model = None

    @staticmethod
    def _load_icd9_descriptions(path: Union[str, Path]) -> Dict[str, str]:
        """Load ICD-9 code descriptions from file."""
        descriptions = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    code, description = parts[0], parts[1]
                    descriptions[code] = description
        logger.info(f"Loaded {len(descriptions)} ICD-9 descriptions")
        return descriptions

    def _preprocess_labels(
        self,
        example: dict,
        precision_mode: bool = False,
    ) -> dict:
        """Preprocess ICD-9 labels from string format to list."""
        labels_str = example.get("LABELS", "")
        if labels_str is None or labels_str == "":
            example["LABELS"] = []
            return example

        # Split labels and clean them
        labels = [label.strip() for label in labels_str.split(";")]

        # Optional: truncate to root code (e.g., "250.01" -> "250")
        if precision_mode:
            labels = [label.split(".")[0] for label in labels]

        # Map codes to descriptions if available
        if self.icd9_descriptions:
            labels = [
                self.icd9_descriptions[label]
                for label in labels
                if label in self.icd9_descriptions
            ]

        example["LABELS"] = labels
        return example

    def _build_label_mappings(self, dataset: Dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build class2id and id2class mappings from dataset."""
        all_labels = [label for labels in dataset["LABELS"] for label in labels]

        if self.top_k_labels != -1:
            most_common = Counter(all_labels).most_common(self.top_k_labels)
            all_labels = [label for label, _ in most_common]

        unique_labels = sorted(set(all_labels))
        class2id = {class_: idx for idx, class_ in enumerate(unique_labels)}
        id2class = {idx: class_ for class_, idx in class2id.items()}

        logger.info(f"Built label mappings for {len(unique_labels)} ICD-9 codes")
        return class2id, id2class

    def _tokenize_and_encode(
        self,
        example: dict,
        text_column: str = "ground_texts",
    ) -> dict:
        """Tokenize text and create multi-hot encoded labels."""
        # Create multi-hot label vector
        labels = [0.0] * len(self.class2id)
        for label in example["LABELS"]:
            if label in self.class2id:
                label_id = self.class2id[label]
                labels[label_id] = 1.0

        # Tokenize text
        tokenized = self.tokenizer(example[text_column], truncation=True)

        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels,
        }

    def _tokenize_keywords(self, example: dict, prompt_column: str = "prompts") -> dict:
        """Tokenize keywords extracted from prompts (for baseline training)."""
        # Extract keywords from prompt
        try:
            keywords = example[prompt_column].split("Keywords: ")[1].removesuffix("[/INST]\n")
        except (IndexError, AttributeError):
            keywords = ""

        # Create multi-hot label vector
        labels = [0.0] * len(self.class2id)
        for label in example["LABELS"]:
            if label in self.class2id:
                label_id = self.class2id[label]
                labels[label_id] = 1.0

        # Tokenize keywords
        tokenized = self.tokenizer(keywords, truncation=True)

        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels,
        }

    def prepare_dataset(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        dataset_filter: Optional[str] = None,
        percentile_threshold: Optional[float] = None,
        score_column: Optional[str] = None,
        precision_mode: bool = False,
        text_column: str = "ground_texts",
        use_keywords: bool = False,
    ) -> Tuple[Dataset, Dataset]:
        """Prepare training and test datasets for ICD classification.

        Args:
            train_data: Path to CSV file or DataFrame with training data
            test_data: Path to CSV file or DataFrame with test data
            dataset_filter: Filter for dataset_ids column (e.g., "0.06-0", "combined")
            percentile_threshold: Percentile threshold for score-based filtering (0-100)
            score_column: Column containing quality scores for filtering
            precision_mode: If True, truncate ICD codes to root (e.g., "250.01" -> "250")
            text_column: Column containing text to classify
            use_keywords: If True, extract and use keywords from prompts instead of full text

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Load datasets
        if isinstance(train_data, pd.DataFrame):
            train_ds = Dataset.from_pandas(train_data)
        else:
            train_ds = load_dataset("csv", data_files=str(train_data))["train"]

        if isinstance(test_data, pd.DataFrame):
            test_ds = Dataset.from_pandas(test_data)
        else:
            test_ds = load_dataset("csv", data_files=str(test_data))["train"]

        # Preprocess labels
        train_ds = train_ds.filter(lambda x: x.get("LABELS") is not None).map(
            lambda x: self._preprocess_labels(x, precision_mode)
        )
        test_ds = test_ds.map(lambda x: self._preprocess_labels(x, precision_mode))

        # Apply dataset filter if specified
        if dataset_filter and "dataset_ids" in train_ds.column_names:
            if dataset_filter == "combined":
                allowed_ids = ["0.06-0", "0.06-1-ofzh3aqu", "0.06-2-ofzh3aqu"]
                train_ds = train_ds.filter(lambda x: x["dataset_ids"] in allowed_ids)
            elif dataset_filter == "combined-4":
                allowed_ids = ["0.04-0", "0.04-1-mru97w7c", "0.04-2-mru97w7c"]
                train_ds = train_ds.filter(lambda x: x["dataset_ids"] in allowed_ids)
            else:
                train_ds = train_ds.filter(lambda x: x["dataset_ids"] == dataset_filter)

        # Build label mappings from test set
        self.class2id, self.id2class = self._build_label_mappings(test_ds)

        # Filter to only include samples with at least one valid label
        def has_valid_labels(example):
            return any(label in self.class2id for label in example["LABELS"])

        train_ds = train_ds.filter(has_valid_labels)
        test_ds = test_ds.filter(has_valid_labels)

        # Apply score-based filtering if specified
        if percentile_threshold is not None and score_column:
            if score_column in train_ds.column_names:
                threshold_value = np.percentile(train_ds[score_column], percentile_threshold)
                train_ds = train_ds.filter(lambda x: x[score_column] >= threshold_value)
                logger.info(
                    f"Filtered training data by {percentile_threshold}th percentile "
                    f"(threshold={threshold_value:.4f})"
                )

        # Tokenize datasets
        tokenize_fn = self._tokenize_keywords if use_keywords else self._tokenize_and_encode
        train_ds = train_ds.map(lambda x: tokenize_fn(x)).select_columns(["input_ids", "labels"])
        test_ds = test_ds.map(
            lambda x: self._tokenize_and_encode(x, text_column)
        ).select_columns(["input_ids", "labels"])

        logger.info(f"Prepared {len(train_ds)} training and {len(test_ds)} test samples")
        return train_ds, test_ds

    def _compute_metrics(self, eval_pred, threshold: float = 0.5):
        """Compute classification metrics."""
        predictions, labels = eval_pred

        # Apply sigmoid and threshold
        predictions = 1 / (1 + np.exp(-predictions))
        predictions = (predictions > threshold).astype(float)
        labels = labels.astype(float)

        # Compute metrics
        metrics_result = self.metrics.compute(
            predictions=predictions.reshape(-1),
            references=labels.reshape(-1),
        )

        return metrics_result

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArguments,
        threshold: float = 0.5,
    ) -> Trainer:
        """Train ICD-9 classification model.

        Args:
            train_dataset: Prepared training dataset
            eval_dataset: Prepared evaluation dataset
            training_args: HuggingFace TrainingArguments
            threshold: Classification threshold for binary predictions

        Returns:
            Trained Trainer object
        """
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.class2id),
            id2label=self.id2class,
            label2id=self.class2id,
            problem_type="multi_label_classification",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: self._compute_metrics(x, threshold),
        )

        # Train
        logger.info("Starting ICD classification training...")
        trainer.train()
        logger.info("Training complete!")

        return trainer

    def evaluate(
        self,
        dataset: Dataset,
        threshold: float = 0.5,
    ) -> dict:
        """Evaluate trained model on dataset.

        Args:
            dataset: Prepared evaluation dataset
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: self._compute_metrics(x, threshold),
        )

        results = trainer.evaluate(dataset)
        return results
