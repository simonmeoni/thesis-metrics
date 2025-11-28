"""Named Entity Recognition for medical utility evaluation"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class NERClassifier:
    """Token-level Named Entity Recognition for medical text evaluation.

    This classifier trains and evaluates models that identify and label medical
    entities (e.g., diseases, medications, symptoms) in clinical notes. It can
    be used to assess whether synthetic data preserves entity-level information.

    Args:
        model_name: HuggingFace model name for token classification
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        seed: int = 42,
    ):
        self.model_name = model_name
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seqeval_metric = evaluate.load("seqeval")

        self.class2id: Dict[str, int] = {}
        self.id2class: Dict[int, str] = {}
        self.model = None
        self.current_max_f1 = [0.0]  # Track best F1 during training

    def _build_label_mappings(self, dataset: Dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build class2id and id2class mappings from dataset entities."""
        all_labels = sorted(set(token for entities in dataset["entities"] for token in entities))

        class2id = {label: idx for idx, label in enumerate(all_labels)}
        id2class = {idx: label for label, idx in class2id.items()}

        logger.info(f"Built label mappings for {len(all_labels)} entity types")
        return class2id, id2class

    def _tokenize_and_align_labels(self, examples: dict) -> dict:
        """Tokenize words and align NER labels with subword tokens.

        This handles the case where the tokenizer splits words into multiple
        subword tokens. Only the first subword token gets the label, others
        get -100 (ignored in loss calculation).
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens get -100
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # Subsequent subword tokens get -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(
        self,
        train_data: Union[str, Path, pd.DataFrame, List[str]],
        test_data: Union[str, Path, pd.DataFrame],
        dataset_filter: Optional[str] = None,
        percentile_threshold: Optional[float] = None,
        score_column: str = "score",
        random_sampling: bool = False,
    ) -> Tuple[Dataset, Dataset]:
        """Prepare training and test datasets for NER.

        Args:
            train_data: Path(s) to parquet files, DataFrame, or list of dataset IDs
            test_data: Path to test parquet file or DataFrame
            dataset_filter: Filter for dataset combinations ("combined", "combined-4", etc.)
            percentile_threshold: Percentile threshold for score-based filtering (0-100)
            score_column: Column containing quality scores
            random_sampling: If True, use random sampling instead of score-based filtering

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Load test dataset
        if isinstance(test_data, pd.DataFrame):
            test_ds = Dataset.from_pandas(test_data)
        else:
            test_ds = load_dataset("parquet", data_files=str(test_data))["train"]

        # Load training dataset(s)
        if isinstance(train_data, list):
            # Multiple datasets to combine
            train_datasets = []
            for dataset_id in train_data:
                ds_path = f"ner-{dataset_id}-train.parquet"
                ds = load_dataset("parquet", data_files=ds_path)["train"]
                train_datasets.append(ds)
            train_ds = concatenate_datasets(train_datasets)
        elif isinstance(train_data, pd.DataFrame):
            train_ds = Dataset.from_pandas(train_data)
        else:
            train_ds = load_dataset("parquet", data_files=str(train_data))["train"]

        # Apply dataset filter if specified
        if dataset_filter == "combined":
            allowed_ids = ["0.06-0", "0.06-1-ofzh3aqu", "0.06-2-ofzh3aqu"]
            if "dataset_ids" in train_ds.column_names:
                train_ds = train_ds.filter(lambda x: x["dataset_ids"] in allowed_ids)
        elif dataset_filter == "combined-4":
            allowed_ids = ["0.04-0", "0.04-1-mru97w7c", "0.04-2-mru97w7c"]
            if "dataset_ids" in train_ds.column_names:
                train_ds = train_ds.filter(lambda x: x["dataset_ids"] in allowed_ids)

        # Build label mappings from test set
        self.class2id, self.id2class = self._build_label_mappings(test_ds)

        # Apply score-based or random filtering
        if percentile_threshold is not None and score_column in train_ds.column_names:
            if random_sampling:
                # Random sampling: split and take test portion
                if percentile_threshold > 0:
                    split_ratio = 1 - (percentile_threshold / 100)
                    train_ds = train_ds.train_test_split(test_size=split_ratio)["test"]
                    logger.info(f"Random sampling: kept {len(train_ds)} samples")
            else:
                # Score-based filtering
                threshold_value = np.percentile(train_ds[score_column], percentile_threshold)
                train_ds = train_ds.filter(lambda x: x[score_column] >= threshold_value)
                logger.info(
                    f"Score-based filtering: {percentile_threshold}th percentile "
                    f"(threshold={threshold_value:.4f}), kept {len(train_ds)} samples"
                )

        # Rename columns to standard format
        if "words" in train_ds.column_names:
            train_ds = train_ds.rename_column("words", "tokens")
        if "words" in test_ds.column_names:
            test_ds = test_ds.rename_column("words", "tokens")

        if "entities" in train_ds.column_names:
            train_ds = train_ds.rename_column("entities", "ner_tags")
        if "entities" in test_ds.column_names:
            test_ds = test_ds.rename_column("entities", "ner_tags")

        # Remove score column if present
        if score_column in train_ds.column_names:
            train_ds = train_ds.remove_columns([score_column])
        if score_column in test_ds.column_names:
            test_ds = test_ds.remove_columns([score_column])

        # Map entity strings to IDs
        def map_tags_to_ids(example):
            # Handle IOB2 format: extract base tag before space
            ner_tags = []
            for tag in example["ner_tags"]:
                base_tag = tag.split(" ")[0]
                tag_id = self.class2id.get(base_tag, self.class2id.get("O", 0))
                ner_tags.append(tag_id)
            return {"ner_tags": ner_tags}

        train_ds = train_ds.map(map_tags_to_ids)
        test_ds = test_ds.map(map_tags_to_ids)

        # Tokenize and align labels
        train_ds = train_ds.map(self._tokenize_and_align_labels, batched=True)
        test_ds = test_ds.map(self._tokenize_and_align_labels, batched=True)

        logger.info(f"Prepared {len(train_ds)} training and {len(test_ds)} test samples")
        return train_ds, test_ds

    def _compute_metrics(self, eval_pred):
        """Compute NER metrics using seqeval."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored labels (-100) and convert IDs back to labels
        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            pred = [self.id2class[p] for p, l in zip(prediction, label) if l != -100]
            lab = [self.id2class[l] for p, l in zip(prediction, label) if l != -100]
            true_predictions.append(pred)
            true_labels.append(lab)

        # Compute seqeval metrics
        results = self.seqeval_metric.compute(
            predictions=true_predictions,
            references=true_labels,
            scheme="IOB2",
        )

        # Track maximum F1
        self.current_max_f1[0] = max(self.current_max_f1[0], results["overall_f1"])

        # Get detailed classification report
        report = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            mode="strict",
            scheme=IOB2,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "max_f1": self.current_max_f1[0],
            "classification_report": report,
        }

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: TrainingArguments,
    ) -> Trainer:
        """Train NER model.

        Args:
            train_dataset: Prepared training dataset
            eval_dataset: Prepared evaluation dataset
            training_args: HuggingFace TrainingArguments

        Returns:
            Trained Trainer object
        """
        # Reset max F1 tracker
        self.current_max_f1 = [0.0]

        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.class2id),
            id2label=self.id2class,
            label2id=self.class2id,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        # Train
        logger.info("Starting NER training...")
        trainer.train()
        logger.info(f"Training complete! Best F1: {self.current_max_f1[0]:.4f}")

        return trainer

    def evaluate(self, dataset: Dataset) -> dict:
        """Evaluate trained model on dataset.

        Args:
            dataset: Prepared evaluation dataset

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        results = trainer.evaluate(dataset)
        return results
