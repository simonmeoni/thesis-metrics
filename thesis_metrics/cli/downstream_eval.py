"""CLI for running downstream task evaluations"""

import logging
from pathlib import Path

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from thesis_metrics.core.downstream import (
    ICDClassifier,
    NERClassifier,
    SemanticSimilarityEvaluator,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs/ds_stream", config_name="default")
def main(cfg: DictConfig):
    """Run downstream task evaluation based on configuration.

    The task type is determined by the config structure:
    - task.type: "semantic_similarity", "icd_classification", or "ner_classification"
    """
    # Set random seed
    if "seed" in cfg:
        set_seed(cfg.seed)

    # Initialize wandb if configured
    if cfg.get("wandb_project"):
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    task_type = cfg.task.type

    if task_type == "semantic_similarity":
        run_semantic_similarity(cfg)
    elif task_type == "icd_classification":
        run_icd_classification(cfg)
    elif task_type == "ner_classification":
        run_ner_classification(cfg)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if wandb.run:
        wandb.finish()


def run_semantic_similarity(cfg: DictConfig):
    """Run semantic similarity evaluation."""
    logger.info("Running semantic similarity evaluation")

    # Initialize evaluator
    evaluator = SemanticSimilarityEvaluator(
        model_name=cfg.task.model_name,
        model_checkpoint=cfg.task.get("model_checkpoint"),
        batch_size=cfg.task.get("batch_size", 32),
    )

    # Load dataset
    dataset_path = Path(cfg.task.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path)
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    # Run evaluation
    if cfg.task.get("use_keywords", False):
        results = evaluator.evaluate_keywords(
            dataset,
            original_column=cfg.task.original_column,
            prompt_column=cfg.task.prompt_column,
        )
    else:
        results = evaluator.evaluate(
            dataset,
            original_column=cfg.task.original_column,
            synthetic_column=cfg.task.synthetic_column,
        )

    # Log results
    logger.info(f"Results: {results}")

    if wandb.run:
        wandb.log(
            {
                "mean_similarity": results["mean_similarity"],
                "median_similarity": results["median_similarity"],
                "std_similarity": results["std_similarity"],
                "min_similarity": results["min_similarity"],
                "max_similarity": results["max_similarity"],
            }
        )


def run_icd_classification(cfg: DictConfig):
    """Run ICD-9 classification training and evaluation."""
    logger.info("Running ICD-9 classification")

    # Initialize classifier
    classifier = ICDClassifier(
        model_name=cfg.task.model_name,
        icd9_descriptions_path=cfg.task.get("icd9_descriptions_path"),
        top_k_labels=cfg.task.get("top_k_labels", 20),
        seed=cfg.get("seed", 42),
    )

    # Prepare datasets
    train_ds, test_ds = classifier.prepare_dataset(
        train_data=cfg.task.train_data_path,
        test_data=cfg.task.test_data_path,
        dataset_filter=cfg.task.get("dataset_filter"),
        percentile_threshold=cfg.task.get("percentile_threshold"),
        score_column=cfg.task.get("score_column"),
        precision_mode=cfg.task.get("precision_mode", False),
        text_column=cfg.task.get("text_column", "ground_texts"),
        use_keywords=cfg.task.get("use_keywords", False),
    )

    # Log dataset statistics
    logger.info(f"Training samples: {len(train_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    if wandb.run:
        # Log label distributions
        train_labels = [sum(binary) for binary in zip(*train_ds["labels"])]
        train_label_data = [
            [classifier.id2class[idx], count] for idx, count in enumerate(train_labels)
        ]
        table = wandb.Table(data=train_label_data, columns=["ICD", "count"])
        wandb.log({"train_label_distribution": wandb.plot.bar(table, "ICD", "count")})

    # Create training arguments
    training_args = hydra.utils.instantiate(cfg.task.training_args)

    # Train model
    trainer = classifier.train(
        train_dataset=train_ds,
        eval_dataset=test_ds,
        training_args=training_args,
        threshold=cfg.task.get("threshold", 0.5),
    )

    # Evaluate
    eval_results = trainer.evaluate(test_ds)
    logger.info(f"Evaluation results: {eval_results}")

    if wandb.run:
        wandb.log(eval_results)


def run_ner_classification(cfg: DictConfig):
    """Run NER training and evaluation."""
    logger.info("Running NER classification")

    # Initialize classifier
    classifier = NERClassifier(
        model_name=cfg.task.model_name,
        seed=cfg.get("seed", 42),
    )

    # Prepare datasets
    train_ds, test_ds = classifier.prepare_dataset(
        train_data=cfg.task.train_data_path,
        test_data=cfg.task.test_data_path,
        dataset_filter=cfg.task.get("dataset_filter"),
        percentile_threshold=cfg.task.get("percentile_threshold"),
        score_column=cfg.task.get("score_column", "score"),
        random_sampling=cfg.task.get("random_sampling", False),
    )

    # Log dataset statistics
    logger.info(f"Training samples: {len(train_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")

    # Create training arguments
    training_args = hydra.utils.instantiate(cfg.task.training_args)

    # Train model
    trainer = classifier.train(
        train_dataset=train_ds,
        eval_dataset=test_ds,
        training_args=training_args,
    )

    # Evaluate
    eval_results = trainer.evaluate(test_ds)
    logger.info(f"Evaluation results: {eval_results}")

    if wandb.run:
        wandb.log(
            {
                "precision": eval_results["eval_precision"],
                "recall": eval_results["eval_recall"],
                "f1": eval_results["eval_f1"],
                "accuracy": eval_results["eval_accuracy"],
                "max_f1": eval_results["eval_max_f1"],
            }
        )


if __name__ == "__main__":
    main()
