"""Text rephrasing module for data augmentation"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
from groq import AsyncGroq
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm.asyncio import tqdm_asyncio

from thesis_metrics.utils import extract_keywords_from_instruction

logger = logging.getLogger(__name__)


class Rephraser:
    """Text rephrasing for data augmentation"""

    def __init__(self, cfg: DictConfig, api_key: str = None):
        """
        Initialize rephraser

        Args:
            cfg: Hydra configuration
            api_key: Groq API key (optional, will use GROQ_API_KEY env var if not provided)
        """
        self.cfg = cfg
        self.client = AsyncGroq(api_key=api_key) if api_key else AsyncGroq()

    async def rephrase_single_text(
        self,
        text: str,
        prompt_template: str,
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
        keywords: str = None
    ) -> str:
        """
        Rephrase a single text with retry logic

        Args:
            text: Text to rephrase
            prompt_template: Prompt template with {text} placeholder
            semaphore: Asyncio semaphore for rate limiting
            max_retries: Maximum retry attempts
            keywords: Optional keywords to pass to the prompt template

        Returns:
            Rephrased text or original text if rephrasing fails
        """
        if pd.isna(text) or text == "":
            return text

        async with semaphore:
            for attempt in range(max_retries):
                try:
                    # Format prompt with keywords if provided
                    if keywords:
                        prompt_content = prompt_template.format(text=text, keywords=keywords)
                    else:
                        prompt_content = prompt_template.format(text=text)

                    completion = await self.client.chat.completions.create(
                        model=self.cfg.rephrasing.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that rephrases text while preserving meaning.",
                            },
                            {"role": "user", "content": prompt_content},
                        ],
                        temperature=self.cfg.rephrasing.temperature,
                        max_tokens=self.cfg.rephrasing.max_tokens,
                    )
                    return completion.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Failed to rephrase after {max_retries} attempts. Using original.")
                        return text

        return text

    async def rephrase_batch(
        self,
        texts: list[str],
        prompt_template: str,
        desc: str = "Rephrasing",
        keywords_list: list[str] = None
    ) -> list[str]:
        """
        Rephrase multiple texts concurrently

        Args:
            texts: List of texts to rephrase
            prompt_template: Prompt template
            desc: Description for progress bar
            keywords_list: Optional list of keywords for each text

        Returns:
            List of rephrased texts
        """
        semaphore = asyncio.Semaphore(self.cfg.rephrasing.max_concurrent)

        if keywords_list:
            tasks = [
                self.rephrase_single_text(text, prompt_template, semaphore, keywords=kw)
                for text, kw in zip(texts, keywords_list)
            ]
        else:
            tasks = [
                self.rephrase_single_text(text, prompt_template, semaphore)
                for text in texts
            ]
        return await tqdm_asyncio.gather(*tasks, desc=desc)

    async def run(self) -> pd.DataFrame:
        """
        Run rephrasing on specified column

        Returns:
            DataFrame with rephrased data
        """
        # Load data
        logger.info(f"Loading data from {self.cfg.rephrasing.input_path}")
        df = pd.read_parquet(self.cfg.rephrasing.input_path)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

        # Step 1: Get texts to rephrase
        texts_to_rephrase = []

        # Check if we need to select best generation first
        if self.cfg.rephrasing.get("select_best_generation", False):
            logger.info("Selecting best-scoring generations...")

            # Find generation and score columns
            gen_prefix = self.cfg.rephrasing.get("generation_prefix", "generation_")
            score_prefix = self.cfg.rephrasing.get("score_prefix", "eval_sem_scores_")

            generation_cols = [
                col for col in df.columns
                if col.startswith(gen_prefix) and col.replace(gen_prefix, '').isdigit()
            ]
            score_cols = [
                col for col in df.columns
                if col.startswith(score_prefix) and col.replace(score_prefix, '').isdigit()
            ]

            logger.info(f"Found {len(generation_cols)} generation columns")
            logger.info(f"Found {len(score_cols)} score columns")

            if not generation_cols or not score_cols:
                raise ValueError(f"No {gen_prefix}* or {score_prefix}* columns found")

            # Find best generation for each row
            for _, row in df.iterrows():
                scores = []
                for score_col in score_cols:
                    score = row[score_col]
                    scores.append(score if pd.notna(score) else float('-inf'))

                best_idx = scores.index(max(scores)) if scores else 0
                best_gen_col = f'{gen_prefix}{best_idx}'
                best_text = row[best_gen_col] if best_gen_col in df.columns else ""
                texts_to_rephrase.append(best_text)

            logger.info(f"Selected best generations for {len(texts_to_rephrase)} rows")
        else:
            # Just use the specified column
            column = self.cfg.rephrasing.column
            logger.info(f"Rephrasing column: {column}")

            if column not in df.columns:
                raise ValueError(
                    f"Column '{column}' not found. Available columns: {list(df.columns)}"
                )

            texts_to_rephrase = df[column].tolist()

        # Step 2: Prepare keyword-aware rephrasing if enabled
        use_keyword_prompt = self.cfg.rephrasing.get("use_keyword_prompt", False)
        keywords_list = None

        if use_keyword_prompt:
            logger.info("Keyword-aware rephrasing enabled")

            # Check if dataset has a keywords column
            if 'keywords' in df.columns:
                logger.info("Using pre-existing keywords from 'keywords' column")
                keywords_list = df['keywords'].tolist()
            else:
                logger.info("No keywords column found, extracting keywords from texts")
                keywords_list = [extract_keywords_from_instruction(text) for text in texts_to_rephrase]

            prompt_template = self.cfg.rephrasing.prompt_with_keywords
            logger.info(f"Using keyword-aware prompt with {len(keywords_list)} keyword sets")
        else:
            prompt_template = self.cfg.rephrasing.prompt

        # Step 3: Rephrase in batches
        all_rephrased = []
        batch_size = self.cfg.rephrasing.batch_size

        for i in range(0, len(texts_to_rephrase), batch_size):
            batch_end = min(i + batch_size, len(texts_to_rephrase))
            logger.info(f"Processing batch {i//batch_size + 1}: rows {i} to {batch_end}")

            batch_texts = texts_to_rephrase[i:batch_end]
            batch_keywords = keywords_list[i:batch_end] if keywords_list else None

            rephrased_batch = await self.rephrase_batch(
                texts=batch_texts,
                prompt_template=prompt_template,
                desc="Rephrasing" + (" (keyword-aware)" if use_keyword_prompt else ""),
                keywords_list=batch_keywords
            )
            all_rephrased.extend(rephrased_batch)

        # Step 4: Save output
        df_output = df.copy()
        output_column = self.cfg.rephrasing.get("output_column") or self.cfg.rephrasing.column
        df_output[output_column] = all_rephrased
        logger.info(f"Saved rephrased text to column: {output_column}")

        output_path = Path(self.cfg.rephrasing.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_output.to_parquet(output_path, index=False)
        logger.info(f"Saved rephrased data to {output_path}")
        logger.info(f"Original rows: {len(df)}, Output rows: {len(df_output)}")

        return df_output


def run_rephrasing_sync(cfg: DictConfig) -> pd.DataFrame:
    """
    Synchronous wrapper for running rephrasing

    Args:
        cfg: Hydra configuration

    Returns:
        DataFrame with rephrased data
    """
    rephraser = Rephraser(cfg)
    return asyncio.run(rephraser.run())
