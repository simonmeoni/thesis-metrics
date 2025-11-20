"""Text rephrasing module for data augmentation"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
from groq import AsyncGroq
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm.asyncio import tqdm_asyncio

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
        max_retries: int = 3
    ) -> str:
        """
        Rephrase a single text with retry logic

        Args:
            text: Text to rephrase
            prompt_template: Prompt template with {text} placeholder
            semaphore: Asyncio semaphore for rate limiting
            max_retries: Maximum retry attempts

        Returns:
            Rephrased text or original text if rephrasing fails
        """
        if pd.isna(text) or text == "":
            return text

        async with semaphore:
            for attempt in range(max_retries):
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.cfg.rephrasing.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that rephrases text while preserving meaning.",
                            },
                            {"role": "user", "content": prompt_template.format(text=text)},
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
        desc: str = "Rephrasing"
    ) -> list[str]:
        """
        Rephrase multiple texts concurrently

        Args:
            texts: List of texts to rephrase
            prompt_template: Prompt template
            desc: Description for progress bar

        Returns:
            List of rephrased texts
        """
        semaphore = asyncio.Semaphore(self.cfg.rephrasing.max_concurrent)
        tasks = [
            self.rephrase_single_text(text, prompt_template, semaphore)
            for text in texts
        ]
        return await tqdm_asyncio.gather(*tasks, desc=desc)

    async def rephrase_instruction_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rephrase a specific column (instruction mode)

        Args:
            df: Input dataframe

        Returns:
            DataFrame with rephrased column
        """
        column = self.cfg.rephrasing.instruction_column
        logger.info(f"Rephrasing column: {column}")

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {list(df.columns)}"
            )

        # Process in batches
        all_rephrased = []
        batch_size = self.cfg.rephrasing.batch_size

        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            logger.info(f"Processing batch {i//batch_size + 1}: rows {i} to {batch_end}")

            batch_texts = df[column].iloc[i:batch_end].tolist()
            rephrased_batch = await self.rephrase_batch(
                texts=batch_texts,
                prompt_template=self.cfg.rephrasing.instruction_prompt,
                desc=f"Rephrasing {column}"
            )
            all_rephrased.extend(rephrased_batch)

        # Create output dataframe
        df_output = df.copy()
        df_output[column] = all_rephrased

        return df_output

    async def rephrase_generation_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find best-scoring generation and rephrase it (generation mode)

        Args:
            df: Input dataframe with generation_* and eval_sem_scores_* columns

        Returns:
            DataFrame with new generation_rephrased column
        """
        logger.info("Finding best-scoring generations...")

        # Find generation and score columns
        generation_cols = [
            col for col in df.columns
            if col.startswith('generation_') and col.replace('generation_', '').isdigit()
        ]
        score_cols = [
            col for col in df.columns
            if col.startswith('eval_sem_scores_') and col.replace('eval_sem_scores_', '').isdigit()
        ]

        logger.info(f"Found {len(generation_cols)} generation columns")
        logger.info(f"Found {len(score_cols)} score columns")

        if not generation_cols or not score_cols:
            raise ValueError("No generation_* or eval_sem_scores_* columns found")

        # Find best generation for each row
        best_texts = []
        for _, row in df.iterrows():
            scores = []
            for score_col in score_cols:
                score = row[score_col]
                scores.append(score if pd.notna(score) else float('-inf'))

            best_idx = scores.index(max(scores)) if scores else 0
            best_gen_col = f'generation_{best_idx}'
            best_text = row[best_gen_col] if best_gen_col in df.columns else ""
            best_texts.append(best_text)

        logger.info(f"Found best generations for {len(best_texts)} rows")

        # Rephrase in batches
        all_rephrased = []
        batch_size = self.cfg.rephrasing.batch_size

        for i in range(0, len(best_texts), batch_size):
            batch_end = min(i + batch_size, len(best_texts))
            logger.info(f"Processing batch {i//batch_size + 1}: rows {i} to {batch_end}")

            batch_texts = best_texts[i:batch_end]
            rephrased_batch = await self.rephrase_batch(
                texts=batch_texts,
                prompt_template=self.cfg.rephrasing.generation_prompt,
                desc="Rephrasing best generations"
            )
            all_rephrased.extend(rephrased_batch)

        # Add new column
        df_output = df.copy()
        df_output['generation_rephrased'] = all_rephrased

        return df_output

    async def run(self) -> pd.DataFrame:
        """
        Run rephrasing based on mode

        Returns:
            DataFrame with rephrased data
        """
        # Load data
        logger.info(f"Loading data from {self.cfg.rephrasing.input_path}")
        df = pd.read_parquet(self.cfg.rephrasing.input_path)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

        # Process based on mode
        mode = self.cfg.rephrasing.mode
        logger.info(f"Running in {mode} mode")

        if mode == "instruction":
            df_output = await self.rephrase_instruction_mode(df)
        elif mode == "generation":
            df_output = await self.rephrase_generation_mode(df)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'instruction' or 'generation'")

        # Save output
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
