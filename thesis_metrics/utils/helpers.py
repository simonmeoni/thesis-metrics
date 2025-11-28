"""Utility functions for the thesis metrics package"""

import logging

from omegaconf import DictConfig
from rich.logging import RichHandler


def setup_logging(cfg: DictConfig):
    """
    Setup logging configuration

    Args:
        cfg: Hydra configuration
    """
    log_level = getattr(logging, cfg.logging.level.upper())

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def extract_keywords_from_instruction(instruction_text: str) -> str:
    """
    Extract keywords from instruction text

    Args:
        instruction_text: Instruction containing keywords

    Returns:
        Extracted keywords as string
    """
    if "### Keywords:" in instruction_text:
        keywords = (
            instruction_text.split("### Keywords:")[1].split("### Knowledge Base")[0].strip()
        )
        return keywords
    return instruction_text
