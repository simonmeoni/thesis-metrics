#!/usr/bin/env python3
"""
Rephrasing CLI

Command-line interface for rephrasing data using Groq API.
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from thesis_metrics.rephrasing import run_rephrasing_sync
from thesis_metrics.utils import setup_logging

console = Console()


def display_banner():
    """Display welcome banner"""
    banner = """
    [bold cyan]Text Rephrasing[/bold cyan]
    [dim]Data augmentation using Groq API[/dim]
    """
    console.print(Panel(banner, border_style="cyan"))


def display_config(cfg: DictConfig):
    """Display configuration summary"""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Input Path", str(cfg.rephrasing.input_path))
    table.add_row("Output Path", str(cfg.rephrasing.output_path))
    table.add_row("Column", str(cfg.rephrasing.column))

    if cfg.rephrasing.get("select_best_generation", False):
        table.add_row("Best Generation", "Enabled")
        table.add_row("Output Column", str(cfg.rephrasing.get("output_column", cfg.rephrasing.column)))

    if cfg.rephrasing.get("use_keyword_prompt", False):
        table.add_row("Keyword-Aware", "Enabled")

    table.add_row("Model", cfg.rephrasing.model)
    table.add_row("Temperature", str(cfg.rephrasing.temperature))
    table.add_row("Max Tokens", str(cfg.rephrasing.max_tokens))
    table.add_row("Batch Size", str(cfg.rephrasing.batch_size))
    table.add_row("Max Concurrent", str(cfg.rephrasing.max_concurrent))

    console.print(table)
    console.print()


@hydra.main(version_base=None, config_path="../configs", config_name="rephrase_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for rephrasing

    Args:
        cfg: Hydra configuration
    """

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]Error:[/bold red] GROQ_API_KEY environment variable not set")
        console.print("Please set your Groq API key:")
        console.print("  export GROQ_API_KEY='your-api-key-here'")
        return

    # Setup logging
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # Display banner
    display_banner()

    # Display configuration
    display_config(cfg)

    # Validate input file
    input_path = Path(cfg.rephrasing.input_path)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        return

    # Run rephrasing
    console.print("\n[bold green]Starting rephrasing...[/bold green]\n")

    try:
        result_df = run_rephrasing_sync(cfg)

        console.print(f"\n[green]✓[/green] Rephrasing complete!")
        console.print(f"[green]✓[/green] Processed {len(result_df)} rows")
        console.print(f"[green]✓[/green] Output saved to: {cfg.rephrasing.output_path}")

    except Exception as e:
        logger.exception("Error during rephrasing")
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        return

    console.print("\n[bold green]✓ Done![/bold green]\n")


if __name__ == "__main__":
    main()
