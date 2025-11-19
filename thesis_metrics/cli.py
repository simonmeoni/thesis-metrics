#!/usr/bin/env python3
"""
Privacy Metrics Evaluation CLI

Clean command-line interface for evaluating privacy metrics on synthetic medical data.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from thesis_metrics.evaluation import PrivacyEvaluator
from thesis_metrics.utils import setup_logging
from thesis_metrics.visualization import TerminalVisualizer

console = Console()


def display_banner():
    """Display welcome banner"""
    banner = """
    [bold cyan]Privacy Metrics Evaluation[/bold cyan]
    [dim]Evaluating privacy leakage in synthetic medical data[/dim]
    """
    console.print(Panel(banner, border_style="cyan"))


def display_config(cfg: DictConfig):
    """Display configuration summary"""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Dataset", cfg.dataset.name)
    table.add_row("Source Path", str(cfg.dataset.source_path))
    table.add_row("Synthetic Path", str(cfg.dataset.synthetic_path))
    table.add_row("Attacks", ", ".join(cfg.attacks.enabled))
    table.add_row("WandB", "✓ Enabled" if cfg.wandb.enabled else "✗ Disabled")

    console.print(table)
    console.print()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for privacy metrics evaluation

    Args:
        cfg: Hydra configuration
    """

    # Setup logging
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    # Display banner
    display_banner()

    # Display configuration
    display_config(cfg)

    # Initialize evaluator
    console.print("[bold green]Initializing evaluator...[/bold green]")
    evaluator = PrivacyEvaluator(cfg)

    # Run evaluation
    console.print("\n[bold green]Running privacy attacks...[/bold green]\n")
    results = evaluator.run()

    # Visualize results
    if cfg.visualization.terminal_plots:
        console.print("\n[bold green]Visualization Results:[/bold green]\n")
        visualizer = TerminalVisualizer()
        visualizer.display_results(results, cfg.dataset.name)

    # Save results
    if cfg.output.save_report:
        output_dir = Path(cfg.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        results_file = output_dir / "results.csv"
        results.to_csv(results_file)
        console.print(f"\n[green]✓[/green] Results saved to: {results_file}")

        # Save config
        config_file = output_dir / "config.yaml"
        OmegaConf.save(cfg, config_file)
        console.print(f"[green]✓[/green] Config saved to: {config_file}")

    # Log to WandB if enabled
    if cfg.wandb.enabled:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.log(results.to_dict())
        wandb.finish()
        console.print("\n[green]✓[/green] Results logged to WandB")

    console.print("\n[bold green]✓ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    main()
