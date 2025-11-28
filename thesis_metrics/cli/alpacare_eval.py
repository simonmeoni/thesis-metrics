#!/usr/bin/env python3
"""
AlpaCare Evaluation CLI

Command-line interface for AlpaCare health dataset evaluation pipeline.
Runs evaluation pipeline including AlpaCare generation, SFT training, and preference evaluation.
"""

import argparse
import logging
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from thesis_metrics.utils.slurm import SlurmJobManager

console = Console()


def display_banner():
    """Display welcome banner"""
    banner = """
    [bold cyan]AlpaCare Evaluation Pipeline[/bold cyan]
    [dim]Evaluate synthetic health datasets using AlpaCare baseline[/dim]
    """
    console.print(Panel(banner, border_style="cyan"))


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run AlpaCare health dataset evaluation pipeline"
    )

    # Required arguments
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier (run_id from training)",
    )
    parser.add_argument(
        "--downstream_ds_path",
        type=str,
        required=True,
        help="Path to the scored evaluation dataset (parquet format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base path for saving evaluation outputs",
    )

    # Optional arguments
    parser.add_argument(
        "--group_id",
        type=str,
        default=None,
        help="Group ID for tracking related experiments (default: auto-generated)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="eval",
        help="Training step identifier (default: 'eval')",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=60000,
        help="Dataset size (default: 60000)",
    )
    parser.add_argument(
        "--suffix_run_name",
        type=str,
        default=None,
        help="Suffix for the run name (default: step value)",
    )
    parser.add_argument(
        "--sts_model",
        type=str,
        default="FremyCompany/BioLORD-2023",
        help="STS model used for scoring (default: FremyCompany/BioLORD-2023)",
    )
    parser.add_argument(
        "--slurm_dir",
        type=str,
        default=None,
        help="Path to SLURM scripts directory (default: alpacare-evaluation/slurm/)",
    )

    return parser.parse_args()


def display_config(args):
    """Display configuration summary"""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Model ID", args.model_id)
    table.add_row("Downstream Dataset", args.downstream_ds_path)
    table.add_row("Output Path", args.output_path)
    table.add_row("Group ID", args.group_id)
    table.add_row("Step", args.step)
    table.add_row("Size", str(args.size))
    table.add_row("Suffix", args.suffix_run_name)
    table.add_row("STS Model", args.sts_model)

    console.print(table)
    console.print()


def main():
    """Main entry point for AlpaCare evaluation"""
    args = parse_args()

    # Display banner
    display_banner()

    # Set defaults
    if args.group_id is None:
        args.group_id = str(uuid.uuid4())[:7]

    if args.suffix_run_name is None:
        args.suffix_run_name = args.step

    # Determine SLURM scripts directory
    if args.slurm_dir is None:
        # Default to original alpacare-evaluation directory
        project_root = Path(__file__).parent.parent.parent
        args.slurm_dir = project_root / "alpacare-evaluation" / "slurm"
    else:
        args.slurm_dir = Path(args.slurm_dir)

    if not args.slurm_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] SLURM scripts directory not found: {args.slurm_dir}"
        )
        console.print(
            "\nPlease specify --slurm_dir or ensure alpacare-evaluation/slurm/ exists"
        )
        return

    # Display configuration
    display_config(args)

    # Validate input file
    input_path = Path(args.downstream_ds_path)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input file not found: {input_path}")
        return

    console.print("[bold green]Starting AlpaCare evaluation pipeline...[/bold green]\n")

    # Initialize job manager
    job_mgr = SlurmJobManager()

    # Step 1: Submit health evaluation job (AlpaCare gen + SFT training + model gen)
    console.print(
        "[bold]Step 1:[/bold] Submitting health evaluation job "
        "(AlpaCare generation + SFT training + model generation)..."
    )

    eval_gen_script = args.slurm_dir / "health-evaluation.slurm"
    eval_gen_cmd = (
        f"{eval_gen_script} "
        f"--DOWNSTREAM_DS_PATH {args.downstream_ds_path} "
        f"--OUTPUT_PATH {args.output_path} "
        f"--GROUP_ID {args.group_id}"
    )

    try:
        eval_job_id = job_mgr.submit(eval_gen_cmd, dependency=False)
        console.print(f"[green]✓[/green] Job submitted: {eval_job_id}\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to submit job: {e}")
        return

    # Step 2: Submit preference evaluation job (depends on step 1)
    console.print(
        "[bold]Step 2:[/bold] Submitting preference evaluation job "
        "(depends on health evaluation)..."
    )

    eval_pref_script = args.slurm_dir / "health-preference-eval.slurm"
    eval_pref_cmd = (
        f"{eval_pref_script} "
        f"--MODEL_ID {args.model_id} "
        f"--STEP {args.step} "
        f"--SIZE {args.size} "
        f"--SUFFIX_RUN_NAME {args.suffix_run_name} "
        f"--GROUP_ID {args.group_id}"
    )

    try:
        pref_job_id = job_mgr.submit(eval_pref_cmd, dependency=True)
        console.print(f"[green]✓[/green] Job submitted: {pref_job_id}\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to submit job: {e}")
        return

    # Success summary
    console.print(Panel(f"""
[bold green]✓ Evaluation pipeline jobs submitted successfully![/bold green]

[bold]Group ID:[/bold] {args.group_id}
[bold]Evaluation Job:[/bold] {eval_job_id}
[bold]Preference Job:[/bold] {pref_job_id}

[bold cyan]Monitor jobs with:[/bold cyan]
  squeue -u $USER

[bold cyan]View logs:[/bold cyan]
  ls -lt log/slurm-*.out | head

[bold cyan]Results will be logged to Weights & Biases:[/bold cyan]
  Run name: eval_{args.model_id}_{args.suffix_run_name}
  Group: {args.group_id}
""", border_style="green"))


if __name__ == "__main__":
    main()
