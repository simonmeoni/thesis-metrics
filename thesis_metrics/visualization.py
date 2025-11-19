"""Terminal visualization module using plotext and rich"""

import pandas as pd
import plotext as plt
from rich.console import Console
from rich.table import Table

console = Console()


class TerminalVisualizer:
    """Terminal-based visualizer for privacy metrics"""

    def display_results(self, results_df: pd.DataFrame, dataset_name: str):
        """
        Display results in terminal with tables and plots

        Args:
            results_df: DataFrame with results
            dataset_name: Name of the dataset
        """

        # Display results table
        self.display_results_table(results_df, dataset_name)

        # Display bar chart
        self.display_bar_chart(results_df, dataset_name)

    def display_results_table(self, results_df: pd.DataFrame, dataset_name: str):
        """Display results as a rich table"""

        table = Table(
            title=f"Privacy Metrics Results - {dataset_name}",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right")

        # Add rows for each metric
        for col in results_df.columns:
            if col != "dataset":
                value = results_df[col].iloc[0]
                if isinstance(value, float):
                    table.add_row(col, f"{value:.4f}")
                else:
                    table.add_row(col, str(value))

        console.print()
        console.print(table)
        console.print()

    def display_bar_chart(self, results_df: pd.DataFrame, dataset_name: str):
        """Display bar chart of accuracy metrics in terminal"""

        # Filter only accuracy metrics
        accuracy_cols = [col for col in results_df.columns if "accuracy" in col.lower()]
        if len(accuracy_cols) == 0:
            return

        # Get metric names (clean labels) and values
        metrics = [col.replace("/accuracy", "").replace("_", " ").title() for col in accuracy_cols]
        values = [results_df[col].iloc[0] for col in accuracy_cols]

        # Create horizontal bar chart
        plt.clear_figure()
        plt.bar(metrics, values, orientation="h", width=0.3)
        plt.title(f"Privacy Attack Accuracy - {dataset_name}")
        plt.xlabel("Accuracy")
        plt.ylabel("Attack Type")
        plt.xlim(0, 1.0)  # Accuracy is between 0 and 1
        plt.theme("pro")
        plt.plotsize(100, 20)
        plt.show()
        console.print()
