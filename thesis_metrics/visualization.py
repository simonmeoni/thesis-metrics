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
        """Display bar chart of metrics in terminal"""

        # Filter numeric columns only
        numeric_cols = results_df.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) == 0:
            return

        # Get metric names and values
        metrics = [col for col in numeric_cols if col != "dataset"]
        values = [results_df[col].iloc[0] for col in metrics]

        # Create horizontal bar chart
        plt.clear_figure()
        plt.bar(metrics, values, orientation="h", width=0.3)
        plt.title(f"Privacy Attack Results - {dataset_name}")
        plt.xlabel("Score")
        plt.ylabel("Attack")
        plt.theme("pro")
        plt.plotsize(100, 20)
        plt.show()
        console.print()
