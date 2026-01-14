import os
import pandas as pd
from .config import TaskVariant


class ResultTracker:
    """Manages CSV output and result tracking"""

    def __init__(self, config):
        """
        Initialize result tracker.

        Args:
            config: ExperimentConfig instance
        """
        self.config = config

    def get_headers(self):
        """Get CSV headers based on task variant"""
        if self.config.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
            return [
                "Variation",
                "Regeneration",
                "Train_input",
                "Train_output",
                "Test_input",
                "Test_output",
                "Full",
                "Score",
                "Transformation Type Selected"
            ]
        elif self.config.task_variant == TaskVariant.CORRECTCROSS:
            return [
                "Variation",
                "Regeneration",
                "Train_input",
                "Train_output",
                "Full",
                "MCResponse",
                "Response"
            ]
        else:  # FULL, CORRECTWITHIN, NOCHANGE
            return [
                "Variation",
                "Regeneration",
                "Train_input",
                "Train_output",
                "Test_input",
                "Test_output",
                "Full#1",
                "Full#2",
                "Full#3",
                "MCResponse#1",
                "MCResponse#2",
                "MCResponse#3",
                "Response#1",
                "Response#2",
                "Response#3"
            ]

    def init_concept_result(self):
        """Initialize empty result dictionary"""
        headers = self.get_headers()
        return {header: [] for header in headers}

    def save_results(self, concept_result, output_file):
        """
        Save results to CSV, appending if file exists.

        Args:
            concept_result: Dictionary of results to save
            output_file: Path to output CSV file
        """
        df_to_add = pd.DataFrame(concept_result)

        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df = pd.concat([df, df_to_add], ignore_index=True)
            df.to_csv(output_file, index=False)
        else:
            df_to_add.to_csv(output_file, index=False)

    def check_already_processed(self, output_file, variation, regeneration):
        """
        Check if variation/regeneration combo already processed.

        Args:
            output_file: Path to output CSV file
            variation: Variation number
            regeneration: Regeneration number

        Returns:
            bool: True if already processed
        """
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            return len(df[(df["Variation"] == variation) & (df["Regeneration"] == regeneration)]) > 0
        return False
