#!/usr/bin/env python3
"""
KiVA Experiment Runner - Unified Entry Point

This script replaces 21 separate chat_system files with a single configurable interface.
Consolidates all visual analogical reasoning experiments into one parameterized system.

Usage:
    python run_experiment.py --concept 2DRotation --model gpt4o --image-mode multi --dataset kiva

Examples:
    # Full evaluation with GPT-4o on KiVA dataset
    python run_experiment.py --concept 2DRotation --model gpt4o --image-mode multi --dataset kiva

    # Correctcross ablation with Mantis
    python run_experiment.py --concept Colour --model mantis --image-mode multi --dataset kiva --task-variant correctcross

    # KiVA-adults evaluation
    python run_experiment.py --concept Counting --model gpt4o --image-mode single --dataset kiva-adults
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_systems.core import create_argument_parser, ExperimentConfig, TrialRunner


def main():
    """Main entry point for KiVA experiments"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configuration
    try:
        config = ExperimentConfig(args)
    except ValueError as e:
        print(f"\nConfiguration error: {e}\n")
        parser.print_help()
        return 1

    # Display configuration
    print("=" * 60)
    print("KiVA Experiment Configuration")
    print("=" * 60)
    print(f"Concept:        {config.concept}")
    print(f"Model:          {config.model_type.value}")
    print(f"Image Mode:     {config.image_mode.value}")
    print(f"Dataset:        {config.dataset.value}")
    print(f"Task Variant:   {config.task_variant.value}")
    print(f"Query Repeats:  {config.query_repeats if config.query_repeats else 'Max available'}")
    print(f"\nOutput Directory: {config.output_directory}")
    print("=" * 60)
    print()

    # Run experiment
    try:
        runner = TrialRunner(config)
        runner.run()
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nExperiment failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
