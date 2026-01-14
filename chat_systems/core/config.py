import argparse
import os
from enum import Enum


class ImageMode(Enum):
    SINGLE = "single"
    MULTI = "multi"


class Dataset(Enum):
    KIVA = "kiva"
    KIVA_ADULTS = "kiva-adults"


class ModelType(Enum):
    GPT4 = "gpt4"
    GPT4O = "gpt4o"
    LLAVA = "llava"
    MANTIS = "mantis"


class TaskVariant(Enum):
    FULL = "full"
    CORRECTCROSS = "correctcross"
    CORRECTWITHIN = "correctwithin"
    NOCHANGE = "nochange"
    EXTRAPOLATION_ONLY = "extrapolation_only"


class ExperimentConfig:
    """
    Main configuration class for KiVA experiments.
    """

    def __init__(self, args):
        self.image_mode = ImageMode(args.image_mode)
        self.dataset = Dataset(args.dataset)
        self.model_type = ModelType(args.model)
        self.task_variant = TaskVariant(args.task_variant)
        self.concept = args.concept
        self.api_key = args.api_key
        self.query_repeats = args.query_repeats

        # Validate configuration
        self._validate()

        # Set up all paths
        self._setup_paths()

    def _validate(self):
        """Validate configuration combinations"""
        # Mantis only works with multi-image
        if self.model_type == ModelType.MANTIS and self.image_mode != ImageMode.MULTI:
            raise ValueError("Mantis model requires multi-image mode (--image-mode multi)")

        # LLaVA only works with single-image
        if self.model_type == ModelType.LLAVA and self.image_mode != ImageMode.SINGLE:
            raise ValueError("LLaVA model requires single-image mode (--image-mode single)")

    def _setup_paths(self):
        """Configure all directory paths based on settings"""
        # Dataset name for paths
        if self.dataset == Dataset.KIVA_ADULTS:
            dataset_name = "KiVA-adults"
        else:
            dataset_name = "KiVA"

        # Image type for paths
        if self.image_mode == ImageMode.SINGLE:
            image_type = "single_image"
        else:
            image_type = "multi_image"

        # Stimuli directories
        self.stimuli_directory = f"stimuli/{dataset_name}/{self.concept}"
        self.text_files_dir = f"stimuli/{dataset_name}/trial_tracker/"

        # Output directory varies by task variant
        if self.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
            base_output = f"output/extrapolation_only/{image_type}"
        elif self.task_variant in [TaskVariant.CORRECTCROSS, TaskVariant.CORRECTWITHIN, TaskVariant.NOCHANGE]:
            base_output = f"output/rebuttal_{self.task_variant.value}/{image_type}"
        else:
            base_output = f"output/{image_type}"

        # Adjust for KiVA-adults dataset with single-image mode
        if self.dataset == Dataset.KIVA_ADULTS and self.image_mode == ImageMode.SINGLE:
            base_output += "_adults"

        # Model-specific output suffix
        model_suffix = "mantis" if self.model_type == ModelType.MANTIS else self.model_type.value

        self.output_directory = f"{base_output}/output_{model_suffix}/{self.concept}"
        self.stitched_images_directory = f"{self.output_directory}/{self.concept}_stitch"

    def is_mantis(self):
        return self.model_type == ModelType.MANTIS

    def is_multi_image(self):
        return self.image_mode == ImageMode.MULTI

    def is_kiva_adults(self):
        return self.dataset == Dataset.KIVA_ADULTS


def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='KiVA Experiment Runner - Unified entry point for visual analogical reasoning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Full evaluation with GPT-4o on KiVA dataset
        python run_experiment.py --concept 2DRotation --model gpt4o --image-mode multi --dataset kiva

        # Correctcross ablation with Mantis
        python run_experiment.py --concept Colour --model mantis --image-mode multi --dataset kiva --task-variant correctcross

        # KiVA-adults evaluation
        python run_experiment.py --concept Counting --model gpt4o --image-mode single --dataset kiva-adults
        """
    )

    parser.add_argument(
        '--concept',
        type=str,
        required=True,
        choices=['2DRotation', 'Colour', 'Counting', 'Reflect', 'Resize'],
        help='Concept to test'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt4o',
        choices=['gpt4', 'gpt4o', 'llava', 'mantis'],
        help='Model to use (default: gpt4o)'
    )

    parser.add_argument(
        '--image-mode',
        type=str,
        default='single',
        choices=['single', 'multi'],
        help='Image presentation mode (default: single)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='kiva',
        choices=['kiva', 'kiva-adults'],
        help='Dataset to use (default: kiva)'
    )

    parser.add_argument(
        '--task-variant',
        type=str,
        default='full',
        choices=['full', 'correctcross', 'correctwithin', 'nochange', 'extrapolation_only'],
        help='Task variant for ablation studies (default: full)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default='API-KEY',
        help='API key for OpenAI models (default: API-KEY)'
    )

    parser.add_argument(
        '--query-repeats',
        type=int,
        default=None,
        help='Number of query repeats (default: None for maximum available)'
    )

    return parser
