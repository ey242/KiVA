import os
import random
from .config import TaskVariant


class StimulusManager:
    """Manages stimulus file discovery, loading, and stitching"""

    def __init__(self, config, concept_mapper):
        """
        Initialize stimulus manager.

        Args:
            config: ExperimentConfig instance
            concept_mapper: ConceptMapper instance
        """
        self.config = config
        self.concept_mapper = concept_mapper

        if config.is_multi_image():
            from utils_multi import stitch_images_train, stitch_images_test, read_image
            self.stitch_train = stitch_images_train
            self.stitch_test = stitch_images_test
        else:
            from utils_single import stitch_images_train, stitch_images_test, read_image, stitch_final_images
            self.stitch_train = stitch_images_train
            self.stitch_test = stitch_images_test
            self.stitch_final = stitch_final_images

        self.read_image = read_image

    def get_indexed_files(self, param):
        """
        Get indexed files for a parameter.

        Args:
            param: Parameter string (e.g., "+90", "Red")

        Returns:
            Dictionary mapping indices to file lists
        """
        indexed_files = {}
        beginning = self.config.concept + str(param)

        for filename in os.listdir(self.config.stimuli_directory):
            if filename.startswith(beginning + "_"):
                index = int(filename.split('_')[1])
                if index not in indexed_files:
                    indexed_files[index] = []
                indexed_files[index].append(filename)

        return indexed_files

    def format_files_by_type(self, indexed_files, index, file_type):
        """
        Format files by type (train/test).

        Args:
            indexed_files: Dictionary mapping indices to file lists
            index: Index to extract
            file_type: 'train' or 'test'

        Returns:
            List of formatted files
        """
        train_files = [filename for filename in indexed_files[index] if 'train' in filename]

        if file_type == 'train':
            # Create pairs of input and output files
            input_filename = None
            output_filename = None
            for filename in train_files:
                if 'input' in filename:
                    input_filename = filename
                elif 'output' in filename:
                    output_filename = filename
            return [input_filename, output_filename]

        elif file_type == 'test':
            test_files = [filename for filename in indexed_files[index] if 'test' in filename]
            return sorted(test_files)

    def prepare_train_stimuli(self, stimuli_set, query, param, regeneration):
        """
        Prepare training stimuli images.

        Args:
            stimuli_set: Dictionary of indexed files
            query: Query/variation number
            param: Parameter string
            regeneration: Regeneration number

        Returns:
            Tuple of (train_image, train_image_path)
        """
        train_stimuli_set = self.format_files_by_type(stimuli_set, query, 'train')

        # For nochange variant, use same image twice
        if self.config.task_variant == TaskVariant.NOCHANGE:
            input_img = self.read_image(f"{self.config.stimuli_directory}/{train_stimuli_set[0]}").convert("RGB")
            if self.config.is_multi_image():
                train_image = self.stitch_train(input_img, input_img)
            else:
                train_image = self.stitch_train(input_img, input_img, case_num=1)
        else:
            input_img = self.read_image(f"{self.config.stimuli_directory}/{train_stimuli_set[0]}").convert("RGB")
            output_img = self.read_image(f"{self.config.stimuli_directory}/{train_stimuli_set[1]}").convert("RGB")
            if self.config.is_multi_image():
                train_image = self.stitch_train(input_img, output_img)
            else:
                train_image = self.stitch_train(input_img, output_img, case_num=1)

        # Save stitched train image
        train_image_path = f"{self.config.stitched_images_directory}/{self.config.concept}{param}_{query}_{regeneration}_train.jpg"
        train_image.save(train_image_path)

        return train_image, train_image_path

    def prepare_test_stimuli(self, stimuli_set, query, param, regeneration):
        """
        Prepare test stimuli images and shuffle.

        Args:
            stimuli_set: Dictionary of indexed files
            query: Query/variation number
            param: Parameter string
            regeneration: Regeneration number

        Returns:
            For multi-image: (test_image_paths, test_stimuli_set, (correct_idx, incorrect_idx, nochange_idx))
            For single-image: (test_image, test_stimuli_set, (correct_idx, incorrect_idx, nochange_idx))
        """
        test_stimuli_set = self.format_files_by_type(stimuli_set, query, 'test')
        test_stimuli_input = test_stimuli_set[0]
        test_stimuli_outputs = test_stimuli_set[1:]

        # Append input to outputs
        test_stimuli_outputs.append(test_stimuli_input)

        # Identify files before shuffling
        correct_file = test_stimuli_outputs[0]
        incorrect_param_file = test_stimuli_outputs[1]
        no_change_file = test_stimuli_outputs[2]

        # Shuffle
        random.shuffle(test_stimuli_outputs)

        # Track indices after shuffling
        correct_file_index = test_stimuli_outputs.index(correct_file)
        incorrect_file_index = test_stimuli_outputs.index(incorrect_param_file)
        no_change_file_index = test_stimuli_outputs.index(no_change_file)

        # Stitch test images
        stitched_images = [self.read_image(f"{self.config.stimuli_directory}/{test_stimuli_input}").convert("RGB")]
        for test_stimuli in test_stimuli_outputs:
            img = self.read_image(f"{self.config.stimuli_directory}/{test_stimuli}").convert("RGB")
            stitched_images.append(img)

        if self.config.is_multi_image():
            # Multi-image: create separate images for each test option
            stitched_test_images = self.stitch_test(stitched_images)
            test_stimuli_image_paths = []
            for num, test_img in enumerate(stitched_test_images):
                path = f"{self.config.stitched_images_directory}/{self.config.concept}{param}_{query}_{regeneration}_test{num}.jpg"
                test_img.save(path)
                test_stimuli_image_paths.append(path)
            return test_stimuli_image_paths, test_stimuli_outputs, (correct_file_index, incorrect_file_index, no_change_file_index)
        else:
            # Single-image: create one composite test image
            test_stimuli_image = self.stitch_test(stitched_images)
            return test_stimuli_image, test_stimuli_outputs, (correct_file_index, incorrect_file_index, no_change_file_index)

    def create_final_single_image(self, train_image, test_image, param, query, regeneration, correct_letter):
        """
        Create final stitched image for single-image mode.

        Args:
            train_image: Training image
            test_image: Test image
            param: Parameter string
            query: Query number
            regeneration: Regeneration number
            correct_letter: Correct answer letter

        Returns:
            Path to final stitched image
        """
        final_image = self.stitch_final(train_image, test_image)
        final_image_path = f"{self.config.stitched_images_directory}/{self.config.concept}{param}_{query}_{regeneration}_{correct_letter}.jpg"
        final_image.save(final_image_path)
        return final_image_path
