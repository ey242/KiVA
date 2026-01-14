import os
import random
from .config import TaskVariant
from .concepts import ConceptMapper
from .prompts import PromptManager
from .models import ModelFactory
from .responses import ResponseEvaluator
from .results import ResultTracker
from .stimuli import StimulusManager


class TrialRunner:
    """Main trial execution"""

    def __init__(self, config):
        """
        Initialize trial runner.

        Args:
            config: ExperimentConfig instance
        """
        self.config = config
        self.concept_mapper = ConceptMapper(config.dataset)
        self.prompt_manager = PromptManager(config.image_mode, config.task_variant, config.concept, self.concept_mapper)
        self.stimulus_manager = StimulusManager(config, self.concept_mapper)
        self.response_evaluator = ResponseEvaluator(config)
        self.result_tracker = ResultTracker(config)

        # Create model with system prompt
        self.model = ModelFactory.create_model(config, self.prompt_manager.system_prompt)

        # Create output directories
        os.makedirs(config.output_directory, exist_ok=True)
        os.makedirs(config.stitched_images_directory, exist_ok=True)

    def run(self):
        """Main execution method"""
        parameters = self.concept_mapper.get_parameters(self.config.concept)

        for param in parameters:
            self._run_parameter(param)

    def _run_parameter(self, param):
        """Run trials for a single parameter"""
        stimuli_set = self.stimulus_manager.get_indexed_files(param)
        output_file = f"{self.config.output_directory}/{self.config.concept}{param}.csv"

        query_repeats = self.config.query_repeats
        if query_repeats is None:
            query_repeats = len(stimuli_set)

        print("----------------------------------------------")
        print(f"Beginning Sub-Concept {self.config.concept} {param}")

        # Sample queries for certain task variants
        if self.config.task_variant in [TaskVariant.CORRECTCROSS, TaskVariant.NOCHANGE] and query_repeats < len(stimuli_set):
            queries = random.sample(range(len(stimuli_set)), query_repeats)
        else:
            queries = list(range(query_repeats))

        for query in queries:
            self._run_query(param, query, stimuli_set, output_file, query_repeats)

    def _run_query(self, param, query, stimuli_set, output_file, query_repeats):
        """Run trials for a single query variation"""
        print("----------------------------------------------")
        print(f"Beginning Variation {query + 1} of {query_repeats}")

        for regeneration in range(3):
            self._run_regeneration(param, query, regeneration, stimuli_set, output_file, query_repeats)

    def _run_regeneration(self, param, query, regeneration, stimuli_set, output_file, query_repeats):
        """Run a single regeneration with retry logic"""
        # Check if already processed
        if self.result_tracker.check_already_processed(output_file, query, regeneration):
            print(f"Skipping Variation {query + 1} of {query_repeats}, already exists.")
            return

        retry_count = 0
        regeneration_successful = False

        while retry_count < 2 and not regeneration_successful:
            print("-----------------------------")
            print(f"Beginning Regeneration {regeneration + 1} of 3")

            # Initialize
            self.model.init_history()
            concept_result = self.result_tracker.init_concept_result()

            # Load trial metadata
            trial_metadata = self._load_trial_metadata(param, query)

            # Add metadata to results
            concept_result["Variation"].append(query)
            concept_result["Regeneration"].append(regeneration)
            concept_result["Train_input"].append(trial_metadata['train_input'])
            concept_result["Train_output"].append(trial_metadata['train_output'])

            if "Test_input" in concept_result:
                concept_result["Test_input"].append(trial_metadata['test_input'])
                concept_result["Test_output"].append(trial_metadata['test_output'])

            # Prepare stimuli
            train_image, train_image_path = self.stimulus_manager.prepare_train_stimuli(
                stimuli_set, query, param, regeneration
            )

            test_data = self.stimulus_manager.prepare_test_stimuli(
                stimuli_set, query, param, regeneration
            )

            # Unpack test data
            if self.config.is_multi_image():
                test_stimuli_image_paths, test_stimuli_outputs, (correct_file_index, incorrect_file_index, no_change_file_index) = test_data
                test_image = None
            else:
                test_image, test_stimuli_outputs, (correct_file_index, incorrect_file_index, no_change_file_index) = test_data
                test_stimuli_image_paths = None

            # Execute MCQs
            self._execute_mcqs(
                param, query, regeneration, train_image, train_image_path, test_image, test_stimuli_image_paths,
                trial_metadata, concept_result, correct_file_index, incorrect_file_index, no_change_file_index
            )

            # Check if regeneration successful (no Null responses)
            null_responses = [val for key, val in concept_result.items()
                            if key.startswith("MCResponse") and val and val[-1] == "Null"]

            if len(null_responses) == 0:
                regeneration_successful = True
            elif retry_count == 2:
                regeneration_successful = True
                print("Max retries reached, proceeding despite null responses")
            else:
                retry_count += 1
                print(f"Retrying due to null response; this is try {retry_count + 1} of 3.")

        # Save results
        self.result_tracker.save_results(concept_result, output_file)
        print("="*20)

    def _load_trial_metadata(self, param, query):
        """Load trial metadata from text file"""
        with open(f"{self.config.text_files_dir}/output_{self.config.concept}{param}.txt", "r") as file:
            lines = file.readlines()

            train_input = lines[0 + (query * 4)].rstrip().split(": ")[1]
            train_output_orig = lines[1 + (query * 4)].rstrip().split(": ")[1]
            test_input = lines[2 + (query * 4)].rstrip().split(": ")[1]
            mc_1 = lines[3 + (query * 4)].rstrip().split(": ")[1]

        # Adjust for nochange variant
        if self.config.task_variant == TaskVariant.NOCHANGE:
            train_output = train_input
        else:
            train_output = train_output_orig

        # Adjust mc_1 for counting concepts
        stimuli_mc_1 = mc_1
        if self.config.concept == "Counting":
            counting_type, option = param[0], param[1:]
            if counting_type == "+":
                stimuli_mc_1 = "-1"
                mc_1 = "-1" if option == "1" else "-2"
            elif counting_type == "-":
                stimuli_mc_1 = "+1"
                mc_1 = "+1" if option == "1" else "+2"
            elif counting_type == "x":
                stimuli_mc_1 = "+1"
                mc_1 = "d2" if option == "2" else "d3"
            elif counting_type == "d":
                stimuli_mc_1 = "-1"
                mc_1 = "x2" if option == "2" else "x3"

        # Calculate test output
        test_output = self._calculate_test_output(param, test_input, train_input)

        return {
            'train_input': train_input,
            'train_output': train_output,
            'test_input': test_input,
            'test_output': test_output,
            'mc_1': mc_1,
            'stimuli_mc_1': stimuli_mc_1
        }

    def _calculate_test_output(self, param, test_input, train_input):
        """Calculate expected test output"""
        if self.config.concept == "2DRotation":
            if self.config.is_kiva_adults() and param != "180":
                return self.concept_mapper.add_angles(test_input, param)
            elif param == "+90":
                return str(int(test_input) + 90)
            else:
                return str(param)
        elif self.config.concept == "Counting":
            counting_type, option = param[0], param[1:]
            if counting_type == "+":
                return str(int(test_input) + int(option))
            elif counting_type == "-":
                return str(int(test_input) - int(option))
            elif counting_type == "x":
                return str(int(test_input) * int(option))
            elif counting_type == "d":
                return str(float(test_input) / int(option))
        elif self.config.concept == "Counting" and self.config.task_variant == TaskVariant.NOCHANGE:
            return test_input
        else:
            return str(param)

    def _execute_mcqs(self, param, query, regeneration, train_image, train_image_path, test_image, test_stimuli_image_paths,
                      trial_metadata, concept_result, correct_file_index, incorrect_file_index, no_change_file_index):
        """Execute all MCQ questions based on task variant"""
        questions = self.prompt_manager.get_mcq_questions()

        add_result = True  # Track if should continue to next questions

        if 'cross' in questions:
            add_result = self._execute_cross_domain_mcq(
                param, train_image, train_image_path, test_image, concept_result
            )

        if 'within' in questions and add_result:
            self._execute_within_domain_mcq(
                param, train_image_path, test_stimuli_image_paths, trial_metadata, concept_result
            )
        elif 'within' not in questions and self.config.task_variant != TaskVariant.CORRECTCROSS:
            # Skip within for certain variants
            if "Full#2" in concept_result:
                concept_result["Full#2"].append("[Skipped mcq#2, task variant]")
                concept_result["MCResponse#2"].append("")
                concept_result["Response#2"].append("")

        if 'extrapolation' in questions:
            self._execute_extrapolation_mcq(
                param, test_stimuli_image_paths, trial_metadata, concept_result,
                correct_file_index, incorrect_file_index, no_change_file_index
            )

    def _execute_cross_domain_mcq(self, param, train_image, train_image_path, test_image, concept_result):
        """Execute cross-domain MCQ (MCQ#1)"""
        correct_answer = self.concept_mapper.get_cross_domain(self.config.concept)

        # For nochange variant, correct answer is "No change between pictures"
        if self.config.task_variant == TaskVariant.NOCHANGE:
            target_answer = "No change between pictures"
        else:
            target_answer = correct_answer

        # Build choices
        all_choices = ["Number of objects", "Size of objects", "Orientation of objects", "Color of objects"]
        all_choices.remove(correct_answer)
        selected_choices = random.sample(all_choices, 2)
        selected_choices.append(correct_answer)
        random.shuffle(selected_choices)
        selected_choices.extend(["No change between pictures", "Doesn't apply"])

        # Label choices
        labels = ["(1) ", "(2) ", "(3) ", "(4) ", "(5) "]
        labeled_choices = [label + choice for label, choice in zip(labels, selected_choices)]

        # Build prompt
        str_prompt = "\n".join(labeled_choices) + "\n"
        full_prompt = self.prompt_manager.general_cross_rule_prompt.format(str_prompt)

        # Get model response
        if self.config.is_multi_image():
            response = self.model.run_model_indiv(full_prompt, train_image, train_image_path)
            self.final_image_path = None  # Not used for multi-image
        else:
            # For single-image, need to create final image first
            alpha = ['A', 'B', 'C', 'D']
            final_image_path = self.stimulus_manager.create_final_single_image(
                train_image, test_image, param, 0, 0, "temp"  # Use temp values for query/regeneration
            )
            self.final_image_path = final_image_path  # Store for later use in extrapolation
            response = self.model.run_model(full_prompt, final_image_path)

        # Post-process for Mantis
        if self.config.is_mantis():
            response['response'] = self.response_evaluator.post_process_mantis_response(response['response'])

        if "Full#1" in concept_result:
            concept_result["Full#1"].append(response["response"])
        elif "Full" in concept_result:
            concept_result["Full"].append(response["response"])
        print("Cross Domain Response:", response["response"])

        # Evaluate
        all_choice_labels = [choice.split(" ")[0] for choice in labeled_choices]
        labeled_correct = [choice.split(" ")[0] for choice in labeled_choices if target_answer in choice][0]
        labeled_incorrect = [choice.split(" ")[0] for choice in labeled_choices if target_answer not in choice]

        context = {'concept_result': concept_result}
        mcq_field = "MCResponse#1" if "MCResponse#1" in concept_result else "MCResponse"
        response_field = "Response#1" if "Response#1" in concept_result else "Response"

        if self.response_evaluator.eval_response(response["response"], [labeled_correct], all_choice_labels, response_field, selected_choices, context):
            concept_result[mcq_field].append("1")
            print("Correct cross response")
            return True
        elif self.response_evaluator.eval_response(response["response"], labeled_incorrect, all_choice_labels):
            concept_result[mcq_field].append("0")
            print("Incorrect cross response")
            return False
        else:
            concept_result[mcq_field].append("Null")
            concept_result[response_field].append("Null")
            print("Uncertain cross response")
            return True

    def _execute_within_domain_mcq(self, param, train_image_path, test_stimuli_image_paths, trial_metadata, concept_result):
        """Execute within-domain MCQ (MCQ#2)"""
        # Build choices based on concept
        params_for_concept = self.concept_mapper.get_parameters(self.config.concept)
        params_for_concept.remove(param) if param in params_for_concept else None

        selected_within_choices = random.sample(params_for_concept, min(2, len(params_for_concept)))
        selected_within_choices.append(param)
        random.shuffle(selected_within_choices)
        selected_within_choices.append(trial_metadata['mc_1'])

        # Convert to worded options
        worded_within_choices = self.concept_mapper.word_mc_options(self.config.concept, selected_within_choices)

        # Label choices
        labels = ["(1) ", "(2) ", "(3) ", "(4) "]
        labeled_within_choices = [label + choice for label, choice in zip(labels, worded_within_choices)]

        # Build prompt
        str_prompt = "\n".join(labeled_within_choices) + "\n"
        full_prompt = self.prompt_manager.general_within_rule_prompt.format(str_prompt)

        # Get model response
        if self.config.is_multi_image():
            from utils_multi import read_image
            train_image = read_image(train_image_path).convert("RGB")
            response = self.model.run_model_indiv(full_prompt, train_image, train_image_path)
        else:
            response = self.model.run_model(full_prompt, train_image_path)  # Should pass final image path

        # Post-process for Mantis
        if self.config.is_mantis():
            response['response'] = self.response_evaluator.post_process_mantis_response(response['response'])

        if "Full#2" in concept_result:
            concept_result["Full#2"].append(response["response"])
        elif "Full" in concept_result:
            # For CORRECTCROSS, only one "Full" field
            pass
        print("Within Domain Response:", response["response"])

        # Evaluate
        all_choice_labels = [choice.split(" ")[0] for choice in labeled_within_choices]
        worded_param = self.concept_mapper.word_mc_options(self.config.concept, [param])[0]
        labeled_correct = [choice.split(" ")[0] for choice in labeled_within_choices if worded_param in choice][0]

        context = {'concept_result': concept_result}
        mcq_field = "MCResponse#2" if "MCResponse#2" in concept_result else "MCResponse"
        response_field = "Response#2" if "Response#2" in concept_result else "Response"

        if self.response_evaluator.eval_response(response["response"], [labeled_correct], all_choice_labels, response_field, worded_within_choices, context):
            concept_result[mcq_field].append("1")
            print("Correct within response")
        elif self.response_evaluator.eval_response(response["response"], [lbl for lbl in all_choice_labels if lbl != labeled_correct], all_choice_labels):
            concept_result[mcq_field].append("0")
            print("Incorrect within response")
        else:
            concept_result[mcq_field].append("Null")
            concept_result[response_field].append("Null")
            print("Uncertain within response")

    def _execute_extrapolation_mcq(self, param, test_stimuli_image_paths, trial_metadata, concept_result,
                                   correct_file_index, incorrect_file_index, no_change_file_index):
        """Execute extrapolation MCQ (MCQ#3)"""
        # Build prompt
        full_prompt = self.prompt_manager.extrapolation_prompt

        # Get model response
        if self.config.is_multi_image():
            from utils_multi import read_image
            test_images = [read_image(path).convert("RGB") for path in test_stimuli_image_paths]
            response = self.model.run_model_multi(full_prompt, test_images, test_stimuli_image_paths)
        else:
            # For single image, use the final_image_path created in cross-domain MCQ
            if hasattr(self, 'final_image_path') and self.final_image_path:
                response = self.model.run_model(full_prompt, self.final_image_path)
            else:
                # Fallback: should not happen if cross-domain executed first
                raise RuntimeError("Single-image mode requires final_image_path from cross-domain MCQ")

        # Post-process for Mantis
        if self.config.is_mantis():
            response['response'] = self.response_evaluator.post_process_mantis_response(response['response'])

        if "Full#3" in concept_result:
            concept_result["Full#3"].append(response["response"])
        elif "Full" in concept_result and self.config.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
            concept_result["Full"].append(response["response"])
        print("Extrapolation Response:", response["response"])

        # Evaluate
        alpha = ['A', 'B', 'C', 'D']
        all_choice_labels = [f"({letter})" for letter in alpha]
        labeled_correct = f"({alpha[correct_file_index]})"

        context = {
            'concept_result': concept_result,
            'param': param,
            'correct_file_index': correct_file_index,
            'incorrect_file_index': incorrect_file_index,
            'no_change_file_index': no_change_file_index,
            'stimuli_mc_1': trial_metadata['stimuli_mc_1']
        }

        mcq_field = "MCResponse#3" if "MCResponse#3" in concept_result else "Score"
        response_field = "Response#3" if "Response#3" in concept_result else "Transformation Type Selected"

        if self.response_evaluator.eval_response(response["response"], [labeled_correct], all_choice_labels, response_field, None, context):
            concept_result[mcq_field].append("1")
            print("Correct extrapolation response")
        elif self.response_evaluator.eval_response(response["response"], [lbl for lbl in all_choice_labels if lbl != labeled_correct], all_choice_labels, response_field, None, context):
            concept_result[mcq_field].append("0")
            print("Incorrect extrapolation response")
        else:
            concept_result[mcq_field].append("Null")
            if response_field in concept_result:
                concept_result[response_field].append("Null")
            print("Uncertain extrapolation response")
