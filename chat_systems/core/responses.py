class ResponseEvaluator:
    """Evaluates model responses against correct answers"""

    def __init__(self, config):
        """
        Initialize response evaluator.

        Args:
            config: ExperimentConfig instance
        """
        self.config = config

    def eval_response(self, response, answers, all_choices, heading=None, all_descriptions=None, context=None):
        """
        Evaluate model response and extract choice.

        Args:
            response: Model's text response
            answers: List of correct answer labels (e.g., ["(1)", "(2)"])
            all_choices: All possible choice labels (e.g., ["(1)", "(2)", "(3)", "(4)"])
            heading: Column name for result tracking (e.g., "Response#1")
            all_descriptions: Descriptions corresponding to choices
            context: Context dict with concept_result, param, file indices, etc.

        Returns:
            bool: True if response matches correct answer(s)
        """
        # Find all choices present in response
        all_available_choices = {}
        for choice in all_choices:
            if choice in response:
                all_available_choices[choice] = response.index(choice)

        if len(all_available_choices) == 0:
            return False

        # Get earliest choice
        extracted_choice = min(all_available_choices, key=all_available_choices.get)

        # Store extracted response if heading and context provided
        if heading is not None and context is not None:
            concept_result = context['concept_result']

            if heading == "Response#3":
                # Special handling for extrapolation response (A/B/C/D)
                alpha = ['A', 'B', 'C', 'D']
                correct_file_index = context['correct_file_index']
                incorrect_file_index = context['incorrect_file_index']
                no_change_file_index = context['no_change_file_index']
                param = context['param']
                stimuli_mc_1 = context['stimuli_mc_1']

                if alpha[correct_file_index] in extracted_choice:
                    concept_result[heading].append(param)
                elif alpha[incorrect_file_index] in extracted_choice:
                    concept_result[heading].append(stimuli_mc_1)
                elif alpha[no_change_file_index] in extracted_choice:
                    concept_result[heading].append("No change")
                elif "D" in extracted_choice:
                    concept_result[heading].append("Doesn't apply")
            else:
                # Regular MCQ response (Response#1 or Response#2)
                extracted_index = int(extracted_choice[1]) - 1
                if 0 <= extracted_index < len(all_descriptions):
                    extracted_choice_description = all_descriptions[extracted_index]
                    concept_result[heading].append(extracted_choice_description)

        # Check if extracted choice matches any correct answer
        for answer in answers:
            if answer == extracted_choice:
                return True

        return False

    def post_process_mantis_response(self, response):
        """
        Post-process Mantis model response.

        Args:
            response: Raw response string from Mantis

        Returns:
            Processed response string
        """
        response = response.replace("Answer:", "")
        response = response.replace("Answer", "")
        response = response.strip()
        response = response.replace("\n", "")

        if response and response[0] != "(":
            response = "(" + response + ")"
        response = response.replace(" ", "")

        return response
