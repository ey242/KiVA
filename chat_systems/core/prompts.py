from .config import ImageMode, TaskVariant


class PromptManager:
    """
    Manages all prompt templates for KiVA experiments.
    Customizes prompts based on image mode, task variant, and concept.
    """

    def __init__(self, image_mode, task_variant, concept, concept_mapper=None):
        """
        Initialize prompt manager.

        Args:
            image_mode: ImageMode enum (SINGLE or MULTI)
            task_variant: TaskVariant enum
            concept: Concept name (e.g., "2DRotation")
            concept_mapper: ConceptMapper instance (needed for CORRECTCROSS variant)
        """
        self.image_mode = image_mode
        self.task_variant = task_variant
        self.concept = concept
        self.concept_mapper = concept_mapper
        self.step_by_step_text = "step-by-step"

        # Build all prompts
        self.system_prompt = self._build_system_prompt()
        self.initi_prompt = self._build_initial_prompt()
        self.general_cross_rule_prompt = self._build_cross_rule_prompt()
        self.general_within_rule_prompt = self._build_within_rule_prompt()
        self.extrapolation_prompt = self._build_extrapolation_prompt()

    def _build_system_prompt(self):
        """Build system prompt (universal across all variants)"""
        prompt = "You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning."
        prompt += f"You will think {self.step_by_step_text} and carefully examine the visual evidence before providing an answer."
        return prompt

    def _build_initial_prompt(self):
        """Build initial prompt based on image mode and task variant"""
        if self.image_mode == ImageMode.SINGLE:
            # Single-image mode
            if self.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
                return ("You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right"
                       "transformations of a different object on the bottom marked by (A) or (B) or (C)."
                       "The transformations involve a change in either the size, orientation, number, or color of an object")
            else:
                return ("You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right"
                       "transformations of a different object on the bottom marked by (A) or (B) or (C)."
                       "The transformations involve a change of either the size, orientation, number, or color of an object")
        else:
            # Multi-image mode
            if self.task_variant == TaskVariant.CORRECTCROSS:
                # Special case: inject correct cross-domain answer
                if self.concept_mapper:
                    cross_domain = self.concept_mapper.get_cross_domain(self.concept).lower()
                    return (f"Observe the left-to-right transformation of an object. The object picture on the left transforms to the object picture on the right."
                           f"Denote this transformation as training transformation. The left-to-right transformation involves the {cross_domain}. ")
                else:
                    # Fallback if concept_mapper not provided
                    return ("Observe the left-to-right transformation of an object. The object picture on the left transforms to the object picture on the right."
                           "Denote this transformation as training transformation. The transformation involves a change of either the size, orientation, number, or color of an object")
            else:
                return ("Observe the left-to-right transformation of an object. The object picture on the left transforms to the object picture on the right."
                       "Denote this transformation as training transformation. The transformation involves a change of either the size, orientation, number, or color of an object")

    def _build_cross_rule_prompt(self):
        """Build cross-domain rule prompt"""
        if self.image_mode == ImageMode.SINGLE:
            prompt = self.initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation on top of the"
                                        "puzzle where the picture on the left transforms to the picture on the right? In your answer start with the correct rule number")
        else:  # MULTI
            prompt = self.initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation"
                                        "where the picture on the left transforms to the picture on the right? In your answer start with the correct rule number")
        prompt += f"surrounded by parentheses, then provide a {self.step_by_step_text} reasoning for your choice."
        return prompt

    def _build_within_rule_prompt(self):
        """Build within-domain rule prompt"""
        if self.image_mode == ImageMode.SINGLE:
            prompt = ("Which one of the following rules {} best describes the left-to-right transformation in the top of the puzzle where the picture"
                     "on the left transforms to the picture on the right?. In your answer start with the correct rule number surrounded by parentheses,")
        else:  # MULTI
            if self.task_variant == TaskVariant.CORRECTCROSS:
                # CORRECTCROSS uses modified initial prompt
                prompt = self.initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation where the picture"
                                            "on the left transforms to the picture on the right?. In your answer start with the correct rule number (1) or (2) or (3) or (4) surrounded by parentheses,")
            else:
                prompt = ("Which one of the following rules {} best describes the left-to-right transformation where the picture"
                         "on the left transforms to the picture on the right?. In your answer start with the correct rule number surrounded by parentheses,")
        prompt += f"then provide a {self.step_by_step_text} reasoning for your choice."
        return prompt

    def _build_extrapolation_prompt(self):
        """Build extrapolation prompt"""
        if self.image_mode == ImageMode.SINGLE:
            if self.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
                prompt = (self.initi_prompt +
                         "Which one of the three left-to-right object transformations (marked by either (A), (B) or (C)) on the bottom of the puzzle is"
                         "the same as the left-to-right transformation on the top of the puzzle?"
                         "In your answer start with the correct letter surrounded by parentheses (or (D) if none of the options apply), ")
            else:
                prompt = ("Which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is"
                         "the same as the left-to-right transformation on the top of the puzzle?"
                         "In your answer start with the correct letter surrounded by parentheses (or (D) if none of the options apply), ")
        else:  # MULTI
            if self.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
                prompt = (self.initi_prompt +
                         "Now you are given three images. Each image contains a left-to-right object transformations (marked by either (A), (B) or (C))."
                         "Which one of these three left-to-right transformations follows the training transformation?"
                         "In your answer start with the correct transformation letter first (A) or (B) or (C). Answer with (D) if none of the options apply.")
            else:
                prompt = ("Now you are given three images. Each image contains a left-to-right object transformations (marked by either (A), (B) or (C) ). "
                         "Which one of these three left-to-right transformations follows the identified transformation. "
                         "In your answer start with the correct transformation letter first (A) or (B) or (C). Answer with (D) if none of options apply.")
        prompt += f"then provide a {self.step_by_step_text} reasoning for your choice."
        return prompt

    def get_mcq_questions(self):
        """
        Return which MCQ questions to ask based on task variant.

        Returns:
            List of question types to ask: 'cross', 'within', 'extrapolation'
        """
        if self.task_variant == TaskVariant.FULL:
            return ['cross', 'within', 'extrapolation']
        elif self.task_variant == TaskVariant.CORRECTCROSS:
            return ['within']  # Only within-domain (cross is given in prompt)
        elif self.task_variant == TaskVariant.CORRECTWITHIN:
            return ['cross', 'extrapolation']  # Skip within-domain
        elif self.task_variant == TaskVariant.NOCHANGE:
            return ['cross', 'within', 'extrapolation']  # All three
        elif self.task_variant == TaskVariant.EXTRAPOLATION_ONLY:
            return ['extrapolation']  # Only extrapolation
        return []
