from .config import Dataset


class ConceptMapper:
    """
    Manages concept-specific logic including parameter mappings,
    cross-domain categorization, and word-based MCQ option formatting.
    """

    def __init__(self, dataset):
        """
        Initialize concept mapper for a specific dataset.

        Args:
            dataset: Dataset enum (KIVA or KIVA_ADULTS)
        """
        self.dataset = dataset
        self.parameters = self._get_parameters()

    def _get_parameters(self):
        """Return parameter mappings based on dataset"""
        if self.dataset == Dataset.KIVA_ADULTS:
            return {
                "2DRotation": ["+45", "-45", "+90", "-90", "+135", "-135", 180],
                "Colour": ["Red", "Yellow", "Green", "Blue", "Grey"],
                "Counting": ["+1", "+2", "-1", "-2", "x2", "x3", "d2", "d3"],
                "Reflect": ["X", "Y", "XY"],
                "Resize": ["0.5X", "0.5Y", "0.5XY", "2X", "2Y", "2XY"]
            }
        else:  # KIVA
            return {
                "2DRotation": ["+90", "-90", 180],
                "Colour": ["Red", "Green", "Blue"],
                "Counting": ["+1", "+2", "-1", "-2"],
                "Reflect": ["X", "Y"],
                "Resize": ["2XY", "0.5XY"]
            }

    def get_parameters(self, concept):
        """Get parameters for a specific concept"""
        return self.parameters[concept]

    def get_cross_domain(self, concept):
        """
        Return correct cross-domain category for a concept.

        Args:
            concept: Concept name (e.g., "2DRotation", "Counting")

        Returns:
            Cross-domain category string
        """
        mapping = {
            "2DRotation": "Orientation of objects",
            "Counting": "Number of objects",
            "Colour": "Color of objects",
            "Reflect": "Orientation of objects",
            "Resize": "Size of objects"
        }
        return mapping[concept]

    def add_angles(self, x, y):
        """
        Add two rotation angles (KiVA-adults specific for 2DRotation).

        Args:
            x: First angle string (e.g., "+90", "-45", "180")
            y: Second angle string (e.g., "+90", "-45", "180")

        Returns:
            Result angle string in same format
        """
        # Parse x
        if x.startswith('+'):
            angle_x = int(x[1:])
        elif x.startswith('-'):
            angle_x = -int(x[1:])
        else:
            angle_x = int(x)

        # Parse y
        if y.startswith('+'):
            angle_y = int(y[1:])
        elif y.startswith('-'):
            angle_y = -int(y[1:])
        else:
            angle_y = int(y)

        result = angle_x + angle_y

        # Format result
        if result == 180:
            result_str = str(result)
        elif result == -180:
            result_str = str(result)[1:]  # Remove negative sign for -180
        else:
            result_str = f"+{result}" if result > 0 else f"{result}"

        return result_str

    def word_mc_options(self, concept, selected_mc_options):
        """
        Convert parameter options to worded descriptions for MCQ.

        Args:
            concept: Concept name (e.g., "2DRotation", "Counting")
            selected_mc_options: List of parameter options to convert

        Returns:
            List of worded descriptions
        """
        worded_options = []

        for option in selected_mc_options:
            if concept == "2DRotation":
                if self.dataset == Dataset.KIVA_ADULTS:
                    # KiVA-adults: specific degree amounts
                    if option == "180" or option == 180:
                        worded_options.append("Objects turn 180 degrees")
                    else:
                        worded_options.append(f"Objects turn {option[1:]} degrees")
                else:
                    # KiVA: simplified (90 or 180)
                    if option == "-90" or option == "+90":
                        worded_options.append("Objects turn 90 degrees")
                    else:
                        worded_options.append("Objects turn 180 degrees")

            elif concept == "Counting":
                counting_type, num = option[0], option[1:]
                if counting_type == "+":
                    worded_options.append(f"Things go up by {num}")
                elif counting_type == "-":
                    worded_options.append(f"Things go down by {num}")
                elif counting_type == "x":
                    worded_options.append(f"Things multiply by {num}")
                elif counting_type == "d":
                    worded_options.append(f"Things divide by {num}")

            elif concept == "Colour":
                worded_options.append(f"Objects turn {option}")

            elif concept == "Reflect":
                if option == "X":
                    worded_options.append("Objects flip upside down")
                elif option == "Y":
                    worded_options.append("Objects flip sideways")
                elif option == "XY":
                    worded_options.append("Objects flip sideways and upside down")

            elif concept == "Resize":
                resize_map = {
                    "0.5X": "Objects become thinner only",
                    "0.5Y": "Objects become shorter only",
                    "0.5XY": "Objects become smaller",
                    "2X": "Objects become wider only",
                    "2Y": "Objects become taller only",
                    "2XY": "Objects become bigger"
                }
                worded_options.append(resize_map[option])

        return worded_options
