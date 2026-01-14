from .config import ModelType


class ModelFactory:
    @staticmethod
    def create_model(config, system_prompt):
        """
        Create and return appropriate model based on configuration.

        Args:
            config: ExperimentConfig instance
            system_prompt: System prompt string

        Returns:
            Initialized model instance
        """
        if config.model_type == ModelType.GPT4:
            if config.is_multi_image():
                from models.gpt4_model_multi import GPT4Model
                return GPT4Model(system_prompt, api_key=config.api_key, max_token=300)
            else:
                from models.gpt4_model import GPT4Model
                return GPT4Model(system_prompt, api_key=config.api_key, max_token=300)

        elif config.model_type == ModelType.GPT4O:
            if config.is_multi_image():
                from models.gpt4o_model_multi import GPT4OModel
                return GPT4OModel(system_prompt, api_key=config.api_key, max_token=300)
            else:
                from models.gpt4o_model import GPT4OModel
                return GPT4OModel(system_prompt, api_key=config.api_key, max_token=300)

        elif config.model_type == ModelType.LLAVA:
            from models.llava_model import LLavaModel
            return LLavaModel(system_prompt, max_token=300)

        elif config.model_type == ModelType.MANTIS:
            from models.mantis_model import MantisModel
            return MantisModel(system_prompt, max_token=300)

        else:
            raise ValueError(f"Unrecognized model type: {config.model_type}")
