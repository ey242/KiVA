import PIL.Image
from typing import Optional
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

class Model:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        system_prompt: str = "",
        max_new_tokens: int = 200,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt.strip()
        self.max_new_tokens = max_new_tokens

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.init_history()

    # Conversation helpers
    def init_history(self) -> None:
        """Start a fresh conversation list understood by apply_chat_template()."""
        self.conversation = []
        if self.system_prompt:
            self.conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        self.main_img = None 
        self._image_token_inserted = False

    def _add_user_turn(self, prompt: str, with_image: bool) -> None:
        entry = {"role": "user",
                "content": [{"type": "text", "text": prompt.strip()}]}

        # Adds 1 image placeholder per conversation
        if with_image and not self._image_token_inserted:
            entry["content"].append({"type": "image"})
            self._image_token_inserted = True 

        self.conversation.append(entry)

    def _add_assistant_turn(self, reply: str) -> None:
        self.conversation.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reply}],
            }
        )

    # Core call
    def run_model(self, prompt: str, image_path: Optional[str] = None) -> dict:
        # Update / keep image reference
        if image_path is not None:
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0] 
            self.main_img = PIL.Image.open(image_path)

        has_image_token = self.main_img is not None
        self._add_user_turn(prompt, with_image=has_image_token)

        # Build prompt
        prompt_text = self.processor.apply_chat_template(
            self.conversation, add_generation_prompt=True
        )

        # Make the image list length match # of <image> tags
        n_image_tokens = prompt_text.count("<image>")
        images_arg = [self.main_img] * n_image_tokens if n_image_tokens else None

        inputs = self.processor(
            text=prompt_text,
            images=images_arg, 
            return_tensors="pt",
        )

        # Device moves
        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            inputs[k] = v.to(device).half() if k == "pixel_values" else v.to(device)

        # Generate
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            do_sample=False,
        )

        # Decode 
        full_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        reply = full_text.split("ASSISTANT:")[-1].strip()

        self._add_assistant_turn(reply)
        return {"response": reply}
