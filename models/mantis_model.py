from models.Mantis.mantis.models.mllava import chat_mllava
from PIL import Image
import torch
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from models.chat_model import ChatModel

class MantisModel(ChatModel): 
	def __init__(self, system_prompt, max_token=100):
		super().__init__(system_prompt)

		self.new_tokens = max_token
		self.model_data = self.load_model()
		self.model_name = "llava"
		self.history = self.init_history()
		self.n_call = 0 

	def load_model(self): 
	
		processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
		attn_implementation = None # or "flash_attention_2"
		model = LlavaForConditionalGeneration.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)

		generation_kwargs = {
			"max_new_tokens": self.new_tokens,
			"num_beams": 1,
			"do_sample": False
		}
		
		model_data = {
			"model": model,
			"processor": processor,
			"generation_kwargs": generation_kwargs
		}
		
		return model_data

	def run_model(self, prompt, image_paths=None):

		if self.n_call == 0: 
			prompt = self.system_prompt + prompt

		if image_paths is None: 
			image_input = []
			for img in self.img_history:
				image_input.append(img)

			image = None
		else: 
			image_input = []
			for img in self.img_history:
				image_input.append(img)

			for img_path in image_paths:
				image = Image.open(img_path).convert("RGB")
				image_input.append(image)
				self.img_history.append(image)
		
		# print(image_input)
		# print(prompt)
		
		if self.history != "": 
			response, history = chat_mllava(prompt, image_input, self.model_data["model"], self.model_data["processor"], history=self.history, **self.model_data["generation_kwargs"])
		else: 
			response, history = chat_mllava(prompt, image_input, self.model_data["model"], self.model_data["processor"], **self.model_data["generation_kwargs"])

		self.history = history


		self.n_call += 1
		return {
			"response": response,
		}

	def init_history(self): 
		self.history = ""
		self.img_history = []
