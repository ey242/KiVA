
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from models.chat_model import ChatModel
from PIL import Image

class LLavaModel(ChatModel):
	def __init__(self, system_prompt, max_token=100):
		super().__init__(system_prompt)

		self.model_data = self.load_model()
		self.model_name = "llava"
		self.new_tokens = max_token
		self.history = self.init_history()


	def load_model(self):

		model_id = "llava-hf/llava-1.5-13b-hf"
		# model_id = "liuhaotian/llava-v1.6-vicuna-13b"

		model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
		processor = AutoProcessor.from_pretrained(model_id)
		processor.image_processor.do_center_cropq = False
		# processor.image_processor.size = {"height": 336, "width": 336}

		model_data = {
			"model":model,
			"processor":processor,
			"name":"llava"
		}

		return model_data
	
	def update_history(self, text, img=None):
		self.history = text
		if img is not None:
			self.img_history.append(img)
		

	def run_model(self, prompt, image_path=None):

		if self.history != "" and image_path is None:
			prompt = self.system_prompt + "\n" + self.history + "\n" + "USER: \n" + prompt + "\nASSISTANT:"
		elif self.history != "" and image_path is not None:
			prompt = self.system_prompt + "\n" + self.history + "\n" + "USER: <image>\n" + prompt + "\nASSISTANT:"
		else: 
			prompt = self.system_prompt + "\n" + "USER: <image>\n" + prompt + "\nASSISTANT:"

		# else:
		
		if image_path is None: 
			image_input = self.img_history
			image = None
		else: 
			image = Image.open(image_path).convert("RGB")
			image_input = self.img_history + [image]

		inputs = self.model_data["processor"](prompt, images=image_input, return_tensors="pt", padding=True).to("cuda", torch.float16)
		output = self.model_data["model"].generate(**inputs, max_new_tokens=self.new_tokens, do_sample=False)
		output = self.model_data["processor"].batch_decode(output)[0]
		response = output.split("ASSISTANT:")[-1].strip()

		output = output.replace("<s>", "")
		output = output.replace("</s>", "")

		self.update_history(output, image)

		return { 
			"response": response,
		}

	def init_history(self): 
		self.history = ""
		self.img_history = []
