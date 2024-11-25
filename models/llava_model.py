
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
		self.init_history()


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
	
	def update_history(self, text):
		self.history = text

	def get_convo_user_prompt(self, prompt, img=True): 

		if img: 
			self.conversation.append(
				{
					"role": "user",
					"content": [
						{"type": "text", "text": prompt},
						{"type": "image"},
					],
				}
			)
		else: 
			self.conversation.append(
				{
					"role": "user",
					"content": [
						{"type": "text", "text": prompt},
					],
				}
			)
		return self.conversation


	def update_convo_assistant_response(self, response):
		self.conversation.append(
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": response},
				],
			}
		)


	def run_model(self, prompt, image_path=None):

		# if self.history != "" and image_path is None:
		# 	prompt = self.system_prompt + "\n" + self.history + "\n" + "USER: \n" + prompt + "\nASSISTANT:"
		# elif self.history != "" and image_path is not None:
		# 	prompt = self.system_prompt + "\n" + self.history + "\n" + "USER: <image>\n" + prompt + "\nASSISTANT:"
		# else: 
		# 	prompt = self.system_prompt + "\n" + "USER: <image>\n" + prompt + "\nASSISTANT:"
		# else:
		
		if image_path is not None: 
			self.main_img = Image.open(image_path)
			conv = self.get_convo_user_prompt(prompt)
		else: 
			conv = self.get_convo_user_prompt(prompt, img=False)


		conv = self.model_data["processor"].apply_chat_template(conv, add_generation_prompt=True)

		inputs = self.model_data["processor"](text=conv, images=self.main_img, return_tensors="pt", padding=True).to("cuda", torch.float16)
		output = self.model_data["model"].generate(**inputs, max_new_tokens=self.new_tokens, do_sample=False)
		output = self.model_data["processor"].batch_decode(output)[0]
		response = output.split("ASSISTANT:")[-1].strip()

		output = output.replace("<s>", "")
		output = output.replace("</s>", "")

		self.update_convo_assistant_response(response)
		return { 
			"response": response,
		}

	def init_history(self): 
		self.conversation = [
			{
				"role": "system",
				"content": [
					{"type": "text", "text": self.system_prompt},
				]
			}]
		
		self.main_img = None
