from models.chat_model import ChatModel
import requests
import time
import base64


class GPT4Model(ChatModel):
	def __init__(self, system_prompt, api_key, max_token = 100):
		super().__init__(system_prompt)
		self.history = self.init_history()
		self.headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}"
		}

		self.max_token = max_token 

	def init_history(self):
		self.history = {}
		self.n_runs = 0 

	def get_history(self):
		new_history = {} 
		for k, v in self.history.items():
			if type(v) == list:
				new_history[k] = v[:]
				print('hellooooo')
			else:
				new_history[k] = v

		self.history["messages"].append("hi")
		return new_history
	
	def set_history(self, history):
		self.history = history
		
	def first_call(self, encoded_image, prompt):
		payload = {
			"model": "gpt-4-vision-preview",
			"messages": 
			[
				{
					"role": "system",
					"content": [self.system_prompt]
				},
				{
					"role": "user",
					"content": [
						{
							"type": "text",
							"text": prompt
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_image}"
							},
						}
					], 
				},
			],
			"max_tokens": self.max_token
		}
		return payload


	def update_history(self, response):

		response_message = {"role": "assistant", 
							"content": [
								{
									"type": "text",
									"text": response
									}
							]
		}

		self.history["messages"] += [response_message]
		self.n_runs += 1

	def subsequent_call(self, prompt, image=None):
		if image is None: 
			prompt = {"role": "user",
						"content": [
							{
								"type": "text",
								"text": prompt
							}
						]
					}
		else: 
			prompt = {
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": prompt
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{image}"
								}
							}
						]
					}

		self.history["messages"] += [prompt]
		return self.history


	def make_api_call(self, payload):
		while True:
			try:
				response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
				response = response.json()
				response_content = response['choices'][0]['message']['content']

				if 'limit' in response_content.lower():
					print("API limit reached, sleeping for 30 seconds...")
					time.sleep(30)
					continue

				return response_content

			except Exception as e:
				print(f"Encountered an error: {e}, sleeping for 30 seconds...")
				time.sleep(30)

	def encode_image(self, image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')


	def run_model(self, prompt, image_path=None):
		if self.n_runs == 0:
			image = self.encode_image(image_path)
			payload = self.first_call(image, prompt)
			self.history = payload
		else:
			if image_path is not None:
				image = self.encode_image(image_path)
				payload = self.subsequent_call(prompt, image)
			else: 
				payload = self.subsequent_call(prompt)

		response = self.make_api_call(payload)

		self.update_history(response)
		return {"response": response}
