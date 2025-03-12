from models.chat_model import ChatModel
import requests
import time
import base64


class GPTO1Model(ChatModel):
	def __init__(self, system_prompt, api_key, max_token=1000):
		super().__init__(system_prompt)
		self.history = self.init_history()
		self.headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}"
		}
		#self.max_tokens = max_token -- o1 structures completion tokens differently

	def init_history(self):
		self.history = {}
		self.n_runs = 0 
		
	def first_call(self, encoded_image, prompt):
		payload = {
			"model": "o1",
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
			#"max_tokens": self.max_tokens
		}
		return payload

	def first_extrapolation_call(self, prompt, encoded_images):
		payload = {
			"model": "o1",
			"messages": 
			[
				{
					"role": "system",
					"content": [self.system_prompt]
				},
				{"role": "user",
					"content": [
						{
							"type": "text",
							"text": prompt
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[0]}"
							},
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[1]}"
							},
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[2]}"
							},
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[3]}"
							},
						}
					]
				},
			],
			#"max_tokens": self.max_tokens
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

	def subsequent_call(self, prompt):
		prompt = {"role": "user",
					"content": [
						{
							"type": "text",
							"text": prompt
						}
					]
				}

		self.history["messages"] += [prompt]
		return self.history
	
	def last_call(self, prompt, encoded_images):
		prompt = {"role": "user",
					"content": [
						{
							"type": "text",
							"text": prompt
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[0]}"
							},
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[1]}"
							},
						},
						{
							"type": "image_url",
							"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_images[2]}"
							},
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
				print(response)
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


	def run_model_multi(self, prompt, final_image_paths, extrapolation_only=False):
		encoded_images = [self.encode_image(image_path) for image_path in final_image_paths]
		if extrapolation_only == False:
			payload = self.last_call(prompt, encoded_images)
		else:
			payload = self.first_extrapolation_call(prompt, encoded_images)
		self.history = payload

		response = self.make_api_call(payload)

		self.update_history(response)
		return {"response": response}


	def run_model_indiv(self, prompt, image=None, final_image_path=None):
		if self.n_runs == 0:
			image = self.encode_image(final_image_path)
			payload = self.first_call(image, prompt)
			self.history = payload
		else:
			payload = self.subsequent_call(prompt)

		response = self.make_api_call(payload)

		self.update_history(response)
		return {"response": response}
