import base64
import time
import requests

def encode_image(image_path):
    """Encode an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Model:
    def __init__(self, system_prompt, api_key, max_token=100):
        self.system_prompt = system_prompt
        self.history = self.init_history()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        #self.max_token = max_token

    def init_history(self):
        """Initialize conversation history and run counter."""
        self.n_runs = 0
        return {"messages": []}

    def first_call(self, encoded_image, prompt):
        """Construct the payload for the first API call with an image."""
        payload = {
            "model": "o1",
            "messages": [
                {
                    "role": "system",
                    "content": [self.system_prompt]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                },
            ],
            #"max_tokens": self.max_token
        }
        return payload

    def subsequent_call(self, prompt, encoded_images=None):
        """Construct the payload for subsequent API calls."""
        if encoded_images is None:
            user_content = [
                {"type": "text",
                 "text": prompt}]
        else:
            image_items = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                           for img in encoded_images]
            user_content = [
                {"type": "text",
                "text": prompt}] + image_items
        prompt_obj = {
            "role": "user",
            "content": user_content}
        self.history["messages"] += [prompt_obj]
        return self.history

    def make_api_call(self, payload):
        while True:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions",
                                         headers=self.headers, json=payload)
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

    def run_model(self, prompt, image_path=None):
        """
        Execute the model:
         - For the first & second run, expect a singular image.
         - For the third run, expect 4 total images (train + 3 test images).
        """
        if image_path is not None:
            if isinstance(image_path, list):
                encoded_images = [encode_image(p) for p in image_path]
            else:
                encoded_images = [encode_image(image_path)]
        else:
            encoded_images = None

        if self.n_runs == 0:
            payload = self.first_call(encoded_images, prompt)
            self.history = payload
        else:
            payload = self.subsequent_call(prompt, encoded_images)
        response = self.make_api_call(payload)
        self.update_history(response)
        return {"response": response}

    def update_history(self, response):
        response_message = {
            "role": "assistant",
            "content": [
                {"type": "text",
                 "text": response}]
        }
        self.history["messages"] += [response_message]
        self.n_runs += 1
