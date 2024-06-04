import csv
import os
import base64
import random
import argparse
import os
from utils import stitch_images_train,  stitch_images_test, read_image, stitch_final_images
from PIL import Image

model_name = 'gpt4'

system_prompt = "You are a helpful visual assistant!"

if model_name == "llava":
	from models.llava_model_16 import LLavaModel
	chat_model =  LLavaModel(system_prompt)

elif model_name == "gpt4":
	from models.gpt4_model import GPT4Model
	api_key = "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"
	chat_model = GPT4Model(system_prompt, api_key=api_key)

else: 
	raise ValueError("Model name not recognized.")

chat_model.init_history()


resp = chat_model.run_model("how many balls are there is this image? ", image_path="test_images/three_balls.jpg")
print(resp)
resp = chat_model.run_model("add three to that result, what do you get? ")
print(resp)
resp = chat_model.run_model("Now multiply the result by 3, what do you get?")
print(resp)

chat_model.init_history() 
image_one_ball = Image.open("test_images/one_ball.jpg")
image_one_ball = image_one_ball.convert('RGB')

resp = chat_model.run_model("what is the color of these balls? ", image_path="test_images/three_balls.jpg")
print(resp)
resp = chat_model.run_model("Count the balls in this image. What is first image ball count - this image ball count?  ", image_path = "test_images/one_ball.jpg")
print(resp)

