
import os
import base64
import random
import argparse
import pandas as pd
from utils_single import UtilsSingle
from utils_multi import UtilsMulti   

UTILS_CONVERT = { 
	"single": UtilsSingle,
	"multi": UtilsMulti
}

CONCEPT_TO_PARAMETERS = { 
	"KiVA-adult": {
				"2DRotation": (["+45", "-45", "+90", "-90", "+135", "-135", 180]), #7
				"Colour": (["Red", "Yellow", "Green", "Blue", "Grey"]), #5
				"Counting": (["+1","+2","-1","-2","x2","x3","d2","d3"]), #8
				"Reflect": (["X", "Y", "XY"]), #3
				"Resize": (["0.5X", "0.5Y", "0.5XY", "2X", "2Y", "2XY"]) #6
			},

	"KiVA":     {
					"2DRotation": (["+90", "-90", 180]),
					"Counting": (["+1","+2","-1","-2"]),  
					"Colour": (["Green", "Blue", "Red"]), # Red
					"Reflect": (["X", "Y"]), 
					"Resize": (["2XY", "0.5XY" ]) 
				}
}

class KiVAStream(): 


	def __init__(self, concept, image_num, mode, stimuli_directory, text_files_dir, model_name): 

		self.stimuli_directory = stimuli_directory
		self.concept = concept
		self.model_name = model_name
		self.num_trials = 3
		self.retry_num = 2
		self.text_files_dir = text_files_dir
		self.image_num = image_num	
		self.utils_single = UTILS_CONVERT["single"]()
		self.utils_multi = UTILS_CONVERT["multi"]()
		self.step_by_step_text = "step-by-step"

		self.mode = mode
		self.prompts = self.load_prompts() 

		# self.concept_result = self.update_concept_result(self.param)

		output_directory = f"transformed_objects/stitched_{mode}/{image_num}_image/{model_name}/{concept}"
		self.stitched_images_directory = f"{output_directory}/{concept}_stitch"
		os.makedirs(self.stitched_images_directory, exist_ok=True)

	def load_prompts(self): 
		prompts = {}
		prompt_path = f"prompts/{self.image_num}"
		for prompt_file in os.listdir(prompt_path):
			with open(f"{prompt_path}/{prompt_file}", "r") as file:
				p = file.read().strip().replace("{step_by_step_text}", self.step_by_step_text).replace("{options}", "{}")
				prompts[prompt_file.split(".")[0]] = p 
		return prompts

	def encode_image(self, image_path):
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

	def format_files_by_type(self, indexed_files, index, file_type):
		train_files = [filename for filename in indexed_files[index] if 'train' in filename]
		
		if file_type == 'train':  # Create pairs of input and output files
			for filename in train_files:
				if 'input' in filename:
					input_filename = filename
				elif 'output' in filename:
					output_filename = filename

			# Extract the pairs into a list of lists
			formatted_files = [input_filename, output_filename]

		elif file_type == 'test':
			test_files_input = [filename for filename in indexed_files[index] if 'test' in filename]
			formatted_files = sorted(test_files_input)

		return formatted_files

	def get_indexed_files(self, concept, param):
		indexed_files = {}
		beginning = concept + str(param)

		for filename in os.listdir(self.stimuli_directory):
			if filename.startswith(beginning + "_"):
				index = int(filename.split('_')[1])
				if index not in indexed_files:
					indexed_files[index] = []
				indexed_files[index].append(filename)

		return indexed_files


	def correct_cross_domain(self):
		if self.concept == "2DRotation":
			return "Orientation of objects"
		elif self.concept == "Counting":
			return "Number of objects"
		elif self.concept == "Colour":
			return "Color of objects"
		elif self.concept == "Reflect":
			return "Orientation of objects"
		elif self.concept == "Resize":
			return "Size of objects"




	def word_mc_options(self, selected_mc_options):
		worded_options = []

		for option in selected_mc_options:
			if self.mode == "KiVA-adult":
				if self.concept == "2DRotation":
					if option == "180" or option == 180:
						worded_options += ["Objects turn 180 degrees"]
					else: 
						worded_options += [f"Objects turn {option[1:]} degrees"]
				elif self.concept == "Counting":
					counting_type, option = option[0], option[1:]
					if counting_type == "+":
						worded_options += [f"Things go up by {option}"] 
					elif counting_type == "-":
						worded_options += [f"Things go down by {option}"] 
					elif counting_type == "x":
						worded_options += [f"Things multiply by {option}"] 
					elif counting_type == "d":
						worded_options += [f"Things divide by {option}"] 
				elif self.concept == "Colour":
					worded_options += [f"Objects turn {option}"]
				elif self.concept == "Reflect":
					if option == "X":
						worded_options += [f"Objects flip upside down"]
					elif option == "Y":
						worded_options += [f"Objects flip sideways"]
					elif option == "XY":
						worded_options += [f"Objects flip sideways and upside down"]
				elif self.concept == "Resize":
					if option == "0.5X":
						worded_options += [f"Objects become thinner only"]
					elif option == "0.5Y":
						worded_options += [f"Objects become shorter only"]
					elif option == "0.5XY":
						worded_options += [f"Objects become smaller"]
					elif option == "2X":
						worded_options += [f"Objects become wider only"]
					elif option == "2Y":
						worded_options += [f"Objects become taller only"]
					elif option == "2XY":
						worded_options += [f"Objects become bigger"]

			elif self.mode == "KiVA":

				if self.concept == "2DRotation":
					if option == "-90" or option == "+90":
						option = 90
					worded_options += [f"Objects rotate by {option} degrees"]
				elif self.concept == "Counting":
					counting_type, option = option[0], option[1:]
					if counting_type == "+":
						worded_options += [f"Things go up by {option}"] 
					elif counting_type == "-":
						worded_options += [f"Things go down by {option}"] 
				elif self.concept == "Colour":
					worded_options += [f"Objects turn {option}"]
				elif self.concept == "Reflect":
					if option == "X":
						worded_options += [f"Objects flip upside down"]
					elif option == "Y":
						worded_options += [f"Objects flip sideways"]  
					else:
						worded_options += [f"Objects rotate by {option} degrees"]
				elif self.concept == "Resize":
					if option == "0.5XY":
						worded_options += [f"Objects become smaller"]
					elif option == "2XY":
						worded_options += [f"Objects become bigger"]
				
			else: 
				raise ValueError("Invalid mode")
		
		return worded_options

	def __iter__(self):
		for param in CONCEPT_TO_PARAMETERS[self.mode][self.concept]:
			stimuli_set = self.get_indexed_files(self.concept, param)
			query_repeats = len(stimuli_set)
			
			print("----------------------------------------------")
			print(f"Beginning Sub-Concept {self.concept} {param}")

			for query in list(range(query_repeats)):

				print("----------------------------------------------")
				print(f"Beginning Variation {query + 1} of {query_repeats}")

				for regeneration in range(self.num_trials):

					print("-----------------------------")
					print(f"Beginning Regeneration {regeneration + 1} of 3")
					
					with open(f"{self.text_files_dir}/output_{self.concept}{param}.txt", "r") as file:
						lines = file.readlines()

						Train_input = lines[0+(query*4)].rstrip().split(": ")[1]
						Train_output = lines[1+(query*4)].rstrip().split(": ")[1]
						Test_input = lines[2+(query*4)].rstrip().split(": ")[1]
						mc_1 = lines[3+(query*4)].rstrip().split(": ")[1]

					# Alter necessary variables as needed
					if self.concept == "Counting":
						counting_type, option = param[0], param[1:]
						if counting_type == "+":
							if option == "1":
								mc_1 = "-1"
							elif option == "2":
								mc_1 = "-2"
						elif counting_type == "-":
							if option == "1":
								mc_1 = "+1"
							elif option == "2":
								mc_1 = "+2"
						elif counting_type == "x":
							if option == "2":
								mc_1 = "d2"
							elif option == "3":
								mc_1 = "d3"
						elif counting_type == "d":
							if option == "2":
								mc_1 = "x2"
							elif option == "3":
								mc_1 = "x3"

					if self.concept == "2DRotation" and param == "+90":
						test_output_result = int(Test_input) + 90
					elif self.concept == "Counting":
						counting_type, option = param[0], param[1:]
						if counting_type == "+":
							test_output_result = int(Test_input) + int(option)
						elif counting_type == "-":
							test_output_result = int(Test_input) - int(option)
						elif counting_type == "x":
							test_output_result = int(Test_input) * int(option)
						elif counting_type == "d":
							test_output_result = float(Test_input) / int(option)
					else:
						test_output_result = param


					#Creating Training Image
					#========================
					train_stimuli_set = self.format_files_by_type(stimuli_set, query, 'train')
					train_image = self.utils_multi.stitch_images_train(self.utils_multi.read_image(f"{self.stimuli_directory}/{train_stimuli_set[0]}").convert("RGB"), self.utils_multi.read_image(f"{self.stimuli_directory}/{train_stimuli_set[1]}").convert("RGB"))
					#========================
					
					#Saving Training Image for the Multi Image Case
					train_image_path = f"{self.stitched_images_directory}/{self.concept}{param}_{query}_{regeneration}_train.jpg"
					train_image.save(train_image_path)
					#========================


					#Creating Test Images and Test Question D
					test_stimuli_set = self.format_files_by_type(stimuli_set, query, 'test')
					test_stimuli_input, test_stimuli_set = test_stimuli_set[0], test_stimuli_set[1:]
					test_stimuli_set.append(test_stimuli_input)

					stitched_images = [self.utils_single.read_image(f"{self.stimuli_directory}/{test_stimuli_input}").convert("RGB")] 
					for num, test_stimuli in enumerate(test_stimuli_set):
						stitched_image = self.utils_single.read_image(f"{self.stimuli_directory}/{test_stimuli}").convert("RGB")
						stitched_images.append(stitched_image) 
					#========================


					#Creating Test Images Answer Data
					correct_file = test_stimuli_set[0] # Find the correct file filename
					incorrect_param_file = test_stimuli_set[1] # Find the incorrect param file filename
					no_change_file = test_stimuli_set[2] # Find the no change file filename
					random.shuffle(test_stimuli_set) # Shuffle the files in the set
					
					correct_file_index = test_stimuli_set.index(correct_file) # Find the index of the correct filename in the set
					incorrect_file_index = test_stimuli_set.index(incorrect_param_file) # Find the index of the incorrect param filename in the set
					no_change_file_index = test_stimuli_set.index(no_change_file) # Find the index of the no change filename in the set

					extrapolation_labels = ["(A)", "(B)", "(C)", "(D)"]
					extrapolation_correct = extrapolation_labels[correct_file_index]
					#========================

					#Stitching **ALL** Test Image for the Single Image Case
					test_stimuli_image = self.utils_single.stitch_images_test(stitched_images)
					#========================

					#Creating the Final Image for the Single Image Case
					final_image = self.utils_single.stitch_final_images(train_image, test_stimuli_image)
					final_image_path = f"{self.stitched_images_directory}/{self.concept}{param}_{query}_{regeneration}_{extrapolation_correct}.jpg"
					final_image.save(final_image_path)
					#========================

					#Stitching **Individual** Test Images for the Multi Image Case
					stitched_test_stimuli = self.utils_multi.stitch_images_test(stitched_images)
					test_stimuli_image_paths = []
					for num, test_stimuli in enumerate(stitched_test_stimuli):
						test_image_path = f"{self.stitched_images_directory}/{self.concept}{param}_{query}_{regeneration}_test{num}.jpg"
						test_stimuli.save(test_image_path)
						test_stimuli_image_paths += [test_image_path]
					#========================

					#Creating Cross Question Answer Data
					correct_choice_cross = self.correct_cross_domain() # The correct cross-domain MC based on concept, not sub-concept

					cross_domain_all_choices = ["Number of objects", "Size of objects", "Orientation of objects", "Color of objects"]
					cross_domain_all_choices.remove(correct_choice_cross) # Remove correct answer
					selected_cross_choices = random.sample(cross_domain_all_choices, 2)
					selected_cross_choices += [correct_choice_cross] # Add correct answer
					random.shuffle(selected_cross_choices)
					selected_cross_choices += ["No change between pictures", "Doesn't apply"]

					cross_labels = ["(1) ", "(2) ", "(3) ", "(4) ", "(5) "]
					labeled_cross_choices = [label + arg for label, arg in zip(cross_labels, selected_cross_choices)]

					# labeled_correct_cross = [option.split(" ")[0] for option in labeled_cross_choices if correct_choice in option][0]
					# labeled_incorrect_cross = [option.split(" ")[0] for option in labeled_cross_choices if not correct_choice in option]
					# all_choices_cross = [option.split(" ")[0] for option in labeled_cross_choices]

					labeled_correct_cross = [option for option in labeled_cross_choices if correct_choice_cross in option]
					labeled_incorrect_cross = [option for option in labeled_cross_choices if not correct_choice_cross in option]
					all_choices_cross = [option for option in labeled_cross_choices]
					#========================

					#Creating Cross Question Prompt
					str_general_cross_rule_prompt = "\n"
					for lcc in labeled_cross_choices:
						str_general_cross_rule_prompt += lcc + "\n" 

					general_cross_rule_prompt = self.prompts["initi_prompt"] + self.prompts["general_cross_rule_prompt"].format(str_general_cross_rule_prompt)	


					#Creating Witin Question Answer Data
					potential_choices = [str(param), str(mc_1)]
					correct_word_param = self.word_mc_options([param]) # Find correct word option

					random.shuffle(potential_choices)
					potential_choices = self.word_mc_options(potential_choices)
					potential_choices += ["No change between pictures", "Doesn't apply"]

					within_labels = ["(1) ", "(2) ", "(3) ", "(4) "]
					labeled_within_choices = [label + arg for label, arg in zip(within_labels, potential_choices)]

					# labeled_correct_within = [option.split(" ")[0] for option in labeled_within_choices if correct_word_param[0] in option][0]
					# labeled_incorrect_within = [option.split(" ")[0] for option in labeled_within_choices if not correct_word_param[0] in option]
					# all_choices_within = [option.split(" ")[0] for option in labeled_within_choices]

					labeled_correct_within = [option for option in labeled_within_choices if correct_word_param[0] in option]
					labeled_incorrect_within = [option for option in labeled_within_choices if not correct_word_param[0] in option]
					all_choices_within = [option for option in labeled_within_choices]
					#========================

					#Creating Within Question Prompt
					str_general_within_rule_prompt = "\n"
					for lwc in labeled_within_choices:
						str_general_within_rule_prompt += lwc + "\n" 

					general_within_rule_prompt = self.prompts["general_within_rule_prompt"].format(str_general_within_rule_prompt)
					#========================

					#Creating Extrapolation Question Prompt
					extrapolation_prompt = self.prompts["extrapolation_prompt"]
					#========================


					#Creating Extrapolation Question Answer Data
					test_stimuli_set += ["No change between pictures", "Doesn't apply"]
					labeled_correct_extrapolation = [extrapolation_correct]
					labeled_incorrect_extrapolation = [item for item in extrapolation_labels if item != extrapolation_correct]
					all_choices_extrapolation = extrapolation_labels
					
					all_choices_extrapolation_with_params = all_choices_extrapolation[:]
					all_choices_extrapolation_with_params[correct_file_index] += f" ({param})"
					all_choices_extrapolation_with_params[incorrect_file_index] += f" ({mc_1})"
					all_choices_extrapolation_with_params[no_change_file_index] += " (No change)"
					all_choices_extrapolation_with_params[-1] += " Doesn't Apply"

					assert extrapolation_correct not in labeled_incorrect_extrapolation
					#========================


					if self.image_num == "single":
						output_images = [final_image_path]
					elif self.image_num == "multi":
						output_images = [train_image_path]
						output_images.extend(test_stimuli_image_paths)
					else: 
						raise ValueError("Invalid image_num")												

					yield {

						"images": output_images,

						"prompts": {
							"general_cross_rule_prompt": general_cross_rule_prompt,
							"general_within_rule_prompt": general_within_rule_prompt,
							"extrapolation_prompt": extrapolation_prompt,
						},

						"exps_details":  {
								"Variation": query,
								"Regeneration": regeneration,
								"Train_input_param": Train_input,
								"Train_output_param": Train_output,
								"Test_input_param": Test_input,
								"Test_output_param": test_output_result,
								"Param": param
						},

						"answers": {
							"general_cross": {"correct_answers": labeled_correct_cross, "incorrect_answers": labeled_incorrect_cross, "all_answers": all_choices_cross},
							"general_within": {"correct_answers": labeled_correct_within, "incorrect_answers": labeled_incorrect_within, "all_answers": all_choices_within},
							"extrapolation": {"correct_answers": labeled_correct_extrapolation, "incorrect_answers": labeled_incorrect_extrapolation, "all_answers": all_choices_extrapolation, "all_answers_with_params": all_choices_extrapolation_with_params}
						}

					}


# def setup_model(self): 
# 	if self.model_name == "llava":
# 		from models.llava_model import LLavaModel
# 		self.chat_model =  LLavaModel(self.prompts["system_prompt"], max_token = 300)

# 	if self.model_name == "gpt4":
# 		from models.gpt4_model import GPT4Model
# 		assert self.api_key is not None, "API key is required for GPT4 model"
# 		self.chat_model = GPT4Model(self.prompts["system_prompt"], api_key=self.api_key, max_token=300)

# 	elif self.model_name == "gpt4o":
# 		from models.gpt4o_model import GPT4OModel
# 		assert self.api_key is not None, "API key is required for GPT4 model"
# 		self.chat_model = GPT4OModel(self.prompts["system_prompt"], api_key=self.api_key, max_token=300)

# 	else: 
# 		raise ValueError("Model name not recognized.")

