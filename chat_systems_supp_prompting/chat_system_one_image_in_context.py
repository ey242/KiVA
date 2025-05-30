import os
import base64
import random
import argparse
import os
from utils_single import stitch_images_train,  stitch_images_test, read_image, stitch_final_images
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--concept', type=str, default="Counting", help='The concept to be tested')
parser.add_argument('--model', type=str, default="llava", help='model')
parser.add_argument('--api_key', type=str, default="API-KEY", help='gpt4_api_key')

args = parser.parse_args()

concept = args.concept
query_repeats = None # Number of times to repeat process, set to None for max. # of trials with given stimuli
model_name = args.model

num_incontext_images_per_param = 2
stimuli_directory = f"stimuli/KiVA/{concept}" # Insert object file directory
text_files_dir = f"stimuli/KiVA/trial_tracker/"
output_directory = f"output/output_oneimage_incontext/output_{model_name}/output_{concept}"

stitched_images_directory = f"{output_directory}/{concept}_stitch"

os.makedirs(output_directory, exist_ok=True)
os.makedirs(stitched_images_directory, exist_ok=True)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
system_prompt = ("You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning."
	 			"You will think step-by-step and carefully examine the visual evidence before providing an answer.")
initi_prompt =  ("You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right"
				 "transformations of a different object on the bottom marked by (A) or (B) or (C)."
	 			 "The transformations involve a change of either the size, orientation, number, or color of an object")

general_cross_rule_prompt =  initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation on top of the"
							 "puzzle where the picture on the left transforms to the picture on the right? In your answer start with the correct rule number"
							 "surrounded by parentheses, then provide a “step-by step” reasoning for your choice.")
general_within_rule_prompt = ("Which one of the following rules {} best describes the transformation in the top of the puzzle where the picture"
	 						  "on the left transforms to the picture on the right?. In your answer start with the correct rule number surrounded by parentheses,"
							  "then provide a “step-by step” reasoning for your choice. ")

extrapolation_prompt = ("Which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is"
	 				    "the same as the left-to-right transformation on the top of the puzzle?"
						"Note that each transformation involves two pictures (a left picture and a right picture). The move from left to right in the chosen transformation should match that of the top transformation."
						"In your answer start with the correct letter surrounded by parentheses (or (D) if none of the options apply), then provide a step-by-step reasoning for your choice. ")


concept_to_parameters = {
	"2DRotation": (["+90", "-90", 180]),
	"Counting": (["+1","+2","-1","-2"]),  
	"Colour": (["Red", "Green", "Blue"]),
	"Reflect": (["X", "Y"]), 
	"Resize": (["0.5XY", "2XY"])
}

concept_headers = [
	"Variation",
	"Regeneration",
	"Train_input",
	"Train_output",
	"Test_input",
	"Test_output",
	"Test_wrong1",
	"Test_wrong2",
	"Full#1",
	"Full#2",
	"Full#3",
	"MCResponse#1",
	"MCResponse#2",
	"MCResponse#3",
]



def update_concept_result(param):
	concept_result = {
			"Variation": [],
			"Regeneration": [],
			"Train_input": [],
			"Train_output": [],
			"Test_input": [],
			"Test_output": [],
			"Test_wrong1": [],
			"Test_wrong2": [],
			"Full#1": [],
			"Full#2": [],
			"Full#3": [],
			"MCResponse#1": [],
			"MCResponse#2": [],
			"MCResponse#3": [],
		}
	return concept_result


def correct_cross_domain(concept):
	if concept == "2DRotation":
		return "Orientation of objects"
	elif concept == "Counting":
		return "Number of objects"
	elif concept == "Colour":
		return "Color of objects"
	elif concept == "Reflect":
		return "Orientation of objects"
	elif concept == "Resize":
		return "Size of objects"
	

def get_indexed_files(param):
	indexed_files = {}
	beginning = concept + str(param)

	for filename in os.listdir(stimuli_directory):
		if filename.startswith(beginning + "_"):
			index = int(filename.split('_')[1])
			if index not in indexed_files:
				indexed_files[index] = []
			indexed_files[index].append(filename)

	return indexed_files

def format_files_by_type(indexed_files, index, file_type):
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



def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def word_mc_options(selected_mc_options):
	worded_options = []

	for option in selected_mc_options:
		if concept == "2DRotation":
			if option == "-90" or option == "+90":
				option = 90
			worded_options += [f"Objects rotate by {option} degrees"]
		elif concept == "Counting":
			counting_type, option = option[0], option[1:]
			if counting_type == "+":
				worded_options += [f"Things go up by {option}"] 
			elif counting_type == "-":
				worded_options += [f"Things go down by {option}"] 
		elif concept == "Colour":
			worded_options += [f"Objects turn {option}"]
		elif concept == "Reflect":
			if option == "X":
				worded_options += [f"Objects flip upside down"]
			elif option == "Y":
				worded_options += [f"Objects flip sideways"]  
			else:
				worded_options += [f"Objects rotate by {option} degrees"]
		elif concept == "Resize":
			if option == "0.5XY":
				worded_options += [f"Objects become smaller"]
			elif option == "2XY":
				worded_options += [f"Objects become bigger"]
		
	return worded_options


def eval_response(response, answers, all_choices): 
	
	all_available_choices = {}
	for choice in all_choices:
		if choice in response:
			all_available_choices[choice] = response.index(choice)

	if len(all_available_choices) == 0:
		return False
	#get the earliest choice 
	extracted_choice = min(all_available_choices, key=all_available_choices.get)
	
	for answer in answers: 
		if answer == extracted_choice: 
			return True 
	return False

if model_name == "llava":
	from models.llava_model import LLavaModel
	chat_model =  LLavaModel(system_prompt)

elif model_name == "gpt4":
	from models.gpt4_model import GPT4Model
	api_key = args.api_key
	chat_model = GPT4Model(system_prompt, api_key=api_key, max_token = 500)

else: 
	raise ValueError("Model name not recognized.")


def run_trial(param, query, regeneration, chat_model, stimuli_set, concept, in_context_history=None, reflect=False): 


	concept_result = update_concept_result(param)

	if in_context_history is not None:
		copy_of_in_context_history = {} 
		for k, v in in_context_history.items():
			if type(v) == list:
				copy_of_in_context_history[k] = v[:]
			else:
				copy_of_in_context_history[k] = v

		print(len(copy_of_in_context_history["messages"]))
		chat_model.set_history(copy_of_in_context_history)

	results = []

	print("-----------------------------")
	print(f"Beginning Regeneration {regeneration + 1} of 3")

	# Save all predefined variables
	with open(f"{text_files_dir}/output_{concept}{param}.txt", "r") as file:
		lines = file.readlines()

		Train_input = lines[0+(query*4)].rstrip().split(": ")[1]
		Train_output = lines[1+(query*4)].rstrip().split(": ")[1]
		Test_input = lines[2+(query*4)].rstrip().split(": ")[1]
		mc_1 = lines[3+(query*4)].rstrip().split(": ")[1]
		mc_2 = 0

	# Alter necessary variables as needed
	if concept == "Counting":
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

	if concept == "2DRotation" and param == "+90":
		test_output_result = int(Test_input) + 90
	elif concept == "Counting":
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


	# Add pre-saved results to final responses file
	concept_result["Variation"] += [query]
	concept_result["Regeneration"] += [regeneration]
	concept_result["Train_input"] += [Train_input]
	concept_result["Train_output"] += [Train_output]
	concept_result["Test_input"] += [Test_input]
	concept_result["Test_output"] += [test_output_result]
	concept_result["Test_wrong1"] += [mc_1]
	concept_result["Test_wrong2"] += [mc_2]


	# Set up train stimuli
	train_stimuli_set = format_files_by_type(stimuli_set, query, 'train')

	train0_image = stitch_images_train(read_image(f"{stimuli_directory}/{train_stimuli_set[0]}").convert("RGB"), read_image(f"{stimuli_directory}/{train_stimuli_set[1]}").convert("RGB"), case_num=1)

	# Set up test stimuli
	test_stimuli_set = format_files_by_type(stimuli_set, query, 'test')
	test_stimuli_input, test_stimuli_set = test_stimuli_set[0], test_stimuli_set[1:]

	test_stimuli_set.append(test_stimuli_input)
	correct_file = test_stimuli_set[0] # Find the correct file filename
	random.shuffle(test_stimuli_set) # Shuffle the files in the set
	correct_file_index = test_stimuli_set.index(correct_file) # Find the index of the correct filename in the set

	stitched_images = [read_image(f"{stimuli_directory}/{test_stimuli_input}").convert("RGB")] 
	for num, test_stimuli in enumerate(test_stimuli_set):
		stitched_image = read_image(f"{stimuli_directory}/{test_stimuli}").convert("RGB")
		stitched_images.append(stitched_image) 

	test_stimuli_image = stitch_images_test(stitched_images)

	# Set up final stitched train + test stimuli image
	final_image = stitch_final_images(train0_image, test_stimuli_image)
	final_image_path = f"{stitched_images_directory}/{concept}{param}_{query}_{regeneration}.jpg"
	final_image.save(final_image_path)

	# Testing Cross-Domain
	add_result = True
	correct_answer = correct_cross_domain(concept) # The correct cross-domain MC based on concept, not sub-concept

	cross_domain_all_choices = ["Number of objects", "Size of objects", "Orientation of objects", "Color of objects"]
	cross_domain_all_choices.remove(correct_answer) # Remove correct answer
	selected_cross_choices = random.sample(cross_domain_all_choices, 2)
	selected_cross_choices += [correct_answer] # Add correct answer
	random.shuffle(selected_cross_choices)
	selected_cross_choices += ["No change between pictures", "Doesn't apply"]

	labels = ["(1) ", "(2) ", "(3) ", "(4) ", "(5) "]
	labeled_cross_choices = [label + arg for label, arg in zip(labels, selected_cross_choices)]

	labeled_correct_cross = [option.split(" ")[0] for option in labeled_cross_choices if correct_answer in option][0]
	labeled_incorrect_cross = [option.split(" ")[0] for option in labeled_cross_choices if not correct_answer in option]


	str_general_cross_rule_prompt = "\n"
	for lcc in labeled_cross_choices:
		str_general_cross_rule_prompt += lcc + "\n" 

	out_chatmodel = chat_model.run_model(general_cross_rule_prompt.format(str_general_cross_rule_prompt), final_image_path)
	concept_result["Full#1"] += [out_chatmodel["response"]]

	print("Cross Domain Response: ", out_chatmodel["response"])

	all_choices = [option.split(" ")[0] for option in labeled_cross_choices]
	if eval_response(out_chatmodel["response"], [labeled_correct_cross], all_choices):
		concept_result["MCResponse#1"] += ["1"]
		print(f"Correct cross response")
	elif eval_response(out_chatmodel["response"], labeled_incorrect_cross, all_choices): 
		concept_result["MCResponse#1"] += ["0"]
		add_result = False
		print(f"Incorrect cross response")
		out_chatmodel = chat_model.run_model(f"The previous response was incorrect. The correct answer is: {labeled_correct_cross}.")
		print(out_chatmodel["response"])
		print("="*20)

	else:
		concept_result["MCResponse#1"] += ["Null"]
		print(f"Uncertain cross response")

	print("="*20)

	if add_result: 
		potential_choices = [str(param), str(mc_1)]
		correct_word_param = word_mc_options([param]) # Find correct word option

		random.shuffle(potential_choices)
		potential_choices = word_mc_options(potential_choices)
		potential_choices += ["No change between pictures", "Doesn't apply"]


		labels = ["(1) ", "(2) ", "(3) ", "(4) "]
		labeled_within_choices = [label + arg for label, arg in zip(labels, potential_choices)]

		labeled_correct_within = [option.split(" ")[0] for option in labeled_within_choices if correct_word_param[0] in option][0]
		labeled_incorrect_within = [option.split(" ")[0] for option in labeled_within_choices if not correct_word_param[0] in option]

		str_general_within_rule_prompt = "\n"
		for lwc in labeled_within_choices:
			str_general_within_rule_prompt += lwc + "\n" 

		out_chatmodel = chat_model.run_model(general_within_rule_prompt.format(str_general_within_rule_prompt))
		concept_result["Full#2"] += [out_chatmodel["response"]]

		print("Within Response: ", out_chatmodel["response"])

		all_choices = [option.split(" ")[0] for option in labeled_within_choices]
		if eval_response(out_chatmodel["response"], [labeled_correct_within], all_choices):
			concept_result["MCResponse#2"] += ["1"]
			print(f"Correct within response")
		elif eval_response(out_chatmodel["response"], labeled_incorrect_within, all_choices):
			concept_result["MCResponse#2"] += ["0"]
			print(f"Incorrect within response")
			out_chatmodel = chat_model.run_model(f"The previous response was incorrect. The correct answer is: {labeled_correct_within}. Why do you think you were wrong?")
			print(out_chatmodel["response"])
			print("="*20)

		else:
			concept_result["MCResponse#2"] += ["Null"]
			print(f"Uncertain within response")

		print("="*20)

	else:
		concept_result["Full#2"] += ["[Skipped mcq#2, incorrect previous response]"]
		concept_result["MCResponse#2"] += [""]

	out_chatmodel = chat_model.run_model(extrapolation_prompt)            
	test_stimuli_set += ["No change between pictures", "Doesn't apply"]

	labels = ["(A) ", "(B) ", "(C) ", "(D) "]
	labeled_choices = [label + arg for label, arg in zip(labels, test_stimuli_set)]
	extracted_letters = [item.split(" ")[0] for item in labeled_choices]

	concept_result["Full#3"] += [out_chatmodel["response"]]

	print("Extrapolation: ", out_chatmodel["response"])
	incorrect_extracted_letters = [item.split(" ")[0] for item in labeled_choices if item.split(" ")[0] != extracted_letters[correct_file_index]]

	all_choices = [option.split(" ")[0] for option in labeled_choices]
	if eval_response(out_chatmodel["response"], [extracted_letters[correct_file_index]], all_choices):
		concept_result["MCResponse#3"] += ["1"]
		print("Correct response")

	elif eval_response(out_chatmodel["response"], incorrect_extracted_letters, all_choices):
		concept_result["MCResponse#3"] += ["0"]
		print("Incorrect response")
		print("="*20)
		print(f"The previous response was incorrect. The correct answer is: {extracted_letters[correct_file_index]}. Why do you think you were wrong?")

		out_chatmodel = chat_model.run_model(f"The previous response was incorrect. The correct answer is: {extracted_letters[correct_file_index]}. Why do you think you were wrong?")
		print(out_chatmodel["response"])
		print("="*20)

	else:
		concept_result["MCResponse#3"] += ["Null"]
		print("Uncertain response")

	results.append(concept_result)
	print("="*20)

	return concept_result

	
in_context_data = {}
for param in concept_to_parameters[concept]:
	stimuli_set = get_indexed_files(param)
	if query_repeats is None:
		query_repeats = len(stimuli_set)

	for query in range(num_incontext_images_per_param):
		result = run_trial(param, query, 0, chat_model, stimuli_set, concept)


teach_prompt = "Now please use the feedback from the training examples to answer the following question."
response = chat_model.run_model(teach_prompt)
print(response)
print("="*20)

in_context_history = chat_model.get_history()
for param in concept_to_parameters[concept]:
	stimuli_set = get_indexed_files(param)
	output_file = output_directory + f"/{concept}{param}.csv"
	if query_repeats is None:
		query_repeats = len(stimuli_set)

	for query in list(range(10, len(stimuli_set))):

		print("----------------------------------------------")
		print(f"Beginning Variation {query + 1} of {query_repeats}")

		for regeneration in range(3):

			df = pd.read_csv(output_file)
			if len(df[(df["Variation"] == query) & (df["Regeneration"] == regeneration)]) > 0:
				print("Skipping query, already completed")
				continue

			print(len(in_context_history["messages"]))
			concept_result = run_trial(param, query, regeneration, chat_model, stimuli_set, concept, in_context_history)

			# use pandas to open csv 
			if os.path.exists(output_file): 
				df = pd.read_csv(output_file)
				df_to_add = pd.DataFrame(concept_result)
				df = pd.concat([df, df_to_add], ignore_index=True)
				df.to_csv(output_file, index=False)
			else: 
				df = pd.DataFrame(concept_result)
				df.to_csv(output_file, index=False)

