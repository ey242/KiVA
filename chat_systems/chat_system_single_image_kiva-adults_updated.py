import os
import base64
import random
import argparse
import pandas as pd
from utils_single import stitch_images_train, stitch_images_test, read_image, stitch_final_images

parser = argparse.ArgumentParser()

parser.add_argument('--concept', type=str, default="2DRotation", help='The concept to be tested')
parser.add_argument('--model', type=str, default="gpt4o", help='model')
parser.add_argument('--api_key', type=str, default="API-KEY", help='gpt4_api_key')
args = parser.parse_args()

concept = args.concept
query_repeats = None # Number of times to repeat process, set to None for max. # of trials with given stimuli
model_name = args.model

stimuli_directory = f"stimuli/KiVA-adults/{concept}" # Insert object file directory
text_files_dir = f"stimuli/KiVA-adults/trial_tracker/"
output_directory = f"output/single_image_adults/output_{model_name}/{concept}"

stitched_images_directory = f"{output_directory}/{concept}_stitch"

os.makedirs(output_directory, exist_ok=True)
os.makedirs(stitched_images_directory, exist_ok=True)
step_by_step_text = "step-by-step"

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

system_prompt = ("You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning.")
system_prompt += f"You will think {step_by_step_text} and carefully examine the visual evidence before providing an answer."

initi_prompt =  ("You are given a visual puzzle. The puzzle features a left-to-right transformation of an object on top and three left-to-right"
                 "transformations of a different object on the bottom marked by (A) or (B) or (C)."
                  "The transformations involve a change of either the size, orientation, number, or color of an object")

general_cross_rule_prompt =  initi_prompt + ("Which one of the following rules {} best describes the left-to-right transformation on top of the"
                             "puzzle where the picture on the left transforms to the picture on the right? Answer with the correct rule number")
general_cross_rule_prompt += f"surrounded by parentheses, then provide a {step_by_step_text} reasoning for your choice."

general_within_rule_prompt = ("Which one of the following rules {} best describes the left-to-right transformation in the top of the puzzle where the picture"
                               "on the left transforms to the picture on the right?. Answer with the correct rule number surrounded by parentheses,")
general_within_rule_prompt += f"then provide a {step_by_step_text} reasoning for your choice."

extrapolation_prompt = ("Which one of three left-to-right object transformations (marked by either (A), (B) or (C) ) on the bottom of the puzzle is"
                         "the same as the left-to-right transformation on the top of the puzzle?"
                        "Answer with the correct letter surrounded by parentheses (or (D) if none of the options apply), ")
extrapolation_prompt += f"then provide a {step_by_step_text} reasoning for your choice."

concept_to_parameters = {
    "2DRotation": (["+45", "-45", "+90", "-90", "+135", "-135", 180]), #7
    "Colour": (["Red", "Yellow", "Green", "Blue", "Grey"]), #5
    "Counting": (["+1","+2","-1","-2","x2","x3","d2","d3"]), #8
    "Reflect": (["X", "Y", "XY"]), #3
    "Resize": (["0.5X", "0.5Y", "0.5XY", "2X", "2Y", "2XY"]) #6
}

concept_headers = [
    "Variation",
    "Regeneration",
    "Train_input",
    "Train_output",
    "Test_input",
    "Test_output",
    "Full#1",
    "Full#2",
    "Full#3",
    "MCResponse#1",
    "MCResponse#2",
    "MCResponse#3",
    "Response#1",
    "Response#2",
    "Response#3"
]

def update_concept_result():
    concept_result = {
            "Variation": [],
            "Regeneration": [],
            "Train_input": [],
            "Train_output": [],
            "Test_input": [],
            "Test_output": [],
            "Full#1": [],
            "Full#2": [],
            "Full#3": [],
            "MCResponse#1": [],
            "MCResponse#2": [],
            "MCResponse#3": [],
            "Response#1": [],
            "Response#2": [],
            "Response#3": []
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
            if option == "180" or option == 180:
                worded_options += ["Objects turn 180 degrees"]
            else: 
                worded_options += [f"Objects turn {option[1:]} degrees"]
        elif concept == "Counting":
            counting_type, option = option[0], option[1:]
            if counting_type == "+":
                worded_options += [f"Things go up by {option}"] 
            elif counting_type == "-":
                worded_options += [f"Things go down by {option}"] 
            elif counting_type == "x":
                worded_options += [f"Things multiply by {option}"] 
            elif counting_type == "d":
                worded_options += [f"Things divide by {option}"] 
        elif concept == "Colour":
            worded_options += [f"Objects turn {option}"]
        elif concept == "Reflect":
            if option == "X":
                worded_options += [f"Objects flip upside down"]
            elif option == "Y":
                worded_options += [f"Objects flip sideways"]
            elif option == "XY":
                worded_options += [f"Objects flip sideways and upside down"]
        elif concept == "Resize":
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
        
    return worded_options

def add_angles(x, y):
    if x.startswith('+'):
        angle_x = int(x[1:]) 
    elif x.startswith('-'):
        angle_x = -int(x[1:])
    else:
        angle_x = int(x)  
    
    if y.startswith('+'):
        angle_y = int(y[1:]) 
    elif y.startswith('-'):
        angle_y = -int(y[1:])
    else:
        angle_y = int(y)  
    
    result = angle_x + angle_y
    
    if result == 180:
        result_str = str(result)
    elif result == -180:
        result_str = str(result)[1:]
    else:
        result_str = f"+{result}" if result > 0 else f"{result}"
    
    return result_str

def eval_response(response, answers, all_choices, heading=None, all_descriptions=None): 
    all_available_choices = {}
    for choice in all_choices:
        if choice in response:
            all_available_choices[choice] = response.index(choice)

    if len(all_available_choices) == 0:
        return False
    
    # Get the earliest choice
    extracted_choice = min(all_available_choices, key=all_available_choices.get)

    # Add the extracted choice + description to concept result
    if heading == "Response#3":
        alpha = ['A', 'B', 'C', 'D']
        if alpha[correct_file_index] in extracted_choice:
            concept_result[heading] += [param]
        elif alpha[incorrect_file_index] in extracted_choice:
            concept_result[heading] += [stimuli_mc_1]
        elif alpha[no_change_file_index] in extracted_choice:
            concept_result[heading] += ["No change"]
        elif "D" in extracted_choice:
            concept_result[heading] += ["Doesn't apply"]
    elif heading is not None:
        extracted_index = int(extracted_choice[1]) - 1
        # Check if the index is within the range of all_descriptions
        if 0 <= extracted_index < len(all_descriptions):
            extracted_choice_description = all_descriptions[extracted_index]
            concept_result[heading] += [extracted_choice_description]

    for answer in answers: 
        if answer == extracted_choice: 
            return True 

    return False

if model_name == "llava":
    from models.llava_model import LLavaModel
    chat_model =  LLavaModel(system_prompt, max_token = 300)

if model_name == "gpt4":
    from models.gpt4_model import GPT4Model
    chat_model = GPT4Model(system_prompt, api_key='API-KEY', max_token=300)
elif model_name == "gpt4o":
    from models.gpt4o_model import GPT4OModel
    chat_model = GPT4OModel(system_prompt, api_key=args.api_key, max_token=300)

else: 
    raise ValueError("Model name not recognized.")

for param in concept_to_parameters[concept]:
    stimuli_set = get_indexed_files(param)
    output_file = output_directory + f"/{concept}{param}.csv"
    if query_repeats is None:
        query_repeats = len(stimuli_set)
    
    print("----------------------------------------------")
    print(f"Beginning Sub-Concept {concept} {param}")

    for query in list(range(query_repeats)):

        print("----------------------------------------------")
        print(f"Beginning Variation {query + 1} of {query_repeats}")

        for regeneration in range(3):

            if os.path.exists(output_file): 
                df = pd.read_csv(output_file)
                if len(df[(df["Variation"] == query) & (df["Regeneration"] == regeneration)]) > 0:
                    print(f"Skipping Variation {query + 1} of {query_repeats}, already exists.")
                    continue

            retry_count = 0  
            regeneration_successful = False

            while retry_count < 2 and not regeneration_successful: # Redo regeneration if any of model responses are uncertain

                chat_model.init_history()
                concept_result = update_concept_result()

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
                    
                stimuli_mc_1 = mc_1

                # Alter necessary variables as needed
                if concept == "Counting":
                    counting_type, option = param[0], param[1:]
                    if counting_type == "+":
                        stimuli_mc_1 = "-1"
                        if option == "1":
                            mc_1 = "-1"
                        elif option == "2":
                            mc_1 = "-2"
                    elif counting_type == "-":
                        stimuli_mc_1 = "+1"
                        if option == "1":
                            mc_1 = "+1"
                        elif option == "2":
                            mc_1 = "+2"
                    elif counting_type == "x":
                        stimuli_mc_1 = "+1"
                        if option == "2":
                            mc_1 = "d2"
                        elif option == "3":
                            mc_1 = "d3"
                    elif counting_type == "d":
                        stimuli_mc_1 = "-1"
                        if option == "2":
                            mc_1 = "x2"
                        elif option == "3":
                            mc_1 = "x3"

                if concept == "2DRotation" and param != 180:
                    test_output_result = add_angles(Test_input, param)
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


                # Set up train stimuli
                train_stimuli_set = format_files_by_type(stimuli_set, query, 'train')
                train_image = stitch_images_train(read_image(f"{stimuli_directory}/{train_stimuli_set[0]}").convert("RGB"), read_image(f"{stimuli_directory}/{train_stimuli_set[1]}").convert("RGB"))

                # Set up test stimuli
                test_stimuli_set = format_files_by_type(stimuli_set, query, 'test')
                test_stimuli_input, test_stimuli_set = test_stimuli_set[0], test_stimuli_set[1:]

                test_stimuli_set.append(test_stimuli_input)
                correct_file = test_stimuli_set[0] # Find the correct file filename
                incorrect_param_file = test_stimuli_set[1] # Find the incorrect param file filename
                no_change_file = test_stimuli_set[2] # Find the no change file filename
                random.shuffle(test_stimuli_set) # Shuffle the files in the set
                correct_file_index = test_stimuli_set.index(correct_file) # Find the index of the correct filename in the set
                incorrect_file_index = test_stimuli_set.index(incorrect_param_file) # Find the index of the incorrect param filename in the set
                no_change_file_index = test_stimuli_set.index(no_change_file) # Find the index of the no change filename in the set

                stitched_images = [read_image(f"{stimuli_directory}/{test_stimuli_input}").convert("RGB")] 
                for num, test_stimuli in enumerate(test_stimuli_set):
                    stitched_image = read_image(f"{stimuli_directory}/{test_stimuli}").convert("RGB")
                    stitched_images.append(stitched_image) 

                test_stimuli_image = stitch_images_test(stitched_images)

                labels = ["(A) ", "(B) ", "(C) ", "(D) "]
                labeled_choices = [label + arg for label, arg in zip(labels, test_stimuli_set)]
                extracted_letters = [item.split(" ")[0] for item in labeled_choices]
                correct = extracted_letters[correct_file_index]

                # Set up final stitched train + test stimuli image
                final_image = stitch_final_images(train_image, test_stimuli_image)
                final_image_path = f"{stitched_images_directory}/{concept}{param}_{query}_{regeneration}_{correct}.jpg"
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
                if eval_response(out_chatmodel["response"], [labeled_correct_cross], all_choices, "Response#1", selected_cross_choices):
                    concept_result["MCResponse#1"] += ["1"]
                    print(f"Correct cross response")
                elif eval_response(out_chatmodel["response"], labeled_incorrect_cross, all_choices): 
                    concept_result["MCResponse#1"] += ["0"]
                    add_result = False
                    print(f"Incorrect cross response")
                else:
                    concept_result["MCResponse#1"] += ["Null"]
                    concept_result["Response#2"] += ["Null"]
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
                    if eval_response(out_chatmodel["response"], [labeled_correct_within], all_choices, "Response#2", potential_choices):
                        concept_result["MCResponse#2"] += ["1"]
                        print(f"Correct within response")
                    elif eval_response(out_chatmodel["response"], labeled_incorrect_within, all_choices):
                        concept_result["MCResponse#2"] += ["0"]
                        print(f"Incorrect within response")
                    else:
                        concept_result["MCResponse#2"] += ["Null"]
                        concept_result["Response#2"] += ["Null"]
                        print(f"Uncertain within response")

                    print("="*20)

                else:
                    concept_result["Full#2"] += ["[Skipped mcq#2, incorrect previous response]"]
                    concept_result["MCResponse#2"] += [""]
                    concept_result["Response#2"] += [""]

                out_chatmodel = chat_model.run_model(extrapolation_prompt)            
                test_stimuli_set += ["No change between pictures", "Doesn't apply"]

                labels = ["(A) ", "(B) ", "(C) ", "(D) "]
                labeled_choices = [label + arg for label, arg in zip(labels, test_stimuli_set)]
                extracted_letters = [item.split(" ")[0] for item in labeled_choices]

                concept_result["Full#3"] += [out_chatmodel["response"]]
                
                print("Extrapolation: ", out_chatmodel["response"])
                incorrect_extracted_letters = [item.split(" ")[0] for item in labeled_choices if item.split(" ")[0] != extracted_letters[correct_file_index]]

                all_choices = [option.split(" ")[0] for option in labeled_choices]
                if eval_response(out_chatmodel["response"], [extracted_letters[correct_file_index]], all_choices, "Response#3"):
                    concept_result["MCResponse#3"] += ["1"]
                    print("Correct response")
                
                elif eval_response(out_chatmodel["response"], incorrect_extracted_letters, all_choices):
                    concept_result["MCResponse#3"] += ["0"]
                    print("Incorrect response")
                else:
                    concept_result["MCResponse#3"] += ["Null"]
                    concept_result["Response#3"] += ["Null"]
                    print("Uncertain response")


                if "Null" not in [concept_result["MCResponse#1"][-1], concept_result["MCResponse#2"][-1], concept_result["MCResponse#3"][-1]]:
                    regeneration_successful = True
                elif retry_count == 2:
                    regeneration_successful = True
                elif retry_count <= 0:
                    retry_count += 1
                    print(f"Retrying due to null response; this is try {retry_count} of 3.")

            results.append(concept_result)

            print("="*20)

            if os.path.exists(output_file): 
                df = pd.read_csv(output_file)
                df_to_add = pd.DataFrame(concept_result)
                df = pd.concat([df, df_to_add], ignore_index=True)
                df.to_csv(output_file, index=False)
            else: 
                df = pd.DataFrame(concept_result)
                df.to_csv(output_file, index=False)
