import os
import random
import argparse
import pandas as pd
from utils_multi import stitch_images_train, stitch_images_test, read_image

parser = argparse.ArgumentParser()

parser.add_argument('--concept', type=str, default="2DRotation", help='The concept to be tested')
parser.add_argument('--model', type=str, default="gpt4o", help='model')
parser.add_argument('--api_key', type=str, default="API-KEY", help='gpt4_api_key')
args = parser.parse_args()

concept = args.concept
query_repeats = 5 # Number of times to repeat process, set to None for max. # of trials with given stimuli
model_name = args.model

stimuli_directory = f"stimuli/KiVA/{concept}" # Insert object file directory
text_files_dir = f"stimuli/KiVA/trial_tracker/"
output_directory = f"output/multi_image/output_{model_name}/{concept}"

stitched_images_directory = f"{output_directory}/{concept}_stitch"

os.makedirs(output_directory, exist_ok=True)
os.makedirs(stitched_images_directory, exist_ok=True)
step_by_step_text = "step-by-step"

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

system_prompt = ("You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning.")
system_prompt += f"You will think {step_by_step_text} and carefully examine the visual evidence before providing an answer."

extrapolation_prompt = ("Now you are given three images. Each image contains a left-to-right object transformations (marked by either (A), (B) or (C) )."
                        "Which one of these three left-to-right transformations follows the identified transformation."
                        "Answer with the correct transformation letter first (A) or (B) or (C). Answer with (D) if none of options apply.")
extrapolation_prompt += f"then provide a {step_by_step_text} reasoning for your choice."

concept_to_parameters = {
    "2DRotation": (["+90", "-90", 180]),
    "Colour": (["Red", "Green", "Blue"]), 
    "Counting": (["+1","+2","-1","-2"]), 
    "Reflect": (["X", "Y"]), 
    "Resize": (["2XY", "0.5XY"]) 
}

concept_headers = [
    "Variation",
    "Regeneration",
    "Train_input",
    "Train_output",
    "Test_input",
    "Test_output",
    "Full",
    "MCResponse",
    "Response"
]

def update_concept_result(param):
    concept_result = {
            "Variation": [],
            "Regeneration": [],
            "Train_input": [],
            "Train_output": [],
            "Test_input": [],
            "Test_output": [],
            "Full": [],
            "MCResponse": [],
            "Response": [],
        }
    return concept_result
    
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

def word_mc_options(selected_mc_options):
    worded_options = []

    for option in selected_mc_options:
        if concept == "2DRotation":
            if option == "-90" or option == "+90":
                option = 90
            worded_options += [f"rotates by {option} degrees"]
        elif concept == "Counting":
            counting_type, option = option[0], option[1:]
            if counting_type == "+":
                worded_options += [f"increases by {option}"] 
            elif counting_type == "-":
                worded_options += [f"decreases by {option}"] 
        elif concept == "Colour":
            worded_options += [f"turns {option}"]
        elif concept == "Reflect":
            if option == "X":
                worded_options += [f"is reflected upside-down"]
            elif option == "Y":
                worded_options += [f"is reflected left-right"]  
        elif concept == "Resize":
            if option == "0.5XY":
                worded_options += [f"becomes smaller"]
            elif option == "2XY":
                worded_options += [f"becomes bigger"]
        
    return worded_options

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
    if heading is not None:
        alpha = ['A', 'B', 'C', 'D']
        if alpha[correct_file_index] in extracted_choice:
            concept_result[heading] += [param]
        elif alpha[incorrect_file_index] in extracted_choice:
            concept_result[heading] += [stimuli_mc_1]
        elif alpha[no_change_file_index] in extracted_choice:
            concept_result[heading] += ["No change"]
        elif "D" in extracted_choice:
            concept_result[heading] += ["Doesn't apply"]

    for answer in answers: 
        if answer == extracted_choice: 
            return True 

    return False

if model_name == "gpt4":
    from models.gpt4_model_multi import GPT4Model
    chat_model = GPT4Model(system_prompt, api_key=args.api_key, max_token=300)
elif model_name == "gpt4o":
    from models.gpt4o_model_multi import GPT4OModel
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

    random_queries = random.sample(range(len(stimuli_set)), query_repeats) # Randomly sample which variations if not max # variations available

    for query in random_queries:

        print("----------------------------------------------")
        print(f"Beginning Variation {query} of {random_queries}")

        for regeneration in range(3): 

            if os.path.exists(output_file): 
                df = pd.read_csv(output_file)
                if len(df[(df["Variation"] == query) & (df["Regeneration"] == regeneration)]) > 0:
                    print(f"Skipping Variation {query + 1} of {query_repeats}, already exists.")
                    continue

            retry_count = 0  
            regeneration_successful = False

            while retry_count < 2 and not regeneration_successful: # Redo regeneration if any of model responses are uncertain

                concept_result = update_concept_result(param)
                chat_model.init_history()

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

                if concept == "2DRotation" and param == "+90":
                    test_output_result = int(Test_input) + 90
                elif concept == "Counting":
                    counting_type, option = param[0], param[1:]
                    if counting_type == "+":
                        test_output_result = int(Test_input) + int(option)
                    elif counting_type == "-":
                        test_output_result = int(Test_input) - int(option)
                else:
                    test_output_result = param


                # Add pre-saved results to final responses file
                concept_result["Variation"] += [query]
                concept_result["Regeneration"] += [regeneration]
                concept_result["Train_input"] += [Train_input]
                concept_result["Train_output"] += [Train_output]
                concept_result["Test_input"] += [Test_input]
                concept_result["Test_output"] += [test_output_result]

                # Insert param in initi prompt
                initi_prompt =  ("Observe the left-to-right transformation of an object. The object picture on the left transforms to the object picture on the right."
                  f"Denote this transformation as training transformation. The picture on the left {word_mc_options([param])[0]} to become the picture on the right. ")

                # Set up train stimuli
                train_stimuli_set = format_files_by_type(stimuli_set, query, 'train')
                
                train0_image = stitch_images_train(read_image(f"{stimuli_directory}/{train_stimuli_set[0]}").convert("RGB"), read_image(f"{stimuli_directory}/{train_stimuli_set[1]}").convert("RGB"))
                train0_image_path = f"{stitched_images_directory}/{concept}{param}_{query}_{regeneration}_train.jpg"
                train0_image.save(train0_image_path)

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

                stitched_test_stimuli = stitch_images_test(stitched_images)

                test_stimuli_image_paths = [train0_image_path]

                for num, test_stimuli in enumerate(stitched_test_stimuli):
                    test_image_path = f"{stitched_images_directory}/{concept}{param}_{query}_{regeneration}_test{num}.jpg"
                    test_stimuli.save(test_image_path)
                    test_stimuli_image_paths += [test_image_path]

                # Testing ground truth within for extrapolation
                out_chatmodel = chat_model.run_model_multi(initi_prompt + extrapolation_prompt, final_image_paths=test_stimuli_image_paths, extrapolation_only=True)            
                test_stimuli_set += ["No change between pictures", "Doesn't apply"]

                labels = ["(A) ", "(B) ", "(C) ", "(D) "]
                labeled_choices = [label + arg for label, arg in zip(labels, test_stimuli_set)]
                extracted_letters = [item.split(" ")[0] for item in labeled_choices]

                concept_result["Full"] += [out_chatmodel["response"]]
                
                print("Response: ", out_chatmodel["response"])
                incorrect_extracted_letters = [item.split(" ")[0] for item in labeled_choices if item.split(" ")[0] != extracted_letters[correct_file_index]]

                all_choices = [option.split(" ")[0] for option in labeled_choices]
                if eval_response(out_chatmodel["response"], [extracted_letters[correct_file_index]], all_choices, "Response"):
                    concept_result["MCResponse"] += ["1"]
                    print("Correct response")
                
                elif eval_response(out_chatmodel["response"], incorrect_extracted_letters, all_choices):
                    concept_result["MCResponse"] += ["0"]
                    print("Incorrect response")
                else:
                    concept_result["MCResponse"] += ["Null"]
                    concept_result["Response"] += ["Null"]
                    print("Uncertain response")

                if "Null" not in [concept_result["MCResponse"][-1]]:
                    regeneration_successful = True
                elif retry_count == 2:
                    regeneration_successful = True
                elif retry_count <= 0:
                    retry_count += 1
                    print(f"Retrying due to null response; this is try {retry_count+1} of 3.")

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
