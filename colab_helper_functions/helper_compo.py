import PIL
import os
import json
from IPython import display
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import PIL.Image
from PIL import ImageOps
import csv
from itertools import combinations

def prepare_data():
    tasks = ['2DRotation', 'Colour', 'Counting', 'Reflect', 'Resize']
    base_path = "/content/multi/test"
    
    # Dictionary to count how many files have each combination (pair) of tasks.
    combination_counts = {}
    data_dict = {}  # To accumulate JSON data (answer keys)
    processed_count = 0  # Counter for processed JSON files
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                processed_count += 1  # Increment counter when a JSON file is found
                file_path = os.path.join(root, file)
                # Load the JSON file and merge its data into data_dict
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    data_dict.update(file_data)
                    
                filename_no_ext = file[:-5] # removes ".json"
                
                if filename_no_ext.startswith("output_"):  
                    filename_no_ext = filename_no_ext[len("output_"):]
                
                # Split the filename by underscores.
                segments = filename_no_ext.split('_')
                
                present_tasks = []
                # For each segment, check if it starts with a concept
                for seg in segments:
                    for task in tasks:
                        if seg.startswith(task):
                            present_tasks.append(task)
                
                # For files with ≥ 2 tasks, generate every pair of tasks
                if len(present_tasks) >= 2:
                    for i in range(len(present_tasks)):
                        for j in range(i + 1, len(present_tasks)):
                            pair = tuple(sorted([present_tasks[i], present_tasks[j]]))
                            combination_counts[pair] = combination_counts.get(pair, 0) + 1
    
    print("Combination Counts:")
    for pair, count in combination_counts.items():
        print(f"{pair[0]} and {pair[1]}: {count} types of combinations")
    
    print('----------------------------------------')
    #print(f"Total number of answer keys: {len(data_dict)}")
    print(f"Total number of combinations processed: {processed_count}")
    
    return data_dict

def show_concept_example(data_dict, concepts):
    """
    Displays an example image set for the given list of concepts.
    Randomly selects from available img_id values that contain all specified concepts.
    """
    matching_img_ids = [
        img_id for img_id in data_dict.keys()
        if all(concept in img_id for concept in concepts)
    ]

    if not matching_img_ids:
        print(f"No matching images found for concepts: {concepts}")
        return

    # Randomly select one matching img_id to display
    img_id = random.choice(matching_img_ids)
    img_info = data_dict[img_id]
    
    correct_label   = img_info['correct']    
    nochange_label  = img_info['nochange']   
    incorrect_label = img_info['incorrect']  

    # Helper function to map '(A)'/'(B)'/'(C)' to a subplot title
    def get_test_title(letter):
        if letter == correct_label:
            return "Test transformation: Correct"
        elif letter == nochange_label:
            return "Test transformation: No change"
        elif letter == incorrect_label:
            return "Test transformation: Incorrect"
        else:
            return f"Test {letter}"

    # Extract variation and regeneration numbers
    img_parts = img_id.split('_')
    numeric_parts = [part for part in img_parts if part.isdigit()]

    if len(numeric_parts) < 1:
        print(f"Warning: No variation number found in img_id '{img_id}'")
        return
    
    # Variation number (always present)
    variation_number = numeric_parts[0]

    # If we have at least 2 numeric parts, 2nd one is regeneration #
    regeneration_number = numeric_parts[1] if len(numeric_parts) > 1 else None

    # Construct correct train_id incl variation number (but not regeneration number)
    train_id_parts = img_parts[:img_parts.index(variation_number) + 1]
    train_id = '_'.join(train_id_parts)

    # Construct correct test_id including both variation + regeneration numbers
    test_id_parts = img_parts[:img_parts.index(variation_number) + 2] if regeneration_number else train_id_parts
    test_id = '_'.join(test_id_parts)

    # Extract the first two transformation parts
    first_trans = img_parts[0]
    second_trans = img_parts[1]

    print(img_info)

    # Construct image paths
    train1_path = f'/content/multi/{train_id}_train_{first_trans}.jpg'
    train2_path = f'/content/multi/{train_id}_train_{second_trans}.jpg'
    test0_path = f'/content/multi/{test_id}_test_0.jpg'
    test1_path = f'/content/multi/{test_id}_test_1.jpg'
    test2_path = f'/content/multi/{test_id}_test_2.jpg'

    print(f"Training transformation 1: {train1_path}")
    print(f"Training transformation 2: {train2_path}")
    print(f"Test transformation (A): {test0_path}")
    print(f"Test transformation (B): {test1_path}")
    print(f"Test transformation (C): {test2_path}")
    
    train1 = PIL.Image.open(train1_path)
    train2 = PIL.Image.open(train2_path)
    test0 = PIL.Image.open(test0_path)
    test1 = PIL.Image.open(test1_path)
    test2 = PIL.Image.open(test2_path)
    
    # Resize all images to the same max size
    max_size = (300, 300)
    train1.thumbnail(max_size)
    train2.thumbnail(max_size)
    test0.thumbnail(max_size)
    test1.thumbnail(max_size)
    test2.thumbnail(max_size)
    
    # Add a small border around each image
    border_size = 5
    border_color = 'black'
    train1_b = ImageOps.expand(train1, border=border_size, fill=border_color)
    train2_b = ImageOps.expand(train2, border=border_size, fill=border_color)
    test0_b = ImageOps.expand(test0, border=border_size, fill=border_color)
    test1_b = ImageOps.expand(test1, border=border_size, fill=border_color)
    test2_b = ImageOps.expand(test2, border=border_size, fill=border_color)
    
    # 1) DISPLAY THE TRAIN IMAGE
    fig_train, axes_train = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.85, wspace=0.2)
    
    axes_train[0].imshow(train1_b)
    axes_train[0].set_title("First training transformation", fontsize=12)
    axes_train[0].axis('off')
    
    axes_train[1].imshow(train2_b)
    axes_train[1].set_title("Second training transformation", fontsize=12)
    axes_train[1].axis('off')

    plt.show()

    # 2) DISPLAY THE THREE TEST IMAGES
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    
    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=0.8,  # more top space for titles
        wspace=0.2  # spacing between the 3 subplots
    )
    
    axes2[0].imshow(test0_b)
    axes2[0].set_title(get_test_title('(A)'), fontsize=10)
    axes2[0].axis('off')
    
    axes2[1].imshow(test1_b)
    axes2[1].set_title(get_test_title('(B)'), fontsize=10)
    axes2[1].axis('off')
    
    axes2[2].imshow(test2_b)
    axes2[2].set_title(get_test_title('(C)'), fontsize=10)
    axes2[2].axis('off')
    
    plt.show()
    
    print('----------------------------------------')
    print('img ID:', img_id)
    print('image_transform:', img_info['transform'])
    print('Correct Answer:', img_info['correct'])
    print('No Change Answer:', img_info['nochange'])
    print('Incorrect Answer:', img_info['incorrect'])

def display_stimuli(train1_path, train2_path, test0_path, test1_path, test2_path):
    # Open images
    train1 = PIL.Image.open(train1_path)
    train2 = PIL.Image.open(train2_path)
    test0 = PIL.Image.open(test0_path)
    test1 = PIL.Image.open(test1_path)
    test2 = PIL.Image.open(test2_path)
    
    # Resize all images to the same max size
    max_size = (300, 300)
    train1.thumbnail(max_size)
    train2.thumbnail(max_size)
    test0.thumbnail(max_size)
    test1.thumbnail(max_size)
    test2.thumbnail(max_size)
    
    # Add a small border around each image
    border_size = 5
    border_color = 'black'
    train1_b = PIL.ImageOps.expand(train1, border=border_size, fill=border_color)
    train2_b = PIL.ImageOps.expand(train2, border=border_size, fill=border_color)
    test0_b = PIL.ImageOps.expand(test0, border=border_size, fill=border_color)
    test1_b = PIL.ImageOps.expand(test1, border=border_size, fill=border_color)
    test2_b = PIL.ImageOps.expand(test2, border=border_size, fill=border_color)
    
    # 1) DISPLAY THE TRAIN IMAGES SIDE BY SIDE
    fig_train, axes_train = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.85, wspace=0.2)
    
    axes_train[0].imshow(train1_b)
    axes_train[0].set_title("First training transformation", fontsize=12)
    axes_train[0].axis('off')
    
    axes_train[1].imshow(train2_b)
    axes_train[1].set_title("Second training transformation", fontsize=12)
    axes_train[1].axis('off')
    
    plt.show()
    
    # 2) DISPLAY THE 3 TEST IMAGES SIDE BY SIDE
    fig_test, axes_test = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.8, wspace=0.2)
    
    axes_test[0].imshow(test0_b)
    axes_test[0].set_title("Test (A)", fontsize=10)
    axes_test[0].axis('off')
    
    axes_test[1].imshow(test1_b)
    axes_test[1].set_title("Test (B)", fontsize=10)
    axes_test[1].axis('off')
    
    axes_test[2].imshow(test2_b)
    axes_test[2].set_title("Test (C)", fontsize=10)
    axes_test[2].axis('off')
    
    plt.show()
    
def extract_concept_names(img_id):
    concepts = ["Colour", "2DRotation", "Counting", "Resize", "Reflect"]
    present_concepts = [concept for concept in concepts if concept in img_id]
    return present_concepts

def generate_cross_options(img_id,train_path):
    """
    Generate the 5 options based on the extracted concept names (1 correct, 2 incorrect, 1 no change, 1 doesn't apply)

    Parameters:
    img_id (str): The input string containing multiple concepts.

    Returns:
    tuple: A tuple containing the list of 5 options, the number of the correct option (ex, "(1)"), and the correct concepts.
    """

     # Extract the part after the last underscore (before .jpg) to get the relevant training transformation
    base = os.path.basename(train_path)  # e.g., "XXX_YYY_m_n_train_XXX.jpg"
    base_no_ext = os.path.splitext(base)[0]
    parts = base_no_ext.split('_')
    extracted = parts[-1]  

    concept_mapping = {
        "2DRotation": "orientation of objects",
        "Counting": "number of objects",
        "Colour": "color of objects",
        "Reflect": "orientation of objects",
        "Resize": "size of objects"
    }

   # Find which concept key is contained in 'extracted' (partial match).
    matching_concepts = [key for key in concept_mapping if key in extracted]
    
    if not matching_concepts:
        raise ValueError(
            f"No known concept found in extracted substring '{extracted}' for {img_id}.\n"
            "Make sure one of ['2DRotation','Counting','Colour','Reflect','Resize'] appears in the substring."
        )
    if len(matching_concepts) > 1:
        raise ValueError(
            f"Multiple concepts found in extracted substring '{extracted}' for {img_id}: {matching_concepts}"
        )

    chosen_concept = matching_concepts[0]
    correct_option = concept_mapping[chosen_concept]

    # Build distractor pool from other concepts' values
    distractor_pool = [v for k, v in concept_mapping.items() if k != chosen_concept]
    distractors = random.sample(distractor_pool, 2)

    no_change_option = "no change between pictures"
    doesnt_apply_option = "doesn't apply"
    
    options_list = [correct_option] + distractors + [no_change_option, doesnt_apply_option]
    randomized_options = random.sample(options_list, len(options_list))
    formatted_options = [f"({i+1}) {option.capitalize()}" for i, option in enumerate(randomized_options)]
    correct_option_number = f"({randomized_options.index(correct_option) + 1})"
    
    return formatted_options, correct_option_number, chosen_concept

def generate_within_options(img_id, img_info, train_path):
    """
    Generate 5 options for a within-concept scenario based on a single concept+parameter
    extracted from the train file name. For example, if the train file name ends
    in "2DRotation+90", then concept = "2DRotation", parameter = "+90".

    Returns a tuple:
      - A list of 5 formatted options
      - The correct option number as a string (e.g. "(1)")
      - A dictionary of correct parameters for each concept (here just one concept)
    """
    import random, os

    parameters_mapping = {
        "2DRotation": ["+90", "-90", "180"],
        "Counting":   ["+1", "+2", "-1", "-2"],
        "Colour":     ["Red", "Green", "Blue"],
        "Reflect":    ["X", "Y", "XY"],
        "Resize":     ["0.5XY", "2XY"]
    }

    def map_parameter_to_option(concept, parameter):
        if concept == "2DRotation":
            if parameter in ["+90", "-90"]:
                return "objects rotate by 90 degrees"
            else:
                return "objects rotate by 180 degrees"
        elif concept == "Counting":
            counting_type, value = parameter[0], parameter[1:]
            if counting_type == "+":
                return f"things go up by {value}"
            elif counting_type == "-":
                return f"things go down by {value}"
        elif concept == "Colour":
            return f"objects turn {parameter.lower()}"
        elif concept == "Reflect":
            if parameter == "X":
                return "objects flip upside down"
            elif parameter == "Y":
                return "objects flip sideways"
            else:
                return "objects rotate by 180 degrees"
        elif concept == "Resize":
            if parameter == "0.5XY":
                return "objects become smaller"
            elif parameter == "2XY":
                return "objects become bigger"
        return f"objects do something with {parameter}"

    # Extract final substring, e.g. "2DRotation+90"
    base = os.path.basename(train_path)
    base_no_ext = os.path.splitext(base)[0]
    parts = base_no_ext.split('_')
    extracted = parts[-1]  # e.g. "2DRotation+90"

    # Find which concept is contained in 'extracted' (partial match).
    possible_concepts = [c for c in parameters_mapping if c in extracted]
    if not possible_concepts:
        raise ValueError(
            f"No known concept found in substring '{extracted}' for {img_id}. "
            f"Must contain one of {list(parameters_mapping.keys())}."
        )
    if len(possible_concepts) > 1:
        raise ValueError(
            f"Multiple concepts found in substring '{extracted}' for {img_id}: {possible_concepts}"
        )

    chosen_concept = possible_concepts[0]

    # The parameter is what's left after removing the concept name from extracted.
    # e.g. "2DRotation+90" -> concept = "2DRotation", param_substring = "+90"
    param_substring = extracted[len(chosen_concept):]  # e.g. "+90"
    if not param_substring:
        raise ValueError(f"No parameter substring left after removing '{chosen_concept}' from '{extracted}'.")

    # Trim any leading or trailing punctuation/spaces
    param_substring = param_substring.strip()

    # Validate that param_substring is in the parameters_mapping for that concept
    possible_params = parameters_mapping[chosen_concept]  # <--- define possible_params here
    if param_substring not in possible_params:
        raise ValueError(
            f"Parameter '{param_substring}' not found among {possible_params} "
            f"for concept '{chosen_concept}' in img_id {img_id}"
        )

    correct_param = param_substring
    correct_option = map_parameter_to_option(chosen_concept, correct_param)

    # Build the pool of alternative (distractor) verbal options for the chosen concept.
    alt_params = [p for p in possible_params if p != correct_param]
    alternatives = [map_parameter_to_option(chosen_concept, p) for p in alt_params]

    # If fewer than 2 alternatives, sample additional options from other concepts.
    if len(alternatives) < 2:
        other_options = []
        for con, params in parameters_mapping.items():
            if con == chosen_concept:
                continue
            other_options.extend([map_parameter_to_option(con, p) for p in params])
        # Remove duplicates and sample as needed.
        other_options = list(set(other_options))
        needed = 2 - len(alternatives)
        if len(other_options) < needed:
            raise ValueError("Not enough distractor options available")
        alternatives += random.sample(other_options, needed)

    # Now choose 2 distractors.
    if len(alternatives) > 2:
        distractors = random.sample(alternatives, 2)
    else:
        distractors = alternatives

    # Add "no change between pictures" and "doesn't apply" if desired
    no_change_option = "no change between pictures"
    doesnt_apply_option = "doesn't apply"

    options_list = [correct_option] + distractors + [no_change_option, doesnt_apply_option]
    randomized_options = random.sample(options_list, len(options_list))

    formatted_options = [f"({i+1}) {opt.capitalize()}" for i, opt in enumerate(randomized_options)]
    correct_option_number = f"({randomized_options.index(correct_option) + 1})"

    correct_parameters = {chosen_concept: correct_param}
    return formatted_options, correct_option_number, correct_parameters


def extract_model_answer(response_text, type):
  if type == "numbers":
    options = ["(1)", "(2)", "(3)", "(4)", "(5)"]
  elif type == "letters":
    options = ["(A)", "(B)", "(C)", "(D)"]

  model_option = None
  earliest_index = len(response_text)

  for option in options:
      idx = response_text.find(option)
      if idx != -1 and idx < earliest_index:
          earliest_index = idx
          model_option = option

  return model_option if model_option else "Null"

def extract_earliest_letter(response_text):
    letters = ["(A)", "(B)", "(C)", "(D)"]
    found_letter = None
    earliest_index = len(response_text)

    for letter in letters:
        idx = response_text.find(letter)
        if idx != -1 and idx < earliest_index:
            earliest_index = idx
            found_letter = letter

    return found_letter if found_letter else "Null"

def is_already_processed(img_id, output_folder, transform):
    """
    Checks if a given image ID has already been processed.
    """
    output_file = os.path.join(output_folder, f"{transform}_results.csv")
    if os.path.exists(output_file):
        existing_results = pd.read_csv(output_file)
        if not existing_results[existing_results["Img_id"] == img_id].empty:
            return True
    return False

def process_cross_domain(img_id, img_path, general_cross_rule_prompt, model):
    """
    Runs the cross-domain processing.
    Returns:
      cross_options: list of answer options.
      correct_cross: the correct answer.
      correct_concept: the underlying concept.
      raw_cross_text: raw text response.
      model_cross_ans: parsed answer from the model.
      full_cross_ans: the full text answer corresponding to the choice.
    """
    # Generate cross-domain options and correct answer
    cross_options, correct_cross, correct_concept = generate_cross_options(img_id,img_path)
    options_str = ", ".join(cross_options)
    prompt = str(general_cross_rule_prompt).format(options_str)
    
    response_dict = model.run_model(prompt, image_path=img_path)
    raw_cross_text = response_dict["response"]
    print("Verbal Classification response:\n", raw_cross_text)
    
    model_cross_ans = extract_model_answer(raw_cross_text, "numbers")
    full_cross_ans = (
        cross_options[int(model_cross_ans.strip("()")) - 1][4:]
        if model_cross_ans != "Null" else "Null"
    )
    
    return cross_options, correct_cross, correct_concept, raw_cross_text, model_cross_ans, full_cross_ans

def process_within_domain(img_id, img_path, img_info, general_within_rule_prompt, model):
    """
    Runs the within-domain processing.
    Returns:
      within_options: list of answer options.
      correct_within: the correct answer.
      correct_param: parameter needed later in extrapolation.
      raw_within_text: raw text response.
      model_within_ans: parsed answer.
      full_within_ans: full text answer corresponding to the choice.
    """
    within_options, correct_within, correct_param = generate_within_options(img_id, img_info, img_path)
    options_str = ", ".join(within_options)
    prompt = str(general_within_rule_prompt).format(options_str)
    
    response_dict = model.run_model(prompt, image_path=img_path)
    raw_within_text = response_dict["response"]
    print("Verbal Specification response:\n", raw_within_text)
    
    model_within_ans = extract_model_answer(raw_within_text, "numbers")
    full_within_ans = (
        within_options[int(model_within_ans.strip("()")) - 1][4:]
        if model_within_ans != "Null" else "Null"
    )
    
    return within_options, correct_within, correct_param, raw_within_text, model_within_ans, full_within_ans

def process_extrapolation(
    img_id,
    img_paths,
    img_info,
    # correct_concept_x,
    # correct_param_x,
    # correct_concept_y,
    # correct_param_y,
    extrapolation_prompt,
    model
):
    """
    Runs the extrapolation step with multiple images.

    Parameters:
      img_id (str): The image identifier.
      img_paths (list of str): List of image paths (e.g., [train1_path, train2_path, test0_path, test1_path, test2_path]).
      img_info (dict): Dictionary of image metadata.
      correct_concept_x (str): The concept for the first transformation (e.g., "2DRotation").
      correct_param_x   (str): The parameter for the first transformation (e.g., "+90").
      correct_concept_y (str): The concept for the second transformation (e.g., "Colour").
      correct_param_y   (str): The parameter for the second transformation (e.g., "Red").
      extrapolation_prompt (str): The fixed prompt for extrapolation.
      model: The model object, which must have methods run_model(..., image_path=...) and the helper to parse answers.

    Returns:
      (raw_extra_text, model_extra_ans, full_extra_ans):
        - raw_extra_text: The raw text response from the model
        - model_extra_ans: The parsed (letter) answer, e.g. "(A)"
        - full_extra_ans: A more detailed interpretation of the chosen answer
    """

    def get_param(concepts, base_concept):
        for concept in concepts:
            if concept.startswith(base_concept):
                return concept[len(base_concept):]
        return None

    # combined_concepts = [f"{correct_concept_x}{correct_param_x}", f"{correct_concept_y}{correct_param_y}"]

    response_dict = model.run_model(extrapolation_prompt, image_path=img_paths)
    raw_extra_text = response_dict["response"]
    print("Visual Extrapolation response:\n", raw_extra_text)
    
    model_extra_ans = extract_model_answer(raw_extra_text, "letters")
    
    # Determine the full answer based on multiple conditions
    if model_extra_ans == "Null":
        full_extra_ans = "Null"
    elif model_extra_ans == img_info['correct']:
        full_extra_ans = img_info['transform'].split('_')
    elif model_extra_ans == img_info['incorrect']:
        concepts = img_info['transform'].split('_')
        incorrect_value = img_info['incorrect_test_output_value']

        if incorrect_value in ["Red", "Green", "Blue"]:
            incorrect_concept = "Colour"
        elif incorrect_value in ["X", "Y"]:
            incorrect_concept = "Reflect"
        elif incorrect_value in ["0.5XY", "2XY"]:
            incorrect_concept = "Resize"
        else:
            # Possibly a rotation or counting mismatch
            numeric_value = float(incorrect_value)
            if numeric_value == 0 or numeric_value >= 45:
                # Any number equal to 0 or >=45 is treated as rotation
                param = get_param(concepts, "2DRotation")
                if param in ["+90", "-90"]:
                    incorrect_concept = "2DRotation"
                    incorrect_value = "180"
                else:
                    incorrect_concept = "2DRotation"
                    incorrect_value = "+/-90"
            else:
                # Otherwise treat it as a counting mismatch
                param = get_param(concepts, "Counting")
                counting_type = param[0] if param else "+"
                opposite_sign = "+" if counting_type == "-" else "-"
                incorrect_concept = "Counting"
                incorrect_value = f"{opposite_sign}{1}"

        filtered_concepts = [c for c in concepts if not c.startswith(incorrect_concept)]
        filtered_concepts.append(f"{incorrect_concept}{incorrect_value}")
        full_extra_ans = filtered_concepts

    elif model_extra_ans == img_info['nochange']:
        full_extra_ans = "No change between pictures"
    elif model_extra_ans == "(D)":
        full_extra_ans = "Doesn't apply"
    else:
        full_extra_ans = None  # fallback if needed
    
    return raw_extra_text, model_extra_ans, full_extra_ans

def save_results(output_folder, transform, img_id, variation, regeneration,
                 raw_cross_text_X,raw_within_text_X,
                 raw_cross_text_Y,raw_within_text_Y, 
                 raw_extra_text,
                 model_cross_ans_X, correct_cross_X,
                 model_cross_ans_Y, correct_cross_Y,
                 model_within_ans_X, correct_within_X,
                 model_within_ans_Y, correct_within_Y,
                 model_extra_ans, correct_extra,
                 full_cross_ans_X, full_cross_ans_Y,
                 full_within_ans_X,full_within_ans_Y,
                 full_extra_ans):
    """
    Saves a single result entry to the CSV file.
    """

    # 1) Function to remove parentheses
    def remove_parentheses(val):
        if isinstance(val, str) and val.startswith("(") and val.endswith(")"):
            return val[1:-1]  # remove first "(" and last ")"
        return val

    # 2) Function to force numeric strings to be treated as text in Excel
    def force_excel_text_if_numeric(val):
        if isinstance(val, str):
            # Remove parentheses first if present
            val = remove_parentheses(val)
            # If now purely numeric (like "3", "12", or "-2"), prepend apostrophe
            # (This also catches a leading minus sign for negative numeric strings)
            # But you may not want negative numeric logic at all. 
            # If you want no negative sign, just handle the parentheses.
            if val.strip().lstrip("-").isdigit():
                return f"{val}"
        return val

    output_file = os.path.join(output_folder, f"{transform}_results.csv")
    if os.path.exists(output_file):
        # Read existing results
        existing_results = pd.read_csv(output_file)
    else:
        existing_results = pd.DataFrame(columns=[
            "Img_id", "Variation", "Regeneration", 
            "Full_Classification_X", "Full_Specifiation_X", 
            "Full_Classification_Y", "Full_Specifiation_Y", 
            "Full_Extrapolation", 
            "MCResponse_Classification_X", "MCCorrect_Classification_X", 
            "MCResponse_Classification_Y", "MCCorrect_Classification_Y", 
            "MCResponse_Specification_X", "MCCorrect_Specification_X", 
            "MCResponse_Specification_Y", "MCCorrect_Specification_Y", 
            "MCResponse_Extrapolation", "MCCorrect_Extrapolation", 
            "Response_Classification_X", "Response_Classification_Y", 
            "Response_Specification_X","Response_Specification_Y",
            "Response_Extrapolation"
        ])
    
    # Convert the model answers by removing parentheses
    model_cross_ans_X_str = force_excel_text_if_numeric(model_cross_ans_X)
    model_cross_ans_Y_str = force_excel_text_if_numeric(model_cross_ans_Y)
    model_within_ans_X_str = force_excel_text_if_numeric(model_within_ans_X)
    model_within_ans_Y_str = force_excel_text_if_numeric(model_within_ans_Y)
    model_extra_ans_str = force_excel_text_if_numeric(model_extra_ans)

    def is_correct(model_ans, correct_ans):
        model_val = model_ans.strip() if isinstance(model_ans, str) else ""
        correct_val = correct_ans.strip() if isinstance(correct_ans, str) else ""
        # If both are empty, return empty string
        if model_val == "" and correct_val == "":
            return ""
        # If one is empty but not the other, consider it incorrect
        if model_val == "" or correct_val == "":
            return 0
        return 1 if model_val == correct_val else 0

    result_entry = {
        "Img_id": img_id,
        "Variation": variation,
        "Regeneration": regeneration,

        # Full text of classification/specification/extrapolation steps
        "Full_Classification_X": raw_cross_text_X,
        "Full_Specification_X":  raw_within_text_X,     
        "Full_Classification_Y": raw_cross_text_Y,
        "Full_Specification_Y":  raw_within_text_Y,   
        "Full_Extrapolation":    raw_extra_text,

        # Model classification answers
        "MCResponse_Classification_X": model_cross_ans_X_str,
        "MCCorrect_Classification_X":  is_correct(model_cross_ans_X, correct_cross_X),
        "MCResponse_Classification_Y": model_cross_ans_Y_str,
        "MCCorrect_Classification_Y":  is_correct(model_cross_ans_Y, correct_cross_Y),

        # Model specification answers
        "MCResponse_Specification_X": model_within_ans_X_str,
        "MCCorrect_Specification_X":  is_correct(model_within_ans_X, correct_within_X),
        "MCResponse_Specification_Y": model_within_ans_Y_str,
        "MCCorrect_Specification_Y":  is_correct(model_within_ans_Y, correct_within_Y),

        # Model extrapolation answers
        "MCResponse_Extrapolation": model_extra_ans_str,
        "MCCorrect_Extrapolation":  is_correct(model_extra_ans, correct_extra),

        # Final interpreted responses
        "Response_Classification_X": full_cross_ans_X,
        "Response_Classification_Y": full_cross_ans_Y,
        "Response_Specification_X":  full_within_ans_X,
        "Response_Specification_Y":  full_within_ans_Y,
        "Response_Extrapolation":    full_extra_ans
    }

    df_to_add = pd.DataFrame([result_entry])
    if os.path.exists(output_file):
        df_to_add.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_to_add.to_csv(output_file, index=False)
    print(f"Saved results for {img_id} in {output_file}")

def display_image(img_path, max_size=(500, 500)):
    img = PIL.Image.open(img_path)
    img.thumbnail(max_size)
    display(img)

def map_single_concept(s: str):
    """
    Map partial strings like 'Counting-1', 'ReflectX' to short labels.
    Returns the concept in the same order it appears.
    """
    if "2DRotation" in s:
        return "Rotation"
    elif "Counting" in s:
        return "Number"
    elif "Colour" in s or "Color" in s:
        return "Color"
    elif "Reflect" in s:
        return "Reflection"
    elif "Resize" in s:
        return "Size"
    else:
        return s  # fallback

def parse_correctness(val):
    """
    Convert a cell to float(0 or 1) or NaN if empty.
    """
    if isinstance(val, float) and np.isnan(val):
        return np.nan
    if isinstance(val, str) and val.strip() == "":
        return np.nan
    return float(val)

def parse_csv_folder(folder_path):
    """
    Reads all *_results.csv in folder_path, extracts pair name, groups by Variation,
    and computes mean & SEM across Variation for each of the 3 tasks:
      - Verbal Classification
      - Verbal Specification
      - Visual Extrapolation

    Returns a long DataFrame with columns: [pair, task, mean, sem].
    """
    import glob
    csv_files = glob.glob(os.path.join(folder_path, "*_results.csv"))

    rows = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)  # e.g. "Counting-1_ReflectX_results.csv"
        core_name = filename.replace("_results.csv", "")  # "Counting-1_ReflectX"
        parts = core_name.split("_")
        if len(parts) != 2:
            print(f"Skipping {filename}: expected exactly 2 underscore parts but got {parts}")
            continue

        concept1 = map_single_concept(parts[0])
        concept2 = map_single_concept(parts[1])
        pair_name = f"{concept1} and {concept2}"

        df = pd.read_csv(file_path)

        # Convert the MCCorrect_* columns to numeric or NaN
        for col in ["MCCorrect_Classification_X", "MCCorrect_Classification_Y",
                    "MCCorrect_Specification_X", "MCCorrect_Specification_Y",
                    "MCCorrect_Extrapolation"]:
            df[col] = df[col].apply(parse_correctness)

        # classification = average of X/Y classification ignoring NaN
        df["classification"] = df[["MCCorrect_Classification_X","MCCorrect_Classification_Y"]].mean(axis=1, skipna=True)
        # specification = average of X/Y specification ignoring NaN
        df["specification"] = df[["MCCorrect_Specification_X","MCCorrect_Specification_Y"]].mean(axis=1, skipna=True)
        # extrapolation
        df["extrapolation"] = df["MCCorrect_Extrapolation"]

        # Group by Variation => average classification/specification/extrapolation
        grouped = df.groupby("Variation", dropna=False).agg({
            "classification": "mean",
            "specification": "mean",
            "extrapolation": "mean"
        }).reset_index()

        # Convert grouped columns to numpy arrays (one data point per Variation)
        cls_vals = grouped["classification"].dropna().values
        spec_vals = grouped["specification"].dropna().values
        extr_vals = grouped["extrapolation"].dropna().values

        # Compute final means
        classification_mean = cls_vals.mean() if len(cls_vals) else np.nan
        specification_mean  = spec_vals.mean() if len(spec_vals) else np.nan
        extrap_mean         = extr_vals.mean() if len(extr_vals) else np.nan

        # Standard error across Variation
        classification_sem = cls_vals.std(ddof=1)/np.sqrt(len(cls_vals)) if len(cls_vals)>1 else np.nan
        specification_sem  = spec_vals.std(ddof=1)/np.sqrt(len(spec_vals)) if len(spec_vals)>1 else np.nan
        extrap_sem         = extr_vals.std(ddof=1)/np.sqrt(len(extr_vals)) if len(extr_vals)>1 else np.nan

        rows.append({
            "pair": pair_name,
            "task": "Verbal Classification",
            "mean": classification_mean,
            "sem": classification_sem
        })
        rows.append({
            "pair": pair_name,
            "task": "Verbal Specification",
            "mean": specification_mean,
            "sem": specification_sem
        })
        rows.append({
            "pair": pair_name,
            "task": "Visual Extrapolation",
            "mean": extrap_mean,
            "sem": extrap_sem
        })

    plot_df = pd.DataFrame(rows).dropna(subset=["mean"], how="all")
    return plot_df

def plot_pair_data(plot_df):
    """
    Plots a grouped bar chart sorted by descending Visual Extrapolation mean.
    'plot_df' must have columns: pair, task, mean, sem.
    """
    import seaborn as sns

    # Order pairs by descending Visual Extrapolation
    ve_means = plot_df[plot_df["task"]=="Visual Extrapolation"][["pair","mean"]].copy()
    ve_means = ve_means.sort_values("mean", ascending=False)
    ordered_pairs = ve_means["pair"].tolist()

    plot_df["pair"] = pd.Categorical(plot_df["pair"], categories=ordered_pairs, ordered=True)

    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=plot_df, x="pair", y="mean", hue="task", order=ordered_pairs, ci=None)

    # Manually add error bars
    hue_order = ["Verbal Classification","Verbal Specification","Visual Extrapolation"]
    for i, row in plot_df.iterrows():
        pair_ = row["pair"]
        task_ = row["task"]
        mean_ = row["mean"]
        sem_  = row["sem"]
        # x pos is index_of(pair_) + offset
        x_idx = ordered_pairs.index(pair_)
        hue_idx = hue_order.index(task_)
        # typical bar width ~0.8, offset each hue by 0.2
        offset = (hue_idx - 1)*0.2
        x_pos = x_idx + offset
        plt.errorbar(x=x_pos, y=mean_, yerr=sem_, fmt="none", ecolor="black", capsize=3)

    plt.xticks(range(len(ordered_pairs)), ordered_pairs, rotation=45, ha="right")
    plt.xlabel("Transformation Pair")
    plt.ylabel("Average Score")
    plt.ylim(0,1.05)
    plt.title("Average Accuracy by Transformation Pair (Averaged Across Variation)")
    plt.legend(title="Task")
    plt.tight_layout()
    plt.show()