import PIL
import os
import json
import csv
import random
from IPython import display
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
from PIL import ImageOps

def prepare_data(presentation_type):
    image_categories = {task: [] for task in ['2DRotation', 'Colour', 'Counting', 'Reflect', 'Resize']}
    base_path = f"/content/{presentation_type}_image"

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                for task in image_categories.keys():
                    if file.startswith(task):
                        image_categories[task].append(os.path.join(root, file))

    data_dict = {}
    json_files = ['Colour.json','Resize.json','Counting.json','2DRotation.json','Reflect.json']
    total_images = 0

    json_base_path = os.path.join(base_path, "test")
    for json_file in json_files:
        with open(os.path.join(json_base_path, json_file), 'r') as f:
            data_dict = data_dict | json.load(f)

    print("Image Categories:")
    for task, images in image_categories.items():
        print(f"{task}: {len(images)} files")
        total_images += len(images)

    print('----------------------------------------')
    print(f"Total number of image files: {total_images}") # 7000 (1 train + 9 separate images per unique trial)
    print(f"Total number of answer keys: {len(data_dict)}") # 2100 (3 regenerations per unique trial)
 
    return data_dict

def show_concept_example(data_dict, concept, presentation_type):
    for img_id in data_dict.keys():
        if extract_concept_name(img_id) == concept:
            img_info = data_dict[img_id]

            if presentation_type == "single":
                img_path = f'/content/single_image/{img_id}_single.jpg'
                print(f"Matching transformation: {img_path}")
                
                img = PIL.Image.open(img_path)
                img.thumbnail((500, 500)) # Resize to be smaller

                border_size, border_color = 5, 'black'
                img_with_border = ImageOps.expand(img, border=border_size, fill=border_color)

                img_padded = ImageOps.expand(img_with_border, border=7, fill='white')

                # Display the image
                fig, ax = plt.subplots(figsize=(6, 6)) 
                ax.imshow(img_padded)
                ax.axis('off') 

                plt.show()

            elif presentation_type == "multi":
                def get_test_title(letter):
                    if letter == img_info['correct']:
                        return f"Test transformation: Correct"
                    elif letter == img_info['nochange']:
                        return f"Test transformation: No change"
                    elif letter == img_info['incorrect']:
                        return f"Test transformation: Incorrect"
                    else:
                        return f"Test {letter}"

                train_id = '_'.join(img_id.split('_')[:2])
                train_path = f'/content/multi_image/{train_id}_train.jpg'
                test0_path = f'/content/multi_image/{img_id}_test_0.jpg'
                test1_path = f'/content/multi_image/{img_id}_test_1.jpg'
                test2_path = f'/content/multi_image/{img_id}_test_2.jpg'

                print(f"Training transformation: {train_path}")
                print(f"Test transformation (A): {test0_path}")
                print(f"Test transformation (B): {test1_path}")
                print(f"Test transformation (C): {test2_path}")
                
                train = PIL.Image.open(train_path)
                test0 = PIL.Image.open(test0_path)
                test1 = PIL.Image.open(test1_path)
                test2 = PIL.Image.open(test2_path)
                
                # Resize all images to the same max size
                max_size = (300, 300)
                train.thumbnail(max_size)
                test0.thumbnail(max_size)
                test1.thumbnail(max_size)
                test2.thumbnail(max_size)
                
                # Add a small border around each image
                border_size, border_color = 5, 'black'
                train_b = ImageOps.expand(train, border=border_size, fill=border_color)
                test0_b = ImageOps.expand(test0, border=border_size, fill=border_color)
                test1_b = ImageOps.expand(test1, border=border_size, fill=border_color)
                test2_b = ImageOps.expand(test2, border=border_size, fill=border_color)
                
                # 1) DISPLAY THE TRAIN IMAGE
                # Make the figure roughly a square that matches the test subplots
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                plt.subplots_adjust(left=0, right=1, bottom=0, top=0.85)
                
                ax1.imshow(train_b)
                ax1.set_title("Training transformation", fontsize=12)
                ax1.axis('off')
                
                plt.show()
                
                # 2) DISPLAY THE THREE TEST IMAGES
                # For 3 images side by side, use width ~3× bigger than height
                fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
                
                plt.subplots_adjust(
                    left=0, right=1, bottom=0, top=0.8,  # More top space for titles
                    wspace=0.2  # Spacing between the 3 subplots
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
            print('img ID: ', img_id)
            print('image_transform: ', img_info['transform'])
            print('Correct Answer: ', img_info['correct'])
            print('No Change Answer: ', img_info['nochange'])
            print('Incorrect Answer: ', img_info['incorrect'])
            break

def display_stimuli(img_paths, presentation_type):
    if presentation_type == "single":
        img = PIL.Image.open(img_paths[0])
        img.thumbnail((500,500))
        img = ImageOps.expand(train, border=border_size, fill=border_color)
        display(img)

    elif presentation_type == "multi":
        train = PIL.Image.open(img_paths[0])
        test0 = PIL.Image.open(img_paths[1])
        test1 = PIL.Image.open(img_paths[2])
        test2 = PIL.Image.open(img_paths[3])
        
        # Resize all images to the same max size
        max_size = (300, 300)
        train.thumbnail(max_size)
        test0.thumbnail(max_size)
        test1.thumbnail(max_size)
        test2.thumbnail(max_size)
        
        # Add a small border around each image
        border_size = 5
        border_color = 'black'
        train_b = ImageOps.expand(train, border=border_size, fill=border_color)
        test0_b = ImageOps.expand(test0, border=border_size, fill=border_color)
        test1_b = ImageOps.expand(test1, border=border_size, fill=border_color)
        test2_b = ImageOps.expand(test2, border=border_size, fill=border_color)
        
        # 1) DISPLAY THE TRAIN IMAGE
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.85)
        
        ax1.imshow(train_b)
        ax1.axis('off')

        plt.show()
        
        # 2) DISPLAY THE 3 TEST IMAGES
        # For 3 images side by side, use width ~3× bigger than height
        fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
        
        plt.subplots_adjust(
            left=0, right=1, bottom=0, top=0.8,  # more top space for titles
            wspace=0.2  # spacing between the 3 subplots
        )
        
        axes2[0].imshow(test0_b)
        axes2[0].axis('off')
        
        axes2[1].imshow(test1_b)
        axes2[1].axis('off')
        
        axes2[2].imshow(test2_b)
        axes2[2].axis('off')
        
        plt.show()
    
def extract_concept_name(img_id):
    concepts = ["Colour", "2DRotation", "Counting", "Resize", "Reflect"]

    for concept in concepts:
        if img_id.startswith(concept):
            return concept

    return ""

def generate_cross_options(img_id):
    """
    Generate the 5 options based on the extracted concept name.

    Parameters:
    img_id (str): The input string containing the concept and parameter.

    Returns:
    tuple: A tuple containing the list of 5 options and the number of the correct option (ex, "(1)").
    """
    concept_mapping = {
        "2DRotation": "Orientation of objects",
        "Counting": "Number of objects",
        "Colour": "Color of objects",
        "Reflect": "Orientation of objects",
        "Resize": "Size of objects"
    }

    correct_concept = extract_concept_name(img_id)

    # Get the correct option text
    correct_option = concept_mapping.get(correct_concept, "")

    # Get 2 incorrect options by sampling from the remaining concepts
    all_options = list(set(concept_mapping.values()))
    all_options.remove(correct_option)
    sampled_incorrect = random.sample(all_options, 2)

    no_change_option = "No change between pictures"
    doesnt_apply_option = "Doesn't apply"

    randomized_first_three = random.sample([correct_option] + sampled_incorrect, 3)
    options = randomized_first_three + [no_change_option, doesnt_apply_option]

    formatted_options = [f"({i+1}) {option}" for i, option in enumerate(options)]

    # Determine the number of the correct option
    correct_option_number = f"({randomized_first_three.index(correct_option) + 1})"

    print(formatted_options)
    return formatted_options, correct_option_number, correct_concept

def generate_within_options(img_id, img_info):
    """
    Generate the 4 options for a within concept scenario based on the parameters.

    Parameters:
    img_id (str): The input string containing the concept and parameter.
    img_info (dict): Dictionary containing metadata about the image, including 'incorrect_test_output_value'.

    Returns:
    tuple: A tuple containing the list of 4 options and the number of the correct option (ex, "(1)").
    """
    parameters_mapping = {
        "2DRotation": ["+90", "-90", "180"],
        "Counting": ["+1", "+2", "-1", "-2"],
        "Colour": ["Red", "Green", "Blue"],
        "Reflect": ["X", "Y"],
        "Resize": ["0.5XY", "2XY"]
    }

    def map_parameter_to_option(concept, parameter):
        if concept == "2DRotation":
            degrees = 90 if parameter in ["+90", "-90"] else 180
            return f"Objects rotate by {degrees} degrees"
        elif concept == "Counting":
            counting_type, value = parameter[0], parameter[1:]
            if counting_type == "+":
                return f"Things go up by {value}"
            elif counting_type == "-":
                return f"Things go down by {value}"
        elif concept == "Colour":
            return f"Objects turn {parameter}"
        elif concept == "Reflect":
            if parameter == "X":
                return "Objects flip upside down"
            elif parameter == "Y":
                return "Objects flip sideways"
            else:
                return "Objects rotate by 180 degrees"
        elif concept == "Resize":
            if parameter == "0.5XY":
                return "Objects become smaller"
            elif parameter == "2XY":
                return "Objects become bigger"

    concept = extract_concept_name(img_id)

    # Get correct parameter
    possible_parameters = parameters_mapping.get(concept, [])
    correct_parameter = next((param for param in possible_parameters if param in img_id), None)

    # Get incorrect parameter
    if concept == "Counting":
        # Special case for "Counting": incorrect parameter is the opposite sign of correct parameter
        counting_type, value = correct_parameter[0], correct_parameter[1:]
        opposite_sign = "+" if counting_type == "-" else "-"
        incorrect_parameter = f"{opposite_sign}{value}"
    else:
        incorrect_parameter = img_info.get('incorrect_test_output_value', None)

    if incorrect_parameter not in possible_parameters:
        raise ValueError(f"Invalid incorrect parameter '{incorrect_parameter}' for concept '{concept}'.")

    # Map parameters to options
    correct_option = map_parameter_to_option(concept, correct_parameter)
    incorrect_option = map_parameter_to_option(concept, incorrect_parameter)

    no_change_option = "No change between pictures"
    doesnt_apply_option = "Doesn't apply"

    randomized_first_two = random.sample([correct_option, incorrect_option], 2)
    options = randomized_first_two + [no_change_option, doesnt_apply_option]

    formatted_options = [f"({i+1}) {option}" for i, option in enumerate(options)]

    # Determine the number of the correct option
    correct_option_number = f"({randomized_first_two.index(correct_option) + 1})"

    print(formatted_options)
    return formatted_options, correct_option_number, correct_parameter

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
    cross_options, correct_cross, correct_concept = generate_cross_options(img_id)
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
    within_options, correct_within, correct_param = generate_within_options(img_id, img_info)
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

def process_extrapolation(img_paths, img_info, correct_concept, correct_param, extrapolation_prompt, model):
    """
    Runs the extrapolation step with multiple images.
    Parameters:
      img_paths (list of str): List of image paths; 1 if "single" & 2 if "multi".
      img_info (dict): Dictionary of image metadata.
      correct_concept (str): The correct concept.
      correct_param (str): The correct parameter.
      extrapolation_prompt (str): The fixed prompt for extrapolation.
      model: The model object, which must have methods encode_image and run_model.
    Returns:
      raw_extra_text: The raw text response.
      model_extra_ans: The parsed (letter) answer.
      full_extra_ans: The final interpreted answer.
    """
    response_dict = model.run_model(extrapolation_prompt, image_path=img_paths)
    raw_extra_text = response_dict["response"]
    print("Visual Extrapolation response:\n", raw_extra_text)
    
    model_extra_ans = extract_model_answer(raw_extra_text, "letters")
    
    # Determine the full answer based on multiple conditions
    if model_extra_ans == "Null":
        full_extra_ans = "Null"
    elif model_extra_ans == img_info['correct']:
        full_extra_ans = correct_param
    elif model_extra_ans == img_info['incorrect']:
        if correct_concept == "2DRotation" and (correct_param == "+90" or correct_param == "-90"):
            full_extra_ans = "180"
        elif correct_concept == "2DRotation" and correct_param == "180":
            full_extra_ans = "90"
        elif correct_concept == "Counting":
            counting_type = correct_param[0]
            opposite_sign = "+" if counting_type == "-" else "-"
            full_extra_ans = f"{opposite_sign}{1}"
        else:
            full_extra_ans = img_info['incorrect_test_output_value']
    elif model_extra_ans == img_info['nochange']:
        full_extra_ans = "No change between pictures"
    elif model_extra_ans == "D":
        full_extra_ans = "Doesn't apply"
    else:
        full_extra_ans = None  # Fallback if needed
    
    return raw_extra_text, model_extra_ans, full_extra_ans

def save_results(output_folder, transform, img_id, variation, regeneration,
                 raw_cross, raw_within, raw_extra,
                 model_cross, correct_cross,
                 model_within, correct_within,
                 model_extra, correct_extra,
                 full_cross, full_within, full_extra):
    """
    Saves a single result entry to the CSV file.
    """
    output_file = os.path.join(output_folder, f"{transform}_results.csv")
    if os.path.exists(output_file):
        # Read existing results
        existing_results = pd.read_csv(output_file)
    else:
        existing_results = pd.DataFrame(columns=[
            "Img_id", "Variation", "Regeneration", 
            "Full#1", "Full#2", "Full#3", 
            "MCResponse#1", "MCResponse#2", "MCResponse#3", 
            "Response#1", "Response#2", "Response#3"
        ])

    result_entry = {
        "Img_id": img_id,
        "Variation": variation,
        "Regeneration": regeneration,
        "Full#1": raw_cross,
        "Full#2": raw_within,
        "Full#3": raw_extra,
        "MCResponse#1": int(model_cross == correct_cross),
        "MCResponse#2": int(model_within == correct_within),
        "MCResponse#3": int(model_extra == correct_extra),
        "Response#1": full_cross,
        "Response#2": full_within,
        "Response#3": full_extra
    }
    df_to_add = pd.DataFrame([result_entry])
    if os.path.exists(output_file):
        df_to_add.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_to_add.to_csv(output_file, index=False)
    print(f"Saved results for {img_id} in {output_file}")

def load_correctness_from_csv(folder_path):
    """
    Reads all CSV files in 'folder_path'. Each CSV has columns:
        Img_id, MCResponse#1, MCResponse#2, MCResponse#3
    with 0/1 indicating correctness for each phase.

    Returns:
        dict: ex {
            'ColourRed_0_1': [0, 1, 0],
            '2DRotation+90_0_0': [1, 1, 0],
            ...
        }
        where each key is the Img_id from the CSV, and the value is
        a list of correctness values [cross, within, extrapolation].
    """
    correctness_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(folder_path, filename)
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row["Img_id"]  # ex "ColourRed_0_1"
                    mc1 = int(row["MCResponse#1"])
                    mc2 = int(row["MCResponse#2"])
                    mc3 = int(row["MCResponse#3"])

                    correctness_dict[img_id] = [mc1, mc2, mc3]

    return correctness_dict

def init_analysis_results(answer_types, transform_types):
    """
    Initialize the nested dictionary for storing analysis results.
    """
    return {
        answer_type: {
            'transform_type_results': {
                # ex {'2DRotation': {'total': 0, 'correct': 0}, ...}
                transform: {'total': 0, 'correct': 0} for transform in transform_types
            },
            'trial_accuracies': {},
            'transform_type_aggregated': {
                transform: [] for transform in transform_types
            }
        }
        for answer_type in answer_types
    }

def update_analysis_results(analysis_results, data_dict, correctness_dict, answer_types):
    """
    Core loop to update analysis_results with correctness counts for
    each (img_id, answer_type). This handles:
      - trial_accuracies
      - transform_type_results
    """
    for img_id, correctness_list in correctness_dict.items():
        # correctness_list might be [1, 0, 1].

        # If data_dict is keyed by the same exact string:
        transform_type = data_dict[img_id]['transform']  # ex '2DRotation'

        # To parse out subcomponents from img_id:
        # ex "ColourRed_0_1" => transform_type_and_subtype="ColourRed", trial_number="0"
        parts = img_id.split("_")
        transform_type_and_subtype = parts[0]  # ex 'ColourRed'
        trial_number = parts[1]               # ex '0'
        # The 3rd part might be '1' if 3 underscores, etc.

        trial_id = f"{transform_type_and_subtype}_{trial_number}"

        for i, answer_type in enumerate(answer_types):
            is_correct_int = correctness_list[i]  # 0 or 1

            answer_analysis = analysis_results[answer_type]
            # Update trial_accuracies
            if trial_id not in answer_analysis['trial_accuracies']:
                answer_analysis['trial_accuracies'][trial_id] = {
                    'correct': 0,
                    'total': 0,
                    'transform_type': transform_type_and_subtype
                }
            answer_analysis['trial_accuracies'][trial_id]['correct'] += is_correct_int
            answer_analysis['trial_accuracies'][trial_id]['total'] += 1

            # Update transform_type_results
            if transform_type not in answer_analysis['transform_type_results']:
                answer_analysis['transform_type_results'][transform_type] = {'total': 0, 'correct': 0}
            answer_analysis['transform_type_results'][transform_type]['total'] += 1
            answer_analysis['transform_type_results'][transform_type]['correct'] += is_correct_int

    return analysis_results

def compute_accuracy_by_unique_trial(analysis_results):
    """
    Compute the per-trial accuracy and store in transform_type_results.
    """
    for answer_type, answer_analysis in analysis_results.items():
        for trial_id, trial_data in answer_analysis['trial_accuracies'].items():
            average_accuracy = trial_data['correct'] / trial_data['total']
            transform_type_and_subtype = trial_id.split("_")[0]

            # Ensure the transform_type_and_subtype key exists
            if transform_type_and_subtype not in answer_analysis['transform_type_results']:
                answer_analysis['transform_type_results'][transform_type_and_subtype] = {
                    'total': 0,
                    'correct': 0,
                    'trial_accuracies': []
                }

            # Ensure 'trial_accuracies' is in that dict
            if 'trial_accuracies' not in answer_analysis['transform_type_results'][transform_type_and_subtype]:
                answer_analysis['transform_type_results'][transform_type_and_subtype]['trial_accuracies'] = []

            # Append average accuracy
            answer_analysis['transform_type_results'][transform_type_and_subtype]['trial_accuracies'].append(average_accuracy)

    return analysis_results

def aggregate_by_transformation_category(analysis_results):
    """
    Aggregate the trial accuracies by their transformation category.
    """
    for answer_type, answer_analysis in analysis_results.items():
        for transform_type_and_subtype, results in answer_analysis['transform_type_results'].items():
            if 'trial_accuracies' in results:
                # Find which top-level transform key this subtype matches
                transformation_type = [
                    key for key in answer_analysis['transform_type_aggregated'].keys() 
                    if transform_type_and_subtype.startswith(key)
                ][0]
                answer_analysis['transform_type_aggregated'][transformation_type].extend(results['trial_accuracies'])

    return analysis_results

def print_subcategory_results(analysis_results):
    def get_label(answer_type):
        if answer_type == 'cross_domain':
            return 'Verbal Classification'
        elif answer_type == 'within_domain':
            return 'Verbal Specification'
        elif answer_type == 'extrapolation':
            return 'Visual Extrapolation'
        else:
            return answer_type # Fallback

    # Iterate over each answer type in the analysis results
    for answer_key, answer_analysis in analysis_results.items():
        # Map to a descriptive label
        label = get_label(answer_key)
        print(f"\n--- {label} -------------------------------------")
        for subcategory, results in answer_analysis['transform_type_results'].items():
            if 'trial_accuracies' in results:
                accuracies = results['trial_accuracies']
                mean_accuracy = np.mean(accuracies)
                std_dev_accuracy = np.std(accuracies)
                print(
                    f"{subcategory}: "
                    f"Mean accuracy = {round(mean_accuracy * 100, 2)}%, "
                    f"Standard deviation = {round(std_dev_accuracy * 100, 2)}%"
                )

def plot_results(analysis_results):
    """
    Plot mean accuracy and standard deviation by transformation category
    for each answer type. Only show a transformation concept if there is data for it.
    """
    # Build a pivoted data structure: for each answer type (mapped to its label),
    # Store transformation concept -> (mean accuracy, std error)
    data_by_answer = {
        "Verbal Classification": {},
        "Verbal Specification": {},
        "Visual Extrapolation": {}
    }

    # Process analysis_results: iterate over each answer type & transformation type
    for answer_key, answer_analysis in analysis_results.items():
        if answer_key == 'cross_domain':
            label = 'Verbal Classification'
        elif answer_key == 'within_domain':
            label = 'Verbal Specification'
        elif answer_key == 'extrapolation':
            label = 'Visual Extrapolation'
        else:
            label = answer_key
        
        for transform_type, accuracies in answer_analysis['transform_type_aggregated'].items():
            if accuracies:  # Only process if there is data
                # Relabel transformation types to a fixed concept name:
                if "colour" in transform_type.lower() or "color" in transform_type.lower():
                    concept = "Color"
                elif "resize" in transform_type.lower():
                    concept = "Size"
                elif "counting" in transform_type.lower():
                    concept = "Number"
                elif "rotation" in transform_type.lower():
                    concept = "Rotation"
                elif "reflect" in transform_type.lower():
                    concept = "Reflection"
                else:
                    concept = transform_type # Fallback: use original name

                mean_acc = np.mean(accuracies)
                std_err = np.std(accuracies) / np.sqrt(len(accuracies))
                data_by_answer[label][concept] = (mean_acc, std_err)
    
    # Define our preferred order of transformation concepts
    custom_order = ["Color", "Size", "Number", "Rotation", "Reflection"]
    # Determine which concepts actually have data (across any answer type)
    available_concepts = set()
    for answer_data in data_by_answer.values():
        available_concepts.update(answer_data.keys())
    # Filter the order so that only concepts with data are shown
    filtered_order = [concept for concept in custom_order if concept in available_concepts]
    
    # Set up the grouped bar chart
    plt.figure(figsize=(18, 10))
    bar_width = 0.25
    indices = np.arange(len(filtered_order))
    
    color_map = {
        "Verbal Classification": "royalblue",
        "Verbal Specification": "orange",
        "Visual Extrapolation": "forestgreen"
    }
    
    # Plot bars for each answer type
    # Loop over the fixed order for x-axis and only plot a bar if data exists
    for i, (answer_type, color) in enumerate(color_map.items()):
        for j, concept in enumerate(filtered_order):
            if concept in data_by_answer[answer_type]:
                mean_acc, std_err = data_by_answer[answer_type][concept]

                #print(f"{answer_type} - {concept}: Mean Accuracy = {mean_acc:.3f}, Std Err = {std_err:.3f}")

                # Only label the answer type once (ex, for the first concept)
                label_arg = answer_type if j == 0 else ""
                plt.bar(j + i * bar_width, mean_acc, yerr=std_err, capsize=5,
                        alpha=0.7, color=color, label=label_arg,
                        width=bar_width, edgecolor='black')
    
    plt.xticks(indices + bar_width, filtered_order)
    plt.xlabel('Transformation Type')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy by Transformation Type and Query Type')
    plt.xlim(-bar_width, len(filtered_order) - 1 + bar_width * len(color_map))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Query Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(pad=5.0)
    plt.subplots_adjust(right=0.85)
    plt.show()
