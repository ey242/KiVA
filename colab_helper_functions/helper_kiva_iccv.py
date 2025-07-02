import PIL
import os
import json
import csv
from IPython import display
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL import ImageOps
import textwrap
import random

def prepare_data(eval):
    extract_path = f"/content/{eval}"

    # Walk through the extracted folder to collect all .jpg files
    image_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".jpg"):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images for {eval} evaluation")
    return image_files

def show_concept_example(eval, image_files):
    if not image_files:
        print(f"No images found for evaluation: {eval}")
        return

    img_path = random.choice(image_files)
    img = PIL.Image.open(img_path)
    img.thumbnail((700, 700))
    display(img)

def display_all_prompts():
    step_by_step_text = "step-by-step"

    system_prompt = (
        "You are an excellent visual puzzle solver! You will be given a visual puzzle that requires using visual analogical reasoning. "
        f"You will think {step_by_step_text} and carefully examine the visual evidence before providing an answer. In each image, observe the left-to-right transformation of an object. "
        "The object picture on the left transforms to the object picture on the right. Denote the transformation in the first image as training transformation. "
        "The transformation involves changes of either the size, orientation, and/or number of an object. "
    )

    extrapolation_prompt = (
        "Now look at the next three images. Each image contains a left-to-right object transformations (marked by either (A), (B) or (C)). "
        "Which one of these three left-to-right transformations follows the identified transformation? "
        "In your answer start with the correct transformation letter first: (A) or (B) or (C). Answer with (D) if none of the options apply. "
        "Make sure to start with the correct letter first, then continue."
    )

    print("--- System Prompt -------------------------------------")
    print(textwrap.fill(system_prompt, width=100))
    print("--- Visual Extrapolation ------------------------------")
    print(textwrap.fill(extrapolation_prompt, width=100))

    return system_prompt, extrapolation_prompt

def display_stimuli(img_path):
    img = PIL.Image.open(img_path)
    img.thumbnail((500,500))
    border_size, border_color = 5, 'black' 
    img = ImageOps.expand(img, border=border_size, fill=border_color)
    display(img)

def extract_model_answer(response_text):
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

def is_already_processed(img_path, output_folder, username, eval):
    """
    Checks if the image (by filename without extension) has already been processed.
    Matches exactly against saved result 'id' fields.
    """
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    output_file = os.path.join(output_folder, f"{username}_{eval}_results.json")

    if not os.path.exists(output_file):
        return False

    with open(output_file, 'r') as file:
        data = json.load(file)

    # Check for exact match in the list of saved IDs
    return any(entry["id"] == image_id for entry in data)

def process_extrapolation(img_path, extrapolation_prompt, model):
    """
    Runs the extrapolation step with multiple images.
    Returns:
      raw_extra_text: The raw text response.
      model_extra_ans: The parsed (letter) answer.
      full_extra_ans: The final interpreted answer.
    """
    response_dict = model.run_model(extrapolation_prompt, image_path=img_path)
    print(response_dict)
    raw_extra_text = response_dict["response"]
    print("Visual Extrapolation response:\n", raw_extra_text)
    
    model_ans = extract_model_answer(raw_extra_text)
    
    return model_ans

def save_results(username, eval, output_folder, randomized_id, answer, image):
    """
    Saves a single result entry to a flat list of dicts:
    [
        {"id": ..., "answer": ...},
        ...
    ]
    """
    output_file = os.path.join(output_folder, f"{username}_{eval}_results.json")

    # Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new result
    data.append({
        "id": randomized_id,
        "answer": answer
    })

    # Save updated data
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print("-" * 80, "\n", f"Model response: {answer}")
    print(f"Saved results in {output_file}",  "\n", "-" * 80)
    display_stimuli(image)

# EVALUATING CORRECTNESS
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
