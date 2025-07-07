import PIL
import os
import json
import csv
from IPython import display
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.projections import PolarAxes
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.projections import register_projection
from matplotlib.spines import Spine 
import PIL.Image
from PIL import ImageOps
import textwrap
import random
from typing import Dict, Any, List, Tuple

# Define constants for data access (ensure these are consistent with your helper file)
LEVEL_KIVA_DB_KEY = 'kiva'
LEVEL_KIVA_FUNCTIONS_DB_KEY = 'kiva-functions'
LEVEL_KIVA_FUNCTIONS_COMPOSITIONALITY_DB_KEY = 'kiva-functions-compositionality'

TRANSFORMATIONS_FOR_SIMPLE_GROUP = ['Counting', 'Resizing', 'Reflect', 'Rotation']
TRANSFORMATIONS_FOR_COMPOSITE_GROUP = sorted(list(
    {'Counting,Reflect', 'Counting,Resizing', 'Counting,Rotation', 'Reflect,Resizing', 'Resizing,Rotation'}
))

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

def setup_kiva_data_set(name, base_data_path):
    """
    Downloads Kiva images and JSON for a given dataset ('train' or 'validation'),
    unzipping images into a flat folder structure within base_data_path,
    loading JSON, and then removing the downloaded zip file and macOS metadata.

    Args:
        name (str): 'train' or 'validation'.
        base_data_path (str): The base path where data should be stored (e.g., './data/').
    """
    print(f"\n--- Setting up {name} data within {base_data_path} ---")

    zip_file_name = f"{name}.zip" # e.g., 'train.zip'
    json_file_name = f"{name}.json" # e.g., 'train.json'

    # Define full paths for where files will reside
    target_img_dir = os.path.join(base_data_path, name) # e.g., './data/train/'
    full_json_path = os.path.join(base_data_path, json_file_name) # e.g., './data/train.json'
    
    # Temporarily download zip to the current working directory, then remove it
    download_zip_location = os.path.join(os.getcwd(), zip_file_name) 

    # 1. Create target directory for images (e.g., ./data/train)
    os.makedirs(target_img_dir, exist_ok=True)

    # 2. Download images zip file
    get_ipython().system(f"wget -q https://storage.googleapis.com/kiva-challenge/{zip_file_name} -O {download_zip_location}")

    # 3. Unzip images into the target image directory (e.g., ./data/train/)
    get_ipython().system(f"unzip -qo {download_zip_location} -d {target_img_dir}")

    # 4. FIX: Flatten nested directory if it occurred (e.g., ./data/train/train/ -> ./data/train/)
    nested_path = os.path.join(target_img_dir, name) # This would be like './data/train/train'
    if os.path.isdir(nested_path) and os.listdir(nested_path):
        get_ipython().system(f'mv {nested_path}/* {target_img_dir}/')
        get_ipython().system(f'rmdir {nested_path}') # Remove the now empty nested folder

    # 5. REMOVE: macOS metadata directories and files
    macos_dir = os.path.join(target_img_dir, '__MACOSX')
    if os.path.exists(macos_dir):
        get_ipython().system(f'rm -rf {macos_dir}')
    
    # Also remove any stray ._ files directly in target_img_dir or its subdirectories
    get_ipython().system(f'find {target_img_dir} -name "._*" -delete')

    # 6. Remove the original zip file
    get_ipython().system(f'rm {download_zip_location}')

    # 7. Download JSON annotations file directly to the data_path
    get_ipython().system(f"wget -q -O {full_json_path} \"https://storage.googleapis.com/kiva-key/{json_file_name}\"")

    # 8. Load JSON data
    with open(full_json_path,'r') as f:
        trials_data = json.load(f)
    print(f"Loaded {len(trials_data)} {name} trials from {json_file_name}")

    # 9. Use helper to prepare stimuli (collect image paths)
    stimuli_data = prepare_data(os.path.relpath(target_img_dir, '/content'))

    return trials_data, stimuli_data

def show_concept_example(eval, image_files):
    if not image_files:
        print(f"No images found for evaluation: {eval}")
        return

    img_path = random.choice(image_files)
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Trial_id: {image_id}")
    img = PIL.Image.open(img_path)
    img.thumbnail((700, 700))
    display(img)
    return image_id

def get_trial_info(trial_id, json_path):
    """
    Load trial metadata from a JSON file and return the entry for the given trial_id.
    
    Parameters:
    - trial_id (str): The ID of the trial (e.g., "0000").
    - json_path (str): Path to the JSON file containing trial data.
    
    Returns:
    - dict: A dictionary containing the trial's metadata, or None if not found.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from: {json_path}")
        return None
    
    entry = data.get(trial_id)
    if entry is None:
        print(f"Trial ID '{trial_id}' not found in {json_path}.")
        return None
    
    # Extract the desired fields
    keys = [
        "level",
        "transformation_domain",
        "concept",
        "correct",
        "incorrect1",
        "incorrect2",
        "train_object_name",
        "test_object_name",
        "train_input_value",
        "train_output_value",
        "test_input_value",
        "correct_test_value",
        "incorrect_test_output_value1",
        "incorrect_test_output_value2"
    ]
    trial_info = {k: entry.get(k) for k in keys}
    return trial_info

def display_all_prompts():
    step_by_step_text = "step-by-step"

    system_prompt = (
    "You are an excellent visual puzzle solver! "
    "You will be given a visual puzzle that requires using visual analogical reasoning. "
    "Each puzzle is presented as a single composite image, split into two regions:  "
    "• A **training example** at the top: a single object shown transforming from left to right.  "
    "• A **test panel** at the bottom: three candidate transformations of a new object, labelled (A), (B), and (C).  "
    "Your job is:  "
    "1. **Carefully inspect** the training example to identify exactly what changed (size, orientation, number, or a combination).  "
    "2. **Find** which one of the three test panels applies that **same** change to the new object. "
    "You will think step-by-step and carefully examine the visual evidence before providing an answer. "
    )
    
    extrapolation_prompt = (
    "Now look at the three bottom panels, labelled (A), (B), and (C). "
    "Which one of the three left-to-right object transformations (marked by either (A), (B) or (C)) "
    "on the bottom of the puzzle is the **same** as the left-to-right transformation on the top of the puzzle? "
    "Answer with the correct letter surrounded by parentheses (or (D) if none of the options apply), then provide a a step-by-step reasoning for your choice."
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

# --- Analysis Visualisation Functions ---
# radar plots
def radar_factory(num_vars: int, frame: str = 'circle'):
  theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
  class RadarTransform(PolarAxes.PolarTransform):
    def transform_path_non_affine(self, path):
      if path._interpolation_steps > 1:
        path = path.interpolated(num_vars)
      return Path(self.transform(path.vertices), path.codes)
  class RadarAxes(PolarAxes):
    name = 'radar'
    RESOLUTION = 1
    PolarTransform = RadarTransform
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.set_theta_zero_location('N')
    def fill(self, *args, closed=True, **kwargs):
      return super().fill(closed=closed, *args, **kwargs)
    def plot(self, *args, **kwargs):
      lines = super().plot(*args, **kwargs)
      for line in lines:
        self._close_line(line)
    def _close_line(self, line):
      x, y = line.get_data()
      if x[0] != x[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        line.set_data(x, y)
    def set_varlabels(self, labels):
      self.set_thetagrids(np.degrees(theta), labels, fontsize=10, linespacing=1.2)
    def _gen_axes_patch(self):
      if frame == 'circle':
        return patches.Circle((0.5, 0.5), 0.5)
      elif frame == 'polygon':
        return patches.RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor='k')
      else:
        raise ValueError("Unknown value for 'frame': %s" % frame)
    def _gen_axes_spines(self):
      if frame == 'circle':
        return super()._gen_axes_spines()
      elif frame == 'polygon':
        spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
        spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
        return {'polar': spine}
      else:
        raise ValueError("Unknown value for 'frame': %s" % frame)
  register_projection(RadarAxes)
  return theta

def radar_plot_pt(scores: Dict[str, List[float]], labels: List[str],
                  title: str, baselines: List[str], save_file: Any = None
                  ) -> None:
  theta = radar_factory(len(labels), frame='polygon')
  fig, axs = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
  fig.subplots_adjust(wspace=0.4, hspace=0.3, top=0.85, bottom=0.05)
  axs.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0]) 
  axs.set_ylim(0, 1) 
  for method, score in scores.items():
    if method in baselines:
      axs.plot(theta, score, color='red', linestyle='dashed', label=method) # Red for random baseline (chance level = 1/3)
    else:
      axs.plot(theta, score, color='blue', label=method) # Blue for 8-shot frequency
      axs.fill(theta, score, facecolor='blue', alpha=0.15, label='_nolegend_')
  axs.set_title(title, size=14, position=(0.1, 1.1), horizontalalignment='center', verticalalignment='center')
  axs.set_varlabels(labels)
  axs.legend(prop={'size': 14}, loc='upper right', bbox_to_anchor=(1.3, 1.))
  if save_file is not None:
    plt.savefig(save_file, bbox_inches='tight')
  plt.show()

# bar plots
def plot_tags(exp_results: Dict[str, float],
              tags: Dict[str, Any], title: str, save_file: Any = None, # Added title parameter
              width: float = 0.8, offset: float = 0.0) -> None:
  _, ax = plt.subplots(figsize=(18, 5))
  labels = [key for key in tags]
  plot_values = [exp_results.get(label, 0.0) for label in labels]

  for idx, label in enumerate(labels):
    ax.bar(idx - offset, plot_values[idx], color='#6495ED', width=width) # Set bars to blue

  ax.set_xticks(np.arange(len(labels)))
  ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
  
  ax.set_ylim(0, 1)
  ax.set_yticks(np.arange(0, 1.1, 0.2)) # Labels from 0 to 1 at intervals of 0.2
  ax.set_ylabel('Accuracy')
  ax.set_title(title, fontsize=14) # Set the title for the bar plot

  handles_exp = [plt.Rectangle((0, 0), 1, 1, color='#6495ED')]
  leg_exp = ax.legend(handles_exp, ['8-shot Frequency'], ncol=1, loc='upper left', bbox_to_anchor=(0.01, 0.99),
                   edgecolor='white')
  plt.gca().add_artist(leg_exp)

  plt.axhline(y=0.33, color='black', linestyle='dashed', label='Random Level (33%)') # Black for random baseline
  plt.legend(loc='upper right') # Ensure the 'Random Level' legend is visible
  plt.margins(x=0)

  if save_file is not None:
    plt.savefig(save_file, bbox_inches='tight')
  plt.show()

# --- Model Query & Evaluation ---

def extract_model_answer1(response_text):
  options = ["(A)", "(B)", "(C)", "(D)"]

  model_option = None
  earliest_index = len(response_text)

  for option in options:
      idx = response_text.find(option)
      if idx != -1 and idx < earliest_index:
          earliest_index = idx
          model_option = option

  return model_option if model_option else "Null"

def extract_model_answer(response_text):
    options = ["A", "B", "C", "D"]
    model_option = None
    earliest_index = len(response_text)

    for opt in options:
        paren_form = f"({opt})"
        bare_form = opt

        for form in [paren_form, bare_form]:
            idx = response_text.find(form)
            if idx != -1 and idx < earliest_index:
                earliest_index = idx
                model_option = paren_form  # Always return in (A) format

    return model_option if model_option is not None else "Null"

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
    
    return model_ans, raw_extra_text

def save_json_results(username, eval, output_folder, randomized_id, answer, image):
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

    #print("-" * 80, "\n", f"Model response: {answer}")
    print("-" * 80, "\n",f"Saved results in {output_file}",  "\n", "-" * 80)
    display_stimuli(image)

def save_csv_results(id, model_response, full_model_response, original_json_path, csv_out_path):
    import json, csv, os

    # Load original results JSON
    with open(original_json_path) as f:
        original_data = json.load(f)

    if id not in original_data:
        print(f"[Warning] ID {id} not found in original JSON.")
        return

    # Load existing IDs in CSV to avoid duplicates
    existing_ids = set()
    if os.path.exists(csv_out_path):
        with open(csv_out_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["id"])

    if id in existing_ids:
        return

    # Prepare row
    entry = original_data[id].copy()
    entry.pop("seed", None)
    entry["model_answer"] = model_response
    entry["model_response"] = full_model_response
    entry["id"] = id  # Put 'id' explicitly

    # Custom field order
    base_fields = list(entry.keys())
    if "model_response" in base_fields:
        base_fields.remove("model_response")
    if "model_answer" in base_fields:
        base_fields.remove("model_answer")

    # Insert model_answer before model_response
    if "correct" in base_fields:
        insert_idx = base_fields.index("correct")
    else:
        insert_idx = len(base_fields)

    base_fields.insert(insert_idx, "model_answer")
    base_fields.insert(insert_idx + 1, "model_response")

    fieldnames = base_fields

    # Write to CSV
    write_header = not os.path.exists(csv_out_path)
    with open(csv_out_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(entry)

    print(f"Saved results in {csv_out_path}", "\n", "-" * 80)

# EVALUATING CORRECTNESS
def load_correctness_from_csv(username, eval, output_folder):
    """
    
    """
    if eval != "train":
        print("Performance is unavailable here for non-train evaluations.")
        return {}

    # for a json file in correct_path = f"/content/kiva_{eval}"
    correct_path = f"/content/kiva_{eval}"
    answer_path = os.path.join(output_folder, f"{username}_{eval}_results.json")

    if not os.path.exists(correct_path):
        print(f"Correct path {correct_path} does not exist.")
        return {}

    correctness_dict = {}

    for filename in os.listdir(correct_path):
        if filename.endswith(".json"):
            json_path = os.path.join(correct_path, filename)
            with open(json_path, mode='r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    img_id = entry["id"]  # ex "ColourRed_0_1"
                    mc1 = int(entry["MCResponse#1"])
                    mc2 = int(entry["MCResponse#2"])
                    mc3 = int(entry["MCResponse#3"])

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
