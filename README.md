# KiVA: Kid-inspired Visual Analogies

KiVA: Kid-inspired Visual Analogies is a visual analogical reasoning benchmark designed to probe fundamental visual pattern recognition and abstraction skills in large multimodal models. The benchmark features real-world object transformations that are supported by developmental psychology and are tested to be solvable by young children as young as 3 years old. 


## Dataset Artifacts


* For *KiVA*: we provide the exact stitched image format (stitching images of objects before and after transformation) shown to models & humans in every trial under the stimuli/stitched_KiVA folder.
* We provide the individual (without stitching) images of each object before and after transformation under the stimuli/KiVA folder.

* For *KiVA-adults*: we provide the exact stitched image format (stitching images of objects before and after transformation) shown to models & humans in every trial under the stimuli/stitched_KiVA-adults folder.
* We provide the individual (without stitching) images of each object before and after transformation under the stimuli/KiVA-adults folder.
  
* We provide the individual PNG images for each object (sourced from [1] and [2]) in its original format under the objects folder. 

The python scripts that run the experiments (as outlined below) use the images under the stimuli folder to construct our trial.

## Running the Transformation Scripts

The instructions to run the scripts for stimuli creation are as follows. 

### Prerequisites

Ensure the following Python libraries are installed:

```
pip install Pillow numpy
```

### Step-by-Step Guide
* Prepare the images for transformation in the input directory
* Create an output directory for which the transformed images will be saved

Run the scripts
* Use the following commands to run the transformation scripts:

#### Basic KiVA Transformations
```
python Transformations-KiVA.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials>
```

#### KiVA Compositionality
```
python Transformations-KiVA-compositionality.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --next_index <start_index> \
    --trials <num_trials> \
    --colour <colour_transformation> \
    --rotate <rotation_angle> \
    --count <math_operation> \
    --resize <resize_factor> \
    --reflect <reflection_axis> \
    --randomize <randomize_parameters>
```

#### KiVA Adults
```
python Transformations-KiVA-adults.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials>
```

## Benchmarking the Models

Please find below instructions on how to reproduce our results for each model. 

The scripts that are prefixed with "chat_system" generates the input data to the models, including the exact images and text on which they are evaluated. 

### GPT4

We support both single image and multiple image format for GPT4 on both KiVA (easy) and KiVA-adults (hard) experiments.

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: KiVA/KiVA-adults (easy/hard)
* SETTING: single/multi

Please run: 

```
python chat_system_[SETTING]_image_[DIFFICULTY].py --concept [CONCEPT] --model gpt4 --api_key ****
```

To evaluate GPT4's performance on the non-verbal visual extrapolation task only, please run:

```
python chat_system_[SETTING]_image_[DIFFICULTY])_extrapolation_only.py --concept [CONCEPT] --model gpt4 --api_key ****
```

### LLaVA

We support single image format for LLaVA-1.5. Please follow the [HuggingFace Tutorial](https://huggingface.co/liuhaotian/llava-v1.5-13b) for model installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: KiVA/KiVA-adults (easy/hard)

```
python chat_system_single_image_[DIFFICULTY].py --concept [CONCEPT] --model llava
```

### MANTIS

We support multiple image format for MANTIS. Please follow the [Model Github](https://tiger-ai-lab.github.io/Mantis/) for installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: KiVA/KiVA-adults (easy/hard)

```
python chat_system_multi_image_mantis_[DIFFICULTY].py --concept [CONCEPT]
```

In summary,

* to run KiVA on GPT-4V and LLaVA with a given training transformation and test transformation options altogether as a single image, run chat_system_single_image_kiva.py
* to run KiVA on GPT-4V with a given training transformation and test transformation options as multiple separate images, run chat_system_multi_image_kiva.py
* to run KiVA on MANTIS with a given training transformation and test transformation options as multiple separate images, run chat_system_multi_image_mantis_kiva.py

* to run KiVA-adults on GPT-4V and LLaVA with a given training transformation and test transformation options altogether as a single image, run chat_system_single_image_kiva-adults.py
* to run KiVA-adults on GPT-4V with a given training transformation and test transformation options as multiple separate images, run chat_system_multi_image_kiva-adults.py
* to run KiVA-adults on MANTIS with a given training transformation and test transformation options as multiple separate images, run chat_system_multi_image_mantis_kiva-adults.py

## Supplementary Prompts 

As outlined in our paper, we test additional prompt settings (instruction prompting, reflection prompting, code prompting and in-context learning) for KiVA. You can find the scripts for these attempts under the chat_systems_supp_prompting folder. 

## Model Data

We provide full output responses and scores of GPT4 (in both single-image and multi_image settings), LLaVA and MANTIS for KiVA and KiVA-adults under the model_data folder. See gpt4_multi_image_extrapolation and gpt4_single_image_extrapolation for results on the non-verbal visual extrapolation task only. 

## Human Data

We provide scores and reaction times of child and adult participants for KiVA (KiVA_children and KiVA_adults); scores and response times of adult participants for KiVA-adults (KiVA-adults_adults) under the human_data folder. Participant identifiers are removed for confidentiality.

## Basic Analysis

We provide some basic analyses that compare model and human performance on KiVA under the analysis folder. 

* Comparing models and humans: We provide two tables that summarize the mean accuracies and standard errors across 14 different transformations and 3 question types (cross-domain, within-domain and extrapolation). 
Performance_14Transformations_fullbenchmark_withoutkids.xlsx compares models to adults on the full benchmark. Performance_14Transformations_400samples_withkids.xlsx compares models to both children and adults on a random subset of the benchmark completed by children.

* Analyzing model errors and biases: We provide a table of the frequency of option labels selected by each model in Model_option_frequency.xlsx. We provide a table of the frequency of different types of wrong responses (incorrect transformation, "No change", "Doesn't apply") made by each model in Model_wrong_frequency.xlsx.

[1] Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., ... & Vanhoucke, V. (2022, May). Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2553-2560). IEEE.

[2] Stojanov, S., Mishra, S., Thai, N. A., Dhanda, N., Humayun, A., Yu, C., ... & Rehg, J. M. (2019). Incremental object learning from contiguous views. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition (pp. 8777-8786).
