# KiVA: Kid-inspired Visual Analogies

KiVA: Kid-inspired Visual Analogies is a visual analogical reasoning benchmark designed to probe fundamental visual pattern recognition and abstraction skills in large multimodal models. The benchmark features real-world object transformations that are supported by developmental psychology and are tested to be solvable by young children as young as 3 years old. 


## Dataset Artifacts


* For *KiVA*: we provide the exact stitched image format (stitching images of objects before and after transformation) shown to models & humans in every trial under the transformed objects/stitched_KiVA folder.
* We provide the individual (without stitching) images of each object before and after transformation under the transformed objects/KiVA folder.

* For *KiVA-adults*: we provide the exact stitched image format (stitching images of objects before and after transformation) shown to models & humans in every trial under the transformed objects/stitched_KiVA-adults folder.
* We provide the individual (without stitching) images of each object before and after transformation under the transformed objects/KiVA-adults folder.
  
* We provide the individual PNG images for each object (sourced from [1] and [2]) in its original format under the untransformed objects folder. In particular, we used achiral objects for rotation and reflection (Achiral Objects for Reflect, 2DRotation folder), planar objects for resize (Planar Objects for Resize folder) to avoid ambiguous object transformations.

The python scripts that run the experiments (as outlined below) use the images under the transformed objects folder to construct our trial.

## Benchmarking the Models

Please find below instructions on how to reproduce our results for each model. 

The scripts that are in the "chat_systems" folder and are prefixed with "chat_system" generate the input data to the models, including the exact images and text on which they are evaluated. 

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

## Generating your own dataset with the Transformation scripts

Our transformation scripts take in any .png image file (note transparent background for color transformations) and perform transformations specified by the user, automatically saving output files annotated with the ground truth. The instructions to run the scripts for creating the transformed objects are as follows. 

### Prerequisites

Ensure the following Python libraries are installed:

```
pip install Pillow numpy
```

### PyTorch Transformations file prerequisites

Ensure the following Python libraries are installed:

```
pip install torch torchvision
```

### Step-by-Step Guide
* Prepare the images for transformation in the input directory
* Create an output directory for which the transformed images will be saved

Use the following commands to run the transformation scripts. Detailed documentation on the input parameters can be found in the code.

#### KiVA
This benchmark features 5 domains (14 subdomains) of transformations. It is solvable by young children.
```
python Transformations-KiVA.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials>
```
This is an example trial of 2D Rotation (+90°) in KiVA. Both train object (duck) and test object (boy) have the same input value of orientation (starting upright, 0°):

<img src="transformed%20objects/stitched_KiVA/single_image/gpt4/2DRotation_stitch/2DRotation%2B90_13_0.jpg" alt="2D Rotation +90° Example" width="60%">
The correct answer is (A). 

#### PyTorch KiVA
We also provide a PyTorch version of KiVA to enable users to generate new transformed objects on the fly.
```
python pytorch_transformations_kiva.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials> \
    --shuffle <true_or_false>
```

#### KiVA-adults
This benchmark involves more transformation subdomains (29 subdomains) and different input values, demanding further abstraction and generalization. It is solvable by adults.
```
python Transformations-KiVA-adults.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials>
```
This is an example trial of 2D Rotation (+45°, parameter not included in KiVA) in KiVA-adults. Both train object (duck) and test object (boy) have different input values of orientation (starting -135° for train, -90° for test):

<img src="transformed%20objects/stitched_KiVA-adults/single_image/llava/2DRotation_stitch/2DRotation%2B45_13_1.jpg" alt="2D Rotation +45° Example" width="60%">
The correct answer is (C). 

#### PyTorch KiVA-adults
We also provide a PyTorch version of KiVA-adults to enable users to generate new transformed objects on the fly.
```
python pytorch_transformations_kiva-adults.py \
    --input_directory <input_dir> \
    --output_directory <output_dir> \
    --transformation <transformation_type> \
    --parameter <parameter> \
    --next_index <start_index> \
    --trials <num_trials> \
    --shuffle <true_or_false>
```

#### KiVA-compositionality
This benchmark combines various transformation domains of KiVA to test compositionality. The user can specify the parameters of color, rotation, number, size and reflection in the following code:
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



[1] Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., ... & Vanhoucke, V. (2022, May). Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2553-2560). IEEE.

[2] Stojanov, S., Mishra, S., Thai, N. A., Dhanda, N., Humayun, A., Yu, C., ... & Rehg, J. M. (2019). Incremental object learning from contiguous views. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition (pp. 8777-8786).
