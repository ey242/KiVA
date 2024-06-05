# KiVA: Kid-inspired Visual Analogies

KiVA: Kid-inspired Visual Analogies is a visual analogical reasoning benchmark designed to probe fundamental visual pattern recognition and abstraction skills in large multimodal models. The benchmark features real-world object transformations that are supported by developmental psychology and are tested to be solvable by young children at the age of 3. 


## Dataset Artifacts

* We provide the individual PNG images for each object (sourced from [1] and [2]) in its original format under *Objects*. 
* We provide the transformed images of each object required for running *KiVA* experiments under Stimuli/KiVA_stimuli. 
* We provide the transformed images of each object required for running the *KiVA-adults* experiments under Stimuli/KiVA_adults_stimuli.

The python scripts that run the experiments (as outlined below) use the images under Stimuli to construct our trial.


## Benchmarking the Models

Please find below instructions on how to reproduce our results for each model. 

### GPT4

We support both single image and multiple image format for GPT4 on both KiVA (easy) and KiVA-adults (hard) experiments.

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard
* SETTING: single/multi

Please run: 

```
python chat_system_[SETTING]_image_[DIFFICULTY].py --concept [CONCEPT] --model gpt4 --api_key ****
```

### LLaVA

We support single image format for LLaVA-1.5. Please follow the [HuggingFace Tutorial](https://huggingface.co/liuhaotian/llava-v1.5-13b) for model installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard

```
python chat_system_single_image_[DIFFICULTY].py --concept [CONCEPT] --model llava
```

### MANTIS

We support multiple image format for MANTIS. Please follow the [Model Github](https://tiger-ai-lab.github.io/Mantis/) for installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard

```
python chat_system_multi_image_mantis_[DIFFICULTY].py --concept [CONCEPT]
```

## Supplemntary Prompts 

As outlined in our paper, we test additional prompt settings for KiVA. You can find the scripts for these attempts under chat_systems_supp folder. 


[1] Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., ... & Vanhoucke, V. (2022, May). Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2553-2560). IEEE.
[2] Stojanov, S., Mishra, S., Thai, N. A., Dhanda, N., Humayun, A., Yu, C., ... & Rehg, J. M. (2019). Incremental object learning from contiguous views. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition (pp. 8777-8786).
