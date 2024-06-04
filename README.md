# KiVA: Kid-inspired Visual Analogies

KiVA: Kid-inspired Visual Analogies is a visual analogical reasoning benchmark designed to probe fundamental visual pattern recognition and abstraction skills in large multimodal models. The benchmark features real-world object transformations that are supported by developmental psychology and are tested to be solvable by young children at the age of 3. 


## Dataset Artifacts

* We provide the individual PNG images for each object in the dataset in its original format under *Objects100*. 
* We provide the transformed images of each object required for the *easy* experiment under Stimuli/Stimuli_50. 
* We provide the transformed images of each object required for the *hard* experiment under Stimuli/Stimuli_50_Hard.

The python scripts that run the experiments (as outlined below) use the images under Stimuli to construct our puzzle.


## Benchmarking the Models

Please find below instructions on how to reproduce our results for each model. 

### GPT4

We support both single image and multiple image format for GPT4 on both the easy and hard experiments. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard
* SETTING: single/multi

Please run: 

```
python chat_system_[SETTING]_image_[DIFFICULTY].py --concept [CONCEPT] --model gpt4 --api_key ****
```

### LLaVA

We support single image format for LLaVA 1.5. Please Follow the [HuggingFace Tutorial](https://huggingface.co/liuhaotian/llava-v1.5-13b) for model installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard

```
python chat_system_single_image_[DIFFICULTY].py --concept [CONCEPT] --model llava
```

### MANTIS

We support multiple image format for Mantis. Please follow the [Model Github](https://tiger-ai-lab.github.io/Mantis/) for installation. 

Given: 
* CONCEPT: 2DRotation/Colour/Resize/Reflect/Counting
* DIFFICULTY: easy/hard

```
python chat_system_multi_image_mantis_[DIFFICULTY].py --concept [CONCEPT]
```

## Supplemntary Prompts 

As outlined in our paper, we test additional prompt settings for the easy benchmark. You can find the scripts for these attempts under chat_systems_supp folder. 
