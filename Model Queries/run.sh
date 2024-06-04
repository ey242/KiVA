#!/bin/bash -l

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l gpu_memory=48G
#$ -l h_rt=12:00:00

export PATH=/projectnb/ivc-ml/mqraitem/miniconda3/bin:$PATH
conda activate vlm_typo

python chat_system_one_image_hard.py $args


