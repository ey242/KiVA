#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 4
#$ -j y
#$ -l h_rt=12:00:00

export PATH=/projectnb/ivc-ml/mqraitem/miniconda3/bin:$PATH
conda activate vlm_typo

python chat_system_multi_image.py $args


