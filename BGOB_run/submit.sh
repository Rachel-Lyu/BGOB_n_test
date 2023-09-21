#!/bin/bash
#SBATCH --job-name=BGOB
source ~/.bashrc
conda activate BGOB
python main.py --model_name ECC_90 --dataset demo.csv --delta_t 1 --hidden_size 300 --p_hidden 600 --mixing 0.0001 --epoch 300 --whole_dataset
