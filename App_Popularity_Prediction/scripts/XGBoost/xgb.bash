#!/bin/sh
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --mem=250000
#SBATCH --output=%j-xgb.out
srun xgbalgo.py