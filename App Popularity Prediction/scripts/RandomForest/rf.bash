#!/bin/sh
#SBATCH --cpus-per-task=64
#SBATCH --mem=250000
#SBATCH --output=%j-rf.out
#SBATCH --nodes=2
srun random_forest.py