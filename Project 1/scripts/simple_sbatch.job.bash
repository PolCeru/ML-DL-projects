!/bin/bash

SBATCH -A master        # Replace with the desired account name
SBATCH -p normal        # Replace with the desired partition name
SBATCH --output=output_mopout_01_16_frame.out
SBATCH --gres=gpu:2     # Request 1 GPU
SBATCH --nodelist=hpc6  # Replace with the desired node name

# Additional SLURM optio configuration
SBATCH -t 10-00:00:0    # Set a time limit for the job (e.g., 10 minutes)
SBATCH --mem-per-cpu=4G # Request memory per CPU (e.g., 4GB)

# Load any necessary modate a virtual environment
module load cuda        # Example: Load CUDA module

./cart.py
