#!/bin/bash
#SBATCH --job-name=gen_imgs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH --time=48:00:00
#SBATCH --partition=shubhamlong
#SBATCH --nodelist=grogu-2-12
#SBATCH --output=download_re10k_2.log
#SBATCH --error=download_re10k_2.log

# Change to the script directory
cd /home/yuwu3/prope

# Initialize and activate conda environment
source /home/yuwu3/miniconda3/etc/profile.d/conda.sh
conda activate prope

# Run the python script
python ./scripts/gen_imgs.py
    
