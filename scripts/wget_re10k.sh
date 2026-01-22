#!/bin/bash
#SBATCH --account=cis240058p
#SBATCH --job-name=wget_re10k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=48:00:00
#SBATCH --partition=RM-shared
#SBATCH --output=wget_re10k.log
#SBATCH --error=wget_re10k.log

wget -c "http://schadenfreude.csail.mit.edu:8000/re10k.zip" -P /ocean/projects/cis240058p/ywu15