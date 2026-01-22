#!/bin/bash
#SBATCH --account=cis240058p
#SBATCH --job-name=lvsm_process_data_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=48:00:00
#SBATCH --partition=RM-shared
#SBATCH --output=lvsm_transform_data2.log
#SBATCH --error=lvsm_transform_data2.log

# Activate conda environment
source ~/.bashrc
conda activate prope
# export PYTHONVERBOSE=-1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1

cd /jet/home/ywu15/prope

# python scripts/lvsm_process_data.py --base_path /ocean/projects/cis240058p/ywu15/re10k_raw --output_dir /ocean/projects/cis240058p/ywu15/re10k --mode 'train'

python scripts/re10k_lvsm2prope_data.py
# sbatch scripts/lvsm_process_data.sh

rsync -ah --info=progress2,stats --partial --inplace \
    yuwu3@grogu-2-9:/grogu/user/yuwu3/objaverse80k_sp/ \
    /ocean/projects/cis240058p/ywu15/objaverse80k_sp

rsync -ah --info=progress2,stats --partial --inplace \
    ywu15@bridges2.psc.edu:/ocean/projects/cis240034p/ywu15/prope_log/ \
    /grogu/user/yuwu3/psc_prope_log/