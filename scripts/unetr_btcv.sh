#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unetr_btcv.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# export PATH="/lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/bin:$PATH"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

python main.py \
    --task unetr_btcv \
    --logname train-unetr.log\
    --data_dir /lustre/orion/bif146/world-shared/enzhi/btcv/data \
    --batch_size 4 \
    --num_workers 4 \
    --num_epochs 30