#!/bin/bash
#SBATCH -A bif146
#SBATCH -o vit_imagenet.o%J
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
    --task vit_imagenet \
    --data_dir /lustre/orion/bif146/world-shared/enzhi/imagenet2012 \
    --batch_size 32 \
    --num_workers 32 \
    --num_epochs 10