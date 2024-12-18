#!/bin/bash
#SBATCH -A nro108
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
module load gcc-native/12.3
module load rocm/5.7.0

python main.py \
    --task vit_imagenet_ap \
    --data_dir /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
    --batch_size 512 \
    --num_workers 32 \
    --num_epochs 100