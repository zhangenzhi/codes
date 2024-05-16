#!/bin/bash
#SBATCH -A bif146
#SBATCH -o vit-ddp2.o%J
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# export PATH="/lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/bin:$PATH"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

source ./export_ddp_envs.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

srun -N 2 -n 16 --ntasks-per-node 8 -c 8 python main.py \
    --task vit_ddp \
    --gpus 8\
    --nodes 2\
    --data_dir /lustre/orion/bif146/world-shared/enzhi/imagenet2012 \
    --batch_size 32 \
    --num_workers 16 \
    --num_epochs 30