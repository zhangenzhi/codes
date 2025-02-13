#!/bin/bash
#SBATCH -A nro108
#SBATCH -o vit_imagenet_af_ddp.o%J
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

module load PrgEnv-gnu
module load gcc-native/12.3
module load rocm/6.2.0

srun -N 2 -n 16 --ntasks-per-node 8 python main.py \
    --task vit_imagenet_af_ddp \
    --data_dir /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
    --batch_size 128 \
    --num_workers 32 \
    --num_epochs 100 \
    --seq_length 514 \
    --savefile cep-514-n2