#!/bin/bash
#SBATCH -A nro108
#SBATCH -o mae_imagenet.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

module load PrgEnv-gnu
module load gcc-native/12.3
module load rocm/6.2.0

python main.py \
    --task vit_imagenet \
    --data_dir ../dataset/imagenet2012 \
    --batch_size 512 \
    --num_workers 32 \
    --num_epochs 100