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
    --task mae_s8d_ap \
    --data_dir /lustre/orion/nro108/world-shared/enzhi/spring8data/demo \
    --batch_size 256 \
    --num_workers 32 \
    --num_epochs 100