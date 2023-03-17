#!/bin/bash

#SBATCH -J ours_IMGR
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=7-0
#SBATCH -o %x_%j_%a.out
#SBATCH -e %x_%j_%a.err

date
ulimit -n 65536
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=$SLURM_NNODES

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /data/moonjunyyy/init.sh
conda activate Trainer

conda --version
python --version

seeds=(1 21 42 3473 10741 32450 93462 85015 64648 71950 87557 99668 55552 4811 10741)

for i in seeds
do
    python main.py train \
     --model ouroboros \
     --batch_size 128 \
     --optimizer adam --lr 8e-4 \
     --scheduler warmup_cosine --warmup-epochs 5 --T-max 150 \
     --epochs 300 \
     --dataset CIFAR100 --datapath /local_datasets \
     --save_path ./Ouroboros \
     --transforms Resize AutoAugment --size 224
done