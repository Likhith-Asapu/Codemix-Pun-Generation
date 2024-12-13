#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=10
#SBATCH --mail-user=likhith.a@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH -w gnode084
export CUDA_VISIBLE_DEVICES=0

python3 finetune_mbart.py