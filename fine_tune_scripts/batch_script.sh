#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=20
#SBATCH --mail-user=likhith.a@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:2
#SBATCH -w gnode081

export CUDA_VISIBLE_DEVICES=0,1

python3 causal_finetune_indicbart.py
