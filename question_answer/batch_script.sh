#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --mincpus=40
#SBATCH --mail-user=likhith.a@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:4
#SBATCH -w gnode047

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 airavata_train.py