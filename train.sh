#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -J "train_ssm"
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -C A100|H100|L40S|H200

module load cuda
module load ffmpeg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/ffmpeg-6.1.1-cup2q2rgvykfd5goili4a7wwuex2zv33/lib
uv run train_classifier.py
