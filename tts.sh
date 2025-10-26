#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -J "tts"
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -C A100|H100|L40S|H200

module load cuda
uv run tts.py
