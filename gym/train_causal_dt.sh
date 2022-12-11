#!/bin/bash
#SBATCH --job-name=dt
#SBATCH --output=logs/dt.log
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export PYTHONPATH=${PWD}
python -u experiment.py --env craft --dataset fourreward --model_type dt --max_iters 20 -w True