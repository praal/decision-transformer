#!/bin/bash
#SBATCH --job-name=dt-causal
#SBATCH --output=logs/dt-causal.log
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export PYTHONPATH=${PWD}
# python -u experiment.py --env craft --dataset four-causal-structure --model_type dt --max_iters 20 -w True --causal_dim 6 --causal_version v1
python -u experiment.py --env craft --dataset fourreward --model_type dt --max_iters 20 -w True