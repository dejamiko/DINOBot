#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=md1823
#SBATCH --output=out_transfer_%j.out


source /vol/bitbucket/md1823/taskmaster/DINOBot/venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 -m demo_transfer
