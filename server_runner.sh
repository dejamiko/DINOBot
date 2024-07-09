#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=md1823
#SBATCH --output=out_server_%j.out
#SBATCH --nodelist=linnet


source /vol/bitbucket/md1823/taskmaster/DINOBot/venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 -m DINOserver.server
