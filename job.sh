#!/bin/bash
#SBATCH --job-name=testing_models
#SBATCH --nodes=1
#SBATCH --partition=gpu_inter
#SBATCH --time=00:23:00
#SBATCH --output=./logslurms/stylegan3-%A_%a.out
#SBATCH --error=./logslurms/stylegan3-%A_%a.err

python3 -m virtualenv $TMPDIR/venv
source $TMPDIR/venv/bin/activate
python -m pip install -r requirements.txt

python main.py config_matt_CNN_3_blocks.yaml
python main.py config_matt_CNN_4_blocks.yaml


# Move logs to logslurm folder
mkdir -p ./logslurm
mv stylegan3-${SLURM_JOB_ID}.out ./logslurm/
mv stylegan3-${SLURM_JOB_ID}.err ./logslurm/
