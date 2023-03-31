#!/bin/bash

#SBATCH --job-name=dist --requeue --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=2 --partition=v100

set -evx

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=0

export SLURM_RANK_INDICATOR=chief
export LOGLEVEL=WARNING

export PYTHONPATH=$HOME/workspace

source /etc/profile.d/conda.sh
conda activate "$USER-tml"
set +e
srun \
  --output=$HOME/workspace/tml/slurm-dist-0.out \
  --error=$HOME/workspace/tml/slurm-dist-0.err \
  --partition=v100 \
  --wait=60 \
  --kill-on-bad-exit=1 \
  $HOME/workspace/tml/projects/twhin/scripts/run_in_slurm.sh
exitcode=$?
set -e


echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${SLURM_RESTART_COUNT:-0}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
