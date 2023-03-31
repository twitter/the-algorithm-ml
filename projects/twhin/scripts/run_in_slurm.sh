#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)" || exit
cd ..
export PYTHONPATH="$(pwd)"

export TASK_TYPE="chief"

ENDPOINT=$(python tml/machines/list_ops.py --op=select --input_list="$SLURM_JOB_NODELIST" --elem=0 --sep=',')
NNODES=$(python tml/machines/list_ops.py --op=len --input_list="$SLURM_JOB_NODELIST" --sep=',')

echo "NNODES: $NNODES"

if [[ $NNODES == "1" ]];
then
  echo "Single Trainer."
  torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 2 \
    tml/projects/twhin/run.py \
    --config_yaml_path="/home/$USER/workspace/tml/projects/twhin/config/slurm.yaml" \
else
  echo "Multiple Trainers: ${NNODES} nodes."
torchrun \
  --rdzv_backend c10d \
  --rdzv_endpoint $ENDPOINT \
  --rdzv_id "$SLURM_JOB_ID" \
  --nnodes "$NNODES" \
  --nproc_per_node 2 \
  --role "" \
  tml/projects/twhin/run.py \
  --config_yaml_path="/home/$USER/workspace/tml/projects/twhin/config/slurm.yaml"
fi
