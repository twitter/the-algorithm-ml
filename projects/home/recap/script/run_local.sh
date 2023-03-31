#!/usr/bin/env bash

# Runs from inside venv
rm -rf $HOME/tmp/runs/recap_local_debug
mkdir -p $HOME/tmp/runs/recap_local_debug
python -m tml.machines.is_venv || exit 1
export TML_BASE="$(git rev-parse --show-toplevel)"

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 1 \
  projects/home/recap/main.py \
  --config_path $(pwd)/projects/home/recap/config/local_prod.yaml \
  $@
