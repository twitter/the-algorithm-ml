#!/usr/bin/env bash

# Runs from inside venv

rm -rf $HOME/tmp/runs/recap_local_random_data
python -m tml.machines.is_venv || exit 1
export TML_BASE="$(git rev-parse --show-toplevel)"

mkdir -p $HOME/tmp/recap_local_random_data
python projects/home/recap/data/generate_random_data.py --config_path $(pwd)/projects/home/recap/config/local_prod.yaml
