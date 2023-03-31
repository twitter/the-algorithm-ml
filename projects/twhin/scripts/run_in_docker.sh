#! /bin/sh

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 2 \
  /usr/src/app/tml/projects/twhin/run.py \
  --config_yaml_path="/usr/src/app/tml/projects/twhin/config/local.yaml" \
  --save_dir="/some/save/dir"
