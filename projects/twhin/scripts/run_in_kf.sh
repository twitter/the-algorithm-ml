#! /bin/sh

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node $(echo $MACHINES_CONFIG | jq .chief.num_accelerators) \
  /usr/src/app/tml/projects/twhin/run.py \
  --config_yaml_path="/usr/src/app/tml/projects/twhin/config/base_twhin.yaml" \
  --save_dir="gs://scratch-user.$USER.dp.gcp.twttr.net/twhin-training/$RANDOM"
