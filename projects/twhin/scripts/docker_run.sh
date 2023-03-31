#! /bin/sh

docker run -it --rm \
  -v $HOME/workspace/tml:/usr/src/app/tml \
  -v $HOME/.config:/root/.config \
  -w /usr/src/app \
  -e PYTHONPATH="/usr/src/app/" \
  --network host \
  -e SPEC_TYPE=chief \
  local/torch \
  bash tml/projects/twhin/scripts/run_in_docker.sh
