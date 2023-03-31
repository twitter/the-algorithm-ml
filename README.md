This project open sources some of the ML models used at Twitter.

Currently these are:

1. The "For You" Heavy Ranker (projects/home/recap).

2. TwHIN embeddings (projects/twhin) https://arxiv.org/abs/2202.05387


This project can be run inside a python virtualenv. We have only tried this on Linux machines and because we use torchrec it works best with an Nvidia GPU. To setup run

`./images/init_venv.sh` (Linux only).

The READMEs of each project contain instructions about how to run each project.
