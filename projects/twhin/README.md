Twhin in torchrec

This project contains code for pretraining dense vector embedding features for Twitter entities. Within Twitter, these embeddings are used for candidate retrieval and as model features in a variety of recommender system models.

We obtain entity embeddings based on a variety of graph data within Twitter such as:
  "User follows User"
  "User favorites Tweet"
  "User clicks Advertisement"

While we cannot release the graph data used to train TwHIN embeddings due to privacy restrictions, heavily subsampled, anonymized open-sourced graph data can used:
https://huggingface.co/datasets/Twitter/TwitterFollowGraph
https://huggingface.co/datasets/Twitter/TwitterFaveGraph

The code expects parquet files with three columns: lhs, rel, rhs that refer to the vocab index of the left-hand-side node, relation type, and right-hand-side node of each edge in a graph respectively.

The location of the data must be specified in the configuration yaml files in projects/twhin/configs.


Workflow
========
- Build local development images `./scripts/build_images.sh`
- Run with `./scripts/docker_run.sh`
- Iterate in image with `./scripts/idocker.sh`
- Run tests with `./scripts/docker_test.sh`
