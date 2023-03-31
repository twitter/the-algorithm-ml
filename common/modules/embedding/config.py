from typing import List
from enum import Enum

import tml.core.config as base_config
from tml.optimizers.config import OptimizerConfig

import pydantic


class DataType(str, Enum):
  FP32 = "fp32"
  FP16 = "fp16"


class EmbeddingSnapshot(base_config.BaseConfig):
  """Configuration for Embedding snapshot"""

  emb_name: str = pydantic.Field(
    ..., description="Name of the embedding table from the loaded snapshot"
  )
  embedding_snapshot_uri: str = pydantic.Field(
    ..., description="Path to torchsnapshot of the embedding"
  )


class EmbeddingBagConfig(base_config.BaseConfig):
  """Configuration for EmbeddingBag."""

  name: str = pydantic.Field(..., description="name of embedding bag")
  num_embeddings: int = pydantic.Field(..., description="size of embedding dictionary")
  embedding_dim: int = pydantic.Field(..., description="size of each embedding vector")
  pretrained: EmbeddingSnapshot = pydantic.Field(None, description="Snapshot properties")
  vocab: str = pydantic.Field(
    None, description="Directory to parquet files of mapping from entity ID to table index."
  )
  # make sure to use an optimizer that matches:
  # https://github.com/pytorch/FBGEMM/blob/4c58137529d221390575e47e88d3c05ce65b66fd/fbgemm_gpu/fbgemm_gpu/split_embedding_configs.py#L15
  optimizer: OptimizerConfig
  data_type: DataType


class LargeEmbeddingsConfig(base_config.BaseConfig):
  """Configuration for EmbeddingBagCollection.

  The tables listed in this config are gathered into a single torchrec EmbeddingBagCollection.
  """

  tables: List[EmbeddingBagConfig] = pydantic.Field(..., description="list of embedding tables")
  tables_to_log: List[str] = pydantic.Field(
    None, description="list of embedding table names that we want to log during training"
  )


class Mode(str, Enum):
  """Job modes."""

  TRAIN = "train"
  EVALUATE = "evaluate"
  INFERENCE = "inference"
