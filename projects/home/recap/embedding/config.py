from typing import List, Optional
import tml.core.config as base_config
from tml.optimizers import config as optimizer_config

import pydantic


class EmbeddingSnapshot(base_config.BaseConfig):
  """Configuration for Embedding snapshot"""

  emb_name: str = pydantic.Field(
    ..., description="Name of the embedding table from the loaded snapshot"
  )
  embedding_snapshot_uri: str = pydantic.Field(
    ..., description="Path to torchsnapshot of the embedding"
  )


# https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_configs.EmbeddingBagConfig
class EmbeddingBagConfig(base_config.BaseConfig):
  """Configuration for EmbeddingBag."""

  name: str = pydantic.Field(..., description="name of embedding bag")
  num_embeddings: int = pydantic.Field(..., description="size of embedding dictionary")
  embedding_dim: int = pydantic.Field(..., description="size of each embedding vector")
  pretrained: EmbeddingSnapshot = pydantic.Field(None, description="Snapshot properties")
  vocab: str = pydantic.Field(
    None, description="Directory to parquet files of mapping from entity ID to table index."
  )


class EmbeddingOptimizerConfig(base_config.BaseConfig):
  learning_rate: optimizer_config.LearningRate = pydantic.Field(
    None, description="learning rate scheduler for the EBC"
  )
  init_learning_rate: float = pydantic.Field(description="initial learning rate for the EBC")
  # NB: Only sgd is supported right now and implicitly.
  # FBGemm only supports simple exact_sgd which only takes LR as an argument.


class LargeEmbeddingsConfig(base_config.BaseConfig):
  """Configuration for EmbeddingBagCollection.

  The tables listed in this config are gathered into a single torchrec EmbeddingBagCollection.
  """

  tables: List[EmbeddingBagConfig] = pydantic.Field(..., description="list of embedding tables")
  optimizer: EmbeddingOptimizerConfig
  tables_to_log: List[str] = pydantic.Field(
    None, description="list of embedding table names that we want to log during training"
  )


class StratifierConfig(base_config.BaseConfig):
  name: str
  index: int
  value: int


class SmallEmbeddingBagConfig(base_config.BaseConfig):
  """Configuration for SmallEmbeddingBag."""

  name: str = pydantic.Field(..., description="name of embedding bag")
  num_embeddings: int = pydantic.Field(..., description="size of embedding dictionary")
  embedding_dim: int = pydantic.Field(..., description="size of each embedding vector")
  index: int = pydantic.Field(..., description="index in the discrete tensor to look for")


class SmallEmbeddingBagConfig(base_config.BaseConfig):
  """Configuration for SmallEmbeddingBag."""

  name: str = pydantic.Field(..., description="name of embedding bag")
  num_embeddings: int = pydantic.Field(..., description="size of embedding dictionary")
  embedding_dim: int = pydantic.Field(..., description="size of each embedding vector")
  index: int = pydantic.Field(..., description="index in the discrete tensor to look for")


class SmallEmbeddingsConfig(base_config.BaseConfig):
  """Configuration for SmallEmbeddingConfig.

  Here we can use discrete features that already are present in our TFRecords generated using
  segdense conversion as "home_recap_2022_discrete__segdense_vals" which are available in
  the model as "discrete_features", and embed a user-defined set of them with configurable
  dimensions and vocabulary sizes.

  Compared with LargeEmbedding, this config is for small embedding tables that can fit inside
  the model, whereas LargeEmbedding usually is meant to be hydrated outside the model at
  serving time due to size (>>1 GB).

  This small embeddings table uses the same optimizer as the rest of the model."""

  tables: List[SmallEmbeddingBagConfig] = pydantic.Field(
    ..., description="list of embedding tables"
  )
