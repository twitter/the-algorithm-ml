from tml.common.modules.embedding.config import LargeEmbeddingsConfig, DataType
from tml.ml_logging.torch_logging import logging

import torch
from torch import nn
import torchrec
from torchrec.modules import embedding_configs
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
import numpy as np


class LargeEmbeddings(nn.Module):
  def __init__(
    self,
    large_embeddings_config: LargeEmbeddingsConfig,
  ):
    super().__init__()

    tables = []
    for table in large_embeddings_config.tables:
      data_type = (
        embedding_configs.DataType.FP32
        if (table.data_type == DataType.FP32)
        else embedding_configs.DataType.FP16
      )

      tables.append(
        EmbeddingBagConfig(
          embedding_dim=table.embedding_dim,
          feature_names=[table.name],  # restricted to 1 feature per table for now
          name=table.name,
          num_embeddings=table.num_embeddings,
          pooling=torchrec.PoolingType.SUM,
          data_type=data_type,
        )
      )

    self.ebc = EmbeddingBagCollection(
      device="meta",
      tables=tables,
    )

    logging.info("********************** EBC named params are **********")
    logging.info(list(self.ebc.named_parameters()))

    # This hook is used to perform post-processing surgery
    # on large_embedding models to prep them for serving
    self.surgery_cut_point = torch.nn.Identity()

  def forward(
    self,
    sparse_features: KeyedJaggedTensor,
  ) -> KeyedTensor:
    pooled_embs = self.ebc(sparse_features)

    # a KeyedTensor
    return self.surgery_cut_point(pooled_embs)
