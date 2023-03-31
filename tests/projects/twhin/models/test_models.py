from tml.projects.twhin.models.config import TwhinEmbeddingsConfig, TwhinModelConfig
from tml.projects.twhin.data.config import TwhinDataConfig
from tml.common.modules.embedding.config import DataType, EmbeddingBagConfig
from tml.optimizers.config import OptimizerConfig, SgdConfig
from tml.model import maybe_shard_model
from tml.projects.twhin.models.models import apply_optimizers, TwhinModel
from tml.projects.twhin.models.config import Operator, Relation
from tml.common.testing_utils import mock_pg

import torch
import torch.nn.functional as F
from pydantic import ValidationError
import pytest


NUM_EMBS = 10_000
EMB_DIM = 128


def twhin_model_config() -> TwhinModelConfig:
  sgd_config_0 = OptimizerConfig(sgd=SgdConfig(lr=0.01))
  sgd_config_1 = OptimizerConfig(sgd=SgdConfig(lr=0.02))

  table0 = EmbeddingBagConfig(
    name="table0",
    num_embeddings=NUM_EMBS,
    embedding_dim=EMB_DIM,
    optimizer=sgd_config_0,
    data_type=DataType.FP32,
  )
  table1 = EmbeddingBagConfig(
    name="table1",
    num_embeddings=NUM_EMBS,
    embedding_dim=EMB_DIM,
    optimizer=sgd_config_1,
    data_type=DataType.FP32,
  )
  embeddings_config = TwhinEmbeddingsConfig(
    tables=[table0, table1],
  )

  model_config = TwhinModelConfig(
    embeddings=embeddings_config,
    translation_optimizer=sgd_config_0,
    relations=[
      Relation(name="rel0", lhs="table0", rhs="table1", operator=Operator.TRANSLATION),
      Relation(name="rel1", lhs="table1", rhs="table0", operator=Operator.TRANSLATION),
    ],
  )

  return model_config


def twhin_data_config() -> TwhinDataConfig:
  data_config = TwhinDataConfig(
    data_root="/",
    per_replica_batch_size=10,
    global_negatives=10,
    in_batch_negatives=10,
    limit=1,
    offset=1,
  )

  return data_config


def test_twhin_model():
  model_config = twhin_model_config()
  loss_fn = F.binary_cross_entropy_with_logits

  with mock_pg():
    data_config = twhin_data_config()
    model = TwhinModel(model_config=model_config, data_config=data_config)

    apply_optimizers(model, model_config)

    for tensor in model.state_dict().values():
      if tensor.size() == (NUM_EMBS, EMB_DIM):
        assert str(tensor.device) == "meta"
      else:
        assert str(tensor.device) == "cpu"

    model = maybe_shard_model(model, device=torch.device("cpu"))


def test_unequal_dims():
  sgd_config_1 = OptimizerConfig(sgd=SgdConfig(lr=0.02))
  sgd_config_2 = OptimizerConfig(sgd=SgdConfig(lr=0.05))
  table0 = EmbeddingBagConfig(
    name="table0",
    num_embeddings=10_000,
    embedding_dim=128,
    optimizer=sgd_config_1,
    data_type=DataType.FP32,
  )
  table1 = EmbeddingBagConfig(
    name="table1",
    num_embeddings=10_000,
    embedding_dim=64,
    optimizer=sgd_config_2,
    data_type=DataType.FP32,
  )

  with pytest.raises(ValidationError):
    _ = TwhinEmbeddingsConfig(
      tables=[table0, table1],
    )
