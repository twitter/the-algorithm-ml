import pytest
import unittest

from tml.projects.twhin.models.models import TwhinModel, apply_optimizers
from tml.projects.twhin.models.test_models import twhin_model_config, twhin_data_config
from tml.projects.twhin.optimizer import build_optimizer
from tml.model import maybe_shard_model
from tml.common.testing_utils import mock_pg


import torch
from torch.nn import functional as F


def test_twhin_optimizer():
  model_config = twhin_model_config()
  data_config = twhin_data_config()

  loss_fn = F.binary_cross_entropy_with_logits
  with mock_pg():
    model = TwhinModel(model_config, data_config)
    apply_optimizers(model, model_config)
    model = maybe_shard_model(model, device=torch.device("cpu"))

    optimizer, _ = build_optimizer(model, model_config)

    # make sure there is one combined fused optimizer and one translation optimizer
    assert len(optimizer.optimizers) == 2
    fused_opt_tup, _ = optimizer.optimizers
    _, fused_opt = fused_opt_tup

    # make sure there are two tables for which the fused opt has parameters
    assert len(fused_opt.param_groups) == 2
