"""Build optimizers and learning rate schedules."""
import bisect
from collections import defaultdict
import functools
import math
import typing
from typing import Optional
import warnings

# from large_embeddings.config import EmbeddingOptimizerConfig
from tml.projects.home.recap import model as model_mod
from tml.optimizers import config
from tml.optimizers import compute_lr
from absl import logging  # type: ignore[attr-defined]

import torch
from torchrec.optim import keyed


_DEFAULT_LR = 24601.0  # NaN the model if we're not using the learning rate.
_BACKBONE = "backbone"
_DENSE_EMBEDDINGS = "dense_ebc"


class RecapLRShim(torch.optim.lr_scheduler._LRScheduler):
  """Shim to get learning rates into a LRScheduler.

  This adheres to the torch.optim scheduler API and can be plugged anywhere that
  e.g. exponential decay can be used.

  """

  def __init__(
    self,
    optimizer,
    lr_dict: typing.Dict[str, config.LearningRate],
    emb_learning_rate,
    last_epoch=-1,
    verbose=False,
  ):
    self.optimizer = optimizer
    self.lr_dict = lr_dict
    self.group_names = list(self.lr_dict.keys())
    self.emb_learning_rate = emb_learning_rate

    # We handle sparse LR scheduling separately, so only validate LR groups against dense param groups
    num_dense_param_groups = sum(
      1
      for _, _optim in optimizer._optims
      for _ in _optim.param_groups
      if isinstance(_optim, keyed.KeyedOptimizerWrapper)
    )
    if num_dense_param_groups != len(lr_dict):
      raise ValueError(
        f"Optimizer had {len(optimizer.param_groups)}, but config had {len(lr_dict)}."
      )
    super().__init__(optimizer, last_epoch, verbose)

  def get_lr(self):
    if not self._get_lr_called_within_step:
      warnings.warn(
        "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
        UserWarning,
      )
    return self._get_closed_form_lr()

  def _get_closed_form_lr(self):
    learning_rates = []

    for lr_config in self.lr_dict.values():
      learning_rates.append(compute_lr(lr_config, self.last_epoch))
    # WARNING: The order of appending is important.
    if self.emb_learning_rate:
      learning_rates.append(compute_lr(self.emb_learning_rate, self.last_epoch))
    return learning_rates


def build_optimizer(
  model: torch.nn.Module,
  optimizer_config: config.OptimizerConfig,
  emb_optimizer_config: None = None,  # Optional[EmbeddingOptimizerConfig] = None,
):
  """Builds an optimizer and scheduler.

  Args:
    model: A torch model, probably with DDP/DMP.
    optimizer_config: An OptimizerConfig object that specifies learning rates per tower.

  Returns:
    A torch.optim instance, and a scheduler instance.
  """
  optimizer_fn = functools.partial(
    torch.optim.Adam,
    lr=_DEFAULT_LR,
    betas=(optimizer_config.adam.beta_1, optimizer_config.adam.beta_2),
    eps=optimizer_config.adam.epsilon,
    maximize=False,
  )
  if optimizer_config.multi_task_learning_rates:
    logging.info("***** Parameter groups for optimization *****")
    # Importantly, we preserve insertion order in dictionaries here.
    parameter_groups: typing.Dict[str, typing.Dict] = defaultdict(dict)
    added_parameters: typing.Set[str] = set()
    for task in optimizer_config.multi_task_learning_rates.tower_learning_rates:
      for name, parameter in model.named_parameters():
        if f".{model_mod.sanitize(task)}." in name:
          parameter_groups[task][name] = parameter
          logging.info(f"{task}: {name}")
          if name in added_parameters:
            raise ValueError(f"Parameter {name} matched multiple tasks.")
          added_parameters.add(name)

    for name, parameter in model.named_parameters():
      if name not in added_parameters and "embedding_bags" not in name:
        parameter_groups[_BACKBONE][name] = parameter
        added_parameters.add(name)
        logging.info(f"{_BACKBONE}: {name}")

    for name, parameter in model.named_parameters():
      if name not in added_parameters and "embedding_bags" in name:
        parameter_groups[_DENSE_EMBEDDINGS][name] = parameter
        logging.info(f"{_DENSE_EMBEDDINGS}: {name}")

    all_learning_rates = optimizer_config.multi_task_learning_rates.tower_learning_rates.copy()
    if optimizer_config.multi_task_learning_rates.backbone_learning_rate is not None:
      all_learning_rates[
        _BACKBONE
      ] = optimizer_config.multi_task_learning_rates.backbone_learning_rate
    if _DENSE_EMBEDDINGS in parameter_groups and emb_optimizer_config:
      all_learning_rates[_DENSE_EMBEDDINGS] = emb_optimizer_config.learning_rate.copy()
  else:
    parameter_groups = dict(model.named_parameters())
    all_learning_rates = {"single_task": optimizer_config.single_task_learning_rate}

  optimizers = [
    keyed.KeyedOptimizerWrapper(param_group, optimizer_fn)
    for param_name, param_group in parameter_groups.items()
    if param_name != _DENSE_EMBEDDINGS
  ]
  # Making EBC optimizer to be SGD to match fused optimiser
  if _DENSE_EMBEDDINGS in parameter_groups:
    optimizers.append(
      keyed.KeyedOptimizerWrapper(
        parameter_groups[_DENSE_EMBEDDINGS],
        functools.partial(torch.optim.SGD, lr=_DEFAULT_LR, maximize=False, momentum=False),
      )
    )

  if not parameter_groups.keys() == all_learning_rates.keys():
    raise ValueError("Learning rates do not match optimizers")

  # If the optimiser is dense, model.fused_optimizer will be empty (but not None)
  emb_learning_rate = None
  if hasattr(model, "fused_optimizer") and model.fused_optimizer.optimizers:
    logging.info(f"Model fused optimiser: {model.fused_optimizer}")
    optimizers.append(model.fused_optimizer)
    if emb_optimizer_config:
      emb_learning_rate = emb_optimizer_config.learning_rate.copy()
    else:
      raise ValueError("Fused kernel exists, but LR is not set")
  logging.info(f"***** Combining optimizers: {optimizers} *****")
  optimizer = keyed.CombinedOptimizer(optimizers)
  scheduler = RecapLRShim(optimizer, all_learning_rates, emb_learning_rate)
  logging.info(f"***** Combined optimizer after init: {optimizer} *****")

  return optimizer, scheduler
