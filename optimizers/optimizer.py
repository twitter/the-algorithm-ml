from typing import Dict, Tuple
import math
import bisect

from tml.optimizers.config import (
  LearningRate,
  OptimizerConfig,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tml.ml_logging.torch_logging import logging


def get_learning_rate(lr_config: LearningRateConfig, step: int) -> float:
    switcher = {
        lr_config.constant is not None: lambda: lr_config.constant,
        lr_config.piecewise_constant is not None: lambda: lr_config.piecewise_constant.learning_rate_values[
            bisect.bisect_right(lr_config.piecewise_constant.learning_rate_boundaries, step)
        ],
        lr_config.linear_ramp_to_constant is not None: lambda: min(lr_config.linear_ramp_to_constant.learning_rate, 
                                                                   (lr_config.linear_ramp_to_constant.learning_rate 
                                                                    / lr_config.linear_ramp_to_constant.num_ramp_steps) 
                                                                    * step),
        lr_config.linear_ramp_to_cosine is not None: lambda: (lr_config.linear_ramp_to_cosine.final_learning_rate 
                                                             + (lr_config.linear_ramp_to_cosine.learning_rate 
                                                                - lr_config.linear_ramp_to_cosine.final_learning_rate) 
                                                                * 0.5 * (1.0 + math.cos(
                                                                  math.pi * (step - lr_config.linear_ramp_to_cosine.num_ramp_steps) 
                                                                  / (lr_config.linear_ramp_to_cosine.final_num_steps - lr_config.linear_ramp_to_cosine.num_ramp_steps))))
    }
    func = switcher.get(True, lambda: f"No option selected in lr_config, passed {lr_config}")
    return func()
return cfg.final_learning_rate
}

class LRShim(_LRScheduler):
  """Shim to get learning rates into a LRScheduler.

  This adheres to the torch.optim scheduler API and can be plugged anywhere that
  e.g. exponential decay can be used.
  """

  def __init__(
    self,
    optimizer,
    lr_dict: Dict[str, LearningRate],
    last_epoch=-1,
    verbose=False,
  ):
    self.optimizer = optimizer
    self.lr_dict = lr_dict
    self.group_names = list(self.lr_dict.keys())

    num_param_groups = sum(1 for _, _optim in optimizer._optims for _ in _optim.param_groups)
    if num_param_groups != len(lr_dict):
      raise ValueError(
        f"Optimizer had {len(optimizer.param_groups)}, but config had {len(lr_dict)}."
      )

    super().__init__(optimizer, last_epoch, verbose)

  def get_lr(self):
    if not self._get_lr_called_within_step:
      logging.warn(
        "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
        UserWarning,
      )
    return self._get_closed_form_lr()

  def _get_closed_form_lr(self):
    return [compute_lr(lr_config, self.last_epoch) for lr_config in self.lr_dict.values()]


def get_optimizer_class(optimizer_config: OptimizerConfig):
  if optimizer_config.adam is not None:
    return torch.optim.Adam
  elif optimizer_config.sgd is not None:
    return torch.optim.SGD
  elif optimizer_config.adagrad is not None:
    return torch.optim.Adagrad


def build_optimizer(
  model: torch.nn.Module, optimizer_config: OptimizerConfig
) -> Tuple[Optimizer, _LRScheduler]:
  """Builds an optimizer and LR scheduler from an OptimizerConfig.
  Note: use this when you want the same optimizer and learning rate schedule for all your parameters.
  """
  optimizer_class = get_optimizer_class(optimizer_config)
  optimizer = optimizer_class(model.parameters(), **optimizer_config.sgd.dict())
  # We're passing everything in as one group here
  scheduler = LRShim(optimizer, lr_dict={"ALL_PARAMS": optimizer_config.learning_rate})
  return optimizer, scheduler
