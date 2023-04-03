"""
This code defines a custom learning rate scheduler for PyTorch by using different learning rate configurations.
The scheduler can compute learning rates based on various strategies,
such as constant, piecewise constant, linear ramp to constant, and linear ramp to cosine.
The code also provides a utility function to build an optimizer and a learning rate scheduler from an OptimizerConfig.
"""

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


def compute_lr(lr_config, step):
    """This function computes the learning rate based on the learning rate configuration (lr_config)
  and the current training step (step). It handles multiple learning rate strategies."""
    if lr_config.constant is not None:
        return lr_config.constant
    elif lr_config.piecewise_constant is not None:
        return lr_config.piecewise_constant.learning_rate_values[
            bisect.bisect_right(lr_config.piecewise_constant.learning_rate_boundaries, step)
        ]
    elif lr_config.linear_ramp_to_constant is not None:
        slope = (
                lr_config.linear_ramp_to_constant.learning_rate
                / lr_config.linear_ramp_to_constant.num_ramp_steps
        )
        return min(lr_config.linear_ramp_to_constant.learning_rate, slope * step)
    elif lr_config.linear_ramp_to_cosine is not None:
        cfg = lr_config.linear_ramp_to_cosine
        if step < cfg.num_ramp_steps:
            slope = cfg.learning_rate / cfg.num_ramp_steps
            return slope * step
        elif step <= cfg.final_num_steps:
            return cfg.final_learning_rate + (cfg.learning_rate - cfg.final_learning_rate) * 0.5 * (
                    1.0
                    + math.cos(
                math.pi * (step - cfg.num_ramp_steps) / (cfg.final_num_steps - cfg.num_ramp_steps)
            )
            )
        else:
            return cfg.final_learning_rate
    else:
        raise ValueError(f"No option selected in lr_config, passed {lr_config}")


class LRShim(_LRScheduler):
    """Shim to get learning rates into a LRScheduler.

  This adheres to the torch.optim scheduler API and can be plugged anywhere that
  e.g. exponential decay can be used.

  This class inherits from PyTorch's _LRScheduler and acts as a shim to compute learning rates according to the
  specified configurations. It also checks if the number of parameter groups in the optimizer matches
  the length of the learning rate dictionary.
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
    """
    This function returns the appropriate PyTorch optimizer class based on the given optimizer_config. It supports Adam, SGD, and Adagrad optimizers.
    """
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

  This function takes a PyTorch model and an OptimizerConfig object as input and returns a tuple containing the created
  optimizer and learning rate scheduler. It creates an optimizer instance of the selected optimizer class and
  initializes the LRShim scheduler with the specified learning rate configurations.
  """
    optimizer_class = get_optimizer_class(optimizer_config)
    optimizer = optimizer_class(model.parameters(), **optimizer_config.sgd.dict())
    # We're passing everything in as one group here
    scheduler = LRShim(optimizer, lr_dict={"ALL_PARAMS": optimizer_config.learning_rate})
    return optimizer, scheduler
