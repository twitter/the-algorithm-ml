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
  """
    Compute the learning rate based on the specified learning rate configuration.

    This function calculates the learning rate according to the given configuration, which can include
    constant learning rates, piecewise constant schedules, linear ramps, and cosine annealing.

    Args:
        lr_config (LearningRate): The learning rate configuration specifying the learning rate schedule.
        step (int): The current training step or iteration.

    Returns:
        float: The computed learning rate for the current step.

    Raises:
        ValueError: If the `lr_config` is invalid or contains conflicting options.

    Example:
        ```python
        lr_schedule = LearningRate(
            constant=0.001,
            piecewise_constant=PiecewiseConstant(
                learning_rate_boundaries=[1000, 2000, 3000],
                learning_rate_values=[0.1, 0.05, 0.01, 0.001]
            )
        )
        current_step = 2500
        learning_rate = compute_lr(lr_schedule, current_step)
        ```
  """
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
  """
    Learning Rate Scheduler Shim to adjust learning rates during training.

    This class acts as a shim to apply different learning rates to individual parameter groups
    within an optimizer. It adheres to the torch.optim scheduler API and can be used with various
    optimizers, allowing fine-grained control over learning rates based on configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which learning rates will be adjusted.
        lr_dict (Dict[str, LearningRate]): A dictionary mapping parameter group names to their
            corresponding learning rate configurations.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
        verbose (bool, optional): If True, prints a warning message when accessing learning rates
            using the deprecated `get_lr()` method. Default is False.

    Raises:
        ValueError: If the number of parameter groups in the optimizer does not match the number
            of learning rate configurations provided.

    Note:
        To obtain the last computed learning rates, please use `get_last_lr()`.

    Example:
        ```python
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_schedule = {
            'main': LearningRate(constant=0.01),
            'auxiliary': LearningRate(piecewise_constant=PiecewiseConstant(
                learning_rate_boundaries=[1000, 2000],
                learning_rate_values=[0.01, 0.001]
            ))
        }
        lr_shim = LRShim(optimizer, lr_schedule)

        for epoch in range(num_epochs):
            # Train the model
            train(...)
            # Update learning rates at the end of each epoch
            lr_shim.step(epoch)

        final_lr_main = lr_shim.get_last_lr()['main']
        final_lr_auxiliary = lr_shim.get_last_lr()['auxiliary']
        ```

    See Also:
        - `LearningRate`: Configuration for specifying learning rates.
        - `PiecewiseConstant`: Configuration for piecewise constant learning rate schedules.
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
  """
    Build an optimizer and learning rate scheduler based on the provided optimizer configuration.

    Args:
        model (torch.nn.Module): The PyTorch model for which the optimizer will be created.
        optimizer_config (OptimizerConfig): The optimizer configuration specifying the optimizer
            algorithm and learning rate settings.

    Returns:
        Tuple[Optimizer, _LRScheduler]: A tuple containing the optimizer and learning rate scheduler
            objects.

    Note:
        This function is intended for cases where you want the same optimizer and learning rate
        schedule for all model parameters.

    Example:
        ```python
        model = MyModel()
        optimizer_config = OptimizerConfig(
            learning_rate=LearningRate(constant=0.01),
            sgd=SgdConfig(lr=0.01, momentum=0.9)
        )
        optimizer, scheduler = build_optimizer(model, optimizer_config)

        for epoch in range(num_epochs):
            # Train the model with the optimizer
            train(model, optimizer, ...)
            # Update learning rates at the end of each epoch
            scheduler.step(epoch)
        ```

    See Also:
        - `OptimizerConfig`: Configuration for specifying optimizer settings.
        - `LRShim`: Learning rate scheduler shim for fine-grained learning rate control.
    """
  optimizer_class = get_optimizer_class(optimizer_config)
  optimizer = optimizer_class(model.parameters(), **optimizer_config.sgd.dict())
  # We're passing everything in as one group here
  scheduler = LRShim(optimizer, lr_dict={"ALL_PARAMS": optimizer_config.learning_rate})
  return optimizer, scheduler
