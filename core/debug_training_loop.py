"""This is a very limited feature training loop useful for interactive debugging.

It is not intended for actual model tranining (it is not fast, doesn't compile the model).
It does not support checkpointing.

suggested use:

from tml.core import debug_training_loop
debug_training_loop.train(...)
"""

from typing import Iterable, Optional, Dict, Callable, List
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchmetrics as tm

from tml.ml_logging.torch_logging import logging


def train(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  train_steps: int,
  dataset: Iterable,
  scheduler: _LRScheduler = None,
  # Accept any arguments (to be compatible with the real training loop)
  # but just ignore them.
  *args,
  **kwargs,
) -> None:

  logging.warning("Running debug training loop, don't use for model training.")

  data_iter = iter(dataset)
  for step in range(0, train_steps + 1):
    x = next(data_iter)
    optimizer.zero_grad()
    loss, outputs = model.forward(x)
    loss.backward()
    optimizer.step()

    if scheduler:
      scheduler.step()

    logging.info(f"Step {step} completed. Loss = {loss}")
