"""Loss functions -- including multi task ones."""

import typing

from tml.core.loss_type import LossType
from tml.ml_logging.torch_logging import logging

import torch


def _maybe_warn(reduction: str):
  """
    Emit a warning if the reduction method is different from 'mean'.

    Args:
        reduction (str): The reduction method being used.
    """
  if reduction != "mean":
    logging.warn(
      f"For the same global_batch_size, the gradient in DDP is guaranteed to be equal,"
      f"to the gradient without DDP only for mean reduction. If you need this property for"
      f"the provided reduction {reduction}, it needs to be implemented."
    )


def build_loss(
  loss_type: LossType,
  reduction="mean",
):
  """
    Build a loss function based on the specified loss type and reduction method.

    Args:
        loss_type (LossType): The type of loss to build.
        reduction (str): The reduction method for the loss (default: 'mean').

    Returns:
        Callable: A loss function that takes logits and labels as input.
    """
  _maybe_warn(reduction)
  f = _LOSS_TYPE_TO_FUNCTION[loss_type]

  def loss_fn(logits, labels):
    return f(logits, labels.type_as(logits), reduction=reduction)

  return loss_fn


def get_global_loss_detached(local_loss, reduction="mean"):
  """
    Perform all_reduce to obtain the global loss function using the provided reduction.

    Args:
        local_loss (torch.Tensor): The local loss of the current rank.
        reduction (str): The reduction to use for all_reduce. Should match the reduction used by DDP.

    Returns:
        torch.Tensor: The reduced and detached global loss.
    """
  if reduction != "mean":
    logging.warn(
      f"The reduction used in this function should be the same as the one used by "
      f"the DDP model. By default DDP uses mean, So ensure that DDP is appropriately"
      f"modified for reduction {reduction}."
    )

  if reduction not in ["mean", "sum"]:
    raise ValueError(f"Reduction {reduction} is currently unsupported.")

  global_loss = local_loss.detach()

  if reduction == "mean":
    global_loss.div_(torch.distributed.get_world_size())

  torch.distributed.all_reduce(global_loss)
  return global_loss


def build_multi_task_loss(
  loss_type: LossType,
  tasks: typing.List[str],
  task_loss_reduction="mean",
  global_reduction="mean",
  pos_weights=None,
):
  """
    Build a multi-task loss function based on the specified loss type and configurations.

    Args:
        loss_type (LossType): The type of loss to build.
        tasks (typing.List[str]): List of task names.
        task_loss_reduction (str): Reduction method for task-specific losses (default: 'mean').
        global_reduction (str): Reduction method for the global loss (default: 'mean').
        pos_weights (Optional): Positive class weights for tasks (default: None).

    Returns:
        Callable: A multi-task loss function that takes logits, labels, and weights as input.
    """ 
  _maybe_warn(global_reduction)
  _maybe_warn(task_loss_reduction)
  f = _LOSS_TYPE_TO_FUNCTION[loss_type]

  loss_reduction_fns = {
    "mean": torch.mean,
    "sum": torch.sum,
    "min": torch.min,
    "max": torch.max,
    "median": torch.median,
  }

  def loss_fn(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    if pos_weights is None:
      torch_weights = torch.ones([len(tasks)])
    else:
      torch_weights = torch.tensor(pos_weights)

    losses = {}
    for task_idx, task in enumerate(tasks):
      task_logits = logits[:, task_idx]
      label = labels[:, task_idx].type_as(task_logits)

      loss = f(
        task_logits,
        label,
        reduction=task_loss_reduction,
        pos_weight=torch_weights[task_idx],
        weight=weights[:, task_idx],
      )
      losses[f"loss/{task}"] = loss

    losses["loss"] = loss_reduction_fns[global_reduction](torch.stack(list(losses.values())))
    return losses

  return loss_fn


_LOSS_TYPE_TO_FUNCTION = {
  LossType.BCE_WITH_LOGITS: torch.nn.functional.binary_cross_entropy_with_logits
}
