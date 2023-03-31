"""
Contains aggregation metrics.
"""
from typing import Tuple, Union

import torch
import torchmetrics


def update_mean(
  current_mean: torch.Tensor,
  current_weight_sum: torch.Tensor,
  value: torch.Tensor,
  weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Update the mean according to Welford formula:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version.
  See also https://nullbuffer.com/articles/welford_algorithm.html for more information.
  Args:
    current_mean: The value of the current accumulated mean.
    current_weight_sum: The current weighted sum.
    value: The new value that needs to be added to get a new mean.
    weight: The weights for the new value.

  Returns: The updated mean and updated weighted sum.

  """
  weight = torch.broadcast_to(weight, value.shape)

  # Avoiding (on purpose) in-place operation when using += in case
  # current_mean and current_weight_sum share the same storage
  current_weight_sum = current_weight_sum + torch.sum(weight)
  current_mean = current_mean + torch.sum((weight / current_weight_sum) * (value - current_mean))
  return current_mean, current_weight_sum


def stable_mean_dist_reduce_fn(state: torch.Tensor) -> torch.Tensor:
  """
  Merge the state from multiple workers.
  Args:
    state: A tensor with the first dimension indicating workers.

  Returns: The accumulated mean from all workers.

  """
  mean, weight_sum = update_mean(
    current_mean=torch.as_tensor(0.0, dtype=state.dtype, device=state.device),
    current_weight_sum=torch.as_tensor(0.0, dtype=state.dtype, device=state.device),
    value=state[:, 0],
    weight=state[:, 1],
  )
  return torch.stack([mean, weight_sum])


class StableMean(torchmetrics.Metric):
  """
  This implements a numerical stable mean metrics computation using Welford algorithm according to
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_batched_version.
  For example when using float32, the algorithm will give a valid output even if the "sum" is larger
   than the maximum float32 as far as the mean is within the limit of float32.
  See also https://nullbuffer.com/articles/welford_algorithm.html for more information.
  """

  def __init__(self, **kwargs):
    """
    Args:
      **kwargs: Additional parameters supported by all torchmetrics.Metric.
    """
    super().__init__(**kwargs)
    self.add_state(
      "mean_and_weight_sum",
      default=torch.zeros(2),
      dist_reduce_fx=stable_mean_dist_reduce_fn,
    )

  def update(self, value: torch.Tensor, weight: Union[float, torch.Tensor] = 1.0) -> None:
    """
    Update the current mean.
    Args:
      value: Value to update the mean with.
      weight: weight to use. Shape should be broadcastable to that of value.
    """
    mean, weight_sum = self.mean_and_weight_sum[0], self.mean_and_weight_sum[1]

    if not isinstance(weight, torch.Tensor):
      weight = torch.as_tensor(weight, dtype=value.dtype, device=value.device)

    self.mean_and_weight_sum[0], self.mean_and_weight_sum[1] = update_mean(
      mean, weight_sum, value, torch.as_tensor(weight)
    )

  def compute(self) -> torch.Tensor:
    """
    Compute and return the accumulated mean.
    """
    return self.mean_and_weight_sum[0]
