"""
AUROC metrics.
"""
from typing import Union

from tml.ml_logging.torch_logging import logging

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat


def _compute_helper(
  predictions: torch.Tensor,
  target: torch.Tensor,
  weights: torch.Tensor,
  max_positive_negative_weighted_sum: torch.Tensor,
  min_positive_negative_weighted_sum: torch.Tensor,
  equal_predictions_as_incorrect: bool,
) -> torch.Tensor:
  """
  Compute AUROC.
  Args:
    predictions: The predictions probabilities.
    target: The target.
    weights: The sample weights to assign to each sample in the batch.
    max_positive_negative_weighted_sum: The sum of the weights for the positive labels.
    min_positive_negative_weighted_sum:
    equal_predictions_as_incorrect: For positive & negative labels having identical scores,
     we assume that they are correct prediction (i.e weight = 1) when ths is False. Otherwise,
     we assume that they are correct prediction (i.e weight = 0).
  """
  dim = 0

  # Sort predictions based on key (score, true_label). The order is ascending for score.
  # For true_label, order is ascending if equal_predictions_as_incorrect is True;
  # otherwise it is descending.
  target_order = torch.argsort(target, dim=dim, descending=equal_predictions_as_incorrect)
  score_order = torch.sort(torch.gather(predictions, dim, target_order), stable=True, dim=dim)[1]
  score_order = torch.gather(target_order, dim, score_order)
  sorted_target = torch.gather(target, dim, score_order)
  sorted_weights = torch.gather(weights, dim, score_order)

  negatives_from_left = torch.cumsum((1.0 - sorted_target) * sorted_weights, 0)

  numerator = torch.sum(
    sorted_weights * (sorted_target * negatives_from_left / max_positive_negative_weighted_sum)
  )

  return numerator / min_positive_negative_weighted_sum


class AUROCWithMWU(torchmetrics.Metric):
  """
  AUROC using Mann-Whitney U-test.
  See https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve.

  This AUROC implementation is well suited to (non-zero) low-CTR. In particular it will return
  the correct AUROC even if the predicted probabilities are all close to 0.
  Currently only support binary classification.
  """

  def __init__(self, label_threshold: float = 0.5, raise_missing_class: bool = False, **kwargs):
    """

    Args:
      label_threshold: Labels strictly above this threshold are considered positive labels,
                       otherwise, they are considered negative.
      raise_missing_class: If True, an error will be raise if negative or positive class is missing.
        Otherwise, we will simply log a warning.
      **kwargs: Additional parameters supported by all torchmetrics.Metric.
    """
    super().__init__(**kwargs)
    self.add_state("predictions", default=[], dist_reduce_fx="cat")
    self.add_state("target", default=[], dist_reduce_fx="cat")
    self.add_state("weights", default=[], dist_reduce_fx="cat")

    self.label_threshold = label_threshold
    self.raise_missing_class = raise_missing_class

  def update(
    self,
    predictions: torch.Tensor,
    target: torch.Tensor,
    weight: Union[float, torch.Tensor] = 1.0,
  ) -> None:
    """
    Update the current auroc.
    Args:
      predictions: Predicted values, 1D Tensor or 2D Tensor of shape batch_size x 1.
      target: Ground truth. Must have same shape as predictions.
      weight: The weight to use for the predicted values. Shape should be
      broadcastable to that of predictions.
    """
    self.predictions.append(predictions)
    self.target.append(target)
    if not isinstance(weight, torch.Tensor):
      weight = torch.as_tensor(weight, dtype=predictions.dtype, device=target.device)
    self.weights.append(torch.broadcast_to(weight, predictions.size()))

  def compute(self) -> torch.Tensor:
    """
    Compute and return the accumulated AUROC.
    """
    weights = dim_zero_cat(self.weights)
    predictions = dim_zero_cat(self.predictions)
    target = dim_zero_cat(self.target).type_as(predictions)

    negative_mask = target <= self.label_threshold
    positive_mask = torch.logical_not(negative_mask)

    if not negative_mask.any():
      msg = "Negative class missing. AUROC returned will be meaningless."
      if self.raise_missing_class:
        raise ValueError(msg)
      else:
        logging.warn(msg)
    if not positive_mask.any():
      msg = "Positive class missing. AUROC returned will be meaningless."
      if self.raise_missing_class:
        raise ValueError(msg)
      else:
        logging.warn(msg)

    weighted_actual_negative_sum = torch.sum(
      torch.where(negative_mask, weights, torch.zeros_like(weights))
    )

    weighted_actual_positive_sum = torch.sum(
      torch.where(positive_mask, weights, torch.zeros_like(weights))
    )

    max_positive_negative_weighted_sum = torch.max(
      weighted_actual_negative_sum, weighted_actual_positive_sum
    )

    min_positive_negative_weighted_sum = torch.min(
      weighted_actual_negative_sum, weighted_actual_positive_sum
    )

    # Compute auroc with the weight set to 1 when positive & negative have identical scores.
    auroc_le = _compute_helper(
      target=target,
      weights=weights,
      predictions=predictions,
      min_positive_negative_weighted_sum=min_positive_negative_weighted_sum,
      max_positive_negative_weighted_sum=max_positive_negative_weighted_sum,
      equal_predictions_as_incorrect=False,
    )

    # Compute auroc with the weight set to 0 when positive & negative have identical scores.
    auroc_lt = _compute_helper(
      target=target,
      weights=weights,
      predictions=predictions,
      min_positive_negative_weighted_sum=min_positive_negative_weighted_sum,
      max_positive_negative_weighted_sum=max_positive_negative_weighted_sum,
      equal_predictions_as_incorrect=True,
    )

    # Compute auroc with the weight set to 1/2 when positive & negative have identical scores.
    return auroc_le - (auroc_le - auroc_lt) / 2.0
