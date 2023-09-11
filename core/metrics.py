"""Common metrics that also support multi task.

We assume multi task models will output [task_idx, ...] predictions

"""
from typing import Any, Dict

from tml.core.metric_mixin import MetricMixin, StratifyMixin, TaskMixin

import torch
import torchmetrics as tm


def probs_and_labels(
  outputs: Dict[str, torch.Tensor],
  task_idx: int,
) -> Dict[str, torch.Tensor]:
  """
    Extract probabilities and labels from model outputs.

    Args:
        outputs (Dict[str, torch.Tensor]): Model outputs.
        task_idx (int): Index of the task.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'preds' and 'target' tensors.
    """
  preds = outputs["probabilities"]
  target = outputs["labels"]
  if task_idx >= 0:
    preds = preds[:, task_idx]
    target = target[:, task_idx]
  return {
    "preds": preds,
    "target": target.int(),
  }


class Count(StratifyMixin, TaskMixin, MetricMixin, tm.SumMetric):
  def transform(self, outputs):
    """
    Count metric class that inherits from StratifyMixin, TaskMixin, MetricMixin, and SumMetric.

    This metric counts values after potential stratification and task selection.
    """
    outputs = self.maybe_apply_stratification(outputs, ["labels"])
    value = outputs["labels"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Ctr(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    Ctr (Click-Through Rate) metric class that inherits from StratifyMixin, TaskMixin, MetricMixin, and MeanMetric.

    This metric calculates the mean metric value after potential stratification and task selection.
    """

  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["labels"])
    value = outputs["labels"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Pctr(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    Pctr (Predicted Click-Through Rate) metric class that inherits from StratifyMixin, TaskMixin, MetricMixin, and MeanMetric.

    This metric calculates the mean metric value using probabilities after potential stratification and task selection.
    """
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities"])
    value = outputs["probabilities"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Precision(StratifyMixin, TaskMixin, MetricMixin, tm.Precision):
  """
    Precision metric class that inherits from StratifyMixin, TaskMixin, MetricMixin, and Precision.

    This metric computes precision after potential stratification and task selection.
    """
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class Recall(StratifyMixin, TaskMixin, MetricMixin, tm.Recall):
  """
    Recall metric class that inherits from StratifyMixin, TaskMixin, MetricMixin, and Recall.

    This metric computes recall after potential stratification and task selection.
    """
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class TorchMetricsRocauc(StratifyMixin, TaskMixin, MetricMixin, tm.AUROC):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class Auc(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    AUC (Area Under the ROC Curve) metric class.

    This metric computes the AUC metric based on the logits and labels in the model outputs.

    Args:
        num_samples (int): The number of samples used to compute AUC.
        **kwargs: Additional keyword arguments.
    
  Based on:
  https://github.com/facebookresearch/PyTorch-BigGraph/blob/a11ff0eb644b7e4cb569067c280112b47f40ef62/torchbiggraph/util.py#L420
  """

  def __init__(self, num_samples, **kwargs):
    super().__init__(**kwargs)
    self.num_samples = num_samples

  def transform(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    scores, labels = outputs["logits"], outputs["labels"]
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    result = {
      "value": pos_scores[torch.randint(len(pos_scores), (self.num_samples,))]
      > neg_scores[torch.randint(len(neg_scores), (self.num_samples,))]
    }
    return result


class PosRanks(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    PosRanks metric class.

    This metric computes the ranks of all positive examples based on the logits and labels
    in the model outputs.

    Args:
        **kwargs: Additional keyword arguments.

  https://github.com/facebookresearch/PyTorch-BigGraph/blob/a11ff0eb644b7e4cb569067c280112b47f40ef62/torchbiggraph/eval.py#L73
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def transform(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    scores, labels = outputs["logits"], outputs["labels"]
    _, sorted_indices = scores.sort(descending=True)
    pos_ranks = labels[sorted_indices].nonzero(as_tuple=True)[0] + 1  # all ranks start from 1
    result = {"value": pos_ranks}
    return result


class ReciprocalRank(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    ReciprocalRank metric class.

    This metric computes the reciprocal of the ranks of all positive examples based on the logits and labels
    in the model outputs.

    Args:
        **kwargs: Additional keyword arguments.
  https://github.com/facebookresearch/PyTorch-BigGraph/blob/a11ff0eb644b7e4cb569067c280112b47f40ef62/torchbiggraph/eval.py#L74
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def transform(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    scores, labels = outputs["logits"], outputs["labels"]
    _, sorted_indices = scores.sort(descending=True)
    pos_ranks = labels[sorted_indices].nonzero(as_tuple=True)[0] + 1  # all ranks start from 1
    result = {"value": torch.div(torch.ones_like(pos_ranks), pos_ranks)}
    return result


class HitAtK(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
    HitAtK metric class.

    This metric computes the fraction of positive examples that rank in the top K among their negatives,
    which is equivalent to precision@K.

    Args:
        k (int): The value of K.
        **kwargs: Additional keyword arguments.
  https://github.com/facebookresearch/PyTorch-BigGraph/blob/a11ff0eb644b7e4cb569067c280112b47f40ef62/torchbiggraph/eval.py#L75
  """

  def __init__(self, k: int, **kwargs):
    super().__init__(**kwargs)
    self.k = k

  def transform(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    scores, labels = outputs["logits"], outputs["labels"]
    _, sorted_indices = scores.sort(descending=True)
    pos_ranks = labels[sorted_indices].nonzero(as_tuple=True)[0] + 1  # all ranks start from 1
    result = {"value": (pos_ranks <= self.k).float()}
    return result
