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
    outputs = self.maybe_apply_stratification(outputs, ["labels"])
    value = outputs["labels"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Ctr(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["labels"])
    value = outputs["labels"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Pctr(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities"])
    value = outputs["probabilities"]
    if self._task_idx >= 0:
      value = value[:, self._task_idx]
    return {"value": value}


class Precision(StratifyMixin, TaskMixin, MetricMixin, tm.Precision):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class Recall(StratifyMixin, TaskMixin, MetricMixin, tm.Recall):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class TorchMetricsRocauc(StratifyMixin, TaskMixin, MetricMixin, tm.AUROC):
  def transform(self, outputs):
    outputs = self.maybe_apply_stratification(outputs, ["probabilities", "labels"])
    return probs_and_labels(outputs, self._task_idx)


class Auc(StratifyMixin, TaskMixin, MetricMixin, tm.MeanMetric):
  """
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
  The ranks of all positives
  Based on:
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
  The reciprocal of the ranks of all
  Based on:
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
  The fraction of positives that rank in the top K among their negatives
  Note that this is basically precision@k
  Based on:
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
