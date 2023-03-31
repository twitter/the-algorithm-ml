"""
Mixin that requires a transform to munge output dictionary of tensors a
model produces to a form that the torchmetrics.Metric.update expects.

By unifying on our signature for `update`, we can also now use
torchmetrics.MetricCollection which requires all metrics have
the same call signature.

To use, override this with a transform that munges `outputs`
into a kwargs dict that the inherited metric.update accepts.

Here are two examples of how to extend torchmetrics.SumMetric so that it accepts
an output dictionary of tensors and munges it to what SumMetric expects (single `value`)
for its update method.

1. Using as a mixin to inherit from or define a new metric class.

  class Count(MetricMixin, SumMetric):
    def transform(self, outputs):
      return {'value': 1}

2. Redefine an existing metric class.

  SumMetric = prepend_transform(SumMetric, lambda outputs: {'value': 1})

"""
from abc import abstractmethod
from typing import Callable, Dict, List

from tml.ml_logging.torch_logging import logging  # type: ignore[attr-defined]

import torch
import torchmetrics


class MetricMixin:
  @abstractmethod
  def transform(self, outputs: Dict[str, torch.Tensor]) -> Dict:
    ...

  def update(self, outputs: Dict[str, torch.Tensor]):
    results = self.transform(outputs)
    # Do not try to update if any tensor is empty as a result of stratification.
    for value in results.values():
      if torch.is_tensor(value) and not value.nelement():
        return
    super().update(**results)


class TaskMixin:
  def __init__(self, task_idx: int = -1, **kwargs):
    super().__init__(**kwargs)
    self._task_idx = task_idx


class StratifyMixin:
  def __init__(
    self,
    stratifier=None,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self._stratifier = stratifier

  def maybe_apply_stratification(
    self, outputs: Dict[str, torch.Tensor], value_names: List[str]
  ) -> Dict[str, torch.Tensor]:
    """Pick out examples with values for which the stratifier feature is equal to a specific stratifier indicator value."""
    outputs = outputs.copy()
    if not self._stratifier:
      return outputs
    stratifiers = outputs.get("stratifiers")
    if not stratifiers:
      return outputs
    if stratifiers.get(self._stratifier.name) is None:
      return outputs

    mask = torch.flatten(outputs["stratifiers"][self._stratifier.name] == self._stratifier.value)
    target_slice = torch.squeeze(mask.nonzero(), -1)
    for value_name in value_names:
      target = outputs[value_name]
      outputs[value_name] = torch.index_select(target, 0, target_slice)
    return outputs


def prepend_transform(base_metric: torchmetrics.Metric, transform: Callable):
  """Returns new class using MetricMixin and given base_metric.

  Functionally the same using inheritance, just saves some lines of code
  if no need for class attributes.

  """

  def transform_method(_self, *args, **kwargs):
    return transform(*args, **kwargs)

  return type(
    base_metric.__name__,
    (
      MetricMixin,
      base_metric,
    ),
    {"transform": transform_method},
  )
