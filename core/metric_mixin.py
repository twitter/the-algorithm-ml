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
    """
        Abstract method to transform model outputs into a dictionary of metrics.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs.

        Returns:
            Dict: A dictionary of computed metrics.
        """
    ...

  def update(self, outputs: Dict[str, torch.Tensor]):
    """
        Update the metrics based on model outputs.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs.
        """
    results = self.transform(outputs)
    # Do not try to update if any tensor is empty as a result of stratification.
    for value in results.values():
      if torch.is_tensor(value) and not value.nelement():
        return
    super().update(**results)


class TaskMixin:
  def __init__(self, task_idx: int = -1, **kwargs):
    """
        Initialize a TaskMixin instance.

        Args:
            task_idx (int): Index of the task associated with this mixin (default: -1).
            **kwargs: Additional keyword arguments.
        """
    super().__init__(**kwargs)
    self._task_idx = task_idx


class StratifyMixin:
  def __init__(
    self,
    stratifier=None,
    **kwargs,
  ):
    """
        Initialize a StratifyMixin instance.

        Args:
            stratifier: A stratifier for filtering outputs (default: None).
            **kwargs: Additional keyword arguments.
        """
    super().__init__(**kwargs)
    self._stratifier = stratifier

  def maybe_apply_stratification(
    self, outputs: Dict[str, torch.Tensor], value_names: List[str]
  ) -> Dict[str, torch.Tensor]:
    """
        Apply stratification to filter examples in the outputs.

        Pick out examples with values for which the stratifier feature is equal to a specific stratifier indicator value.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs.
            value_names (List[str]): Names of values to filter.

        Returns:
            Dict[str, torch.Tensor]: Filtered outputs.
        """
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
  """
    Returns a new class using MetricMixin and the given base_metric.

    Functionally the same as using inheritance, but it saves some lines of code
    if there's no need for class attributes.

    Args:
        base_metric (torchmetrics.Metric): The base metric class to prepend the transform to.
        transform (Callable): The transformation function to prepend to the metric.

    Returns:
        Type: A new class that includes MetricMixin and the provided base_metric
        with the specified transformation method.
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
