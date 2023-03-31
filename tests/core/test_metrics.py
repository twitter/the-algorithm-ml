from dataclasses import dataclass

from tml.core import metrics as core_metrics
from tml.core.metric_mixin import MetricMixin, prepend_transform

import torch
from torchmetrics import MaxMetric, MetricCollection, SumMetric


@dataclass
class MockStratifierConfig:
  name: str
  index: int
  value: int


class Count(MetricMixin, SumMetric):
  def transform(self, outputs):
    return {"value": 1}


Max = prepend_transform(MaxMetric, lambda outputs: {"value": outputs["value"]})


def test_count_metric():
  num_examples = 123
  examples = [
    {"stuff": 0},
  ] * num_examples

  metric = Count()
  for outputs in examples:
    metric.update(outputs)

  assert metric.compute().item() == num_examples


def test_collections():
  max_metric = Max()
  count_metric = Count()
  metric = MetricCollection([max_metric, count_metric])

  examples = [{"value": idx} for idx in range(123)]
  for outputs in examples:
    metric.update(outputs)

  assert metric.compute() == {
    max_metric.__class__.__name__: len(examples) - 1,
    count_metric.__class__.__name__: len(examples),
  }


def test_task_dependent_ctr():
  num_examples = 144
  batch_size = 1024
  outputs = [
    {
      "stuff": 0,
      "labels": torch.arange(0, 6).repeat(batch_size, 1),
    }
    for idx in range(num_examples)
  ]

  for task_idx in range(5):
    metric = core_metrics.Ctr(task_idx=task_idx)
    for output in outputs:
      metric.update(output)
    assert metric.compute().item() == task_idx


def test_stratified_ctr():
  outputs = [
    {
      "stuff": 0,
      # [bsz, tasks]
      "labels": torch.tensor(
        [
          [0, 1, 2, 3],
          [1, 2, 3, 4],
          [2, 3, 4, 0],
        ]
      ),
      "stratifiers": {
        # [bsz]
        "level": torch.tensor(
          [9, 0, 9],
        ),
      },
    }
  ]

  stratifier = MockStratifierConfig(name="level", index=2, value=9)
  for task_idx in range(5):
    metric = core_metrics.Ctr(task_idx=1, stratifier=stratifier)
    for output in outputs:
      metric.update(output)
    # From the dataset of:
    # [
    #   [0, 1, 2, 3],
    #   [1, 2, 3, 4],
    #   [2, 3, 4, 0],
    # ]
    # we pick out
    # [
    #   [0, 1, 2, 3],
    #   [2, 3, 4, 0],
    # ]
    # and with Ctr task_idx, we pick out
    # [
    #   [1,],
    #   [3,],
    # ]
    assert metric.compute().item() == (1 + 3) / 2


def test_auc():
  num_samples = 10000
  metric = core_metrics.Auc(num_samples)
  target = torch.tensor([0, 0, 1, 1, 1])
  preds_correct = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
  outputs_correct = {"logits": preds_correct, "labels": target}
  preds_bad = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
  outputs_bad = {"logits": preds_bad, "labels": target}

  metric.update(outputs_correct)
  assert metric.compute().item() == 1.0

  metric.reset()
  metric.update(outputs_bad)
  assert metric.compute().item() == 0.0


def test_pos_rank():
  metric = core_metrics.PosRanks()
  target = torch.tensor([0, 0, 1, 1, 1])
  preds_correct = torch.tensor([-1.0, -1.0, 0.5, 1.0, 1.5])
  outputs_correct = {"logits": preds_correct, "labels": target}
  preds_bad = torch.tensor([1.0, 1.0, -1.5, -1.0, -0.5])
  outputs_bad = {"logits": preds_bad, "labels": target}

  metric.update(outputs_correct)
  assert metric.compute().item() == 2.0

  metric.reset()
  metric.update(outputs_bad)
  assert metric.compute().item() == 4.0


def test_reciprocal_rank():
  metric = core_metrics.ReciprocalRank()
  target = torch.tensor([0, 0, 1, 1, 1])
  preds_correct = torch.tensor([-1.0, -1.0, 0.5, 1.0, 1.5])
  outputs_correct = {"logits": preds_correct, "labels": target}
  preds_bad = torch.tensor([1.0, 1.0, -1.5, -1.0, -0.5])
  outputs_bad = {"logits": preds_bad, "labels": target}

  metric.update(outputs_correct)
  assert abs(metric.compute().item() - 0.6111) < 0.001

  metric.reset()
  metric.update(outputs_bad)
  assert abs(metric.compute().item() == 0.2611) < 0.001


def test_hit_k():
  hit1_metric = core_metrics.HitAtK(1)
  target = torch.tensor([0, 0, 1, 1, 1])
  preds_correct = torch.tensor([-1.0, 1.0, 0.5, -0.1, 1.5])
  outputs_correct = {"logits": preds_correct, "labels": target}
  preds_bad = torch.tensor([1.0, 1.0, -1.5, -1.0, -0.5])
  outputs_bad = {"logits": preds_bad, "labels": target}

  hit1_metric.update(outputs_correct)
  assert abs(hit1_metric.compute().item() - 0.3333) < 0.0001

  hit1_metric.reset()
  hit1_metric.update(outputs_bad)

  assert hit1_metric.compute().item() == 0

  hit3_metric = core_metrics.HitAtK(3)
  hit3_metric.update(outputs_correct)
  assert (hit3_metric.compute().item() - 0.66666) < 0.0001

  hit3_metric.reset()
  hit3_metric.update(outputs_bad)
  assert abs(hit3_metric.compute().item() - 0.3333) < 0.0001
