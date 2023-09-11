from dataclasses import dataclass
from typing import Tuple

from tml.common.batch import DataclassBatch
from tml.common.testing_utils import mock_pg
from tml.core import train_pipeline

import torch
from torchrec.distributed import DistributedModelParallel


@dataclass
class MockDataclassBatch(DataclassBatch):
  """
    Mock data class batch for testing purposes.

    This class represents a batch of data with continuous features and labels.

    Attributes:
        continuous_features (torch.Tensor): Tensor containing continuous feature data.
        labels (torch.Tensor): Tensor containing label data.
    """
  continuous_features: torch.Tensor
  labels: torch.Tensor


class MockModule(torch.nn.Module):
  """
    Mock PyTorch module for testing purposes.

    This module defines a simple neural network model with a linear layer
    followed by a BCEWithLogitsLoss loss function.

    Attributes:
        model (torch.nn.Linear): The linear model layer.
        loss_fn (torch.nn.BCEWithLogitsLoss): Binary cross-entropy loss function.
    """
  def __init__(self) -> None:
    super().__init__()
    self.model = torch.nn.Linear(10, 1)
    self.loss_fn = torch.nn.BCEWithLogitsLoss()

  def forward(self, batch: MockDataclassBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Forward pass of the mock module.

        Args:
            batch (MockDataclassBatch): Input data batch with continuous features and labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss and predictions.
        """
    pred = self.model(batch.continuous_features)
    loss = self.loss_fn(pred, batch.labels)
    return (loss, pred)


def create_batch(bsz: int):
  """
    Create a mock data batch with random continuous features and labels.

    Args:
        bsz (int): Batch size.

    Returns:
        MockDataclassBatch: A batch of data with continuous features and labels.
    """
  return MockDataclassBatch(
    continuous_features=torch.rand(bsz, 10).float(),
    labels=torch.bernoulli(torch.empty(bsz, 1).uniform_(0, 1)).float(),
  )


def test_sparse_pipeline():
  """
    Test function for the sparse pipeline with distributed model parallelism.

    This function tests the behavior of the sparse training pipeline using
    a mock module and data.
    """

  device = torch.device("cpu")
  model = MockModule().to(device)

  steps = 8
  example = create_batch(1)
  dataloader = iter(example for _ in range(steps + 2))

  results = []
  with mock_pg():
    d_model = DistributedModelParallel(model)
    pipeline = train_pipeline.TrainPipelineSparseDist(
      model=d_model,
      optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
      device=device,
      grad_accum=2,
    )
    for _ in range(steps):
      results.append(pipeline.progress(dataloader))

  results = [elem.detach().numpy() for elem in results]
  # Check gradients are accumulated, i.e. results do not change for every 0th and 1th.
  for first, second in zip(results[::2], results[1::2]):
    assert first == second, results

  # Check we do update gradients, i.e. results do change for every 1th and 2nd.
  for first, second in zip(results[1::2], results[2::2]):
    assert first != second, results


def test_amp():
  """
    Test automatic mixed-precision (AMP) training with the sparse pipeline.

    This function tests the behavior of the sparse training pipeline with
    automatic mixed-precision (AMP) enabled, using a mock module and data.

    AMP allows for faster training by using lower-precision data types, such as
    torch.bfloat16, while maintaining model accuracy.
  """
  device = torch.device("cpu")
  model = MockModule().to(device)

  steps = 8
  example = create_batch(1)
  dataloader = iter(example for _ in range(steps + 2))

  results = []
  with mock_pg():
    d_model = DistributedModelParallel(model)
    pipeline = train_pipeline.TrainPipelineSparseDist(
      model=d_model,
      optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
      device=device,
      enable_amp=True,
      # Not supported on CPU.
      enable_grad_scaling=False,
    )
    for _ in range(steps):
      results.append(pipeline.progress(dataloader))

  results = [elem.detach() for elem in results]
  for value in results:
    assert value.dtype == torch.bfloat16
