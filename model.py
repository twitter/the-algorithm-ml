"""Wraps servable model in loss and RecapBatch passing to be trainable."""
# flake8: noqa
from typing import Callable

from tml.ml_logging.torch_logging import logging  # type: ignore[attr-defined]

import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel


class ModelAndLoss(torch.nn.Module):
  # Reconsider our approach at a later date: https://ppwwyyxx.com/blog/2022/Loss-Function-Separation/

  def __init__(
    self,
    model,
    loss_fn: Callable,
  ) -> None:
    """
    Args:
      model: torch module to wrap.
      loss_fn: Function for calculating loss, should accept logits and labels.
    """
    super().__init__()
    self.model = model
    self.loss_fn = loss_fn

  def forward(self, batch: "RecapBatch"):  # type: ignore[name-defined]
    """Runs model forward and calculates loss according to given loss_fn.

    NOTE: The input signature here needs to be a Pipelineable object for
    prefetching purposes during training using torchrec's pipeline.  However
    the underlying model signature needs to be exportable to onnx, requiring
    generic python types.  see https://pytorch.org/docs/stable/onnx.html#types.

    """
    outputs = self.model(batch)
    losses = self.loss_fn(outputs["logits"], batch.labels.float(), batch.weights.float())

    outputs.update(
      {
        "loss": losses,
        "labels": batch.labels,
        "weights": batch.weights,
      }
    )

    # Allow multiple losses.
    return losses, outputs


def maybe_shard_model(
  model,
  device: torch.device,
):
  """Set up and apply DistributedModelParallel to a model if running in a distributed environment.

    If in a distributed environment, constructs Topology, sharders, and ShardingPlan, then applies
    DistributedModelParallel.

  If not in a distributed environment, returns model directly.
  """
  if dist.is_initialized():
    logging.info("***** Wrapping in DistributedModelParallel *****")
    logging.info(f"Model before wrapping: {model}")
    model = DistributedModelParallel(
      module=model,
      device=device,
    )
    logging.info(f"Model after wrapping: {model}")

  return model


def log_sharded_tensor_content(weight_name: str, table_name: str, weight_tensor) -> None:
  """Handy function to log the content of EBC embedding layer.
     Only works for single GPU machines.

  Args:
      weight_name: name of tensor, as defined in model
      table_name: name of the EBC table the weight is taken from
      weight_tensor: embedding weight tensor
  """
  logging.info(f"{weight_name}, {table_name}", rank=-1)
  logging.info(f"{weight_tensor.metadata()}", rank=-1)
  output_tensor = torch.zeros(*weight_tensor.size(), device=torch.device("cuda:0"))
  weight_tensor.gather(out=output_tensor)
  logging.info(f"{output_tensor}", rank=-1)
