from typing import Callable, Optional, List
from tml.projects.home.recap.embedding import config as embedding_config_mod
import torch
from absl import logging


class ModelAndLoss(torch.nn.Module):
  def __init__(
    self,
    model,
    loss_fn: Callable,
    stratifiers: Optional[List[embedding_config_mod.StratifierConfig]] = None,
  ) -> None:
    """
    Args:
      model: torch module to wrap.
      loss_fn: Function for calculating loss, should accept logits and labels.
      straitifiers: mapping of stratifier name and index of discrete features to emit for metrics stratification.
    """
    super().__init__()
    self.model = model
    self.loss_fn = loss_fn
    self.stratifiers = stratifiers

  def forward(self, batch: "RecapBatch"):  # type: ignore[name-defined]
    """Runs model forward and calculates loss according to given loss_fn.

    NOTE: The input signature here needs to be a Pipelineable object for
    prefetching purposes during training using torchrec's pipeline.  However
    the underlying model signature needs to be exportable to onnx, requiring
    generic python types.  see https://pytorch.org/docs/stable/onnx.html#types.

    """
    outputs = self.model(
      continuous_features=batch.continuous_features,
      binary_features=batch.binary_features,
      discrete_features=batch.discrete_features,
      sparse_features=batch.sparse_features,
      user_embedding=batch.user_embedding,
      user_eng_embedding=batch.user_eng_embedding,
      author_embedding=batch.author_embedding,
      labels=batch.labels,
      weights=batch.weights,
    )
    losses = self.loss_fn(outputs["logits"], batch.labels.float(), batch.weights.float())

    if self.stratifiers:
      logging.info(f"***** Adding stratifiers *****\n {self.stratifiers}")
      outputs["stratifiers"] = {}
      for stratifier in self.stratifiers:
        outputs["stratifiers"][stratifier.name] = batch.discrete_features[:, stratifier.index]

    # In general, we can have a large number of losses returned by our loss function.
    if isinstance(losses, dict):
      return losses["loss"], {
        **outputs,
        **losses,
        "labels": batch.labels,
        "weights": batch.weights,
      }
    else:  # Assume that this is a float.
      return losses, {
        **outputs,
        "loss": losses,
        "labels": batch.labels,
        "weights": batch.weights,
      }
