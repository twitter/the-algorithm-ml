from typing import Callable, Optional, List
from tml.projects.home.recap.embedding import config as embedding_config_mod
import torch
from absl import logging


class ModelAndLoss(torch.nn.Module):
  """
    PyTorch module that combines a neural network model and loss function.

    This module wraps a neural network model and facilitates the forward pass through the model
    while also calculating the loss based on the model's predictions and provided labels.

    Args:
        model: The torch module to wrap.
        loss_fn (Callable): Function for calculating the loss, which should accept logits and labels.
        stratifiers (Optional[List[embedding_config_mod.StratifierConfig]]): A list of stratifier configurations
            for metrics stratification. Each stratifier config includes the name and index of discrete features
            to emit for stratification.

    Example:
        To use `ModelAndLoss` in a PyTorch training loop, you can create an instance of it and pass your model
        and loss function as arguments:

        ```python
        # Create a neural network model
        model = YourNeuralNetworkModel()

        # Define a loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create an instance of ModelAndLoss
        model_and_loss = ModelAndLoss(model, loss_fn)

        # Generate a batch of training data (e.g., RecapBatch)
        batch = generate_training_batch()

        # Perform a forward pass through the model and calculate the loss
        loss, outputs = model_and_loss(batch)

        # You can now backpropagate and optimize using the computed loss
        loss.backward()
        optimizer.step()
        ```

    Note:
        The `ModelAndLoss` class simplifies the process of running forward passes through a model and
        calculating loss, making it easier to integrate the model into your training loop. Additionally,
        it supports the addition of stratifiers for metrics stratification, if needed.

    Warning:
        This class is intended for internal use within neural network architectures and should not be
        directly accessed or modified by external code.
    """
  def __init__(
    self,
    model,
    loss_fn: Callable,
    stratifiers: Optional[List[embedding_config_mod.StratifierConfig]] = None,
  ) -> None:
    """
        Initializes the ModelAndLoss module.

        Args:
            model: The torch module to wrap.
            loss_fn (Callable): Function for calculating the loss, which should accept logits and labels.
            stratifiers (Optional[List[embedding_config_mod.StratifierConfig]]): A list of stratifier configurations
                for metrics stratification.
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
