from typing import Callable
import math

from tml.projects.twhin.data.edges import EdgeBatch
from tml.projects.twhin.models.config import TwhinModelConfig
from tml.projects.twhin.data.config import TwhinDataConfig
from tml.common.modules.embedding.embedding import LargeEmbeddings
from tml.optimizers.optimizer import get_optimizer_class
from tml.optimizers.config import get_optimizer_algorithm_config

import torch
from torch import nn
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward


class TwhinModel(nn.Module):
  """
    Twhin model for graph-based entity embeddings and translation.

    This class defines the Twhin model, which is used for learning embeddings of entities in a graph
    and applying translations to these embeddings based on graph relationships.

    Args:
        model_config (TwhinModelConfig): Configuration for the Twhin model.
        data_config (TwhinDataConfig): Configuration for the data used by the model.

    Attributes:
        batch_size (int): The batch size used for training.
        table_names (List[str]): Names of tables in the model's embeddings.
        large_embeddings (LargeEmbeddings): LargeEmbeddings instance for entity embeddings.
        embedding_dim (int): Dimensionality of entity embeddings.
        num_tables (int): Number of tables in the model's embeddings.
        in_batch_negatives (int): Number of in-batch negative samples to use during training.
        global_negatives (int): Number of global negative samples to use during training.
        num_relations (int): Number of graph relationships in the model.
        all_trans_embs (torch.nn.Parameter): Parameter tensor for translation embeddings.

    """
  def __init__(self, model_config: TwhinModelConfig, data_config: TwhinDataConfig):
    super().__init__()
    self.batch_size = data_config.per_replica_batch_size
    self.table_names = [table.name for table in model_config.embeddings.tables]
    self.large_embeddings = LargeEmbeddings(model_config.embeddings)
    self.embedding_dim = model_config.embeddings.tables[0].embedding_dim
    self.num_tables = len(model_config.embeddings.tables)
    self.in_batch_negatives = data_config.in_batch_negatives
    self.global_negatives = data_config.global_negatives
    self.num_relations = len(model_config.relations)

    # one bias per relation
    self.all_trans_embs = torch.nn.parameter.Parameter(
      torch.nn.init.uniform_(torch.empty(self.num_relations, self.embedding_dim))
    )

  def forward(self, batch: EdgeBatch):
    """
        Forward pass of the Twhin model.

        Args:
            batch (EdgeBatch): Input batch containing graph edge information.

        Returns:
            dict: A dictionary containing model output with "logits" and "probabilities".
                - "logits" (torch.Tensor): Logit scores.
                - "probabilities" (torch.Tensor): Sigmoid probabilities.
        """
    # B x D
    trans_embs = self.all_trans_embs.data[batch.rels]

    # KeyedTensor
    outs = self.large_embeddings(batch.nodes)

    # 2B x TD
    x = outs.values()

    # 2B x T x D
    x = x.reshape(2 * self.batch_size, -1, self.embedding_dim)

    # 2B x D
    x = torch.sum(x, 1)

    # B x 2 x D
    x = x.reshape(self.batch_size, 2, self.embedding_dim)

    # translated
    translated = x[:, 1, :] + trans_embs

    negs = []
    if self.in_batch_negatives:
      # construct dot products for negatives via matmul
      for relation in range(self.num_relations):
        rel_mask = batch.rels == relation
        rel_count = rel_mask.sum()

        if not rel_count:
          continue

        # R x D
        lhs_matrix = x[rel_mask, 0, :]
        rhs_matrix = x[rel_mask, 1, :]

        lhs_perm = torch.randperm(lhs_matrix.shape[0])
        # repeat until we have enough negatives
        lhs_perm = lhs_perm.repeat(math.ceil(float(self.in_batch_negatives) / rel_count))
        lhs_indices = lhs_perm[: self.in_batch_negatives]
        sampled_lhs = lhs_matrix[lhs_indices]

        rhs_perm = torch.randperm(rhs_matrix.shape[0])
        # repeat until we have enough negatives
        rhs_perm = rhs_perm.repeat(math.ceil(float(self.in_batch_negatives) / rel_count))
        rhs_indices = rhs_perm[: self.in_batch_negatives]
        sampled_rhs = rhs_matrix[rhs_indices]

        # RS
        negs_rhs = torch.flatten(torch.matmul(lhs_matrix, sampled_rhs.t()))
        negs_lhs = torch.flatten(torch.matmul(rhs_matrix, sampled_lhs.t()))

        negs.append(negs_lhs)
        negs.append(negs_rhs)

    # dot product for positives
    x = (x[:, 0, :] * translated).sum(-1)

    # concat positives and negatives
    x = torch.cat([x, *negs])
    return {
      "logits": x,
      "probabilities": torch.sigmoid(x),
    }


def apply_optimizers(model: TwhinModel, model_config: TwhinModelConfig):
  """
    Apply optimizers to the Twhin model's embeddings.

    This function applies optimizers to the embeddings of the Twhin model based on the provided configuration.

    Args:
        model (TwhinModel): The Twhin model to apply optimizers to.
        model_config (TwhinModelConfig): Configuration for the Twhin model.

    Returns:
        TwhinModel: The Twhin model with optimizers applied to its embeddings.
    """
  for table in model_config.embeddings.tables:
    optimizer_class = get_optimizer_class(table.optimizer)
    optimizer_kwargs = get_optimizer_algorithm_config(table.optimizer).dict()
    params = [
      param
      for name, param in model.large_embeddings.ebc.named_parameters()
      if (name.startswith(f"embedding_bags.{table.name}"))
    ]
    apply_optimizer_in_backward(
      optimizer_class=optimizer_class,
      params=params,
      optimizer_kwargs=optimizer_kwargs,
    )

  return model


class TwhinModelAndLoss(torch.nn.Module):
  def __init__(
    self,
    model,
    loss_fn: Callable,
    data_config: TwhinDataConfig,
    device: torch.device,
  ) -> None:
    """
        Initialize a TwhinModelAndLoss module.

        Args:
            model: The torch module to wrap.
            loss_fn: A function for calculating loss, should accept logits and labels.
            data_config: Configuration for Twhin data.
            device: The torch device to use for calculations.
        """
    super().__init__()
    self.model = model
    self.loss_fn = loss_fn
    self.batch_size = data_config.per_replica_batch_size
    self.in_batch_negatives = data_config.in_batch_negatives
    self.device = device

  def forward(self, batch: "RecapBatch"):  # type: ignore[name-defined]
    """
        Run the model forward and calculate the loss according to the given loss_fn.

        NOTE: The input signature here needs to be a Pipelineable object for
        prefetching purposes during training using torchrec's pipeline.  However
        the underlying model signature needs to be exportable to onnx, requiring
        generic python types.  see https://pytorch.org/docs/stable/onnx.html#types

        Args:
            batch ("RecapBatch"): The input batch for model inference.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the loss tensor and a dictionary of
            additional outputs including logits, labels, and weights.
        """
    outputs = self.model(batch)
    logits = outputs["logits"]

    num_negatives = 2 * self.batch_size * self.in_batch_negatives
    num_positives = self.batch_size

    neg_weight = float(num_positives) / num_negatives

    labels = torch.cat([batch.labels.float(), torch.ones(num_negatives).to(self.device)])

    weights = torch.cat(
      [batch.weights.float(), (torch.ones(num_negatives) * neg_weight).to(self.device)]
    )

    losses = self.loss_fn(logits, labels, weights)

    outputs.update(
      {
        "loss": losses,
        "labels": labels,
        "weights": weights,
      }
    )

    # Allow multiple losses.
    return losses, outputs
