from __future__ import annotations

from absl import logging
import torch
from typing import Optional, Callable, Mapping, Dict, Sequence, TYPE_CHECKING
from tml.projects.home.recap.model import feature_transform
from tml.projects.home.recap.model import config as model_config_mod
from tml.projects.home.recap.model import mlp
from tml.projects.home.recap.model import mask_net
from tml.projects.home.recap.model import numeric_calibration
from tml.projects.home.recap.model.model_and_loss import ModelAndLoss
import tml.projects.home.recap.model.config as model_config_mod

if TYPE_CHECKING:
  from tml.projects.home.recap import config as config_mod
  from tml.projects.home.recap.data.config import RecapDataConfig
  from tml.projects.home.recap.model.config import ModelConfig


def sanitize(task_name):
  return task_name.replace(".", "__")


def unsanitize(sanitized_task_name):
  return sanitized_task_name.replace("__", ".")


def _build_single_task_model(task: model_config_mod.TaskModel, input_shape: int):
  """ "Builds a model for a single task"""
  if task.mlp_config:
    return mlp.Mlp(in_features=input_shape, mlp_config=task.mlp_config)
  elif task.dcn_config:
    return dcn.Dcn(dcn_config=task.dcn_config, in_features=input_shape)
  elif task.mask_net_config:
    return mask_net.MaskNet(mask_net_config=task.mask_net_config, in_features=input_shape)
  else:
    raise ValueError("This should never be reached.")


class MultiTaskRankingModel(torch.nn.Module):
  """Multi-task ranking model."""

  def __init__(
    self,
    input_shapes: Mapping[str, torch.Size],
    config: ModelConfig,
    data_config: RecapDataConfig,
    return_backbone: bool = False,
  ):
    """Constructor for Multi task learning.

    Assumptions made:
    1. Tasks specified in data config match model architecture.

    These are all validated in config.
    """
    super().__init__()

    self._config = config
    self._data_config = data_config

    self._preprocessor = feature_transform.build_features_preprocessor(
      config.featurization_config, input_shapes
    )

    self.return_backbone = return_backbone

    self.embeddings = None
    self.small_embeddings = None
    embedding_dims = 0
    if config.large_embeddings:
      from large_embeddings.models.learnable_embeddings import LargeEmbeddings

      self.embeddings = LargeEmbeddings(large_embeddings_config=config.large_embeddings)

      embedding_dims += sum([table.embedding_dim for table in config.large_embeddings.tables])
      logging.info(f"Emb dim: {embedding_dims}")

    if config.small_embeddings:
      self.small_embeddings = SmallEmbedding(config.small_embeddings)
      embedding_dims += sum([table.embedding_dim for table in config.small_embeddings.tables])
      logging.info(f"Emb dim (with small embeddings): {embedding_dims}")

    if "user_embedding" in data_config.seg_dense_schema.renamed_features:
      embedding_dims += input_shapes["user_embedding"][-1]
      self._user_embedding_layer_norm = torch.nn.LayerNorm(input_shapes["user_embedding"][-1])
    else:
      self._user_embedding_layer_norm = None
    if "user_eng_embedding" in data_config.seg_dense_schema.renamed_features:
      embedding_dims += input_shapes["user_eng_embedding"][-1]
      self._user_eng_embedding_layer_norm = torch.nn.LayerNorm(
        input_shapes["user_eng_embedding"][-1]
      )
    else:
      self._user_eng_embedding_layer_norm = None
    if "author_embedding" in data_config.seg_dense_schema.renamed_features:
      embedding_dims += input_shapes["author_embedding"][-1]
      self._author_embedding_layer_norm = torch.nn.LayerNorm(input_shapes["author_embedding"][-1])
    else:
      self._author_embedding_layer_norm = None

    input_dims = input_shapes["continuous"][-1] + input_shapes["binary"][-1] + embedding_dims

    if config.position_debias_config:
      self.position_debias_model = PositionDebias(config.position_debias_config)
      input_dims += self.position_debias_model.out_features
    else:
      self.position_debias_model = None
    logging.info(f"input dim: {input_dims}")

    if config.multi_task_type in [
      model_config_mod.MultiTaskType.SHARE_ALL,
      model_config_mod.MultiTaskType.SHARE_PARTIAL,
    ]:
      self._backbone = _build_single_task_model(config.backbone, input_dims)
    else:
      self._backbone = None

    _towers: Dict[str, torch.nn.Module] = {}
    _calibrators: Dict[str, torch.nn.Module] = {}
    _affine_maps: Dict[str, torch.nn.Module] = {}

    for task_name, task_architecture in config.tasks.items():
      safe_name = sanitize(task_name)

      # Complex input dimension calculation.
      if config.multi_task_type == model_config_mod.MultiTaskType.SHARE_NONE:
        num_inputs = input_dims
      elif config.multi_task_type == model_config_mod.MultiTaskType.SHARE_ALL:
        num_inputs = self._backbone.out_features
      elif config.multi_task_type == model_config_mod.MultiTaskType.SHARE_PARTIAL:
        num_inputs = input_dims + self._backbone.out_features
      else:
        raise ValueError("Unreachable branch of enum.")

      # Annoyingly, ModuleDict doesn't allow . inside key names.
      _towers[safe_name] = _build_single_task_model(task_architecture, num_inputs)

      if task_architecture.affine_map:
        affine_map = torch.nn.Linear(1, 1)
        affine_map.weight.data = torch.tensor([[task_architecture.affine_map.scale]])
        affine_map.bias.data = torch.tensor([task_architecture.affine_map.bias])
        _affine_maps[safe_name] = affine_map
      else:
        _affine_maps[safe_name] = torch.nn.Identity()

      _calibrators[safe_name] = numeric_calibration.NumericCalibration(
        pos_downsampling_rate=data_config.tasks[task_name].pos_downsampling_rate,
        neg_downsampling_rate=data_config.tasks[task_name].neg_downsampling_rate,
      )

    self._task_names = list(config.tasks.keys())
    self._towers = torch.nn.ModuleDict(_towers)
    self._affine_maps = torch.nn.ModuleDict(_affine_maps)
    self._calibrators = torch.nn.ModuleDict(_calibrators)

    self._counter = torch.autograd.Variable(torch.tensor(0), requires_grad=False)

  def forward(
    self,
    continuous_features: torch.Tensor,
    binary_features: torch.Tensor,
    discrete_features: Optional[torch.Tensor] = None,
    sparse_features=None,  # Optional[KeyedJaggedTensor]
    user_embedding: Optional[torch.Tensor] = None,
    user_eng_embedding: Optional[torch.Tensor] = None,
    author_embedding: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
  ):
    concat_dense_features = [
      self._preprocessor(continuous_features=continuous_features, binary_features=binary_features)
    ]

    if self.embeddings:
      concat_dense_features.append(self.embeddings(sparse_features))

    # Twhin embedding layer norms
    if self.small_embeddings:
      if discrete_features is None:
        raise ValueError(
          "Forward arg discrete_features is None, but since small_embeddings are used, a Tensor is expected."
        )
      concat_dense_features.append(self.small_embeddings(discrete_features))

    if self._user_embedding_layer_norm:
      if user_embedding is None:
        raise ValueError(
          "Forward arg user_embedding is None, but since Twhin user_embeddings are used by the model, a Tensor is expected."
        )
      concat_dense_features.append(self._user_embedding_layer_norm(user_embedding))

    if self._user_eng_embedding_layer_norm:
      if user_eng_embedding is None:
        raise ValueError(
          "Forward arg user_eng_embedding is None, but since Twhin user_eng_embeddings are used by the model, a Tensor is expected."
        )
      concat_dense_features.append(self._user_eng_embedding_layer_norm(user_eng_embedding))

    if self._author_embedding_layer_norm:
      if author_embedding is None:
        raise ValueError(
          "Forward arg author_embedding is None, but since Twhin author_embeddings are used by the model, a Tensor is expected."
        )
      concat_dense_features.append(self._author_embedding_layer_norm(author_embedding))

    if self.position_debias_model:
      if discrete_features is None:
        raise ValueError(
          "Forward arg discrete_features is None, but since position_debias_model is used, a Tensor is expected."
        )
      concat_dense_features.append(self.position_debias_model(discrete_features))

    if discrete_features is not None and not (self.position_debias_model or self.small_embeddings):
      logging.warning("Forward arg discrete_features is passed, but never used.")

    concat_dense_features = torch.cat(concat_dense_features, dim=1)

    if self._backbone:
      if self._config.multi_task_type == model_config_mod.MultiTaskType.SHARE_ALL:
        net = self._backbone(concat_dense_features)["output"]
      elif self._config.multi_task_type == model_config_mod.MultiTaskType.SHARE_PARTIAL:
        net = torch.cat(
          [concat_dense_features, self._backbone(concat_dense_features)["output"]], dim=1
        )
    else:
      net = concat_dense_features

    backbone_result = net

    all_logits = []
    all_probabilities = []
    all_calibrated_probabilities = []

    for task_name in self._task_names:
      safe_name = sanitize(task_name)
      tower_outputs = self._towers[safe_name](net)
      logits = tower_outputs["output"]
      scaled_logits = self._affine_maps[safe_name](logits)
      probabilities = torch.sigmoid(scaled_logits)
      calibrated_probabilities = self._calibrators[safe_name](probabilities)

      all_logits.append(scaled_logits)
      all_probabilities.append(probabilities)
      all_calibrated_probabilities.append(calibrated_probabilities)

    results = {
      "logits": torch.squeeze(torch.stack(all_logits, dim=1), dim=-1),
      "probabilities": torch.squeeze(torch.stack(all_probabilities, dim=1), dim=-1),
      "calibrated_probabilities": torch.squeeze(
        torch.stack(all_calibrated_probabilities, dim=1), dim=-1
      ),
    }

    # Returning the backbone is intended for stitching post-tf conversion
    # Leaving this on will ~200x the size of the output
    # and could slow things down
    if self.return_backbone:
      results["backbone"] = backbone_result

    return results


def create_ranking_model(
  data_spec,
  # Used for planner to be batch size aware.
  config: config_mod.RecapConfig,
  device: torch.device,
  loss_fn: Optional[Callable] = None,
  data_config=None,
  return_backbone=False,
):

  if list(config.model.tasks.values())[0].dlrm_config:
    raise NotImplementedError()
    model = EmbeddingRankingModel(
      input_shapes=data_spec,
      config=all_config.model,
      data_config=all_config.train_data,
    )
  else:
    model = MultiTaskRankingModel(
      input_shapes=data_spec,
      config=config.model,
      data_config=data_config if data_config is not None else config.train_data,
      return_backbone=return_backbone,
    )

  logging.info("***** Model Architecture *****")
  logging.info(model)

  logging.info("***** Named Parameters *****")
  for elem in model.named_parameters():
    logging.info(elem[0])

  if loss_fn:
    logging.info("***** Wrapping in loss *****")
    model = ModelAndLoss(
      model=model,
      loss_fn=loss_fn,
      stratifiers=config.model.stratifiers,
    )

  return model
