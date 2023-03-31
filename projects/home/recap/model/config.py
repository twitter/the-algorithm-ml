"""Configuration for the main Recap model."""

import enum
from typing import List, Optional, Dict

import tml.core.config as base_config
from tml.projects.home.recap.embedding import config as embedding_config

import pydantic


class DropoutConfig(base_config.BaseConfig):
  """Configuration for the dropout layer."""

  rate: pydantic.PositiveFloat = pydantic.Field(
    0.1, description="Fraction of inputs to be dropped."
  )


class LayerNormConfig(base_config.BaseConfig):
  """Configruation for the layer normalization."""

  epsilon: float = pydantic.Field(
    1e-3, description="Small float added to variance to avoid dividing by zero."
  )
  axis: int = pydantic.Field(-1, description="Axis or axes to normalize across.")
  center: bool = pydantic.Field(True, description="Whether to add learnable center.")
  scale: bool = pydantic.Field(True, description="Whether to add learnable scale.")


class BatchNormConfig(base_config.BaseConfig):
  """Configuration of the batch normalization layer."""

  epsilon: pydantic.PositiveFloat = 1e-5
  momentum: pydantic.PositiveFloat = 0.9
  training_mode_at_inference_time: bool = False
  use_renorm: bool = False
  center: bool = pydantic.Field(True, description="Whether to add learnable center.")
  scale: bool = pydantic.Field(True, description="Whether to add learnable scale.")


class DenseLayerConfig(base_config.BaseConfig):
  layer_size: pydantic.PositiveInt
  dropout: DropoutConfig = pydantic.Field(None, description="Optional dropout config for layer.")


class MlpConfig(base_config.BaseConfig):
  """Configuration for MLP model."""

  layer_sizes: List[pydantic.PositiveInt] = pydantic.Field(None, one_of="mlp_layer_definition")
  layers: List[DenseLayerConfig] = pydantic.Field(None, one_of="mlp_layer_definition")


class BatchNormConfig(base_config.BaseConfig):
  """Configuration for the batch norm layer."""

  affine: bool = pydantic.Field(True, description="Use affine transformation.")
  momentum: pydantic.PositiveFloat = pydantic.Field(
    0.1, description="Forgetting parameter in moving average."
  )


class DoubleNormLogConfig(base_config.BaseConfig):
  batch_norm_config: Optional[BatchNormConfig] = pydantic.Field(None)
  clip_magnitude: float = pydantic.Field(
    5.0, description="Threshold to clip the normalized input values."
  )
  layer_norm_config: Optional[LayerNormConfig] = pydantic.Field(None)


class Log1pAbsConfig(base_config.BaseConfig):
  """Simple configuration where only the log transform is performed."""


class ClipLog1pAbsConfig(base_config.BaseConfig):
  clip_magnitude: pydantic.NonNegativeFloat = pydantic.Field(
    3e38, description="Threshold to clip the input values."
  )


class ZScoreLogConfig(base_config.BaseConfig):
  analysis_path: str
  schema_path: str = pydantic.Field(
    None,
    description="Schema path which feaure statistics are generated with. Can be different from scehma in data config.",
  )
  clip_magnitude: float = pydantic.Field(
    5.0, description="Threshold to clip the normalized input values."
  )
  use_batch_norm: bool = pydantic.Field(
    False, description="Option to use batch normalization on the inputs."
  )
  use_renorm: bool = pydantic.Field(
    False, description="Option to use batch renormalization for trainig and serving consistency."
  )
  use_bq_stats: bool = pydantic.Field(
    False, description="Option to load the partitioned json files from BQ as statistics."
  )


class FeaturizationConfig(base_config.BaseConfig):
  """Configuration for featurization."""

  log1p_abs_config: Log1pAbsConfig = pydantic.Field(None, one_of="featurization")
  clip_log1p_abs_config: ClipLog1pAbsConfig = pydantic.Field(None, one_of="featurization")
  z_score_log_config: ZScoreLogConfig = pydantic.Field(None, one_of="featurization")
  double_norm_log_config: DoubleNormLogConfig = pydantic.Field(None, one_of="featurization")
  feature_names_to_concat: List[str] = pydantic.Field(
    ["binary"], description="Feature names to concatenate as raw values with continuous features."
  )


class DropoutConfig(base_config.BaseConfig):
  """Configuration for the dropout layer."""

  rate: pydantic.PositiveFloat = pydantic.Field(
    0.1, description="Fraction of inputs to be dropped."
  )


class MlpConfig(base_config.BaseConfig):
  """Configuration for MLP model."""

  layer_sizes: List[pydantic.PositiveInt]
  batch_norm: BatchNormConfig = pydantic.Field(
    None, description="Optional batch norm configuration."
  )
  dropout: DropoutConfig = pydantic.Field(None, description="Optional dropout configuration.")
  final_layer_activation: bool = pydantic.Field(
    False, description="Whether to include activation on final layer."
  )


class DcnConfig(base_config.BaseConfig):
  """Config for DCN model."""

  poly_degree: pydantic.PositiveInt
  projection_dim: pydantic.PositiveInt = pydantic.Field(
    None, description="Factorizes main DCN matmul with projection."
  )

  parallel_mlp: Optional[MlpConfig] = pydantic.Field(
    None, description="Config for the mlp if used. If None, only the cross layers are used."
  )
  use_parallel: bool = pydantic.Field(True, description="Whether to use parallel DCN.")

  output_mlp: Optional[MlpConfig] = pydantic.Field(None, description="Config for the output mlp.")


class MaskBlockConfig(base_config.BaseConfig):
  output_size: int
  reduction_factor: Optional[pydantic.PositiveFloat] = pydantic.Field(
    None, one_of="aggregation_size"
  )
  aggregation_size: Optional[pydantic.PositiveInt] = pydantic.Field(
    None, description="Specify the aggregation size directly.", one_of="aggregation_size"
  )
  input_layer_norm: bool


class MaskNetConfig(base_config.BaseConfig):
  mask_blocks: List[MaskBlockConfig]
  mlp: Optional[MlpConfig] = pydantic.Field(None, description="MLP Configuration for parallel")
  use_parallel: bool = pydantic.Field(False, description="Whether to use parallel MaskNet.")


class PositionDebiasConfig(base_config.BaseConfig):
  """
  Configuration for Position Debias.
  """

  max_position: int = pydantic.Field(256, description="Bucket all later positions.")
  num_dims: pydantic.PositiveInt = pydantic.Field(
    64, description="Number of dimensions in embedding."
  )
  drop_probability: float = pydantic.Field(0.5, description="Probability of dropping position.")

  # Currently it should be 51 based on dataset being tested at the time of writing this model
  # However, no default provided here to make sure user of the model is aware of its importance.
  position_feature_index: int = pydantic.Field(
    description="The index of the position feature in the discrete features"
  )


class AffineMap(base_config.BaseConfig):
  """An affine map that scales the logits into the appropriate range."""

  scale: float = pydantic.Field(1.0)
  bias: float = pydantic.Field(0.0)


class DLRMConfig(base_config.BaseConfig):
  bottom_mlp: MlpConfig = pydantic.Field(
    ...,
    description="Bottom mlp, the output to be combined with sparse features and feed to interaction",
  )
  top_mlp: MlpConfig = pydantic.Field(..., description="Top mlp, generate the final output")


class TaskModel(base_config.BaseConfig):
  mlp_config: MlpConfig = pydantic.Field(None, one_of="architecture")
  dcn_config: DcnConfig = pydantic.Field(None, one_of="architecture")
  dlrm_config: DLRMConfig = pydantic.Field(None, one_of="architecture")
  mask_net_config: MaskNetConfig = pydantic.Field(None, one_of="architecture")

  affine_map: AffineMap = pydantic.Field(
    None,
    description="Affine map applied to logits so we can represent a broader range of probabilities.",
  )
  # DANGER DANGER: not implemented yet.
  # loss_weight: float = pydantic.Field(1.0, description="Weight for task in loss.")
  pos_weight: float = pydantic.Field(1.0, description="Weight of positive in loss.")


class MultiTaskType(str, enum.Enum):
  SHARE_NONE = "share_none"  # Tasks are separate.
  SHARE_ALL = "share_all"  # Tasks share same backbone.
  SHARE_PARTIAL = "share_partial"  # Tasks share some backbone, but have their own portions.


class ModelConfig(base_config.BaseConfig):
  """Specify model architecture."""

  tasks: Dict[str, TaskModel] = pydantic.Field(
    description="Specification of architecture per task."
  )

  large_embeddings: embedding_config.LargeEmbeddingsConfig = pydantic.Field(None)
  small_embeddings: embedding_config.SmallEmbeddingsConfig = pydantic.Field(None)
  # Not implemented yet.
  # multi_task_loss_reduction_fn: str = "mean"

  position_debias_config: PositionDebiasConfig = pydantic.Field(
    default=None, description="position debias model configuration"
  )

  featurization_config: FeaturizationConfig = pydantic.Field(None)

  multi_task_type: MultiTaskType = pydantic.Field(
    MultiTaskType.SHARE_NONE, description="Multi task architecture"
  )

  backbone: TaskModel = pydantic.Field(None, description="Type of architecture for the backbone.")
  stratifiers: List[embedding_config.StratifierConfig] = pydantic.Field(
    default=None, description="Discrete features and values to stratify metrics by."
  )

  @pydantic.root_validator()
  def _validate_mtl(cls, values):
    if values.get("multi_task_type", None) is None:
      return values
    elif values["multi_task_type"] in [MultiTaskType.SHARE_ALL, MultiTaskType.SHARE_PARTIAL]:
      if values.get("backbone", None) is None:
        raise ValueError("Require `backbone` for SHARE_ALL and SHARE_PARTIAL.")
    elif values["multi_task_type"] in [
      MultiTaskType.SHARE_NONE,
    ]:
      if values.get("backbone", None) is not None:
        raise ValueError("Can not have backbone if the share type is SHARE_NONE")
    return values
