from typing import Mapping, Sequence, Union

from tml.projects.home.recap.model.config import (
  BatchNormConfig,
  DoubleNormLogConfig,
  FeaturizationConfig,
  LayerNormConfig,
)

import torch


def log_transform(x: torch.Tensor) -> torch.Tensor:
  """
    Safe log transform that works across both negative, zero, and positive floats.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transformed tensor with log1p applied to absolute values.
    """
  return torch.sign(x) * torch.log1p(torch.abs(x))


class BatchNorm(torch.nn.Module):
  def __init__(self, num_features: int, config: BatchNormConfig):
    """
    Batch normalization layer.

    Args:
      num_features (int): Number of input features.
      config (BatchNormConfig): Configuration for batch normalization.
    """
    super().__init__()
    self.layer = torch.nn.BatchNorm1d(num_features, affine=config.affine, momentum=config.momentum)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the batch normalization layer.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after batch normalization.
    """
    return self.layer(x)


class LayerNorm(torch.nn.Module):
  def __init__(self, normalized_shape: Union[int, Sequence[int]], config: LayerNormConfig):
    """
    Layer normalization layer.

    Args:
      normalized_shape (Union[int, Sequence[int]]): Size or shape of the input tensor.
      config (LayerNormConfig): Configuration for layer normalization.
    """
    super().__init__()
    if config.axis != -1:
      raise NotImplementedError
    if config.center != config.scale:
      raise ValueError(
        f"Center and scale must match in torch, received {config.center}, {config.scale}"
      )
    self.layer = torch.nn.LayerNorm(
      normalized_shape, eps=config.epsilon, elementwise_affine=config.center
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the layer normalization layer.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Output tensor after layer normalization.
    """

    return self.layer(x)


class Log1pAbs(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass that applies a log transformation to the input tensor.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Transformed tensor with log applied to absolute values.
    """

    return log_transform(x)


class InputNonFinite(torch.nn.Module):
  def __init__(self, fill_value: float = 0):
    """
    Replaces non-finite (NaN and Inf) values in the input tensor with a specified fill value.

    Args:
      fill_value (float): The value to fill non-finite elements with. Default is 0.
    """
    super().__init__()

    self.register_buffer(
      "fill_value", torch.as_tensor(fill_value, dtype=torch.float32), persistent=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass that replaces non-finite values in the input tensor with the specified fill value.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Transformed tensor with non-finite values replaced.
    """
    return torch.where(torch.isfinite(x), x, self.fill_value)


class Clamp(torch.nn.Module):
  def __init__(self, min_value: float, max_value: float):
    """
    Applies element-wise clamping to a tensor, ensuring that values are within a specified range.

    Args:
      min_value (float): The minimum value to clamp elements to.
      max_value (float): The maximum value to clamp elements to.
    """
    super().__init__()
    # Using buffer to make sure they are on correct device (and not moved every time).
    # Will also be part of state_dict.
    self.register_buffer(
      "min_value", torch.as_tensor(min_value, dtype=torch.float32), persistent=True
    )
    self.register_buffer(
      "max_value", torch.as_tensor(max_value, dtype=torch.float32), persistent=True
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass that clamps the input tensor element-wise within the specified range.

    Args:
      x (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Transformed tensor with elements clamped within the specified range.
    """
    return torch.clamp(x, min=self.min_value, max=self.max_value)


class DoubleNormLog(torch.nn.Module):
  """
  Performs a batch norm and clamp on continuous features followed by a layer norm on binary and continuous features.

  Args:
    input_shapes (Mapping[str, Sequence[int]]): A mapping of input feature names to their corresponding shapes.
    config (DoubleNormLogConfig): Configuration for the DoubleNormLog module.

  Attributes:
    _before_concat_layers (torch.nn.Sequential): Sequential layers for batch normalization, log transformation,
                                                      batch normalization (optional), and clamping.
    layer_norm (LayerNorm or None): Layer normalization layer for binary and continuous features (optional).
  """
  def __init__(
    self,
    input_shapes: Mapping[str, Sequence[int]],
    config: DoubleNormLogConfig,
  ):
    super().__init__()

    _before_concat_layers = [
      InputNonFinite(),
      Log1pAbs(),
    ]
    if config.batch_norm_config:
      _before_concat_layers.append(
        BatchNorm(input_shapes["continuous"][-1], config.batch_norm_config)
      )
    _before_concat_layers.append(
      Clamp(min_value=-config.clip_magnitude, max_value=config.clip_magnitude)
    )
    self._before_concat_layers = torch.nn.Sequential(*_before_concat_layers)

    self.layer_norm = None
    if config.layer_norm_config:
      last_dim = input_shapes["continuous"][-1] + input_shapes["binary"][-1]
      self.layer_norm = LayerNorm(last_dim, config.layer_norm_config)

  def forward(
    self, continuous_features: torch.Tensor, binary_features: torch.Tensor
  ) -> torch.Tensor:
    """
    Forward pass that processes continuous and binary features using batch normalization, log transformation,
    optional batch normalization (if configured), clamping, and layer normalization (if configured).

    Args:
      continuous_features (torch.Tensor): Input tensor of continuous features.
      binary_features (torch.Tensor): Input tensor of binary features.

    Returns:
      torch.Tensor: Transformed tensor containing both continuous and binary features.
    """
    x = self._before_concat_layers(continuous_features)
    x = torch.cat([x, binary_features], dim=1)
    if self.layer_norm:
      return self.layer_norm(x)
    return x


def build_features_preprocessor(
  config: FeaturizationConfig, input_shapes: Mapping[str, Sequence[int]]
):
  """
  Build a feature preprocessor module based on the provided configuration.
  Trivial right now, but we will change in the future.

  Args:
    config (FeaturizationConfig): Configuration for feature preprocessing.
    input_shapes (Mapping[str, Sequence[int]]): A mapping of input feature names to their corresponding shapes.

  Returns:
    DoubleNormLog: An instance of the DoubleNormLog feature preprocessor.
  """
  return DoubleNormLog(input_shapes, config.double_norm_log_config)
