from typing import Mapping, Sequence, Union

from tml.projects.home.recap.model.config import (
  BatchNormConfig,
  DoubleNormLogConfig,
  FeaturizationConfig,
  LayerNormConfig,
)

import torch


def log_transform(x: torch.Tensor) -> torch.Tensor:
  """Safe log transform that works across both negative, zero, and positive floats."""
  return torch.sign(x) * torch.log1p(torch.abs(x))


class BatchNorm(torch.nn.Module):
  def __init__(self, num_features: int, config: BatchNormConfig):
    super().__init__()
    self.layer = torch.nn.BatchNorm1d(num_features, affine=config.affine, momentum=config.momentum)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer(x)


class LayerNorm(torch.nn.Module):
  def __init__(self, normalized_shape: Union[int, Sequence[int]], config: LayerNormConfig):
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
    return self.layer(x)


class Log1pAbs(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return log_transform(x)


class InputNonFinite(torch.nn.Module):
  def __init__(self, fill_value: float = 0):
    super().__init__()

    self.register_buffer(
      "fill_value", torch.as_tensor(fill_value, dtype=torch.float32), persistent=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, self.fill_value)


class Clamp(torch.nn.Module):
  def __init__(self, min_value: float, max_value: float):
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
    return torch.clamp(x, min=self.min_value, max=self.max_value)


class DoubleNormLog(torch.nn.Module):
  """Performs a batch norm and clamp on continuous features followed by a layer norm on binary and continuous features."""

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
    x = self._before_concat_layers(continuous_features)
    x = torch.cat([x, binary_features], dim=1)
    if self.layer_norm:
      return self.layer_norm(x)
    return x


def build_features_preprocessor(
  config: FeaturizationConfig, input_shapes: Mapping[str, Sequence[int]]
):
  """Trivial right now, but we will change in the future."""
  return DoubleNormLog(input_shapes, config.double_norm_log_config)
