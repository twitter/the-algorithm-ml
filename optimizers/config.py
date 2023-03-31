"""Optimization configurations for models."""

import typing

import tml.core.config as base_config

import pydantic


class PiecewiseConstant(base_config.BaseConfig):
  learning_rate_boundaries: typing.List[int] = pydantic.Field(None)
  learning_rate_values: typing.List[float] = pydantic.Field(None)


class LinearRampToConstant(base_config.BaseConfig):
  learning_rate: float
  num_ramp_steps: pydantic.PositiveInt = pydantic.Field(
    description="Number of steps to ramp this up from zero."
  )


class LinearRampToCosine(base_config.BaseConfig):
  learning_rate: float
  final_learning_rate: float
  num_ramp_steps: pydantic.PositiveInt = pydantic.Field(
    description="Number of steps to ramp this up from zero."
  )
  final_num_steps: pydantic.PositiveInt = pydantic.Field(
    description="Final number of steps where decay stops."
  )


class LearningRate(base_config.BaseConfig):
  constant: float = pydantic.Field(None, one_of="lr")
  linear_ramp_to_cosine: LinearRampToCosine = pydantic.Field(None, one_of="lr")
  linear_ramp_to_constant: LinearRampToConstant = pydantic.Field(None, one_of="lr")
  piecewise_constant: PiecewiseConstant = pydantic.Field(None, one_of="lr")


class OptimizerAlgorithmConfig(base_config.BaseConfig):
  """Base class for optimizer configurations."""

  lr: float
  ...


class AdamConfig(OptimizerAlgorithmConfig):
  # see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
  lr: float
  betas: typing.Tuple[float, float] = [0.9, 0.999]
  eps: float = 1e-7  # Numerical stability in denominator.


class SgdConfig(OptimizerAlgorithmConfig):
  lr: float
  momentum: float = 0.0


class AdagradConfig(OptimizerAlgorithmConfig):
  lr: float
  eps: float = 0


class OptimizerConfig(base_config.BaseConfig):
  learning_rate: LearningRate = pydantic.Field(
    None,
    description="Constant learning rates",
  )
  adam: AdamConfig = pydantic.Field(None, one_of="optimizer")
  sgd: SgdConfig = pydantic.Field(None, one_of="optimizer")
  adagrad: AdagradConfig = pydantic.Field(None, one_of="optimizer")


def get_optimizer_algorithm_config(optimizer_config: OptimizerConfig):
  if optimizer_config.adam is not None:
    return optimizer_config.adam
  elif optimizer_config.sgd is not None:
    return optimizer_config.sgd
  elif optimizer_config.adagrad is not None:
    return optimizer_config.adagrad
  else:
    raise ValueError(f"No optimizer selected in optimizer_config, passed {optimizer_config}")
