"""Optimization configurations for models."""

import typing

import tml.core.config as base_config
import tml.optimizers.config as optimizers_config_mod

import pydantic


class RecapAdamConfig(base_config.BaseConfig):
  beta_1: float = 0.9  # Momentum term.
  beta_2: float = 0.999  # Exponential weighted decay factor.
  epsilon: float = 1e-7  # Numerical stability in denominator.


class MultiTaskLearningRates(base_config.BaseConfig):
  tower_learning_rates: typing.Dict[str, optimizers_config_mod.LearningRate] = pydantic.Field(
    description="Learning rates for different towers of the model."
  )

  backbone_learning_rate: optimizers_config_mod.LearningRate = pydantic.Field(
    None, description="Learning rate for backbone of the model."
  )


class RecapOptimizerConfig(base_config.BaseConfig):
  multi_task_learning_rates: MultiTaskLearningRates = pydantic.Field(
    None, description="Multiple learning rates for different tasks.", one_of="lr"
  )

  single_task_learning_rate: optimizers_config_mod.LearningRate = pydantic.Field(
    None, description="Single task learning rates", one_of="lr"
  )

  adam: RecapAdamConfig = pydantic.Field(one_of="optimizer")
