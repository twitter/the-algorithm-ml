"""Optimization configurations for models."""

import typing

import tml.core.config as base_config
import tml.optimizers.config as optimizers_config_mod

import pydantic


class RecapAdamConfig(base_config.BaseConfig):
  """
    Configuration settings for the Adam optimizer used in Recap.

    Args:
        beta_1 (float): Momentum term (default: 0.9).
        beta_2 (float): Exponential weighted decay factor (default: 0.999).
        epsilon (float): Numerical stability in the denominator (default: 1e-7).

    Example:
        To define an Adam optimizer configuration for Recap, use:

        ```python
        adam_config = RecapAdamConfig(beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        ```

    Note:
        This class configures the parameters of the Adam optimizer, which is commonly used for optimizing neural networks.

    Warning:
        This class is intended for internal use within Recap and should not be directly accessed or modified by external code.
    """
  
  beta_1: float = 0.9  # Momentum term.
  beta_2: float = 0.999  # Exponential weighted decay factor.
  epsilon: float = 1e-7  # Numerical stability in denominator.


class MultiTaskLearningRates(base_config.BaseConfig):
  """
    Configuration settings for multiple learning rates in Recap.

    Args:
        tower_learning_rates (Dict[str, optimizers_config_mod.LearningRate]): Learning rates for different towers of the model.
        backbone_learning_rate (optimizers_config_mod.LearningRate): Learning rate for the model's backbone (default: None).

    Example:
        To define multiple learning rates for different towers in Recap, use:

        ```python
        multi_task_lr = MultiTaskLearningRates(
            tower_learning_rates={
                'task1': learning_rate1,
                'task2': learning_rate2,
            },
            backbone_learning_rate=backbone_lr,
        )
        ```

    Note:
        This class allows specifying different learning rates for different parts of the model, including task-specific towers and the backbone.

    Warning:
        This class is intended for internal use within Recap and should not be directly accessed or modified by external code.
    """
  tower_learning_rates: typing.Dict[str, optimizers_config_mod.LearningRate] = pydantic.Field(
    description="Learning rates for different towers of the model."
  )

  backbone_learning_rate: optimizers_config_mod.LearningRate = pydantic.Field(
    None, description="Learning rate for backbone of the model."
  )


class RecapOptimizerConfig(base_config.BaseConfig):
  """
    Configuration settings for the Recap optimizer.

    Args:
        multi_task_learning_rates (MultiTaskLearningRates): Multiple learning rates for different tasks (optional).
        single_task_learning_rate (optimizers_config_mod.LearningRate): Learning rate for a single task (optional).
        adam (RecapAdamConfig): Configuration settings for the Adam optimizer.

    Example:
        To define an optimizer configuration for training with Recap, use:

        ```python
        optimizer_config = RecapOptimizerConfig(
            multi_task_learning_rates=multi_task_lr,
            single_task_learning_rate=single_task_lr,
            adam=adam_config,
        )
        ```

    Warning:
        This class is intended for internal use to configure the optimizer settings within Recap and should not be
        directly accessed by external code.
    """

  multi_task_learning_rates: MultiTaskLearningRates = pydantic.Field(
    None, description="Multiple learning rates for different tasks.", one_of="lr"
  )

  single_task_learning_rate: optimizers_config_mod.LearningRate = pydantic.Field(
    None, description="Single task learning rates", one_of="lr"
  )

  adam: RecapAdamConfig = pydantic.Field(one_of="optimizer")
