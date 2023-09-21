"""Optimization configurations for models."""

import typing

import tml.core.config as base_config

import pydantic


class PiecewiseConstant(base_config.BaseConfig):
  """
    Configuration for a piecewise constant learning rate schedule.

    This configuration class allows you to specify a piecewise constant learning rate schedule
    by defining boundaries and corresponding learning rate values.

    Attributes:
        learning_rate_boundaries (List[int], optional): List of step boundaries at which
            the learning rate will change. If None, no boundaries are defined.
        learning_rate_values (List[float], optional): List of learning rate values
            corresponding to the boundaries. If None, no values are defined.

    Example:
        To configure a piecewise constant learning rate schedule, create an instance of this class
        and set the attributes accordingly. For example:

        ```python
        piecewise_lr = PiecewiseConstant(
            learning_rate_boundaries=[1000, 2000, 3000],
            learning_rate_values=[0.1, 0.05, 0.01, 0.001]
        )
        ```

    Note:
        The number of learning rate values should be one more than the number of boundaries.

    """
  learning_rate_boundaries: typing.List[int] = pydantic.Field(None)
  learning_rate_values: typing.List[float] = pydantic.Field(None)


class LinearRampToConstant(base_config.BaseConfig):
  """
    Configuration for a linear ramp-up to constant learning rate schedule.

    This configuration class allows you to specify a learning rate schedule that ramps up linearly
    from zero to a constant value over a specified number of steps.

    Attributes:
        learning_rate (float): The final constant learning rate.
        num_ramp_steps (PositiveInt): Number of steps to ramp up the learning rate from zero.

    Example:
        To configure a linear ramp-up to a constant learning rate, create an instance of this class
        and set the attributes accordingly. For example:

        ```python
        linear_ramp_lr = LinearRampToConstant(
            learning_rate=0.1,
            num_ramp_steps=1000
        )
        ```

    """
  learning_rate: float
  num_ramp_steps: pydantic.PositiveInt = pydantic.Field(
    description="Number of steps to ramp this up from zero."
  )


class LinearRampToCosine(base_config.BaseConfig):
  """
    Configuration for a linear ramp-up to cosine decay learning rate schedule.

    This configuration class allows you to specify a learning rate schedule that ramps up linearly
    from zero, then decays following a cosine schedule to a final constant learning rate.

    Attributes:
        learning_rate (float): The initial learning rate at the start of ramp-up.
        final_learning_rate (float): The final constant learning rate after decay.
        num_ramp_steps (PositiveInt): Number of steps to ramp up the learning rate from zero.
        final_num_steps (PositiveInt): Final number of steps where decay stops.

    Example:
        To configure a linear ramp-up to cosine decay learning rate, create an instance of this
        class and set the attributes accordingly. For example:

        ```python
        ramp_to_cosine_lr = LinearRampToCosine(
            learning_rate=0.01,
            final_learning_rate=0.001,
            num_ramp_steps=1000,
            final_num_steps=5000
        )
        ```

    """
  learning_rate: float
  final_learning_rate: float
  num_ramp_steps: pydantic.PositiveInt = pydantic.Field(
    description="Number of steps to ramp this up from zero."
  )
  final_num_steps: pydantic.PositiveInt = pydantic.Field(
    description="Final number of steps where decay stops."
  )


class LearningRate(base_config.BaseConfig):
  """
    Learning rate configuration for training.

    This configuration class allows you to specify different learning rate schedules
    for your training process.

    Attributes:
        constant (float, optional): Constant learning rate to be used throughout training.
        linear_ramp_to_cosine (LinearRampToCosine, optional): Learning rate that ramps up linearly
            and then decays following a cosine schedule.
        linear_ramp_to_constant (LinearRampToConstant, optional): Learning rate that ramps up
            linearly and then remains constant.
        piecewise_constant (PiecewiseConstant, optional): Learning rate that changes at specified
            boundaries with corresponding values.

    Example:
        To configure a learning rate schedule, create an instance of this class and set the
        attributes accordingly. For example:

        ```python
        learning_rate = LearningRate(
            constant=0.01,
            linear_ramp_to_cosine=LinearRampToCosine(
                learning_rate=0.1,
                final_learning_rate=0.001,
                num_ramp_steps=1000,
                final_num_steps=5000
            )
        )
        ```

    Note:
        Each learning rate schedule attribute can be set to `None` if not needed.

    """
  constant: float = pydantic.Field(None, one_of="lr")
  linear_ramp_to_cosine: LinearRampToCosine = pydantic.Field(None, one_of="lr")
  linear_ramp_to_constant: LinearRampToConstant = pydantic.Field(None, one_of="lr")
  piecewise_constant: PiecewiseConstant = pydantic.Field(None, one_of="lr")


class OptimizerAlgorithmConfig(base_config.BaseConfig):
  """
    Base class for optimizer configurations.

    This base configuration class provides a structure for specifying various optimizer-related
    settings, including the learning rate and different learning rate schedules.

    Attributes:
        lr (float): The base learning rate used by the optimizer.

    Subclasses should inherit from this base class and define additional attributes specific to
    the optimizer algorithm they represent.

    Example:
        To create a custom optimizer configuration, create a subclass of this base class and
        define the necessary attributes. For example:

        ```python
        class MyOptimizerConfig(OptimizerAlgorithmConfig):
            momentum: float = pydantic.Field(0.9, description="Momentum value for SGD.")
        ```

    Note:
        This base class does not include specific optimizer settings. Subclasses should define
        the optimizer-specific attributes as needed.

    """

  lr: float
  ...


class AdamConfig(OptimizerAlgorithmConfig):
  """
    Configuration for the Adam optimizer.

    This configuration class allows you to specify the hyperparameters for the Adam optimizer.

    Attributes:
        lr (float): The learning rate for optimization.
        betas (Tuple[float, float], optional): Coefficients used for computing running averages
            of gradient and squared gradient. Defaults to (0.9, 0.999).
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Defaults to 1e-7.

    Example:
        To configure the Adam optimizer, create an instance of this class and set the attributes
        accordingly. For example:

        ```python
        adam_optimizer = AdamConfig(
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        ```

    See Also:
        [PyTorch Adam Documentation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)

    """
  lr: float
  betas: typing.Tuple[float, float] = [0.9, 0.999]
  eps: float = 1e-7  # Numerical stability in denominator.


class SgdConfig(OptimizerAlgorithmConfig):
  """
    Configuration for the Stochastic Gradient Descent (SGD) optimizer.

    This configuration class allows you to specify the hyperparameters for the SGD optimizer.

    Attributes:
        lr (float): The learning rate for optimization.
        momentum (float, optional): The momentum factor for SGD. Defaults to 0.0.

    Example:
        To configure the SGD optimizer, create an instance of this class and set the attributes
        accordingly. For example:

        ```python
        sgd_optimizer = SgdConfig(
            lr=0.01,
            momentum=0.9
        )
        ```

    """
  lr: float
  momentum: float = 0.0


class AdagradConfig(OptimizerAlgorithmConfig):
  """
    Configuration for the optimizer used during training.

    This configuration class allows you to specify the optimizer for training, including
    options for various optimizer algorithms.

    Attributes:
        learning_rate (LearningRate, optional): Learning rate configuration. Defaults to None.
        adam (AdamConfig, optional): Configuration for the Adam optimizer. Defaults to None.
        sgd (SgdConfig, optional): Configuration for the Stochastic Gradient Descent (SGD) optimizer.
            Defaults to None.
        adagrad (AdagradConfig, optional): Configuration for the Adagrad optimizer. Defaults to None.

    Example:
        To configure the optimizer for training, create an instance of this class and set the
        attributes accordingly. For example:

        ```python
        optimizer_config = OptimizerConfig(
            learning_rate=LearningRate(constant=0.001),
            adam=AdamConfig(lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        )
        ```

    """
  lr: float
  eps: float = 0


class OptimizerConfig(base_config.BaseConfig):
  """
    Configuration for defining different optimizer algorithms and their parameters.

    This class allows you to configure various optimizer algorithms such as Adam, SGD, and Adagrad,
    along with their respective hyperparameters.

    Args:
        learning_rate (LearningRate): The learning rate configuration, which can include
            constant learning rates or other learning rate schedules.
        adam (AdamConfig): Configuration for the Adam optimizer.
        sgd (SgdConfig): Configuration for the Stochastic Gradient Descent (SGD) optimizer.
        adagrad (AdagradConfig): Configuration for the Adagrad optimizer.

    Example:
        ```python
        optimizer_config = OptimizerConfig(
            learning_rate=LearningRate(constant=0.001),
            adam=AdamConfig(lr=0.001, betas=(0.9, 0.999), eps=1e-8),
        )
        ```

    Attributes:
        learning_rate (LearningRate): The learning rate configuration.
        adam (AdamConfig): Configuration for the Adam optimizer.
        sgd (SgdConfig): Configuration for the Stochastic Gradient Descent (SGD) optimizer.
        adagrad (AdagradConfig): Configuration for the Adagrad optimizer.

    Note:
        You can specify only one of the optimizer configurations (adam, sgd, or adagrad) in an
        `OptimizerConfig` instance.

    See Also:
        - `LearningRate`: Configuration for specifying learning rates.
        - `AdamConfig`: Configuration for the Adam optimizer.
        - `SgdConfig`: Configuration for the Stochastic Gradient Descent (SGD) optimizer.
        - `AdagradConfig`: Configuration for the Adagrad optimizer.

    """
  learning_rate: LearningRate = pydantic.Field(
    None,
    description="Constant learning rates",
  )
  adam: AdamConfig = pydantic.Field(None, one_of="optimizer")
  sgd: SgdConfig = pydantic.Field(None, one_of="optimizer")
  adagrad: AdagradConfig = pydantic.Field(None, one_of="optimizer")


def get_optimizer_algorithm_config(optimizer_config: OptimizerConfig):
  """
    Get the optimizer algorithm configuration from the given `OptimizerConfig`.

    This function extracts and returns the specific optimizer algorithm configuration
    (e.g., Adam, SGD, or Adagrad) from the provided `OptimizerConfig`.

    Args:
        optimizer_config (OptimizerConfig): The optimizer configuration object containing
            one of the optimizer algorithm configurations.

    Returns:
        Union[AdamConfig, SgdConfig, AdagradConfig]: The specific optimizer algorithm
        configuration extracted from `optimizer_config`.

    Raises:
        ValueError: If no optimizer algorithm is selected in `optimizer_config`.

    Example:
        ```python
        optimizer_config = OptimizerConfig(
            adam=AdamConfig(lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        )
        algorithm_config = get_optimizer_algorithm_config(optimizer_config)
        # `algorithm_config` will be an instance of `AdamConfig`.
        ```

    """
  if optimizer_config.adam is not None:
    return optimizer_config.adam
  elif optimizer_config.sgd is not None:
    return optimizer_config.sgd
  elif optimizer_config.adagrad is not None:
    return optimizer_config.adagrad
  else:
    raise ValueError(f"No optimizer selected in optimizer_config, passed {optimizer_config}")
