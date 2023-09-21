from tml.core import config as config_mod
import tml.projects.home.recap.data.config as data_config
import tml.projects.home.recap.model.config as model_config
import tml.projects.home.recap.optimizer.config as optimizer_config

from enum import Enum
from typing import Dict, Optional
import pydantic


class TrainingConfig(config_mod.BaseConfig):
  """
    Configuration settings for the training process.

    This class defines various training-related settings, including the directory to save checkpoints, the number
    of training steps, logging intervals, and other training parameters.

    Attributes:
        save_dir (str): The directory where checkpoints and training artifacts will be saved.
        num_train_steps (pydantic.PositiveInt): The total number of training steps to run.
        initial_checkpoint_dir (str): The directory containing initial checkpoints (optional).
        checkpoint_every_n (pydantic.PositiveInt): Frequency of saving checkpoints during training.
        checkpoint_max_to_keep (pydantic.PositiveInt): Maximum number of checkpoints to keep (optional).
        train_log_every_n (pydantic.PositiveInt): Frequency of logging training progress.
        num_eval_steps (int): Number of evaluation steps. Use a negative value to evaluate the entire dataset.
        eval_log_every_n (pydantic.PositiveInt): Frequency of logging evaluation progress.
        eval_timeout_in_s (pydantic.PositiveFloat): Maximum time (in seconds) allowed for evaluation.
        gradient_accumulation (int): Number of replica steps to accumulate gradients (optional).

    Example:
        To configure training with checkpoints saved every 1000 steps, use the following settings:

        ```python
        TrainingConfig(
            save_dir="/tmp/model",
            num_train_steps=1000000,
            checkpoint_every_n=1000,
            train_log_every_n=1000,
        )
        ```
    """
  save_dir: str = "/tmp/model"
  num_train_steps: pydantic.PositiveInt = 1000000
  initial_checkpoint_dir: str = pydantic.Field(
    None, description="Directory of initial checkpoints", at_most_one_of="initialization"
  )
  checkpoint_every_n: pydantic.PositiveInt = 1000
  checkpoint_max_to_keep: pydantic.PositiveInt = pydantic.Field(
    None, description="Maximum number of checkpoints to keep. Defaults to keeping all."
  )
  train_log_every_n: pydantic.PositiveInt = 1000
  num_eval_steps: int = pydantic.Field(
    16384, description="Number of evaluation steps. If < 0 the entire dataset " "will be used."
  )
  eval_log_every_n: pydantic.PositiveInt = 5000

  eval_timeout_in_s: pydantic.PositiveFloat = 60 * 60

  gradient_accumulation: int = pydantic.Field(
    None, description="Number of replica steps to accumulate gradients."
  )


class RecapConfig(config_mod.BaseConfig):
  """
    Configuration settings for the Recap model training process.

    This class defines the overall configuration for the training process of a Recap model. It includes settings for
    training, model architecture, data, optimization, and evaluation.

    Attributes:
        training (TrainingConfig): Configuration settings for the training process.
        model (model_config.ModelConfig): Configuration settings for the Recap model architecture.
        train_data (data_config.RecapDataConfig): Configuration settings for training data.
        validation_data (Dict[str, data_config.RecapDataConfig]): Configuration settings for validation data.
        optimizer (optimizer_config.RecapOptimizerConfig): Configuration settings for optimization.
        which_metrics (Optional[str]): Optional specification of which metrics to pick.

    Note:
        This class encapsulates all the necessary configurations to train a Recap model. It defines settings for
        training, the model architecture, data loading, optimization, and evaluation.

    Example:
        To configure a Recap model training process, use the following settings:

        ```python
        RecapConfig(
            training=TrainingConfig(
                save_dir="/tmp/model",
                num_train_steps=1000000,
                checkpoint_every_n=1000,
                train_log_every_n=1000,
            ),
            model=model_config.ModelConfig(...),
            train_data=data_config.RecapDataConfig(...),
            validation_data={"dev": data_config.RecapDataConfig(...)},
            optimizer=optimizer_config.RecapOptimizerConfig(...),
        )
        ```
    """
  training: TrainingConfig = pydantic.Field(TrainingConfig())
  model: model_config.ModelConfig
  train_data: data_config.RecapDataConfig
  validation_data: Dict[str, data_config.RecapDataConfig]
  optimizer: optimizer_config.RecapOptimizerConfig

  which_metrics: Optional[str] = pydantic.Field(None, description="which metrics to pick.")

  # DANGER DANGER! You might expect validators here to ensure that multi task learning setups are
  # the same as the data. Unfortunately, this throws opaque errors when the model configuration is
  # invalid. In our judgement, that is a more frequency and worse occurrence than tasks not matching
  # the data.


class JobMode(str, Enum):
  """Job modes."""

  TRAIN = "train"
  EVALUATE = "evaluate"
  INFERENCE = "inference"
