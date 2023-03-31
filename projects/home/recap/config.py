from tml.core import config as config_mod
import tml.projects.home.recap.data.config as data_config
import tml.projects.home.recap.model.config as model_config
import tml.projects.home.recap.optimizer.config as optimizer_config

from enum import Enum
from typing import Dict, Optional
import pydantic


class TrainingConfig(config_mod.BaseConfig):
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
