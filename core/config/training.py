from typing import Any, Dict, List, Optional

from tml.common.wandb import WandbConfig
from tml.core.config import base_config
from tml.projects.twhin.data.config import TwhinDataConfig
from tml.projects.twhin.models.config import TwhinModelConfig

import pydantic


class RuntimeConfig(base_config.BaseConfig):
  wandb: WandbConfig = pydantic.Field(None)
  enable_tensorfloat32: bool = pydantic.Field(
    False, description="Use tensorfloat32 if on Ampere devices."
  )
  enable_amp: bool = pydantic.Field(False, description="Enable automatic mixed precision.")


class TrainingConfig(base_config.BaseConfig):
  save_dir: str = pydantic.Field("/tmp/model", description="Directory to save checkpoints.")
  num_train_steps: pydantic.PositiveInt = 10000
  initial_checkpoint_dir: str = pydantic.Field(
    None, description="Directory of initial checkpoints", at_most_one_of="initialization"
  )
  checkpoint_every_n: pydantic.PositiveInt = 1000
  checkpoint_max_to_keep: pydantic.PositiveInt = pydantic.Field(
    None, description="Maximum number of checkpoints to keep. Defaults to keeping all."
  )
  train_log_every_n: pydantic.PositiveInt = 1000
  num_eval_steps: int = pydantic.Field(
    16384, description="Number of evaluation steps. If < 0 the entire dataset will be used."
  )
  eval_log_every_n: pydantic.PositiveInt = 5000

  eval_timeout_in_s: pydantic.PositiveFloat = 60 * 60

  gradient_accumulation: int = pydantic.Field(
    None, description="Number of replica steps to accumulate gradients."
  )
  num_epochs: pydantic.PositiveInt = 1
