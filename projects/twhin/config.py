from tml.core.config import base_config
from tml.projects.twhin.data.config import TwhinDataConfig
from tml.projects.twhin.models.config import TwhinModelConfig
from tml.core.config.training import RuntimeConfig, TrainingConfig

import pydantic


class TwhinConfig(base_config.BaseConfig):
  runtime: RuntimeConfig = pydantic.Field(RuntimeConfig())
  training: TrainingConfig = pydantic.Field(TrainingConfig())
  model: TwhinModelConfig
  train_data: TwhinDataConfig
  validation_data: TwhinDataConfig
