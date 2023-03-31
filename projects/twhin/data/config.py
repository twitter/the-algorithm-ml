from tml.core.config import base_config

import pydantic


class TwhinDataConfig(base_config.BaseConfig):
  data_root: str
  per_replica_batch_size: pydantic.PositiveInt
  global_negatives: int
  in_batch_negatives: int
  limit: pydantic.PositiveInt
  offset: pydantic.PositiveInt = pydantic.Field(
    None, description="The offset to start reading from."
  )
