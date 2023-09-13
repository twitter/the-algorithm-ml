from tml.core.config import base_config

import pydantic


class TwhinDataConfig(base_config.BaseConfig):
  """
    Configuration for Twhin model training data.

    Args:
        data_root (str): The root directory for the training data.
        per_replica_batch_size (pydantic.PositiveInt): Batch size per replica.
        global_negatives (int): The number of global negatives.
        in_batch_negatives (int): The number of in-batch negatives.
        limit (pydantic.PositiveInt): The limit on the number of data points to use.
        offset (pydantic.PositiveInt, optional): The offset to start reading from. Default is None.
    """
  data_root: str
  per_replica_batch_size: pydantic.PositiveInt
  global_negatives: int
  in_batch_negatives: int
  limit: pydantic.PositiveInt
  offset: pydantic.PositiveInt = pydantic.Field(
    None, description="The offset to start reading from."
  )
