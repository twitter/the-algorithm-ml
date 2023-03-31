from contextlib import contextmanager
import datetime
import os
from unittest.mock import patch

import torch.distributed as dist
from tml.ml_logging.torch_logging import logging


MOCK_ENV = {
  "LOCAL_RANK": "0",
  "WORLD_SIZE": "1",
  "LOCAL_WORLD_SIZE": "1",
  "MASTER_ADDR": "localhost",
  "MASTER_PORT": "29501",
  "RANK": "0",
}


@contextmanager
def mock_pg():
  with patch.dict(os.environ, MOCK_ENV):
    try:
      dist.init_process_group(
        backend="gloo",
        timeout=datetime.timedelta(1),
      )
      yield
    except:
      dist.destroy_process_group()
      raise
    finally:
      dist.destroy_process_group()
