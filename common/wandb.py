from typing import Any, Dict, List

import tml.core.config as base_config

import pydantic


class WandbConfig(base_config.BaseConfig):
  host: str = pydantic.Field(
    "https://https--wandb--prod--wandb.service.qus1.twitter.biz/",
    description="Host of Weights and Biases instance, passed to login.",
  )
  key_path: str = pydantic.Field(description="Path to key file.")

  name: str = pydantic.Field(None, description="Name of the experiment, passed to init.")
  entity: str = pydantic.Field(None, description="Name of user/service account, passed to init.")
  project: str = pydantic.Field(None, description="Name of wandb project, passed to init.")
  tags: List[str] = pydantic.Field([], description="List of tags, passed to init.")
  notes: str = pydantic.Field(None, description="Notes, passed to init.")
  metadata: Dict[str, Any] = pydantic.Field(None, description="Additional metadata to log.")
