from typing import Any, Dict, List

import tml.core.config as base_config

import pydantic


class WandbConfig(base_config.BaseConfig):
  """
    Configuration for integrating with Weights and Biases (WandB).

    Attributes:
        host (str): Host of the Weights and Biases instance, passed to login.
        key_path (str): Path to the key file.
        name (str): Name of the experiment, passed to init.
        entity (str): Name of the user/service account, passed to init.
        project (str): Name of the WandB project, passed to init.
        tags (List[str]): List of tags, passed to init.
        notes (str): Notes, passed to init.
        metadata (Dict[str, Any]): Additional metadata to log.

    Example:
        ```python
        wandb_config = WandbConfig(
            host="https://wandb.example.com",
            key_path="/path/to/key",
            name="experiment_1",
            entity="user123",
            project="my_project",
            tags=["experiment", "ml"],
            notes="This is a test experiment.",
            metadata={"version": "1.0"}
        )
        ```
    """
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
