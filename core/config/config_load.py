import yaml
import string
import getpass
import os
from typing import Type

from tml.core.config.base_config import BaseConfig


def load_config_from_yaml(config_type: Type[BaseConfig], yaml_path: str):
  """Recommend method to load a config file (a yaml file) and parse it.

  Because we have a shared filesystem the recommended route to running jobs it put modified config
  files with the desired parameters somewhere on the filesytem and run jobs pointing to them.
  """

  def _substitute(s):
    return string.Template(s).safe_substitute(os.environ, USER=getpass.getuser())

  with open(yaml_path, "r") as f:
    raw_contents = f.read()
    obj = yaml.safe_load(_substitute(raw_contents))

  return config_type.parse_obj(obj)
