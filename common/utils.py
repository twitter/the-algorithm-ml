import yaml
import getpass
import os
import string
from typing import Tuple, Type, TypeVar

from tml.core.config import base_config

import fsspec

C = TypeVar("C", bound=base_config.BaseConfig)


def _read_file(f):
  with fsspec.open(f) as f:
    return f.read()


def setup_configuration(
  config_type: Type[C],
  yaml_path: str,
  substitute_env_variable: bool = False,
) -> Tuple[C, str]:
  """Resolves a config at a yaml path.

  Args:
    config_type: Pydantic config class to load.
    yaml_path: yaml path of the config file.
    substitute_env_variable: If True substitute string in the format $VAR or ${VAR} by their
    environment variable value whenever possible. If an environment variable doesn't exist,
    the string is left unchanged.

  Returns:
    The pydantic config object.
  """

  def _substitute(s):
    if substitute_env_variable:
      return string.Template(s).safe_substitute(os.environ, USER=getpass.getuser())
    return s

  assert config_type is not None, "can't use all_config without config_type"
  content = _substitute(yaml.safe_load(_read_file(yaml_path)))
  return config_type.parse_obj(content)
