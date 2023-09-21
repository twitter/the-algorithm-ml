import yaml
import string
import getpass
import os
from typing import Type

from tml.core.config.base_config import BaseConfig


def load_config_from_yaml(config_type: Type[BaseConfig], yaml_path: str):
  """
    Recommend method to Load and parse a configuration from a YAML file.

    This function loads a configuration from a YAML file, parses it, and returns an instance of the
    specified config type.

    Because we have a shared filesystem the recommended route to running jobs it put modified config
    files with the desired parameters somewhere on the filesytem and run jobs pointing to them.

    Args:
        config_type (Type[BaseConfig]): The Pydantic config class to load.
        yaml_path (str): The path to the YAML configuration file.

    Returns:
        BaseConfig: An instance of the specified config type populated with values from the YAML file.

    Example:
        Suppose you have a YAML file 'my_config.yaml' containing the following:

        ```yaml
        x: 42
        y: "hello"
        ```

        You can load and parse it using this function as follows:

        ```python
        my_config = load_config_from_yaml(MyConfigClass, 'my_config.yaml')
        ```

    Note:
        This function performs environment variable substitution in the YAML file. It replaces
        occurrences of the format '$VAR' or '${VAR}' with their corresponding environment variable
        values. If an environment variable does not exist, the string is left unchanged.

    """

  def _substitute(s):
    return string.Template(s).safe_substitute(os.environ, USER=getpass.getuser())

  with open(yaml_path, "r") as f:
    raw_contents = f.read()
    obj = yaml.safe_load(_substitute(raw_contents))

  return config_type.parse_obj(obj)
