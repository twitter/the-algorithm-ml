from unittest import TestCase

from tml.core.config import BaseConfig, load_config_from_yaml

import pydantic
import getpass
import pydantic


class _PointlessConfig(BaseConfig):
      
  a: int
  user: str


def test_load_config_from_yaml(tmp_path):
  """Test loading a configuration from a YAML file and verifying its values.

    This test function checks the functionality of the `load_config_from_yaml` function by creating
    a temporary YAML configuration file, loading it, and asserting that the loaded config object
    has the expected values.

    Args:
        tmp_path: A temporary directory provided by the `pytest` framework.

    Test Steps:
        1. Create a temporary YAML file containing configuration data.
        2. Use the `load_config_from_yaml` function to load the configuration from the YAML file.
        3. Assert that the loaded configuration object has the expected values.

    """
  yaml_path = tmp_path.joinpath("test.yaml").as_posix()
  with open(yaml_path, "w") as yaml_file:
    yaml_file.write("""a: 3\nuser: ${USER}\n""")

  pointless_config = load_config_from_yaml(_PointlessConfig, yaml_path)

  assert pointless_config.a == 3
  assert pointless_config.user == getpass.getuser()
