from unittest import TestCase

from tml.core.config import BaseConfig, load_config_from_yaml

import pydantic
import getpass
import pydantic


class _PointlessConfig(BaseConfig):
  a: int
  user: str


def test_load_config_from_yaml(tmp_path):
  yaml_path = tmp_path.joinpath("test.yaml").as_posix()
  with open(yaml_path, "w") as yaml_file:
    yaml_file.write("""a: 3\nuser: ${USER}\n""")

  pointless_config = load_config_from_yaml(_PointlessConfig, yaml_path)

  assert pointless_config.a == 3
  assert pointless_config.user == getpass.getuser()
