from unittest import TestCase

from tml.core.config import BaseConfig

import pydantic


class BaseConfigTest(TestCase):
  def test_extra_forbidden(self):
    class Config(BaseConfig):
      x: int

    Config(x=1)
    with self.assertRaises(pydantic.ValidationError):
      Config(x=1, y=2)

  def test_one_of(self):
    class Config(BaseConfig):
      x: int = pydantic.Field(None, one_of="f")
      y: int = pydantic.Field(None, one_of="f")

    with self.assertRaises(pydantic.ValidationError):
      Config()
    Config(x=1)
    Config(y=1)
    with self.assertRaises(pydantic.ValidationError):
      Config(x=1, y=3)

  def test_at_most_one_of(self):
    class Config(BaseConfig):
      x: int = pydantic.Field(None, at_most_one_of="f")
      y: str = pydantic.Field(None, at_most_one_of="f")

    Config()
    Config(x=1)
    Config(y="a")
    with self.assertRaises(pydantic.ValidationError):
      Config(x=1, y="a")
