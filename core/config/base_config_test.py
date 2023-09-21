from unittest import TestCase

from tml.core.config import BaseConfig

import pydantic


class BaseConfigTest(TestCase):
    """
    Unit tests for the BaseConfig class.
    """

    def test_extra_forbidden(self):
        """
        Test that extra fields are forbidden when creating a Config instance.

        This test case checks whether the `BaseConfig` class correctly raises a
        `pydantic.ValidationError` when extra fields are provided when creating a
        `Config` instance.

        Raises:
            AssertionError: If the test fails.
        """
        class Config(BaseConfig):
            x: int

        Config(x=1)
        with self.assertRaises(pydantic.ValidationError):
            Config(x=1, y=2)

    def test_one_of(self):
        """
        Test the use of the `one_of` attribute for fields in a Config instance.

        This test case checks the behavior of the `one_of` attribute in a `Config`
        instance. It verifies that the `pydantic.Field` correctly enforces the
        specified constraint.

        Raises:
            AssertionError: If the test fails.
        """
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
        """
        Test the use of the `at_most_one_of` attribute for fields in a Config instance.

        This test case checks the behavior of the `at_most_one_of` attribute in a
        `Config` instance. It verifies that the `pydantic.Field` enforces the
        constraint where at most one of the specified fields can be provided.

        Raises:
            AssertionError: If the test fails.
        """
        class Config(BaseConfig):
            x: int = pydantic.Field(None, at_most_one_of="f")
            y: str = pydantic.Field(None, at_most_one_of="f")

        Config()
        Config(x=1)
        Config(y="a")
        with self.assertRaises(pydantic.ValidationError):
            Config(x=1, y="a")
