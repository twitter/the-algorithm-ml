"""Base class for all config (forbids extra fields)."""

import collections
import functools
import yaml

import pydantic


class BaseConfig(pydantic.BaseModel):
    """Base class for all derived config classes.

    This class provides convenient functionality and constraints for derived config classes:

    - Disallows extra fields when constructing an object. User errors due to extraneous arguments
      are minimized.
    - "one_of" fields: Subclasses can group optional fields and enforce that only one of the fields
      be set. For example:

      ```python
      class ExampleConfig(BaseConfig):
          x: int = Field(None, one_of="group_1")
          y: int = Field(None, one_of="group_1")

      ExampleConfig(x=1)  # OK
      ExampleConfig(y=1)  # OK
      ExampleConfig(x=1, y=1)  # Raises an error
      ```

    Attributes:
        Config (class): Configuration options for this class, forbidding extra fields.

    Methods:
        _field_data_map(cls, field_data_name): Create a map of fields with the provided field data.
        _one_of_check(cls, values): Validate that all 'one of' fields appear exactly once.
        _at_most_one_of_check(cls, values): Validate that all 'at_most_one_of' fields appear at most once.
        pretty_print(self): Return a human-readable (YAML) representation of the config useful for logging.

    """
    class Config:
        """Configuration options that forbid extra fields."""
        extra = pydantic.Extra.forbid  # noqa

    @classmethod
    @functools.lru_cache()
    def _field_data_map(cls, field_data_name):
        """Create a map of fields with the provided field data."""
        schema = cls.schema()
        one_of = collections.defaultdict(list)
        for field, fdata in schema["properties"].items():
            if field_data_name in fdata:
                one_of[fdata[field_data_name]].append(field)
        return one_of

    @pydantic.root_validator
    def _one_of_check(cls, values):
        """Validate that all 'one of' fields appear exactly once."""
        one_of_map = cls._field_data_map("one_of")
        for one_of, field_names in one_of_map.items():
            if sum([values.get(n, None) is not None for n in field_names]) != 1:
                raise ValueError(f"Exactly one of {','.join(field_names)} required.")
        return values

    @pydantic.root_validator
    def _at_most_one_of_check(cls, values):
        """Validate that all 'at_most_one_of' fields appear at most once."""
        at_most_one_of_map = cls._field_data_map("at_most_one_of")
        for one_of, field_names in at_most_one_of_map.items():
            if sum([values.get(n, None) is not None for n in field_names]) > 1:
                raise ValueError(f"At most one of {','.join(field_names)} can be set.")

        return values

    def pretty_print(self) -> str:
        """Return a human-readable (YAML) representation of the config useful for logging."""
        return yaml.dump(self.dict())
