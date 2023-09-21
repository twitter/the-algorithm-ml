"""Extension of torchrec.dataset.utils.Batch to cover any dataset.
"""
# flake8: noqa
from __future__ import annotations
from typing import Dict
import abc
from dataclasses import dataclass
import dataclasses

import torch
from torchrec.streamable import Pipelineable


class BatchBase(Pipelineable, abc.ABC):
  """
    A base class for batches used in pipelines.

    Attributes:
        None

    """
  @abc.abstractmethod
  def as_dict(self) -> Dict:
    """
        Convert the batch into a dictionary representation.

        Returns:
            Dict: A dictionary representation of the batch.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """
    raise NotImplementedError

  def to(self, device: torch.device, non_blocking: bool = False):
    """
        Move the batch to the specified device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): Whether to use non-blocking transfers. Defaults to False.

        Returns:
            BatchBase: A new batch on the target device.

        """
    args = {}
    for feature_name, feature_value in self.as_dict().items():
      args[feature_name] = feature_value.to(device=device, non_blocking=non_blocking)
    return self.__class__(**args)

  def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
    """
        Record a CUDA stream for all tensors in the batch.

        Args:
            stream (torch.cuda.streams.Stream): The CUDA stream to record.

        Returns:
            None

        """
    for feature_value in self.as_dict().values():
      feature_value.record_stream(stream)

  def pin_memory(self):
    """
        Pin memory for all tensors in the batch.

        Returns:
            BatchBase: A new batch with pinned memory.

        """
    args = {}
    for feature_name, feature_value in self.as_dict().items():
      args[feature_name] = feature_value.pin_memory()
    return self.__class__(**args)

  def __repr__(self) -> str:
    """
        Generate a string representation of the batch.

        Returns:
            str: A string representation of the batch.

        """
    def obj2str(v):
      return f"{v.size()}" if hasattr(v, "size") else f"{v.length_per_key()}"

    return "\n".join([f"{k}: {obj2str(v)}," for k, v in self.as_dict().items()])

  @property
  def batch_size(self) -> int:
    """
        Get the batch size from the tensors in the batch.

        Returns:
            int: The batch size.

        Raises:
            Exception: If the batch size cannot be determined from the tensors.

        """
    for tensor in self.as_dict().values():
      if tensor is None:
        continue
      if not isinstance(tensor, torch.Tensor):
        continue
      return tensor.shape[0]
    raise Exception("Could not determine batch size from tensors.")


@dataclass
class DataclassBatch(BatchBase):
  """
    A batch class that uses dataclasses to define its fields.

    Attributes:
        None

    """
  @classmethod
  def feature_names(cls):
    """
        Get the feature names of the dataclass.

        Returns:
            List[str]: A list of feature names.

        """
    return list(cls.__dataclass_fields__.keys())

  def as_dict(self):
    """
        Convert the dataclass batch into a dictionary representation.

        Returns:
            Dict: A dictionary representation of the batch.

        """
    return {
      feature_name: getattr(self, feature_name)
      for feature_name in self.feature_names()
      if hasattr(self, feature_name)
    }

  @staticmethod
  def from_schema(name: str, schema):
    """
        Instantiate a custom batch subclass if all columns can be represented as a torch.Tensor.

        Args:
            name (str): The name of the custom batch class.
            schema: The schema or structure of the batch.

        Returns:
            Type[DataclassBatch]: A custom batch class.

        """

    return dataclasses.make_dataclass(
      cls_name=name,
      fields=[(name, torch.Tensor, dataclasses.field(default=None)) for name in schema.names],
      bases=(DataclassBatch,),
    )

  @staticmethod
  def from_fields(name: str, fields: dict):
    """
        Create a custom batch subclass from a set of fields.

        Args:
            name (str): The name of the custom batch class.
            fields (dict): A dictionary specifying the fields and their types.

        Returns:
            Type[DataclassBatch]: A custom batch class.

        """
    return dataclasses.make_dataclass(
      cls_name=name,
      fields=[(_name, _type, dataclasses.field(default=None)) for _name, _type in fields.items()],
      bases=(DataclassBatch,),
    )


class DictionaryBatch(BatchBase, dict):
  """
    A batch class that represents data as a dictionary.

    Attributes:
        None

    """
  def as_dict(self) -> Dict:
    """
        Convert the dictionary batch into a dictionary representation.

        Returns:
            Dict: A dictionary representation of the batch.

        """
    return self
