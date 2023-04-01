"""Extension of torchrec.dataset.utils.Batch to cover any dataset.
"""
# flake8: noqa
from __future__ import (
  annotations,
)

import abc
import dataclasses
from collections import (
  UserDict,
)
from dataclasses import (
  dataclass,
)
from typing import (
  Any,
  Dict,
  List,
  TypeVar,
)

import torch
from torchrec.streamable import (
  Pipelineable,
)

_KT = TypeVar("_KT")  #  key type
_VT = TypeVar("_VT")  #  value type


class BatchBase(Pipelineable, abc.ABC):
  @abc.abstractmethod
  def as_dict(self) -> Dict[str, Any]:
    raise NotImplementedError

  def to(self, device: torch.device, non_blocking: bool = False) -> BatchBase:
    args = {}
    for feature_name, feature_value in self.as_dict().items():
      args[feature_name] = feature_value.to(device=device, non_blocking=non_blocking)
    return self.__class__(**args)

  def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
    for feature_value in self.as_dict().values():
      feature_value.record_stream(stream)

  def pin_memory(self) -> BatchBase:
    args = {}
    for feature_name, feature_value in self.as_dict().items():
      args[feature_name] = feature_value.pin_memory()
    return self.__class__(**args)

  def __repr__(self) -> str:
    def obj2str(v: Any) -> str:
      return f"{v.size()}" if hasattr(v, "size") else f"{v.length_per_key()}"

    return "\n".join([f"{k}: {obj2str(v)}," for k, v in self.as_dict().items()])

  @property
  def batch_size(self) -> int:
    for tensor in self.as_dict().values():
      if tensor is None:
        continue
      if not isinstance(tensor, torch.Tensor):
        continue
      return tensor.shape[0]
    raise Exception("Could not determine batch size from tensors.")


@dataclass
class DataclassBatch(BatchBase):
  @classmethod
  def feature_names(cls) -> List[str]:
    return list(cls.__dataclass_fields__.keys())

  def as_dict(self) -> Dict[str, Any]:
    return {
      feature_name: getattr(self, feature_name)
      for feature_name in self.feature_names()
      if hasattr(self, feature_name)
    }

  @staticmethod
  def from_schema(name: str, schema: Any) -> type:
    """Instantiates a custom batch subclass if all columns can be represented as a torch.Tensor."""
    return dataclasses.make_dataclass(
      cls_name=name,
      fields=[(name, torch.Tensor, dataclasses.field(default=None)) for name in schema.names],
      bases=(DataclassBatch,),
    )

  @staticmethod
  def from_fields(name: str, fields: Dict[str, Any]) -> type:
    return dataclasses.make_dataclass(
      cls_name=name,
      fields=[(_name, _type, dataclasses.field(default=None)) for _name, _type in fields.items()],
      bases=(DataclassBatch,),
    )


class DictionaryBatch(BatchBase, UserDict[_KT, _VT]):
  def as_dict(self) -> Dict[str, Any]:
    return self
