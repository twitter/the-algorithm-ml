"""Reader utilities."""
import itertools
import time
from typing import Optional

from tml.common.batch import DataclassBatch
from tml.ml_logging.torch_logging import logging

import pyarrow as pa
import torch


def roundrobin(*iterables):
  """Round robin through provided iterables, useful for simple load balancing.

  Adapted from https://docs.python.org/3/library/itertools.html.

  """
  num_active = len(iterables)
  nexts = itertools.cycle(iter(it).__next__ for it in iterables)
  while num_active:
    try:
      for _next in nexts:
        yield _next()
    except StopIteration:
      # Remove the iterator we just exhausted from the cycle.
      num_active -= 1
      nexts = itertools.cycle(itertools.islice(nexts, num_active))
      logging.warning(f"Iterable exhausted, {num_active} iterables left.")
    except Exception as exc:
      logging.warning(f"Iterable raised exception {exc}, ignoring.")
      # continue
      raise


def speed_check(data_loader, max_steps: int, frequency: int, peek: Optional[int]):
  num_examples = 0
  prev = time.perf_counter()
  for idx, batch in enumerate(data_loader):
    if idx > max_steps:
      break
    if peek and idx % peek == 0:
      logging.info(f"Batch: {batch}")
    num_examples += batch.batch_size
    if idx % frequency == 0:
      now = time.perf_counter()
      elapsed = now - prev
      logging.info(
        f"step: {idx}, "
        f"elapsed(s): {elapsed}, "
        f"examples: {num_examples}, "
        f"ex/s: {num_examples / elapsed}, "
      )
      prev = now
      num_examples = 0


def pa_to_torch(array: pa.array) -> torch.Tensor:
  return torch.from_numpy(array.to_numpy())


def create_default_pa_to_batch(schema) -> DataclassBatch:
  """ """
  _CustomBatch = DataclassBatch.from_schema("DefaultBatch", schema=schema)

  def get_imputation_value(pa_type):
    type_map = {
      pa.float64(): pa.scalar(0, type=pa.float64()),
      pa.int64(): pa.scalar(0, type=pa.int64()),
      pa.string(): pa.scalar("", type=pa.string()),
    }
    if pa_type not in type_map:
      raise Exception(f"Imputation for type {pa_type} not supported.")
    return type_map[pa_type]

  def _impute(array: pa.array) -> pa.array:
    return array.fill_null(get_imputation_value(array.type))

  def _column_to_tensor(record_batch: pa.RecordBatch):
    tensors = {
      col_name: pa_to_torch(_impute(record_batch.column(col_name)))
      for col_name in record_batch.schema.names
    }
    return _CustomBatch(**tensors)

  return _column_to_tensor
