"""Reader utilities."""
import itertools
import time
from typing import Optional

from tml.common.batch import DataclassBatch
from tml.ml_logging.torch_logging import logging

import pyarrow as pa
import torch


def roundrobin(*iterables):
  """
    Iterate through provided iterables in a round-robin fashion.

    This function takes multiple iterables and returns an iterator that yields elements from
    each iterable in a round-robin manner. It continues cycling through the iterables until
    all of them are exhausted.

    Adapted from https://docs.python.org/3/library/itertools.html.

    Args:
        *iterables: One or more iterable objects to iterate through.

    Yields:
        Elements from the provided iterables in a round-robin fashion.

    Raises:
        StopIteration: If all provided iterables are exhausted.

    Example:
        ```python
        iterable1 = [1, 2, 3]
        iterable2 = ['a', 'b', 'c']
        iterable3 = [0.1, 0.2, 0.3]

        for item in roundrobin(iterable1, iterable2, iterable3):
            print(item)

        # Output:
        # 1
        # 'a'
        # 0.1
        # 2
        # 'b'
        # 0.2
        # 3
        # 'c'
        # 0.3
        ```

    Note:
        - If one of the provided iterables is shorter than the others, the function will
          continue iterating through the remaining iterables until all are exhausted.
        - If an iterable raises an exception during iteration, a warning message is logged,
          and the function continues with the next iterable.

    See Also:
        - `itertools.cycle`: A function that repeatedly cycles through elements of an iterable.
        - `itertools.islice`: A function to slice an iterable to limit the number of iterations.
    """
  num_active = len(iterables)
  nexts = itertools.cycle(iter(it).__next__ for it in iterables)
  while num_active:
    try:
      for _next in nexts:
        result = _next()
        yield result
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
  """
    Monitor the speed and progress of data loading using a data loader.

    This function iterates through a data loader for a specified number of steps or until
    the end of the data loader is reached, periodically logging progress information.

    Args:
        data_loader: The data loader to monitor.
        max_steps: The maximum number of steps to iterate through the data loader.
        frequency: The frequency (in steps) at which to log progress.
        peek (optional): If specified, it indicates the frequency (in steps) at which to log
            batch contents for inspection.

    Example:
        ```python
        import torch
        from torch.utils.data import DataLoader

        # Create a data loader (replace with your own DataLoader configuration)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Monitor data loading speed and progress
        speed_check(data_loader, max_steps=1000, frequency=50, peek=500)
        ```

    Args:
        data_loader: The data loader to monitor.
        max_steps: The maximum number of steps to iterate through the data loader.
        frequency: The frequency (in steps) at which to log progress.
        peek (optional): If specified, it indicates the frequency (in steps) at which to log
            batch contents for inspection.

    Note:
        - The function logs information about elapsed time, the number of examples processed,
          and the processing speed in examples per second.
        - If `peek` is provided, batch contents will be logged for inspection at the specified
          frequency.

    See Also:
        - `torch.utils.data.DataLoader`: PyTorch's data loading utility for batching and
          iterating through datasets.
    """
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
  """
    Convert a PyArrow Array to a PyTorch Tensor.

    Args:
        array (pa.array): The PyArrow Array to convert.

    Returns:
        torch.Tensor: A PyTorch Tensor containing the data from the input PyArrow Array.

    Example:
        ```python
        import pyarrow as pa
        import torch

        # Create a PyArrow Array
        arrow_array = pa.array([1, 2, 3])

        # Convert it to a PyTorch Tensor
        torch_tensor = pa_to_torch(arrow_array)
        ```
    """
  return torch.from_numpy(array.to_numpy())


def create_default_pa_to_batch(schema) -> DataclassBatch:
  """
    Create a function that converts a PyArrow RecordBatch to a custom DataclassBatch with imputed values for missing data.

    Args:
        schema (pa.Schema): The PyArrow schema describing the data structure of the RecordBatch.

    Returns:
        callable: A function that takes a PyArrow RecordBatch as input and returns a custom DataclassBatch.

    Example:
        ```python
        import pyarrow as pa
        from dataclass_batch import DataclassBatch

        # Define a PyArrow schema
        schema = pa.schema([
            ("feature1", pa.float64()),
            ("feature2", pa.int64()),
            ("label", pa.int64()),
        ])

        # Create the conversion function
        pa_to_batch = create_default_pa_to_batch(schema)

        # Create a PyArrow RecordBatch
        record_batch = pa.RecordBatch.from_pandas(pd.DataFrame({
            "feature1": [1.0, 2.0, None],
            "feature2": [10, 20, 30],
            "label": [0, 1, None],
        }))

        # Convert the RecordBatch to a custom DataclassBatch
        custom_batch = pa_to_batch(record_batch)
        ```
    """
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
