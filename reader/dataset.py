"""Dataset to be overwritten that can work with or without distributed reading.

- Override `pa_to_batch` for dataset specific imputation, negative sampling, or coercion to Batch.
- Readers can be colocated or off trainer machines.

"""
import abc
import functools
import random
from typing import Optional

from fsspec.implementations.local import LocalFileSystem
import pyarrow.dataset as pads
import pyarrow as pa
import pyarrow.parquet
import pyarrow.flight
from pyarrow.ipc import IpcWriteOptions
import torch

from tml.common.batch import DataclassBatch
from tml.machines import environment as env
import tml.reader.utils as reader_utils
from tml.common.filesystem import infer_fs
from tml.ml_logging.torch_logging import logging


class _Reader(pa.flight.FlightServerBase):
  """Distributed reader flight server wrapping a dataset."""

  def __init__(self, location: str, ds: "Dataset"):
    super().__init__(location=location)
    self._location = location
    self._ds = ds

  def do_get(self, _, __):
    # NB: An updated schema (to account for column selection) has to be given the stream.
    schema = next(iter(self._ds.to_batches())).schema
    batches = self._ds.to_batches()
    return pa.flight.RecordBatchStream(
      data_source=pa.RecordBatchReader.from_batches(
        schema=schema,
        batches=batches,
      ),
      options=IpcWriteOptions(use_threads=True),
    )


class Dataset(torch.utils.data.IterableDataset):
  LOCATION = "grpc://0.0.0.0:2222"

  def __init__(self, file_pattern: str, **dataset_kwargs) -> None:
    """Specify batch size and column to select for.

    Refer to https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_dataset.
    """
    self._file_pattern = file_pattern
    self._fs = infer_fs(self._file_pattern)
    self._dataset_kwargs = dataset_kwargs
    logging.info(f"Using dataset_kwargs: {self._dataset_kwargs}")
    self._files = self._fs.glob(self._file_pattern)
    assert len(self._files) > 0, f"No files found at {self._file_pattern}"
    logging.info(f"Found {len(self._files)} files: {', '.join(self._files[:4])}, ...")
    self._schema = pa.parquet.read_schema(self._files[0], filesystem=self._fs)
    self._validate_columns()

  def _validate_columns(self):
    columns = set(self._dataset_kwargs.get("columns", []))
    wrong_columns = set(columns) - set(self._schema.names)
    if wrong_columns:
      raise Exception(f"Specified columns {list(wrong_columns)} not in schema.")

  def serve(self):
    self.reader = _Reader(location=self.LOCATION, ds=self)
    self.reader.serve()

  def _create_dataset(self):
    return pads.dataset(
      source=random.sample(self._files, len(self._files))[0],
      format="parquet",
      filesystem=self._fs,
      exclude_invalid_files=False,
    )

  def to_batches(self):
    """This allows the init to control reading settings.

    Refer to https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_dataset.

    Perform `drop_remainder` behavior to afix batch size.
    This does not shift our data distribution bc of volume and file-level shuffling on every repeat.
    """
    batch_size = self._dataset_kwargs["batch_size"]
    while True:
      ds = self._create_dataset()
      for batch in ds.to_batches(**self._dataset_kwargs):
        if batch.num_rows < batch_size:
          logging.info(f"Dropping remainder ({batch.num_rows}/{batch_size})")
          break
        yield batch

  @abc.abstractmethod
  def pa_to_batch(self, batch: pa.RecordBatch) -> DataclassBatch:
    raise NotImplementedError

  def dataloader(self, remote: bool = False):
    if not remote:
      return map(self.pa_to_batch, self.to_batches())
    readers = get_readers(2)
    return map(self.pa_to_batch, reader_utils.roundrobin(*readers))


GRPC_OPTIONS = [
  ("GRPC_ARG_KEEPALIVE_TIME_MS", 60000),
  ("GRPC_ARG_MIN_RECONNECT_BACKOFF_MS", 2000),
  ("GRPC_ARG_MAX_METADATA_SIZE", 1024 * 1024 * 1024),
]


def get_readers(num_readers_per_worker: int):
  addresses = env.get_flight_server_addresses()

  readers = []
  for worker in addresses:
    logging.info(f"Attempting connection to reader {worker}.")
    client = pa.flight.connect(worker, generic_options=GRPC_OPTIONS)
    client.wait_for_available(60)
    reader = client.do_get(None).to_reader()
    logging.info(f"Connected reader to {worker}.")
    readers.append(reader)
  return readers
