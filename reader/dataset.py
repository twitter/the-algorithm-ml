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
  """
    Distributed reader flight server wrapping a dataset.

    This class implements a Flight server that wraps a dataset, allowing clients to retrieve data
    from the dataset over the Flight protocol. It is designed to be used in a distributed environment
    for efficient data access.

    Args:
        location (str): The location of the Flight server.
        ds (Dataset): The dataset to be wrapped by the Flight server.

    Attributes:
        _location (str): The location of the Flight server.
        _ds (Dataset): The dataset wrapped by the Flight server.

    Methods:
        do_get(_, __): Handles Flight requests for data retrieval.

    Note:
        Flight is an Apache Arrow project that provides a framework for efficient data transfer.
        This class allows clients to retrieve data from the dataset using Flight.

    """

  def __init__(self, location: str, ds: "Dataset"):
    """
        Initialize a new _Reader instance.

        Args:
            location (str): The location of the Flight server.
            ds (Dataset): The dataset to be wrapped by the Flight server.
        """
    super().__init__(location=location)
    self._location = location
    self._ds = ds

  def do_get(self, _, __):
    """
        Handle Flight requests for data retrieval.

        This method retrieves data from the wrapped dataset and provides it to clients over the Flight protocol.

        Args:
            _: Unused argument.
            __: Unused argument.

        Returns:
            pa.flight.RecordBatchStream: A stream of record batches containing data from the dataset.

        Note:
            An updated schema (to account for column selection) must be given to the stream.
        """
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
  """
    A PyTorch IterableDataset wrapping a Parquet dataset for efficient data loading.

    This class enables efficient loading of data from Parquet files using PyArrow.
    It is designed to be used as an IterableDataset in PyTorch for training and inference.

    Args:
        file_pattern (str): A glob pattern specifying the Parquet files to include in the dataset.
        **dataset_kwargs: Additional keyword arguments passed to PyArrow's `to_batches` method.
                         Refer to https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_dataset
                         for more details.

    Attributes:
        LOCATION (str): The default location for the Flight server used for data distribution.
        _file_pattern (str): The glob pattern specifying Parquet files in the dataset.
        _fs: The filesystem object used for file operations.
        _dataset_kwargs (dict): Additional keyword arguments passed to PyArrow's `to_batches` method.
        _files (list): A list of file paths matching the glob pattern.
        _schema (pa.Schema): The schema of the Parquet dataset.

    Methods:
        serve(): Start serving the dataset using a Flight server.
        to_batches(): Generate batches of data from the Parquet dataset.
        pa_to_batch(batch: pa.RecordBatch) -> DataclassBatch: Convert a Parquet RecordBatch to a custom data batch.
        dataloader(remote: bool = False): Create a PyTorch DataLoader for iterating through the dataset.

    Note:
        This class efficiently loads data from Parquet files using PyArrow, and it can be used with PyTorch
        to create DataLoader instances for training or inference.
    """
  LOCATION = "grpc://0.0.0.0:2222"

  def __init__(self, file_pattern: str, **dataset_kwargs) -> None:
    """
    Initialize a new Dataset instance. Specify batch size and column to select for.

    Refer to https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html#pyarrow.dataset.Scanner.from_dataset.


        Args:
            file_pattern (str): A glob pattern specifying the Parquet files to include in the dataset.
            **dataset_kwargs: Additional keyword arguments passed to PyArrow's `to_batches` method.
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
    """
        Validate the specified columns against the dataset schema.

        Raises:
            Exception: If any specified columns are not found in the dataset schema.
        """
    columns = set(self._dataset_kwargs.get("columns", []))
    wrong_columns = set(columns) - set(self._schema.names)
    if wrong_columns:
      raise Exception(f"Specified columns {list(wrong_columns)} not in schema.")

  def serve(self):
    """Start serving the dataset using a Flight server."""
    self.reader = _Reader(location=self.LOCATION, ds=self)
    self.reader.serve()

  def _create_dataset(self):
    """Create a PyArrow dataset for data retrieval."""

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
    """
        Convert a Parquet RecordBatch to a custom data batch.

        Args:
            batch (pa.RecordBatch): A batch of data from the Parquet dataset.

        Returns:
            DataclassBatch: A custom data batch used in PyTorch training.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
    raise NotImplementedError

  def dataloader(self, remote: bool = False):
    """
        Create a PyTorch DataLoader for iterating through the dataset.

        Args:
            remote (bool, optional): If True, create a remote DataLoader using Flight for distributed training.

        Returns:
            DataLoader: A PyTorch DataLoader for iterating through the dataset.

        Note:
            If `remote` is True, a remote DataLoader is created for distributed training using Flight.
        """
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
  """
    Get Flight readers for distributed data loading.

    This function retrieves Flight readers for distributed data loading in a PyTorch environment.

    Args:
        num_readers_per_worker (int): The number of Flight readers to retrieve per worker.

    Returns:
        List[pa.RecordBatchFileReader]: A list of Flight readers for distributed data loading.

    Note:
        Flight readers are used to fetch data in a distributed manner for efficient data loading.

    Example:
        To obtain Flight readers, use the following code:

        >>> readers = get_readers(num_readers_per_worker=2)
    """
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
