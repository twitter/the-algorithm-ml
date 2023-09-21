"""Tests edges dataset functionality."""

from unittest.mock import patch
import os
import tempfile

from tml.projects.twhin.data.edges import EdgesDataset
from tml.projects.twhin.models.config import Relation

from fsspec.implementations.local import LocalFileSystem
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


TABLE_SIZES = {"user": 16, "author": 32}
RELATIONS = [
  Relation(name="fav", lhs="user", rhs="author"),
  Relation(name="engaged_with_reply", lhs="author", rhs="user"),
]


def test_gen():
  """Test function for generating edge-based datasets and dataloaders.

    This function generates a synthetic dataset and tests the creation of an `EdgesDataset`
    instance and a dataloader for it.

    The test includes the following steps:
    1. Create synthetic data with left-hand-side (lhs), right-hand-side (rhs), and relation (rel) columns.
    2. Write the synthetic data to a Parquet file.
    3. Create an `EdgesDataset` instance with the Parquet file pattern, table sizes, relations, and batch size.
    4. Initialize the local file system for the dataset.
    5. Create a dataloader for the dataset and retrieve the first batch.
    6. Assert that the labels in the batch are positive.
    7. Verify that the positive examples in the batch match the expected values.

    This function serves as a test case for the data generation and dataset creation process.

    Raises:
        AssertionError: If any of the test assertions fail.
    """
  import os
  import tempfile

  from fsspec.implementations.local import LocalFileSystem
  import pyarrow as pa
  import pyarrow.parquet as pq

  lhs = pa.array(np.arange(4))
  rhs = pa.array(np.flip(np.arange(4)))
  rel = pa.array([0, 1, 0, 0])
  names = ["lhs", "rhs", "rel"]

  with tempfile.TemporaryDirectory() as tmpdir:
    table = pa.Table.from_arrays([lhs, rhs, rel], names=names)
    writer = pq.ParquetWriter(
      os.path.join(tmpdir, "example.parquet"),
      table.schema,
    )
    writer.write_table(table)
    writer.close()

    ds = EdgesDataset(
      file_pattern=os.path.join(tmpdir, "*"),
      table_sizes=TABLE_SIZES,
      relations=RELATIONS,
      batch_size=4,
    )
    ds.FS = LocalFileSystem()

    dl = ds.dataloader()
    batch = next(iter(dl))

    # labels should be positive
    labels = batch.labels
    assert (labels[:4] == 1).sum() == 4

    # make sure positive examples are what we expect
    kjt_values = batch.nodes.values()
    users, authors = torch.split(kjt_values, 4, dim=0)
    assert torch.equal(users[:4], torch.tensor([0, 2, 2, 3]))
    assert torch.equal(authors[:4], torch.tensor([3, 1, 1, 0]))
