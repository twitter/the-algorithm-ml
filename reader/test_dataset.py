import multiprocessing as mp
import os
from unittest.mock import patch

import tml.reader.utils as reader_utils
from tml.reader.dataset import Dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


def create_dataset(tmpdir):

  table = pa.table(
    {
      "year": [2020, 2022, 2021, 2022, 2019, 2021],
      "n_legs": [2, 2, 4, 4, 5, 100],
    }
  )
  file_path = tmpdir
  pq.write_to_dataset(table, root_path=str(file_path))

  class MockDataset(Dataset):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self._pa_to_batch = reader_utils.create_default_pa_to_batch(self._schema)

    def pa_to_batch(self, batch):
      return self._pa_to_batch(batch)

  return MockDataset(file_pattern=str(file_path / "*"), batch_size=2)


def test_dataset(tmpdir):
  ds = create_dataset(tmpdir)
  batch = next(iter(ds.dataloader(remote=False)))
  assert batch.batch_size == 2
  assert torch.equal(batch.year, torch.Tensor([2020, 2022]))
  assert torch.equal(batch.n_legs, torch.Tensor([2, 2]))


@pytest.mark.skipif(
  os.environ.get("GITHUB_WORKSPACE") is not None,
  reason="Multiprocessing doesn't work on github yet.",
)
def test_distributed_dataset(tmpdir):
  MOCK_ENV = {"TEMP_SLURM_NUM_READERS": "1"}

  def _client():
    with patch.dict(os.environ, MOCK_ENV):
      with patch(
        "tml.reader.dataset.env.get_flight_server_addresses", return_value=["grpc://localhost:2222"]
      ):
        ds = create_dataset(tmpdir)
        batch = next(iter(ds.dataloader(remote=True)))
        assert batch.batch_size == 2
        assert torch.equal(batch.year, torch.Tensor([2020, 2022]))
        assert torch.equal(batch.n_legs, torch.Tensor([2, 2]))

  def _worker():
    ds = create_dataset(tmpdir)
    ds.serve()

  worker = mp.Process(target=_worker)
  client = mp.Process(target=_client)
  worker.start()
  client.start()
  client.join()
  assert not client.exitcode
  worker.kill()
  client.kill()
