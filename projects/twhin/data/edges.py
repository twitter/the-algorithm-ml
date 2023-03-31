from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tml.common.batch import DataclassBatch
from tml.reader.dataset import Dataset
from tml.projects.twhin.models.config import Relation

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class EdgeBatch(DataclassBatch):
  nodes: KeyedJaggedTensor
  labels: torch.Tensor
  rels: torch.Tensor
  weights: torch.Tensor


class EdgesDataset(Dataset):
  rng = np.random.default_rng()

  def __init__(
    self,
    file_pattern: str,
    table_sizes: Dict[str, int],
    relations: List[Relation],
    lhs_column_name: str = "lhs",
    rhs_column_name: str = "rhs",
    rel_column_name: str = "rel",
    **dataset_kwargs
  ):
    self.batch_size = dataset_kwargs["batch_size"]

    self.table_sizes = table_sizes
    self.num_tables = len(table_sizes)
    self.table_names = list(table_sizes.keys())

    self.relations = relations
    self.relations_t = torch.tensor(
      [
        [self.table_names.index(relation.lhs), self.table_names.index(relation.rhs)]
        for relation in relations
      ]
    )

    self.lhs_column_name = lhs_column_name
    self.rhs_column_name = rhs_column_name
    self.rel_column_name = rel_column_name
    self.label_column_name = "label"

    super().__init__(file_pattern=file_pattern, **dataset_kwargs)

  def pa_to_batch(self, batch: pa.RecordBatch):
    lhs = torch.from_numpy(batch.column(self.lhs_column_name).to_numpy())
    rhs = torch.from_numpy(batch.column(self.rhs_column_name).to_numpy())
    rel = torch.from_numpy(batch.column(self.rel_column_name).to_numpy())
    label = torch.from_numpy(batch.column(self.label_column_name).to_numpy())

    nodes = self._to_kjt(lhs, rhs, rel)
    return EdgeBatch(
      nodes=nodes,
      rels=rel,
      labels=label,
      weights=torch.ones(batch.num_rows),
    )

  def _to_kjt(
    self, lhs: torch.Tensor, rhs: torch.Tensor, rel: torch.Tensor
  ) -> Tuple[KeyedJaggedTensor, List[Tuple[int, int]]]:

    """Process edges that contain lhs index, rhs index, relation index.
    Example:

    ```
    tables = ["f0", "f1", "f2", "f3"]
    relations = [["f0", "f1"], ["f1", "f2"], ["f1", "f0"], ["f2", "f1"], ["f0", "f2"]]
    self.relations_t = torch.Tensor([[0, 1], [1, 2], [1, 0], [2, 1], [0, 2]])
    lhs = [1, 6, 3, 1, 8]
    rhs = [6, 3, 4, 4, 9]
    rel = [0, 2, 1, 3, 4]

    This corresponds to the following "edges":
    edges = [
      {"lhs": 1, "rhs": 6, "relation": ["f0", "f1"]},
      {"lhs": 6, "rhs": 3, "relation": ["f1", "f0"]},
      {"lhs": 3, "rhs": 4, "relation": ["f1", "f2"]},
      {"lhs": 1, "rhs": 4, "relation": ["f2", "f1"]},
      {"lhs": 8, "rhs": 9, "relation": ["f0", "f2"]},
    ]
    ```

    Returns a KeyedJaggedTensor used to look up all embeddings.

    Note: We treat the lhs and rhs as though they're separate lookups: `len(lenghts) == 2 * bsz * len(tables)`.
    This differs from the DLRM pattern where we have `len(lengths) = bsz * len(tables)`.

    For the example above:
    ```
    lookups = tensor([
      [0., 1.],
      [1., 6.],
      [1., 6.],
      [0., 3.],
      [1., 3.],
      [2., 4.],
      [2., 1.],
      [1., 4.],
      [0., 8.],
      [2., 9.]
    ])

    kjt = KeyedJaggedTensor(
      features=["f0", "f1", "f2"]
      values=[
        1, 3, 8,      # f0
        6, 6, 3, 4,   # f1
        4, 1, 9       # f2
      ]
      lengths=[
        1, 0, 0, 1, 0, 0, 0, 0, 1, 0,  # f0
        0, 1, 1, 0, 1, 0, 0, 1, 0, 0,  # f1
        0, 0, 0, 0, 0, 1, 1, 0, 0, 1,  # f2
    )
    ```

    Note:
      - values = [values for f0] + [values for f1] + [values for f2]
      - lengths are always 0 or 1, and sum(lengths) = len(values) = 2 * bsz
    """
    lookups = torch.concat((lhs[:, None], self.relations_t[rel], rhs[:, None]), dim=1)
    index = torch.LongTensor([1, 0, 2, 3])
    lookups = lookups[:, index].reshape(2 * self.batch_size, 2)

    # values is just the row indices into each table, ordered by the table indices
    _, indices = torch.sort(lookups[:, 0], dim=0, stable=True)
    values = lookups[indices][:, 1].int()

    # lengths[table_idx * batch_size + i] == whether the ith lookup is for the table with index table_idx
    lengths = torch.arange(self.num_tables)[:, None].eq(lookups[:, 0])
    lengths = lengths.reshape(-1).int()

    return KeyedJaggedTensor(keys=self.table_names, values=values, lengths=lengths)

  def to_batches(self):
    ds = super().to_batches()
    batch_size = self._dataset_kwargs["batch_size"]

    names = [
      self.lhs_column_name,
      self.rhs_column_name,
      self.rel_column_name,
      self.label_column_name,
    ]
    for _, batch in enumerate(ds):
      # Pass along positive edges
      lhs = batch.column(self.lhs_column_name)
      rhs = batch.column(self.rhs_column_name)
      rel = batch.column(self.rel_column_name)
      label = pa.array(np.ones(batch_size, dtype=np.int64))

      yield pa.RecordBatch.from_arrays(
        arrays=[lhs, rhs, rel, label],
        names=names,
      )
