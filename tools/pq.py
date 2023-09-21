"""Local reader of parquet files.

1. Make sure you are initialized locally:
  ```
  ./images/init_venv_macos.sh
  ```
2. Activate
  ```
  source ~/tml_venv/bin/activate
  ```
3. Use tool, e.g.

  `head` prints the first `--num` rows of the dataset.
  ```
  python3 tools/pq.py \
    --num 5 --path "tweet_eng/small/edges/all/*" \
    head
  ```

  `distinct` prints the observed values in the first `--num` rows for the specified columns.
  ```
  python3 tools/pq.py \
    --num 1000000000 --columns '["rel"]' \
    --path "tweet_eng/small/edges/all/*" \
    distinct
  ```

"""
from typing import List, Optional

from tml.common.filesystem import infer_fs

import fire
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq


def _create_dataset(path: str):
  """
    Create a PyArrow dataset from Parquet files located at the specified path.

    Args:
        path (str): The path to the Parquet files.

    Returns:
        pyarrow.dataset.Dataset: The PyArrow dataset.
    """
  fs = infer_fs(path)
  files = fs.glob(path)
  return pads.dataset(files, format="parquet", filesystem=fs)


class PqReader:
  def __init__(
    self, path: str, num: int = 10, batch_size: int = 1024, columns: Optional[List[str]] = None
  ):
    """
        Initialize a Parquet Reader.

        Args:
            path (str): The path to the Parquet files.
            num (int): The maximum number of rows to read.
            batch_size (int): The batch size for reading data.
            columns (Optional[List[str]]): A list of column names to read (default is None, which reads all columns).
        """
    self._ds = _create_dataset(path)
    self._batch_size = batch_size
    self._num = num
    self._columns = columns

  def __iter__(self):
    """
        Iterate through the Parquet data and yield batches of rows.

        Yields:
            pyarrow.RecordBatch: A batch of rows.
        """
    batches = self._ds.to_batches(batch_size=self._batch_size, columns=self._columns)
    rows_seen = 0
    for count, record in enumerate(batches):
      if self._num and rows_seen >= self._num:
        break
      yield record
      rows_seen += record.data.num_rows

  def _head(self):
    """
        Get the first `num` rows of the Parquet data.

        Returns:
            pyarrow.RecordBatch: A batch of rows.
        """
    total_read = self._num * self.bytes_per_row
    if total_read >= int(500e6):
      raise Exception(
        "Sorry you're trying to read more than 500 MB " f"into memory ({total_read} bytes)."
      )
    return self._ds.head(self._num, columns=self._columns)

  @property
  def bytes_per_row(self) -> int:
    """
        Calculate the estimated bytes per row in the dataset.

        Returns:
            int: The estimated bytes per row.
        """
    nbits = 0
    for t in self._ds.schema.types:
      try:
        nbits += t.bit_width
      except:
        # Just estimate size if it is variable
        nbits += 8
    return nbits // 8

  def schema(self):
    """
        Display the schema of the Parquet dataset.
        """
    print(f"\n# Schema\n{self._ds.schema}")

  def head(self):
    """
        Display the first `num` rows of the Parquet data as a pandas DataFrame.
        """
    print(self._head().to_pandas())

  def distinct(self):
    """
        Display unique values seen in specified columns in the first `num` rows.

        Useful for getting an approximate vocabulary for certain columns.
        """
    for col_name, column in zip(self._head().column_names, self._head().columns):
      print(col_name)
      print("unique:", column.unique().to_pylist())


if __name__ == "__main__":
  pd.set_option("display.max_columns", None)
  pd.set_option("display.max_rows", None)
  fire.Fire(PqReader)
