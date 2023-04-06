"""Utilities for interacting with the file systems."""
from typing import (
  Union,
)

import gcsfs
from fsspec.implementations.local import (
  LocalFileSystem,
)

GCS_FS = gcsfs.GCSFileSystem(cache_timeout=-1)
LOCAL_FS = LocalFileSystem()


def infer_fs(path: str) -> Union[LocalFileSystem, gcsfs.core.GCSFileSystem, NotImplementedError]:
  if path.startswith("gs://"):
    return GCS_FS
  elif path.startswith("hdfs://"):
    # We can probably use pyarrow HDFS to support this.
    raise NotImplementedError("HDFS not yet supported")
  else:
    return LOCAL_FS


def is_local_fs(fs: LocalFileSystem) -> bool:
  return fs == LOCAL_FS


def is_gcs_fs(fs: gcsfs.core.GCSFileSystem) -> bool:
  return fs == GCS_FS
