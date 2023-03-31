"""Utilities for interacting with the file systems."""
from fsspec.implementations.local import LocalFileSystem
import gcsfs


GCS_FS = gcsfs.GCSFileSystem(cache_timeout=-1)
LOCAL_FS = LocalFileSystem()


def infer_fs(path: str):
  if path.startswith("gs://"):
    return GCS_FS
  elif path.startswith("hdfs://"):
    # We can probably use pyarrow HDFS to support this.
    raise NotImplementedError("HDFS not yet supported")
  else:
    return LOCAL_FS


def is_local_fs(fs):
  return fs == LOCAL_FS


def is_gcs_fs(fs):
  return fs == GCS_FS
