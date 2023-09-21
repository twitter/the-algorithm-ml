"""Utilities for interacting with the file systems."""
from fsspec.implementations.local import LocalFileSystem
import gcsfs


GCS_FS = gcsfs.GCSFileSystem(cache_timeout=-1)
LOCAL_FS = LocalFileSystem()


def infer_fs(path: str):
  """
    Infer the file system (fs) type based on the given path.

    Args:
        path (str): The file path.

    Returns:
        str: The inferred file system type ("gs://" for Google Cloud Storage, "hdfs://" for Hadoop Distributed File System, or "local" for local file system).
    
    Raises:
        NotImplementedError: If the path indicates Hadoop Distributed File System (HDFS) which is not yet supported.
    """
  if path.startswith("gs://"):
    return GCS_FS
  elif path.startswith("hdfs://"):
    # We can probably use pyarrow HDFS to support this.
    raise NotImplementedError("HDFS not yet supported")
  else:
    return LOCAL_FS


def is_local_fs(fs):
  """
    Check if the given file system is the local file system.

    Args:
        fs (str): The file system type to check.

    Returns:
        bool: True if the file system is the local file system, False otherwise.
    """
  return fs == LOCAL_FS


def is_gcs_fs(fs):
  """
    Check if the given file system is Google Cloud Storage (GCS).

    Args:
        fs (str): The file system type to check.

    Returns:
        bool: True if the file system is GCS, False otherwise.
    """
  return fs == GCS_FS
