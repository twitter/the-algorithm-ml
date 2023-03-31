"""Minimal test for infer_fs.

Mostly a test that it returns an object
"""
from tml.common.filesystem import infer_fs


def test_infer_fs():
  local_path = "/tmp/local_path"
  gcs_path = "gs://somebucket/somepath"

  local_fs = infer_fs(local_path)
  gcs_fs = infer_fs(gcs_path)

  # This should return two different objects
  assert local_fs != gcs_fs
