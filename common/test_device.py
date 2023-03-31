"""Minimal test for device.

Mostly a test that this can be imported properly even tho moved.
"""
from unittest.mock import patch

import tml.common.device as device_utils


def test_device():
  with patch("tml.common.device.dist.init_process_group"):
    device = device_utils.setup_and_get_device(tf_ok=False)
  assert device.type == "cpu"
