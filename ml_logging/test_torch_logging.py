import unittest

from tml.ml_logging.torch_logging import logging


class Testtlogging(unittest.TestCase):
  def test_warn_once(self):
    with self.assertLogs(level="INFO") as captured_logs:
      logging.info("first info")
      logging.warning("first warning")
      logging.warning("first warning")
      logging.info("second info")

    self.assertEqual(
      captured_logs.output,
      [
        "INFO:absl:first info",
        "WARNING:absl:first warning",
        "INFO:absl:second info",
      ],
    )
