import unittest

from tml.ml_logging.torch_logging import logging


class Testtlogging(unittest.TestCase):
  def test_warn_once(self):
    """
        Test that warning messages are logged only once when using the assertLogs context manager.

        This unit test checks the behavior of the logging system when warning messages are issued
        multiple times within the same context. It uses the assertLogs context manager to capture
        log messages at the INFO level and verifies that warning messages are logged only once.

        Example:
            To use this test case, call it using a test runner like unittest:
            ```
            python -m unittest your_test_module.TestLogging.test_warn_once
            ```

        """

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
