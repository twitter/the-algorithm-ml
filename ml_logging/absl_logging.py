"""Sets up logging through absl for training usage.

- Redirects logging to sys.stdout so that severity levels in GCP Stackdriver are accurate.

Usage:
    >>> from twitter.ml.logging.absl_logging import logging
    >>> logging.info(f"Properly logged as INFO level in GCP Stackdriver.")

"""
import logging as py_logging
import sys

from absl import logging as logging


def setup_absl_logging():
  """Make sure that absl logging pushes to stdout rather than stderr."""
  logging.get_absl_handler().python_handler.stream = sys.stdout
  formatter = py_logging.Formatter(
    fmt="[%(module)s.%(funcName)s:%(lineno)s - %(levelname)s] %(message)s"
  )
  logging.get_absl_handler().setFormatter(formatter)
  logging.set_verbosity(logging.INFO)


setup_absl_logging()
