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
  """
    Configure absl-py logging to direct log messages to stdout and apply a custom log message format.

    This function ensures that log messages generated by the absl-py library are written to stdout
    rather than stderr. It also applies a custom log message format that includes module, function,
    line number, log level, and the log message content.

    Note:
        This function should be called once at the beginning of your script or application to
        configure absl-py logging.

    Example:
        To use this function, simply call it at the start of your script:
        ```
        setup_absl_logging()
        ```

    """
  logging.get_absl_handler().python_handler.stream = sys.stdout
  formatter = py_logging.Formatter(
    fmt="[%(module)s.%(funcName)s:%(lineno)s - %(levelname)s] %(message)s"
  )
  logging.get_absl_handler().setFormatter(formatter)
  logging.set_verbosity(logging.INFO)


setup_absl_logging()
