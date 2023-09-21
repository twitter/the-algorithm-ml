"""Overrides absl logger to be rank-aware for distributed pytorch usage.

    >>> # in-bazel import
    >>> from twitter.ml.logging.torch_logging import logging
    >>> # out-bazel import
    >>> from ml.logging.torch_logging import logging
    >>> logging.info(f"This only prints on rank 0 if distributed, otherwise prints normally.")
    >>> logging.info(f"This prints on all ranks if distributed, otherwise prints normally.", rank=-1)

"""
import functools
from typing import Optional

from tml.ml_logging.absl_logging import logging as logging
from absl import logging as absl_logging

import torch.distributed as dist


def rank_specific(logger):
  """
    Customize logger behavior based on the distributed environment and rank.

    This function allows for customizing the behavior of a logger based on the distributed environment and the rank
    of the current process. It overrides standard logging methods (e.g., error, warning) to conditionally log messages
    depending on the rank or limit the number of redundant logs.

    Args:
        logger: The logger object to customize.

    Returns:
        The customized logger.

    Example:
        To use this function with the `logging` module:
        ```python
        import logging
        from rank_specific_logging import rank_specific

        logger = logging.getLogger(__name__)
        rank_specific(logger)
        ```

    Customization:
        - Messages are only logged if the distributed environment is not initialized or if the rank matches.
        - The 'warning' method is limited to logging a single redundant warning.
        - Logging from rank -1 is redirected to include the rank information.

    """
  if hasattr(logger, "_ALREADY_OVERWRITTEN_TO_BE_RANK_SPECIFIC"):
    return logger

  def _if_rank(logger_method, limit: Optional[int] = None):
    if limit:
      # If we are limiting redundant logs, wrap logging call with a cache
      # to not execute if already cached.
      def _wrap(_call):
        @functools.lru_cache(limit)
        def _logger_method(*args, **kwargs):
          _call(*args, **kwargs)

        return _logger_method

      logger_method = _wrap(logger_method)

    def _inner(msg, *args, rank: int = 0, **kwargs):
      if not dist.is_initialized():
        logger_method(msg, *args, **kwargs)
      elif dist.get_rank() == rank:
        logger_method(msg, *args, **kwargs)
      elif rank < 0:
        logger_method(f"Rank{dist.get_rank()}: {msg}", *args, **kwargs)

    # Register this stack frame with absl logging so that it doesn't trample logging lines.
    absl_logging.ABSLLogger.register_frame_to_skip(__file__, _inner.__name__)

    return _inner

  logger.fatal = _if_rank(logger.fatal)
  logger.error = _if_rank(logger.error)
  logger.warning = _if_rank(logger.warning, limit=1)
  logger.info = _if_rank(logger.info)
  logger.debug = _if_rank(logger.debug)
  logger.exception = _if_rank(logger.exception)

  logger._ALREADY_OVERWRITTEN_TO_BE_RANK_SPECIFIC = True


rank_specific(logging)
