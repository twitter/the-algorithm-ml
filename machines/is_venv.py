"""This is intended to be run as a module.
e.g. python -m tml.machines.is_venv

Exits with 0 ii running in venv, otherwise 1.
"""

import sys
import logging


def is_venv():
  # See https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
  return sys.base_prefix != sys.prefix


def _main():
  if is_venv():
    logging.info("In venv %s", sys.prefix)
    sys.exit(0)
  else:
    logging.error("Not in venv")
    sys.exit(1)


if __name__ == "__main__":
  _main()
