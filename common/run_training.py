import os
import subprocess
import sys
from typing import Optional

from tml.ml_logging.torch_logging import logging  # type: ignore[attr-defined]
from twitter.ml.tensorflow.experimental.distributed import utils

import torch
import torch.distributed.run


def is_distributed_worker():
  world_size = os.environ.get("WORLD_SIZE", None)
  rank = os.environ.get("RANK", None)
  return world_size is not None and rank is not None


def maybe_run_training(
  train_fn,
  module_name,
  nproc_per_node: Optional[int] = None,
  num_nodes: Optional[int] = None,
  set_python_path_in_subprocess: bool = False,
  is_chief: Optional[bool] = False,
  **training_kwargs,
):
  """Wrapper function for single node, multi-GPU Pytorch training.

  If the necessary distributed Pytorch environment variables
  (WORLD_SIZE, RANK) have been set, then this function executes
  `train_fn(**training_kwargs)`.

  Otherwise, this function calls torchrun and points at the calling module
  `module_name`.  After this call, the necessary environment variables are set
  and training will commence.

  Args:
    train_fn:  The function that is responsible for training
    module_name:  The name of the module that this function was called from;
       used to indicate torchrun entrypoint.
    nproc_per_node: Number of workers per node; supported values.
    num_nodes: Number of nodes, otherwise inferred from environment.
    is_chief: If process is running on chief.
    set_python_path_in_subprocess: A bool denoting whether to set PYTHONPATH.
  """

  machines = utils.machine_from_env()
  if num_nodes is None:
    num_nodes = 1
    if machines.num_workers:
      num_nodes += machines.num_workers

  if is_distributed_worker():
    # world_size, rank, etc are set; assuming any other env vars are set (checks to come)
    # start the actual training!
    train_fn(**training_kwargs)
  else:
    if nproc_per_node is None:
      if torch.cuda.is_available():
        nproc_per_node = torch.cuda.device_count()
      else:
        nproc_per_node = machines.chief.num_accelerators

    # Rejoin all arguments to send back through torchrec
    # this is a temporary measure, will replace the os.system call
    # with torchrun API calls
    args = list(f"--{key}={val}" for key, val in training_kwargs.items())

    cmd = [
      "--nnodes",
      str(num_nodes),
    ]
    if nproc_per_node:
      cmd.extend(["--nproc_per_node", str(nproc_per_node)])
    if num_nodes > 1:
      cluster_resolver = utils.cluster_resolver()
      backend_address = cluster_resolver.cluster_spec().task_address("chief", 0)
      cmd.extend(
        [
          "--rdzv_backend",
          "c10d",
          "--rdzv_id",
          backend_address,
        ]
      )
      # Set localhost on chief because of https://github.com/pytorch/pytorch/issues/79388
      if is_chief:
        cmd.extend(["--rdzv_endpoint", "localhost:2222"])
      else:
        cmd.extend(["--rdzv_endpoint", backend_address])
    else:
      cmd.append("--standalone")

    cmd.extend(
      [
        str(module_name),
        *args,
      ]
    )
    logging.info(f"""Distributed running with cmd: '{" ".join(cmd)}'""")

    # Call torchrun on this module;  will spawn new processes and re-run this
    # function, eventually calling "train_fn". The following line sets the PYTHONPATH to accommodate
    # bazel stubbing for the main binary.
    if set_python_path_in_subprocess:
      subprocess.run(["torchrun"] + cmd, env={**os.environ, "PYTHONPATH": ":".join(sys.path)})
    else:
      torch.distributed.run.main(cmd)
