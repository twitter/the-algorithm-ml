import os

import torch
import torch.distributed as dist


def maybe_setup_tensorflow():
  try:
    import tensorflow as tf
  except ImportError:
    pass
  else:
    tf.config.set_visible_devices([], "GPU")  # disable tf gpu


def setup_and_get_device(tf_ok: bool = True) -> torch.device:
  if tf_ok:
    maybe_setup_tensorflow()

  device = torch.device("cpu")
  backend = "gloo"
  if torch.cuda.is_available():
    rank = os.environ["LOCAL_RANK"]
    device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
  if not torch.distributed.is_initialized():
    dist.init_process_group(backend)

  return device
