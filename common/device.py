import os

import torch
import torch.distributed as dist


def maybe_setup_tensorflow():
  """
    Try to import TensorFlow and disable GPU devices if TensorFlow is available.

    This function checks if TensorFlow is installed and, if so, disables GPU devices used by TensorFlow to avoid conflicts with PyTorch.

    Returns:
        None

    """
  try:
    import tensorflow as tf
  except ImportError:
    pass
  else:
    tf.config.set_visible_devices([], "GPU")  # disable tf gpu


def setup_and_get_device(tf_ok: bool = True) -> torch.device:
  """
    Set up the distributed environment and get the appropriate torch device.

    This function sets up the distributed environment using PyTorch's `dist.init_process_group` and retrieves the appropriate torch device based on GPU availability and local rank.

    Args:
        tf_ok (bool, optional): Whether to run `maybe_setup_tensorflow` to disable TensorFlow GPU devices. Defaults to True.

    Returns:
        torch.device: The torch device for the current process.

    """
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
