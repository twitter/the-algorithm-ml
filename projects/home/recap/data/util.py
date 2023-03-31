from typing import Mapping, Tuple, Union
import torch
import torchrec
import numpy as np
import tensorflow as tf


def keyed_tensor_from_tensors_dict(
  tensor_map: Mapping[str, torch.Tensor]
) -> "torchrec.KeyedTensor":
  """
  Convert a dictionary of torch tensor to torchrec keyed tensor
  Args:
    tensor_map:

  Returns:

  """
  keys = list(tensor_map.keys())
  # We expect batch size to be first dim. However, if we get a shape [Batch_size],
  # KeyedTensor will not find the correct batch_size. So, in those cases we make sure the shape is
  # [Batch_size x 1].
  values = [
    tensor_map[key] if len(tensor_map[key].shape) > 1 else torch.unsqueeze(tensor_map[key], -1)
    for key in keys
  ]
  return torchrec.KeyedTensor.from_tensor_list(keys, values)


def _compute_jagged_tensor_from_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  if tensor.is_sparse:
    x = tensor.coalesce()  # Ensure that the indices are ordered.
    lengths = torch.bincount(x.indices()[0])
    values = x.values()
  else:
    values = tensor
    lengths = torch.ones(tensor.shape[0], dtype=torch.int32, device=tensor.device)
  return values, lengths


def jagged_tensor_from_tensor(tensor: torch.Tensor) -> "torchrec.JaggedTensor":
  """
  Convert a torch tensor to torchrec jagged tensor.
  Note: Currently only support shape of [Batch_size] or [Batch_size x N] for dense tensors.
        For sparse tensor the shape of .values() should be [Batch_size] or [Batch_size x N]; the
          dense_shape of the sparse tensor can be arbitrary.
  Args:
    tensor: a torch (sparse) tensor.
  Returns:
  """
  values, lengths = _compute_jagged_tensor_from_tensor(tensor)
  return torchrec.JaggedTensor(values=values, lengths=lengths)


def keyed_jagged_tensor_from_tensors_dict(
  tensor_map: Mapping[str, torch.Tensor]
) -> "torchrec.KeyedJaggedTensor":
  """
  Convert a dictionary of (sparse) torch tensors to torchrec keyed jagged tensor.
  Note: Currently only support shape of [Batch_size] or [Batch_size x 1] for dense tensors.
        For sparse tensor the shape of .values() should be [Batch_size] or [Batch_size x 1]; the
          dense_shape of the sparse tensor can be arbitrary.
  Args:
    tensor_map:

  Returns:

  """

  if not tensor_map:
    return torchrec.KeyedJaggedTensor(
      keys=[],
      values=torch.zeros(0, dtype=torch.int),
      lengths=torch.zeros(0, dtype=torch.int),
    )
  values = []
  lengths = []
  for tensor in tensor_map.values():
    tensor_val, tensor_len = _compute_jagged_tensor_from_tensor(tensor)
    values.append(torch.squeeze(tensor_val))
    lengths.append(tensor_len)

  values = torch.cat(values, axis=0)
  lengths = torch.cat(lengths, axis=0)

  return torchrec.KeyedJaggedTensor(
    keys=list(tensor_map.keys()),
    values=values,
    lengths=lengths,
  )


def _tf_to_numpy(tf_tensor: tf.Tensor) -> np.ndarray:
  return tf_tensor._numpy()  # noqa


def _dense_tf_to_torch(tensor: tf.Tensor, pin_memory: bool) -> torch.Tensor:
  tensor = _tf_to_numpy(tensor)
  # Pytorch does not support bfloat16, up cast to float32 to keep the same number of bits on exponent
  if tensor.dtype.name == "bfloat16":
    tensor = tensor.astype(np.float32)

  tensor = torch.from_numpy(tensor)
  if pin_memory:
    tensor = tensor.pin_memory()
  return tensor


def sparse_or_dense_tf_to_torch(
  tensor: Union[tf.Tensor, tf.SparseTensor], pin_memory: bool
) -> torch.Tensor:
  if isinstance(tensor, tf.SparseTensor):
    tensor = torch.sparse_coo_tensor(
      _dense_tf_to_torch(tensor.indices, pin_memory).t(),
      _dense_tf_to_torch(tensor.values, pin_memory),
      torch.Size(_tf_to_numpy(tensor.dense_shape)),
    )
  else:
    tensor = _dense_tf_to_torch(tensor, pin_memory)
  return tensor
