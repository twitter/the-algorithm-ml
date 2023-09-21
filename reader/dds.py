"""Dataset service orchestrated by a TFJob
"""
from typing import Optional
import uuid

from tml.ml_logging.torch_logging import logging
import tml.machines.environment as env

import packaging.version
import tensorflow as tf

try:
  import tensorflow_io as tfio
except:
  pass
from tensorflow.python.data.experimental.ops.data_service_ops import (
  _from_dataset_id,
  _register_dataset,
)
import torch.distributed as dist


def maybe_start_dataset_service():
  """
    Start the dataset service if readers are available and required dependencies are met.

    This function checks if readers are available and if the required TensorFlow version is >= 2.5.
    If both conditions are met and the current environment is the dispatcher or reader, it starts
    the TensorFlow dataset service.

    Raises:
        Exception: If the required TensorFlow version is not met (>= 2.5).
    """
  if not env.has_readers():
    return

  if packaging.version.parse(tf.__version__) < packaging.version.parse("2.5"):
    raise Exception(f"maybe_distribute_dataset requires TF >= 2.5; got {tf.__version__}")

  if env.is_dispatcher():
    logging.info(f"env.get_reader_port() = {env.get_reader_port()}")
    logging.info(f"env.get_dds_journaling_dir() = {env.get_dds_journaling_dir()}")
    work_dir = env.get_dds_journaling_dir()
    server = tf.data.experimental.service.DispatchServer(
      tf.data.experimental.service.DispatcherConfig(
        port=env.get_reader_port(),
        protocol="grpc",
        work_dir=work_dir,
        fault_tolerant_mode=bool(work_dir),
      )
    )
    server.join()

  elif env.is_reader():
    logging.info(f"env.get_reader_port() = {env.get_reader_port()}")
    logging.info(f"env.get_dds_dispatcher_address() = {env.get_dds_dispatcher_address()}")
    logging.info(f"env.get_dds_worker_address() = {env.get_dds_worker_address()}")
    server = tf.data.experimental.service.WorkerServer(
      tf.data.experimental.service.WorkerConfig(
        port=env.get_reader_port(),
        dispatcher_address=env.get_dds_dispatcher_address(),
        worker_address=env.get_dds_worker_address(),
        protocol="grpc",
      )
    )
    server.join()


def register_dataset(
  dataset: tf.data.Dataset, dataset_service: str, compression: Optional[str] = "AUTO"
):
  """
    Register a dataset with the distributed dataset service.

    This function registers a dataset with the distributed dataset service and broadcasts the dataset ID
    and job name to all processes in the distributed environment.

    Args:
        dataset (tf.data.Dataset): The dataset to be registered.
        dataset_service (str): The name of the dataset service.
        compression (Optional[str]): The compression type for the dataset (default is "AUTO").

    Returns:
        Tuple[int, str]: A tuple containing the dataset ID and job name.

    Note:
        This function should be called on the rank 0 process.

    """
  if dist.get_rank() == 0:
    dataset_id = _register_dataset(
      service=dataset_service,
      dataset=dataset,
      compression=compression,
    )
    job_name = uuid.uuid4().hex[:8]
    id_and_job = [dataset_id.numpy(), job_name]
    logging.info(f"rank{dist.get_rank()}: Created dds job with {dataset_id.numpy()}, {job_name}")
  else:
    id_and_job = [None, None]

  dist.broadcast_object_list(id_and_job, src=0)
  return tuple(id_and_job)


def distribute_from_dataset_id(
  dataset_service: str,
  dataset_id: int,
  job_name: Optional[str],
  compression: Optional[str] = "AUTO",
  prefetch: Optional[int] = tf.data.experimental.AUTOTUNE,
) -> tf.data.Dataset:
  """
    Distribute a dataset from a registered dataset ID.

    This function consumes a dataset from the distributed dataset service using the provided dataset ID
    and job name. It also supports prefetching for improved performance.

    Args:
        dataset_service (str): The name of the dataset service.
        dataset_id (int): The ID of the dataset to be consumed.
        job_name (Optional[str]): The name of the job associated with the dataset (optional).
        compression (Optional[str]): The compression type for the dataset (default is "AUTO").
        prefetch (Optional[int]): The number of elements to prefetch (default is tf.data.experimental.AUTOTUNE).

    Returns:
        tf.data.Dataset: The distributed dataset.

    """
  logging.info(f"rank{dist.get_rank()}: Consuming dds job with {dataset_id}, {job_name}")
  dataset = _from_dataset_id(
    processing_mode="parallel_epochs",
    service=dataset_service,
    dataset_id=dataset_id,
    job_name=job_name,
    element_spec=None,
    compression=compression,
  )
  if prefetch is not None:
    dataset = dataset.prefetch(prefetch)
  return dataset


def maybe_distribute_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """
    Distribute a TensorFlow dataset for Torch-compatible and distributed training-aware consumption.

    This function is used to distribute a dataset in a distributed training environment. It performs the
    following steps:
    - On the rank 0 process, it registers the given dataset with the distributed dataset service.
    - It broadcasts the job name and dataset ID to all rank processes.
    - All rank processes then consume the same dataset from the distributed dataset service.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to be distributed.

    Returns:
        tf.data.Dataset: The distributed TensorFlow dataset.

    Note:
        - If there are no reader processes in the distributed environment, the original dataset is returned
          without any distribution.
        - This function is intended for use in distributed training environments to prevent out-of-memory (OOM)
          issues caused by each rank process trying to serve one job.

    """
  if not env.has_readers():
    return dataset
  dataset_service = env.get_dds()

  logging.info(f"using DDS = {dataset_service}")
  dataset_id, job_name = register_dataset(dataset=dataset, dataset_service=dataset_service)
  dataset = distribute_from_dataset_id(
    dataset_service=dataset_service, dataset_id=dataset_id, job_name=job_name
  )
  return dataset


if __name__ == "__main__":
  maybe_start_dataset_service()
