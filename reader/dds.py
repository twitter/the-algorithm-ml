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
  """Torch-compatible and distributed-training-aware dataset service distributor.

  - rank 0 process will register the given dataset.
  - rank 0 process will broadcast job name and dataset id.
  - all rank processes will consume from the same job/dataset.

  Without this, dataset workers will try to serve 1 job per rank process and OOM.

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
