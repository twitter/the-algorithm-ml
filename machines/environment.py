import json
import os
from typing import List


KF_DDS_PORT: int = 5050
SLURM_DDS_PORT: int = 5051
FLIGHT_SERVER_PORT: int = 2222


def on_kf():
  return "SPEC_TYPE" in os.environ


def has_readers():
  if on_kf():
    machines_config_env = json.loads(os.environ["MACHINES_CONFIG"])
    return machines_config_env["dataset_worker"] is not None
  return os.environ.get("HAS_READERS", "False") == "True"


def get_task_type():
  if on_kf():
    return os.environ["SPEC_TYPE"]
  return os.environ["TASK_TYPE"]


def is_chief() -> bool:
  return get_task_type() == "chief"


def is_reader() -> bool:
  return get_task_type() == "datasetworker"


def is_dispatcher() -> bool:
  return get_task_type() == "datasetdispatcher"


def get_task_index():
  if on_kf():
    pod_name = os.environ["MY_POD_NAME"]
    return int(pod_name.split("-")[-1])
  else:
    raise NotImplementedError


def get_reader_port():
  if on_kf():
    return KF_DDS_PORT
  return SLURM_DDS_PORT


def get_dds():
  if not has_readers():
    return None
  dispatcher_address = get_dds_dispatcher_address()
  if dispatcher_address:
    return f"grpc://{dispatcher_address}"
  else:
    raise ValueError("Job does not have DDS.")


def get_dds_dispatcher_address():
  if not has_readers():
    return None
  if on_kf():
    job_name = os.environ["JOB_NAME"]
    dds_host = f"{job_name}-datasetdispatcher-0"
  else:
    dds_host = os.environ["SLURM_JOB_NODELIST_HET_GROUP_0"]
  return f"{dds_host}:{get_reader_port()}"


def get_dds_worker_address():
  if not has_readers():
    return None
  if on_kf():
    job_name = os.environ["JOB_NAME"]
    task_index = get_task_index()
    return f"{job_name}-datasetworker-{task_index}:{get_reader_port()}"
  else:
    node = os.environ["SLURMD_NODENAME"]
    return f"{node}:{get_reader_port()}"


def get_num_readers():
  if not has_readers():
    return 0
  if on_kf():
    machines_config_env = json.loads(os.environ["MACHINES_CONFIG"])
    return int(machines_config_env["num_dataset_workers"] or 0)
  return len(os.environ["SLURM_JOB_NODELIST_HET_GROUP_1"].split(","))


def get_flight_server_addresses():
  if on_kf():
    job_name = os.environ["JOB_NAME"]
    return [
      f"grpc://{job_name}-datasetworker-{task_index}:{FLIGHT_SERVER_PORT}"
      for task_index in range(get_num_readers())
    ]
  else:
    raise NotImplementedError


def get_dds_journaling_dir():
  return os.environ.get("DATASET_JOURNALING_DIR", None)
