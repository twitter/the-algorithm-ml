import json
import os
from typing import List


KF_DDS_PORT: int = 5050
SLURM_DDS_PORT: int = 5051
FLIGHT_SERVER_PORT: int = 2222


def on_kf():
    """Check if the code is running on Kubernetes with Kubeflow (KF) environment.

    Returns:
        bool: True if running on KF, False otherwise.
    """
    return "SPEC_TYPE" in os.environ


def has_readers():
    """Check if the current task has dataset workers.

    Returns:
        bool: True if the task has dataset workers, False otherwise.
    """
    if on_kf():
        machines_config_env = json.loads(os.environ["MACHINES_CONFIG"])
        return machines_config_env.get("dataset_worker") is not None
    return os.environ.get("HAS_READERS", "False") == "True"


def get_task_type():
    """Get the type of the current task.

    Returns:
        str: Task type, such as 'chief', 'datasetworker', or 'datasetdispatcher'.
    """
    if on_kf():
        return os.environ["SPEC_TYPE"]
    return os.environ["TASK_TYPE"]


def is_chief() -> bool:
    """Check if the current task is the 'chief'.

    Returns:
        bool: True if the current task is the 'chief', False otherwise.
    """
    return get_task_type() == "chief"


def is_reader() -> bool:
    """Check if the current task is a 'datasetworker'.

    Returns:
        bool: True if the current task is a 'datasetworker', False otherwise.
    """
    return get_task_type() == "datasetworker"


def is_dispatcher() -> bool:
    """Check if the current task is a 'datasetdispatcher'.

    Returns:
        bool: True if the current task is a 'datasetdispatcher', False otherwise.
    """
    return get_task_type() == "datasetdispatcher"


def get_task_index():
    """Get the index of the current task.

    Returns:
        int: Task index.
    Raises:
        NotImplementedError: If not running on Kubernetes with Kubeflow (KF) environment.
    """
    if on_kf():
        pod_name = os.environ["MY_POD_NAME"]
        return int(pod_name.split("-")[-1])
    else:
        raise NotImplementedError


def get_reader_port():
    """Get the port used by readers.

    Returns:
        int: Reader port.
    """
    if on_kf():
        return KF_DDS_PORT
    return SLURM_DDS_PORT


def get_dds():
    """Get the Distributed Data Service (DDS) address.

    Returns:
        str: DDS address in the format 'grpc://host:port'.
    Raises:
        ValueError: If the job does not have DDS.
    """
    if not has_readers():
        return None
    dispatcher_address = get_dds_dispatcher_address()
    if dispatcher_address:
        return f"grpc://{dispatcher_address}"
    else:
        raise ValueError("Job does not have DDS.")


def get_dds_dispatcher_address():
    """Get the DDS dispatcher address.

    Returns:
        str: DDS dispatcher address in the format 'host:port'.
    """
    if not has_readers():
        return None
    if on_kf():
        job_name = os.environ["JOB_NAME"]
        dds_host = f"{job_name}-datasetdispatcher-0"
    else:
        dds_host = os.environ["SLURM_JOB_NODELIST_HET_GROUP_0"]
    return f"{dds_host}:{get_reader_port()}"


def get_dds_worker_address():
    """Get the DDS worker address.

    Returns:
        str: DDS worker address in the format 'host:port'.
    """
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
    """Get the number of dataset workers.

    Returns:
        int: Number of dataset workers.
    """
    if not has_readers():
        return 0
    if on_kf():
        machines_config_env = json.loads(os.environ["MACHINES_CONFIG"])
        return int(machines_config_env.get("num_dataset_workers") or 0)
    return len(os.environ["SLURM_JOB_NODELIST_HET_GROUP_1"].split(","))


def get_flight_server_addresses():
    """Get Flight server addresses for dataset workers.

    Returns:
        List[str]: List of Flight server addresses in the format 'grpc://host:port'.
    Raises:
        NotImplementedError: If not running on Kubernetes with Kubeflow (KF) environment.
    """
    if on_kf():
        job_name = os.environ["JOB_NAME"]
        return [
            f"grpc://{job_name}-datasetworker-{task_index}:{FLIGHT_SERVER_PORT}"
            for task_index in range(get_num_readers())
        ]
    else:
        raise NotImplementedError


def get_dds_journaling_dir():
    """Get the DDS journaling directory.

    Returns:
        str: DDS journaling directory.
    """
    return os.environ.get("DATASET_JOURNALING_DIR", None)
