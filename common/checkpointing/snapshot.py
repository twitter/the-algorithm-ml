import os
import time
from typing import Any, Dict, List, Optional

from tml.ml_logging.torch_logging import logging
from tml.common.filesystem import infer_fs, is_gcs_fs

import torchsnapshot


DONE_EVAL_SUBDIR = "evaled_by"
GCS_PREFIX = "gs://"


class Snapshot:
  """Checkpoints using torchsnapshot.

  Also saves step to be updated by the training loop.

  """

  def __init__(self, save_dir: str, state: Dict[str, Any]) -> None:
    self.save_dir = save_dir
    self.state = state
    self.state["extra_state"] = torchsnapshot.StateDict(step=0, walltime=0.0)

  @property
  def step(self):
    return self.state["extra_state"]["step"]

  @step.setter
  def step(self, step: int) -> None:
    self.state["extra_state"]["step"] = step

  @property
  def walltime(self):
    return self.state["extra_state"]["walltime"]

  @walltime.setter
  def walltime(self, walltime: float) -> None:
    self.state["extra_state"]["walltime"] = walltime

  def save(self, global_step: int) -> "PendingSnapshot":
    """Saves checkpoint with given global_step."""
    path = os.path.join(self.save_dir, str(global_step))
    logging.info(f"Saving snapshot global_step {global_step} to {path}.")
    start_time = time.time()
    # Take a snapshot in async manner, the snapshot is consistent that state changes after this method returns have no effect on the snapshot. It performs storage I/O in the background.
    snapshot = torchsnapshot.Snapshot.async_take(
      app_state=self.state,
      path=path,
      # commented out because DistributedModelParallel model saving
      # errors with this on multi-GPU. With it removed, CPU, single
      # GPU, and multi-GPU training all successfully checkpoint.
      # replicated=["**"],
    )
    logging.info(f"Snapshot saved to {snapshot.path} ({time.time() - start_time:.05}s")
    return snapshot

  def restore(self, checkpoint: str) -> None:
    """Restores a given checkpoint."""
    snapshot = torchsnapshot.Snapshot(path=checkpoint)
    logging.info(f"Restoring snapshot from {snapshot.path}.")
    start_time = time.time()
    # We can remove the try-except when we are confident that we no longer need to restore from
    # checkpoints from before walltime was added
    try:
      # checkpoints that do not have extra_state[walltime] will fail here
      snapshot.restore(self.state)
    except RuntimeError:
      # extra_state[walltime] does not exist in the checkpoint, but step should be there so restore it
      self.state["extra_state"] = torchsnapshot.StateDict(step=0)
      snapshot.restore(self.state)
      # we still need to ensure that extra_state has walltime in it
      self.state["extra_state"] = torchsnapshot.StateDict(step=self.step, walltime=0.0)

    logging.info(f"Restored snapshot from {snapshot.path}. ({time.time() - start_time:.05}s")

  @classmethod
  def get_torch_snapshot(
    cls,
    snapshot_path: str,
    global_step: Optional[int] = None,
    missing_ok: bool = False,
  ) -> torchsnapshot.Snapshot:
    """Get torch stateless snapshot, without actually loading it.
    Args:
      snapshot_path: path to the model snapshot
      global_step: restores from this checkpoint if specified.
      missing_ok: if True and checkpoints do not exist, returns without restoration.
    """
    path = get_checkpoint(snapshot_path, global_step, missing_ok)
    logging.info(f"Loading snapshot from {path}.")
    return torchsnapshot.Snapshot(path=path)

  @classmethod
  def load_snapshot_to_weight(
    cls,
    embedding_snapshot: torchsnapshot.Snapshot,
    snapshot_emb_name: str,
    weight_tensor,
  ) -> None:
    """Loads pretrained embedding from the snapshot to the model.
       Utilise partial lodaing meachanism from torchsnapshot.
    Args:
      embedding_snapshot: Path to the snapshot containing pretrained embeddings (EBC).
      snapshot_emb_name: Name of the layer in the *snapshot* model, containing the EBC.
      weight_tensor: embeddings tensor of *current* model, where the embeddings will be loaded.
    """
    start_time = time.time()
    manifest = embedding_snapshot.get_manifest()
    for path in manifest.keys():
      if path.startswith("0") and snapshot_emb_name in path:
        snapshot_path_to_load = path
    embedding_snapshot.read_object(snapshot_path_to_load, weight_tensor)
    logging.info(
      f"Loaded embedding snapshot from {snapshot_path_to_load}: {time.time() - start_time:.05}s",
      rank=-1,
    )
    logging.info(f"Snapshot loaded to {weight_tensor.metadata()}", rank=-1)


def _eval_subdir(checkpoint_path: str) -> str:
  return os.path.join(checkpoint_path, DONE_EVAL_SUBDIR)


def _eval_done_path(checkpoint_path: str, eval_partition: str) -> str:
  return os.path.join(_eval_subdir(checkpoint_path), f"{eval_partition}_DONE")


def is_done_eval(checkpoint_path: str, eval_partition: str):
  return get_checkpoint(checkpoint_path).exists(_eval_done_path(checkpoint_path, eval_partition))


def mark_done_eval(checkpoint_path: str, eval_partition: str):
  infer_fs(checkpoint_path).touch(_eval_done_path(checkpoint_path, eval_partition))


def step_from_checkpoint(checkpoint: str) -> int:
  return int(os.path.basename(checkpoint))


def checkpoints_iterator(save_dir: str, seconds_to_sleep: int = 30, timeout: int = 1800):
  """Simplified equivalent of tf.train.checkpoints_iterator.

  Args:
    seconds_to_sleep: time between polling calls.
    timeout: how long to wait for a new checkpoint.

  """

  def _poll(last_checkpoint: Optional[str] = None):
    stop_time = time.time() + timeout
    while True:
      _checkpoint_path = get_checkpoint(save_dir, missing_ok=True)
      if not _checkpoint_path or _checkpoint_path == last_checkpoint:
        if time.time() + seconds_to_sleep > stop_time:
          logging.info(
            f"Timed out waiting for next available checkpoint from {save_dir} for {timeout}s."
          )
          return None
        logging.info(f"Waiting for next available checkpoint from {save_dir}.")
        time.sleep(seconds_to_sleep)
      else:
        logging.info(f"Found latest checkpoint {_checkpoint_path}.")
        return _checkpoint_path

  checkpoint_path = None
  while True:
    new_checkpoint = _poll(checkpoint_path)
    if not new_checkpoint:
      return
    checkpoint_path = new_checkpoint
    yield checkpoint_path


def get_checkpoint(
  save_dir: str,
  global_step: Optional[int] = None,
  missing_ok: bool = False,
) -> str:
  """Gets latest checkpoint or checkpoint at specified global_step.

  Args:
    global_step: Finds this checkpoint if specified.
    missing_ok: if True and checkpoints do not exist, returns without restoration.

  """
  checkpoints = get_checkpoints(save_dir)
  if not checkpoints:
    if not missing_ok:
      raise Exception(f"No checkpoints found at {save_dir}")
    else:
      logging.info(f"No checkpoints found for restoration at {save_dir}.")
      return ""

  if global_step is None:
    return checkpoints[-1]

  logging.info(f"Found checkpoints: {checkpoints}")
  for checkpoint in checkpoints:
    step = step_from_checkpoint(checkpoint)
    if global_step == step:
      chosen_checkpoint = checkpoint
      break
  else:
    raise Exception(f"Desired checkpoint at {global_step} not found in {save_dir}")
  return chosen_checkpoint


def get_checkpoints(save_dir: str) -> List[str]:
  """Gets all checkpoints that have been fully written."""
  checkpoints = []
  fs = infer_fs(save_dir)
  if fs.exists(save_dir):
    prefix = GCS_PREFIX if is_gcs_fs(fs) else ""
    checkpoints = list(f"{prefix}{elem}" for elem in fs.ls(save_dir, detail=False))
    # Only take checkpoints that were fully written.
    checkpoints = list(
      filter(
        lambda path: fs.exists(f"{path}/{torchsnapshot.snapshot.SNAPSHOT_METADATA_FNAME}"),
        checkpoints,
      )
    )
    checkpoints = sorted(checkpoints, key=lambda path: int(os.path.basename(path)))
  return checkpoints


def wait_for_evaluators(
  save_dir: str,
  partition_names: List[str],
  global_step: int,
  timeout: int,
) -> None:
  logging.info("Waiting for all evaluators to finish.")
  start_time = time.time()

  for checkpoint in checkpoints_iterator(save_dir):
    step = step_from_checkpoint(checkpoint)
    logging.info(f"Considering checkpoint {checkpoint} for global step {global_step}.")
    if step == global_step:
      while partition_names:
        if is_done_eval(checkpoint, partition_names[-1]):
          logging.info(
            f"Checkpoint {checkpoint} marked as finished eval for partition {partition_names[-1]} at step {step}, still waiting for {partition_names}."
          )
          partition_names.pop()

        if time.time() - start_time >= timeout:
          logging.warning(
            f"Not all evaluators finished after waiting for {time.time() - start_time}"
          )
          return
        time.sleep(10)
      logging.info("All evaluators finished.")
      return

    if time.time() - start_time >= timeout:
      logging.warning(f"Not all evaluators finished after waiting for {time.time() - start_time}")
      return
