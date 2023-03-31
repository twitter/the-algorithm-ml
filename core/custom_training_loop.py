"""Torch and torchrec specific training and evaluation loops.

Features (go/100_enablements):
    - CUDA data-fetch, compute, gradient-push overlap
    - Large learnable embeddings through torchrec
    - On/off-chief evaluation
    - Warmstart/checkpoint management
    - go/dataset-service 0-copy integration

"""
import datetime
import os
from typing import Callable, Dict, Iterable, List, Mapping, Optional


from tml.common import log_weights
import tml.common.checkpointing.snapshot as snapshot_lib
from tml.core.losses import get_global_loss_detached
from tml.ml_logging.torch_logging import logging  # type: ignore[attr-defined]
from tml.core.train_pipeline import TrainPipelineSparseDist

import tree
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
import torchmetrics as tm


def get_new_iterator(iterable: Iterable):
  """
  This obtain a new iterator from the iterable. If the iterable uses tf.data.Dataset internally,
   getting a new iterator each N steps will avoid memory leak. To avoid the memory leak
   calling iter(iterable) should return a "fresh" iterator using a fresh
   (new instance of) tf.data.Iterator.
   In particular, iterable can be a torch.utils.data.IterableDataset or a
   torch.utils.data.DataLoader.

  When using DDS, performing this reset does not change the order in which elements are received
   (excluding elements already prefetched) provided that iter(iterable) internally uses
   a new instance of tf.data.Dataset created by calling from_dataset_id.
   This requirement is satisfied by RecapDataset.
  :param iterable:
  :return:
  """
  return iter(iterable)


def _get_step_fn(pipeline, data_iterator, training: bool):
  def step_fn():
    # It turns out that model.train() and model.eval() simply switch a single field inside the model
    # class,so it's somewhat safer to wrap in here.
    if training:
      pipeline._model.train()
    else:
      pipeline._model.eval()

    outputs = pipeline.progress(data_iterator)
    return tree.map_structure(lambda elem: elem.detach(), outputs)

  return step_fn


@torch.no_grad()
def _run_evaluation(
  pipeline,
  dataset,
  eval_steps: int,
  metrics: tm.MetricCollection,
  eval_batch_size: int,
  logger=None,
):
  """Runs the evaluation loop over all evaluation iterators."""
  dataset = get_new_iterator(dataset)
  step_fn = _get_step_fn(pipeline, dataset, training=False)
  last_time = datetime.datetime.now()
  logging.info(f"Starting {eval_steps} steps of evaluation.")
  for _ in range(eval_steps):
    outputs = step_fn()
    metrics.update(outputs)
  eval_ex_per_s = (
    eval_batch_size * eval_steps / (datetime.datetime.now() - last_time).total_seconds()
  )
  logging.info(f"eval examples_per_s : {eval_ex_per_s}")
  metrics_result = metrics.compute()
  # Resetting at end to release metrics memory not in use.
  # Reset metrics to prevent accumulation between multiple evaluation splits and not report a
  # running average.
  metrics.reset()
  return metrics_result


def train(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: str,
  save_dir: str,
  logging_interval: int,
  train_steps: int,
  checkpoint_frequency: int,
  dataset: Iterable,
  worker_batch_size: int,
  num_workers: Optional[int] = 0,
  enable_amp: bool = False,
  initial_checkpoint_dir: Optional[str] = None,
  gradient_accumulation: Optional[int] = None,
  logger_initializer: Optional[Callable] = None,
  scheduler: _LRScheduler = None,
  metrics: Optional[tm.MetricCollection] = None,
  parameters_to_log: Optional[Dict[str, Callable]] = None,
  tables_to_log: Optional[List[str]] = None,
) -> None:
  """Runs training and eval on the given TrainPipeline

  Args:
    dataset: data iterator for the training set
    evaluation_iterators: data iterators for the different evaluation sets
    scheduler: optional learning rate scheduler
    output_transform_for_metrics: optional transformation functions to transorm the model
                                  output and labels into a format the metrics can understand
  """

  train_pipeline = TrainPipelineSparseDist(
    model=model,
    optimizer=optimizer,
    device=device,
    enable_amp=enable_amp,
    grad_accum=gradient_accumulation,
  )  # type: ignore[var-annotated]

  # We explicitly initialize optimizer state here so that checkpoint will work properly.
  if hasattr(train_pipeline._optimizer, "init_state"):
    train_pipeline._optimizer.init_state()

  save_state = {
    "model": train_pipeline._model,
    "optimizer": train_pipeline._optimizer,
    "scaler": train_pipeline._grad_scaler,
  }

  chosen_checkpoint = None
  checkpoint_handler = snapshot_lib.Snapshot(
    save_dir=save_dir,
    state=save_state,
  )

  if save_dir:
    chosen_checkpoint = snapshot_lib.get_checkpoint(save_dir=save_dir, missing_ok=True)

  start_step = 0
  start_walltime = 0.0
  if chosen_checkpoint:
    # Skip restoration and exit if we should be finished.
    chosen_checkpoint_global_step = snapshot_lib.step_from_checkpoint(chosen_checkpoint)
    if not chosen_checkpoint_global_step < dist.get_world_size() * train_steps:
      logging.info(
        "Not restoring and finishing training as latest checkpoint "
        f"{chosen_checkpoint} found "
        f"at global_step ({chosen_checkpoint_global_step}) >= "
        f"train_steps ({dist.get_world_size() * train_steps})"
      )
      return
    logging.info(f"Restoring latest checkpoint from global_step {chosen_checkpoint_global_step}")
    checkpoint_handler.restore(chosen_checkpoint)
    start_step = checkpoint_handler.step
    start_walltime = checkpoint_handler.walltime
  elif initial_checkpoint_dir:
    base, ckpt_step = os.path.split(initial_checkpoint_dir)
    warmstart_handler = snapshot_lib.Snapshot(
      save_dir=base,
      state=save_state,
    )
    ckpt = snapshot_lib.get_checkpoint(save_dir=base, missing_ok=False, global_step=int(ckpt_step))
    logging.info(
      f"Restoring from initial_checkpoint_dir: {initial_checkpoint_dir}, but keeping starting step as 0."
    )
    warmstart_handler.restore(ckpt)

  train_logger = logger_initializer(mode="train") if logger_initializer else None
  train_step_fn = _get_step_fn(train_pipeline, get_new_iterator(dataset), training=True)

  # Counting number of parameters in the model directly when creating it.
  nb_param = 0
  for p in model.parameters():
    nb_param += p.numel()
  logging.info(f"Model has {nb_param} parameters")

  last_time = datetime.datetime.now()
  start_time = last_time
  last_pending_snapshot = None
  for step in range(start_step, train_steps + 1):
    checkpoint_handler.step = step
    outputs = train_step_fn()
    step_done_time = datetime.datetime.now()
    checkpoint_handler.walltime = (step_done_time - start_time).total_seconds() + start_walltime

    if scheduler:
      scheduler.step()

    if step % logging_interval == 0:
      interval_time = (step_done_time - last_time).total_seconds()
      steps_per_s = logging_interval / interval_time
      worker_example_per_s = steps_per_s * worker_batch_size
      global_example_per_s = worker_example_per_s * (1 + (num_workers or 0))
      global_step = step

      log_values = {
        "global_step": global_step,
        "loss": get_global_loss_detached(outputs["loss"]),
        "steps_per_s": steps_per_s,
        "global_example_per_s": global_example_per_s,
        "worker_examples_per_s": worker_example_per_s,
        "active_training_walltime": checkpoint_handler.walltime,
      }
      if parameters_to_log:
        log_values.update(
          log_weights.weights_to_log(
            model=model,
            how_to_log=parameters_to_log,
          )
        )
      log_values = tree.map_structure(lambda elem: torch.as_tensor(elem).cpu(), log_values)

      if tables_to_log:
        log_values.update(
          log_weights.log_ebc_norms(
            model_state_dict=train_pipeline._model.state_dict(),
            ebc_keys=tables_to_log,
          )
        )
      if train_logger:
        train_logger.log(log_values, step=global_step)
      log_line = ", ".join(f"{name}: {value}" for name, value in log_values.items())
      logging.info(f"Step: {step}, training. {log_line}")
      last_time = step_done_time

      # If we just restored, do not save again.
      if checkpoint_frequency and step > start_step and step % checkpoint_frequency == 0:
        if last_pending_snapshot and not last_pending_snapshot.done():
          logging.warning(
            "Begin a new snapshot and the last one hasn't finished. That probably indicates "
            "either you're snapshotting really often or something is wrong. Will now block and "
            "wait for snapshot to finish before beginning the next one."
          )
          last_pending_snapshot.wait()
        last_pending_snapshot = checkpoint_handler.save(global_step=step * dist.get_world_size())

  # Save if we did not just save.
  if checkpoint_frequency and step % checkpoint_frequency != 0:
    # For the final save, wait for the checkpoint to write to make sure the process doesn't finish
    # before its completed.
    last_pending_snapshot = checkpoint_handler.save(global_step=step * dist.get_world_size())
  logging.info(f"Finished training steps: {step}, global_steps: {step * dist.get_world_size()}")

  if last_pending_snapshot:
    logging.info(f"Waiting for any checkpoints to finish.")
    last_pending_snapshot.wait()


def log_eval_results(
  results,
  eval_logger,
  partition_name: str,
  step: int,
):
  results = tree.map_structure(lambda elem: torch.as_tensor(elem).cpu(), results)
  logging.info(f"Step: {step}, evaluation ({partition_name}).")
  for metric_name, metric_value in results.items():
    logging.info(f"\t{metric_name}: {metric_value:1.4e}")

  if eval_logger:
    eval_logger.log(results, step=step, commit=True)


def only_evaluate(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: str,
  save_dir: str,
  num_train_steps: int,
  dataset: Iterable,
  eval_batch_size: int,
  num_eval_steps: int,
  eval_timeout_in_s: int,
  eval_logger: Callable,
  partition_name: str,
  metrics: Optional[tm.MetricCollection] = None,
):
  logging.info(f"Evaluating on partition {partition_name}.")
  logging.info("Computing metrics:")
  logging.info(metrics)
  eval_pipeline = TrainPipelineSparseDist(model, optimizer, device)  # type: ignore[var-annotated]
  save_state = {
    "model": eval_pipeline._model,
    "optimizer": eval_pipeline._optimizer,
  }
  checkpoint_handler = snapshot_lib.Snapshot(
    save_dir=save_dir,
    state=save_state,
  )
  for checkpoint_path in snapshot_lib.checkpoints_iterator(save_dir, timeout=eval_timeout_in_s):
    checkpoint_handler.restore(checkpoint_path)
    step = checkpoint_handler.step
    dataset = get_new_iterator(dataset)
    results = _run_evaluation(
      pipeline=eval_pipeline,
      dataset=dataset,
      eval_steps=num_eval_steps,
      eval_batch_size=eval_batch_size,
      metrics=metrics,
    )
    log_eval_results(results, eval_logger, partition_name, step=step)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
      snapshot_lib.mark_done_eval(checkpoint_path, partition_name)
    if step >= num_train_steps:
      return
