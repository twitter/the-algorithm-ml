from absl import app, flags
import json
from typing import Optional
import os
import sys

import torch

# isort: on
from tml.common.device import setup_and_get_device
from tml.common.utils import setup_configuration
import tml.core.custom_training_loop as ctl
import tml.machines.environment as env
from tml.projects.twhin.models.models import apply_optimizers, TwhinModel, TwhinModelAndLoss
from tml.model import maybe_shard_model
from tml.projects.twhin.metrics import create_metrics
from tml.projects.twhin.config import TwhinConfig
from tml.projects.twhin.data.data import create_dataset
from tml.projects.twhin.optimizer import build_optimizer

from tml.ml_logging.torch_logging import logging

import torch.distributed as dist
from torch.nn import functional as F
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.model_parallel import get_module

FLAGS = flags.FLAGS

flags.DEFINE_bool("overwrite_save_dir", False, "Whether to clear preexisting save directories.")
flags.DEFINE_string("save_dir", None, "If provided, overwrites the save directory.")
flags.DEFINE_string("config_yaml_path", None, "Path to hyperparameters for model.")
flags.DEFINE_string("task", None, "Task to run if this is local. Overrides TF_CONFIG etc.")


def run(
  all_config: TwhinConfig,
  save_dir: Optional[str] = None,
):
  train_dataset = create_dataset(all_config.train_data, all_config.model)

  if env.is_reader():
    train_dataset.serve()
  if env.is_chief():
    device = setup_and_get_device(tf_ok=False)
    logging.info(f"device: {device}")
    logging.info(f"WORLD_SIZE: {dist.get_world_size()}")

    # validation_dataset = create_dataset(all_config.validation_data, all_config.model)

    global_batch_size = all_config.train_data.per_replica_batch_size * dist.get_world_size()

    metrics = create_metrics(device)

    model = TwhinModel(all_config.model, all_config.train_data)
    apply_optimizers(model, all_config.model)
    model = maybe_shard_model(model, device=device)
    optimizer, scheduler = build_optimizer(model=model, config=all_config.model)

    loss_fn = F.binary_cross_entropy_with_logits
    model_and_loss = TwhinModelAndLoss(
      model, loss_fn, data_config=all_config.train_data, device=device
    )

    ctl.train(
      model=model_and_loss,
      optimizer=optimizer,
      device=device,
      save_dir=save_dir,
      logging_interval=all_config.training.train_log_every_n,
      train_steps=all_config.training.num_train_steps,
      checkpoint_frequency=all_config.training.checkpoint_every_n,
      dataset=train_dataset.dataloader(remote=False),
      worker_batch_size=global_batch_size,
      num_workers=0,
      scheduler=scheduler,
      initial_checkpoint_dir=all_config.training.initial_checkpoint_dir,
      gradient_accumulation=all_config.training.gradient_accumulation,
    )


def main(argv):
  logging.info("Starting")

  logging.info(f"parsing config from {FLAGS.config_yaml_path}...")
  all_config = setup_configuration(  # type: ignore[var-annotated]
    TwhinConfig,
    yaml_path=FLAGS.config_yaml_path,
  )

  run(
    all_config,
    save_dir=FLAGS.save_dir,
  )


if __name__ == "__main__":
  app.run(main)
