import datetime
import os
from typing import Callable, List, Optional, Tuple
import tensorflow as tf

import tml.common.checkpointing.snapshot as snapshot_lib
from tml.common.device import setup_and_get_device
from tml.core import config as tml_config_mod
import tml.core.custom_training_loop as ctl
from tml.core import debug_training_loop
from tml.core import losses
from tml.core.loss_type import LossType
from tml.model import maybe_shard_model


import tml.projects.home.recap.data.dataset as ds
import tml.projects.home.recap.config as recap_config_mod
import tml.projects.home.recap.optimizer as optimizer_mod


# from tml.projects.home.recap import feature
import tml.projects.home.recap.model as model_mod
import torchmetrics as tm
import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel

from absl import app, flags, logging

flags.DEFINE_string("config_path", None, "Path to hyperparameters for model.")
flags.DEFINE_bool("debug_loop", False, "Run with debug loop (slow)")

FLAGS = flags.FLAGS


def run(unused_argv: str, data_service_dispatcher: Optional[str] = None):
  print("#" * 100)

  config = tml_config_mod.load_config_from_yaml(recap_config_mod.RecapConfig, FLAGS.config_path)
  logging.info("Config: %s", config.pretty_print())

  device = setup_and_get_device()

  # Always enable tensorfloat on supported devices.
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  loss_fn = losses.build_multi_task_loss(
    loss_type=LossType.BCE_WITH_LOGITS,
    tasks=list(config.model.tasks.keys()),
    pos_weights=[task.pos_weight for task in config.model.tasks.values()],
  )

  # Since the prod model doesn't use large embeddings, for now we won't support them.
  assert config.model.large_embeddings is None

  train_dataset = ds.RecapDataset(
    data_config=config.train_data,
    dataset_service=data_service_dispatcher,
    mode=recap_config_mod.JobMode.TRAIN,
    compression=config.train_data.dataset_service_compression,
    vocab_mapper=None,
    repeat=True,
  )

  train_iterator = iter(train_dataset.to_dataloader())

  torch_element_spec = train_dataset.torch_element_spec

  model = model_mod.create_ranking_model(
    data_spec=torch_element_spec[0],
    config=config,
    loss_fn=loss_fn,
    device=device,
  )

  optimizer, scheduler = optimizer_mod.build_optimizer(model, config.optimizer, None)

  model = maybe_shard_model(model, device)

  datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
  print(f"{datetime_str}\n", end="")

  if FLAGS.debug_loop:
    logging.warning("Running debug mode, slow!")
    train_mod = debug_training_loop
  else:
    train_mod = ctl

  train_mod.train(
    model=model,
    optimizer=optimizer,
    device=device,
    save_dir=config.training.save_dir,
    logging_interval=config.training.train_log_every_n,
    train_steps=config.training.num_train_steps,
    checkpoint_frequency=config.training.checkpoint_every_n,
    dataset=train_iterator,
    worker_batch_size=config.train_data.global_batch_size,
    enable_amp=False,
    initial_checkpoint_dir=config.training.initial_checkpoint_dir,
    gradient_accumulation=config.training.gradient_accumulation,
    scheduler=scheduler,
  )


if __name__ == "__main__":
  app.run(run)
