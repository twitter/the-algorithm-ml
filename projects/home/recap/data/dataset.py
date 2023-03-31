from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import functools

import torch
import tensorflow as tf

from tml.common.batch import DataclassBatch
from tml.projects.home.recap.data.config import RecapDataConfig, TaskData
from tml.projects.home.recap.data import preprocessors
from tml.projects.home.recap.config import JobMode
from tml.projects.home.recap.data.tfe_parsing import get_seg_dense_parse_fn
from tml.projects.home.recap.data.util import (
  keyed_jagged_tensor_from_tensors_dict,
  sparse_or_dense_tf_to_torch,
)
from absl import logging
import torch.distributed as dist


@dataclass
class RecapBatch(DataclassBatch):
  """Holds features and labels from the Recap dataset."""

  continuous_features: torch.Tensor
  binary_features: torch.Tensor
  discrete_features: torch.Tensor
  sparse_features: "KeyedJaggedTensor"  # type: ignore[name-defined]  # noqa: F821
  labels: torch.Tensor
  user_embedding: torch.Tensor = None
  user_eng_embedding: torch.Tensor = None
  author_embedding: torch.Tensor = None
  weights: torch.Tensor = None

  def __post_init__(self):
    if self.weights is None:
      self.weights = torch.ones_like(self.labels)
    for feature_name, feature_value in self.as_dict().items():
      if ("embedding" in feature_name) and (feature_value is None):
        setattr(self, feature_name, torch.empty([0, 0]))


def to_batch(x, sparse_feature_names: Optional[List[str]] = None) -> RecapBatch:
  """Converts a torch data loader output into `RecapBatch`."""

  x = tf.nest.map_structure(functools.partial(sparse_or_dense_tf_to_torch, pin_memory=False), x)
  try:
    features_in, labels = x
  except ValueError:
    # For Mode.INFERENCE, we do not expect to recieve labels as part of the input tuple
    features_in, labels = x, None

  sparse_features = keyed_jagged_tensor_from_tensors_dict({})
  if sparse_feature_names:
    sparse_features = keyed_jagged_tensor_from_tensors_dict(
      {embedding_name: features_in[embedding_name] for embedding_name in sparse_feature_names}
    )

  user_embedding, user_eng_embedding, author_embedding = None, None, None
  if "user_embedding" in features_in:
    if sparse_feature_names and "meta__user_id" in sparse_feature_names:
      raise ValueError("Only one source of embedding for user is supported")
    else:
      user_embedding = features_in["user_embedding"]

  if "user_eng_embedding" in features_in:
    if sparse_feature_names and "meta__user_eng_id" in sparse_feature_names:
      raise ValueError("Only one source of embedding for user is supported")
    else:
      user_eng_embedding = features_in["user_eng_embedding"]

  if "author_embedding" in features_in:
    if sparse_feature_names and "meta__author_id" in sparse_feature_names:
      raise ValueError("Only one source of embedding for user is supported")
    else:
      author_embedding = features_in["author_embedding"]

  return RecapBatch(
    continuous_features=features_in["continuous"],
    binary_features=features_in["binary"],
    discrete_features=features_in["discrete"],
    sparse_features=sparse_features,
    user_embedding=user_embedding,
    user_eng_embedding=user_eng_embedding,
    author_embedding=author_embedding,
    labels=labels,
    weights=features_in.get("weights", None),  # Defaults to torch.ones_like(labels)
  )


def _chain(param, f1, f2):
  """
  Reduce multiple functions into one chained function
  _chain(x, f1, f2) -> f2(f1(x))
  """
  output = param
  fns = [f1, f2]
  for f in fns:
    output = f(output)
  return output


def _add_weights(inputs, tasks: Dict[str, TaskData]):
  """Adds weights based on label sampling for positive and negatives.

  This is useful for numeric calibration etc. This mutates inputs.

  Args:
    inputs: A dictionary of strings to tensor-like structures.
    tasks: A dict of string (label) to `TaskData` specifying inputs.

  Returns:
    A tuple of features and labels; weights are added to features.
  """

  weights = []
  for key, task in tasks.items():
    label = inputs[key]
    float_label = tf.cast(label, tf.float32)

    weights.append(
      float_label / task.pos_downsampling_rate + (1.0 - float_label) / task.neg_downsampling_rate
    )

  # Ensure we are batch-major (assumes we batch before this call).
  inputs["weights"] = tf.squeeze(tf.transpose(tf.convert_to_tensor(weights)), axis=0)
  return inputs


def get_datetimes(explicit_datetime_inputs):
  """Compute list datetime strings for train/validation data."""
  datetime_format = "%Y/%m/%d/%H"
  end = datetime.strptime(explicit_datetime_inputs.end_datetime, datetime_format)
  dates = sorted(
    [
      (end - timedelta(hours=i + 1)).strftime(datetime_format)
      for i in range(int(explicit_datetime_inputs.hours))
    ]
  )
  return dates


def get_explicit_datetime_inputs_files(explicit_datetime_inputs):
  """
  Compile list of files for training/validation.

  Used with DataConfigs that use the `explicit_datetime_inputs` format to specify data.
  For each hour of data, if the directory is missing or empty, we increment a counter to keep
  track of the number of missing data hours.
  Returns only files with a `.gz` extension.

  Args:
    explicit_datetime_inputs: An `ExplicitDatetimeInputs` object within a `datasets.DataConfig` object

  Returns:
    data_files: Sorted list of files to read corresponding to data at the desired datetimes
    num_hours_missing: Number of hours that we are missing data

  """
  datetimes = get_datetimes(explicit_datetime_inputs)
  folders = [os.path.join(explicit_datetime_inputs.data_root, datetime) for datetime in datetimes]
  data_files = []
  num_hours_missing = 0
  for folder in folders:
    try:
      files = tf.io.gfile.listdir(folder)
      if not files:
        logging.warning(f"{folder} contained no data files")
        num_hours_missing += 1
      data_files.extend(
        [
          os.path.join(folder, filename)
          for filename in files
          if filename.rsplit(".", 1)[-1].lower() == "gz"
        ]
      )
    except tf.errors.NotFoundError as e:
      num_hours_missing += 1
      logging.warning(f"Cannot find directory {folder}. Missing one hour of data. Error: \n {e}")
  return sorted(data_files), num_hours_missing


def _map_output_for_inference(
  inputs, tasks: Dict[str, TaskData], preprocessor: tf.keras.Model = None, add_weights: bool = False
):
  if preprocessor:
    raise ValueError("No preprocessor should be used at inference time.")
  if add_weights:
    raise NotImplementedError()

  # Add zero weights.
  inputs["weights"] = tf.zeros_like(tf.expand_dims(inputs["continuous"][:, 0], -1))
  for label in tasks:
    del inputs[label]
  return inputs


def _map_output_for_train_eval(
  inputs, tasks: Dict[str, TaskData], preprocessor: tf.keras.Model = None, add_weights: bool = False
):
  if add_weights:
    inputs = _add_weights_based_on_sampling_rates(inputs, tasks)

  # Warning this has to happen first as it changes the input
  if preprocessor:
    inputs = preprocessor(inputs)

  label_values = tf.squeeze(tf.stack([inputs[label] for label in tasks], axis=1), axis=[-1])

  for label in tasks:
    del inputs[label]

  return inputs, label_values


def _add_weights_based_on_sampling_rates(inputs, tasks: Dict[str, TaskData]):
  """Adds weights based on label sampling for positive and negatives.

  This is useful for numeric calibration etc. This mutates inputs.

  Args:
    inputs: A dictionary of strings to tensor-like structures.
    tasks: A dict of string (label) to `TaskData` specifying inputs.

  Returns:
    A tuple of features and labels; weights are added to features.
  """
  weights = []
  for key, task in tasks.items():
    label = inputs[key]
    float_label = tf.cast(label, tf.float32)

    weights.append(
      float_label / task.pos_downsampling_rate + (1.0 - float_label) / task.neg_downsampling_rate
    )

  # Ensure we are batch-major (assumes we batch before this call).
  inputs["weights"] = tf.squeeze(tf.transpose(tf.convert_to_tensor(weights)), axis=0)
  return inputs


class RecapDataset(torch.utils.data.IterableDataset):
  def __init__(
    self,
    data_config: RecapDataConfig,
    dataset_service: Optional[str] = None,
    mode: JobMode = JobMode.TRAIN,
    compression: Optional[str] = "AUTO",
    repeat: bool = False,
    vocab_mapper: tf.keras.Model = None,
  ):
    logging.info("***** Labels *****")
    logging.info(list(data_config.tasks.keys()))

    self._data_config = data_config
    self._parse_fn = get_seg_dense_parse_fn(data_config)
    self._mode = mode
    self._repeat = repeat
    self._num_concurrent_iterators = 1
    self._vocab_mapper = vocab_mapper
    self.dataset_service = dataset_service

    preprocessor = None
    self._batch_size_multiplier = 1
    if data_config.preprocess:
      preprocessor = preprocessors.build_preprocess(data_config.preprocess, mode=mode)
      if data_config.preprocess.downsample_negatives:
        self._batch_size_multiplier = data_config.preprocess.downsample_negatives.batch_multiplier

    self._preprocessor = preprocessor

    if mode == JobMode.INFERENCE:
      if preprocessor is not None:
        raise ValueError("Expect no preprocessor at inference time.")
      should_add_weights = False
      output_map_fn = _map_output_for_inference  # (features,)
    else:
      # Only add weights if there is a reason to! If all weights will
      # be equal to 1.0, save bandwidth between DDS and Chief by simply
      # relying on the fact that weights default to 1.0 in `RecapBatch`
      # WARNING: Weights may still be added as a side effect of a preprocessor
      #          such as `DownsampleNegatives`.
      should_add_weights = any(
        [
          task_cfg.pos_downsampling_rate != 1.0 or task_cfg.neg_downsampling_rate != 1.0
          for task_cfg in data_config.tasks.values()
        ]
      )
      output_map_fn = _map_output_for_train_eval  # (features, labels)

    self._output_map_fn = functools.partial(
      output_map_fn,
      tasks=data_config.tasks,
      preprocessor=preprocessor,
      add_weights=should_add_weights,
    )

    sparse_feature_names = list(vocab_mapper.vocabs.keys()) if vocab_mapper else None

    self._tf_dataset = self._create_tf_dataset()

    self._init_tensor_spec()

  def _init_tensor_spec(self):
    def _tensor_spec_to_torch_shape(spec):
      if spec.shape is None:
        return None
      shape = [x if x is not None else -1 for x in spec.shape]
      return torch.Size(shape)

    self.torch_element_spec = tf.nest.map_structure(
      _tensor_spec_to_torch_shape, self._tf_dataset.element_spec
    )

  def _create_tf_dataset(self):
    if hasattr(self, "_tf_dataset"):
      raise ValueError("Do not call `_create_tf_dataset` more than once.")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_replica_bsz = (
      self._batch_size_multiplier * self._data_config.global_batch_size // world_size
    )

    dataset: tf.data.Dataset = self._create_base_tf_dataset(
      batch_size=per_replica_bsz,
    )

    if self._repeat:
      logging.info("Repeating dataset")
      dataset = dataset.repeat()

    if self.dataset_service:
      if self._num_concurrent_iterators > 1:
        if not self.machines_config:
          raise ValueError(
            "Must supply a machine_config for autotuning in order to use >1 concurrent iterators"
          )
        dataset = dataset_lib.with_auto_tune_budget(
          dataset,
          machine_config=self.machines_config.chief,
          num_concurrent_iterators=self.num_concurrent_iterators,
          on_chief=False,
        )

      self.dataset_id, self.job_name = register_dataset(
        dataset=dataset, dataset_service=self.dataset_service, compression=self.compression
      )
      dataset = distribute_from_dataset_id(
        dataset_id=self.dataset_id,  # type: ignore[arg-type]
        job_name=self.job_name,
        dataset_service=self.dataset_service,
        compression=self.compression,
      )

    elif self._num_concurrent_iterators > 1:
      if not self.machines_config:
        raise ValueError(
          "Must supply a machine_config for autotuning in order to use >1 concurrent iterators"
        )
      dataset = dataset_lib.with_auto_tune_budget(
        dataset,
        machine_config=self.machines_config.chief,
        num_concurrent_iterators=self._num_concurrent_iterators,
        on_chief=True,
      )

    # Vocabulary mapping happens on the training node, not in dds because of size.
    if self._vocab_mapper:
      dataset = dataset.map(self._vocab_mapper)

    return dataset.prefetch(world_size * 2)

  def _create_base_tf_dataset(self, batch_size: int):
    if self._data_config.inputs:
      glob = self._data_config.inputs
      filenames = sorted(tf.io.gfile.glob(glob))
    elif self._data_config.explicit_datetime_inputs:
      num_missing_hours_tol = self._data_config.explicit_datetime_inputs.num_missing_hours_tol
      filenames, num_hours_missing = get_explicit_datetime_inputs_files(
        self._data_config.explicit_datetime_inputs,
        increment="hourly",
      )
      if num_hours_missing > num_missing_hours_tol:
        raise ValueError(
          f"We are missing {num_hours_missing} hours of data"
          f"more than tolerance {num_missing_hours_tol}."
        )
    elif self._data_config.explicit_date_inputs:
      num_missing_days_tol = self._data_config.explicit_date_inputs.num_missing_days_tol
      filenames, num_days_missing = get_explicit_datetime_inputs_files(
        self._data_config.explicit_date_inputs,
        increment="daily",
      )
      if num_days_missing > num_missing_days_tol:
        raise ValueError(
          f"We are missing {num_days_missing} days of data"
          f"more than tolerance {num_missing_days_tol}."
        )
    else:
      raise ValueError(
        "Must specifiy either `inputs`, `explicit_datetime_inputs`, or `explicit_date_inputs` in data_config"
      )

    num_files = len(filenames)
    logging.info(f"Found {num_files} data files")
    if num_files < 1:
      raise ValueError("No data files found")

    if self._data_config.num_files_to_keep is not None:
      filenames = filenames[: self._data_config.num_files_to_keep]
      logging.info(f"Retaining only {len(filenames)} files.")

    filenames_ds = (
      tf.data.Dataset.from_tensor_slices(filenames).shuffle(len(filenames))
      # Because of drop_remainder, if our dataset does not fill
      # up a batch, it will emit nothing without this repeat.
      .repeat(-1)
    )

    if self._data_config.file_batch_size:
      filenames_ds = filenames_ds.batch(self._data_config.file_batch_size)

    def per_shard_dataset(filename):
      ds = tf.data.TFRecordDataset([filename], compression_type="GZIP")
      return ds.prefetch(4)

    ds = filenames_ds.interleave(
      per_shard_dataset,
      block_length=4,
      deterministic=False,
      num_parallel_calls=self._data_config.interleave_num_parallel_calls
      or tf.data.experimental.AUTOTUNE,
    )

    # Combine functions into one map call to reduce overhead.
    map_fn = functools.partial(
      _chain,
      f1=self._parse_fn,
      f2=self._output_map_fn,
    )

    # Shuffle -> Batch -> Parse is the correct ordering
    # Shuffling needs to be performed before batching otherwise there is not much point
    # Batching happens before parsing because tf.Example parsing is actually vectorized
    #     and works much faster overall on batches of data.
    ds = (
      # DANGER DANGER: there is a default shuffle size here.
      ds.shuffle(self._data_config.examples_shuffle_buffer_size)
      .batch(batch_size=batch_size, drop_remainder=True)
      .map(
        map_fn,
        num_parallel_calls=self._data_config.map_num_parallel_calls
        or tf.data.experimental.AUTOTUNE,
      )
    )

    if self._data_config.cache:
      ds = ds.cache()

    if self._data_config.ignore_data_errors:
      ds = ds.apply(tf.data.experimental.ignore_errors())

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    return ds

  def _gen(self):
    for x in self._tf_dataset:
      yield to_batch(x)

  def to_dataloader(self) -> Dict[str, torch.Tensor]:
    return torch.utils.data.DataLoader(self, batch_size=None)

  def __iter__(self):
    return iter(self._gen())
