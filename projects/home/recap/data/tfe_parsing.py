import functools
import json

from tml.projects.home.recap.data import config as recap_data_config

from absl import logging
import tensorflow as tf


DEFAULTS_MAP = {"int64_list": 0, "float_list": 0.0, "bytes_list": ""}
DTYPE_MAP = {"int64_list": tf.int64, "float_list": tf.float32, "bytes_list": tf.string}


def create_tf_example_schema(
  data_config: recap_data_config.SegDenseSchema,
  segdense_schema,
):
  """Generate schema for deseralizing tf.Example.

  Args:
    segdense_schema: List of dicts of segdense features (includes feature_name, dtype, length).
    labels: List of strings denoting labels.

  Returns:
    A dictionary schema suitable for deserializing tf.Example.
  """
  segdense_config = data_config.seg_dense_schema
  labels = list(data_config.tasks.keys())
  used_features = (
    segdense_config.features + list(segdense_config.renamed_features.values()) + labels
  )
  logging.info(used_features)

  tfe_schema = {}
  for entry in segdense_schema:
    feature_name = entry["feature_name"]

    if feature_name in used_features:
      length = entry["length"]
      dtype = entry["dtype"]

      if feature_name in labels:
        logging.info(f"Label: feature name is {feature_name} type is {dtype}")
        tfe_schema[feature_name] = tf.io.FixedLenFeature(
          length, DTYPE_MAP[dtype], DEFAULTS_MAP[dtype]
        )
      elif length == -1:
        tfe_schema[feature_name] = tf.io.VarLenFeature(DTYPE_MAP[dtype])
      else:
        tfe_schema[feature_name] = tf.io.FixedLenFeature(
          length, DTYPE_MAP[dtype], [DEFAULTS_MAP[dtype]] * length
        )
  for feature_name in used_features:
    if feature_name not in tfe_schema:
      raise ValueError(f"{feature_name} missing from schema: {segdense_config.schema_path}.")
  return tfe_schema


@functools.lru_cache(1)
def make_mantissa_mask(mask_length: int) -> tf.Tensor:
  """For experimentating with emulating bfloat16 or less precise types."""
  return tf.constant((1 << 32) - (1 << mask_length), dtype=tf.int32)


def mask_mantissa(tensor: tf.Tensor, mask_length: int) -> tf.Tensor:
  """For experimentating with emulating bfloat16 or less precise types."""
  mask: tf.Tensor = make_mantissa_mask(mask_length)
  return tf.bitcast(tf.bitwise.bitwise_and(tf.bitcast(tensor, tf.int32), mask), tensor.dtype)


def parse_tf_example(
  serialized_example,
  tfe_schema,
  seg_dense_schema_config,
):
  """Parse serialized tf.Example into dict of tensors.

  Args:
    serialized_example: Serialized tf.Example to be parsed.
    tfe_schema: Dictionary schema suitable for deserializing tf.Example.

  Returns:
    Dictionary of tensors to be used as model input.
  """
  inputs = tf.io.parse_example(serialized=serialized_example, features=tfe_schema)

  for new_feature_name, old_feature_name in seg_dense_schema_config.renamed_features.items():
    inputs[new_feature_name] = inputs.pop(old_feature_name)

  # This should not actually be used except for experimentation with low precision floats.
  if "mask_mantissa_features" in seg_dense_schema_config:
    for feature_name, mask_length in seg_dense_schema_config.mask_mantissa_features.items():
      inputs[feature_name] = mask_mantissa(inputs[feature_name], mask_length)

  # DANGER DANGER: This default seems really scary, and it's only here because it has to be visible
  # at TF level.
  # We should not return empty tensors if we dont use embeddings.
  # Otherwise, it breaks numpy->pt conversion
  renamed_keys = list(seg_dense_schema_config.renamed_features.keys())
  for renamed_key in renamed_keys:
    if "embedding" in renamed_key and (renamed_key not in inputs):
      inputs[renamed_key] = tf.zeros([], tf.float32)

  logging.info(f"parsed example and inputs are {inputs}")
  return inputs


def get_seg_dense_parse_fn(data_config: recap_data_config.RecapDataConfig):
  """Placeholder for seg dense.

  In the future, when we use more seg dense variations, we can change this.
  """
  with tf.io.gfile.GFile(data_config.seg_dense_schema.schema_path, "r") as f:
    seg_dense_schema = json.load(f)["schema"]

  tf_example_schema = create_tf_example_schema(
    data_config,
    seg_dense_schema,
  )

  logging.info("***** TF Example Schema *****")
  logging.info(tf_example_schema)

  parse = functools.partial(
    parse_tf_example,
    tfe_schema=tf_example_schema,
    seg_dense_schema_config=data_config.seg_dense_schema,
  )
  return parse
