import os
import json
from absl import app, flags, logging
import tensorflow as tf
from typing import Dict

from tml.projects.home.recap.data import tfe_parsing
from tml.core import config as tml_config_mod
import tml.projects.home.recap.config as recap_config_mod

flags.DEFINE_string("config_path", None, "Path to hyperparameters for model.")
flags.DEFINE_integer("n_examples", 100, "Numer of examples to generate.")

FLAGS = flags.FLAGS


def _generate_random_example(
  tf_example_schema: Dict[str, tf.io.FixedLenFeature]
) -> Dict[str, tf.Tensor]:
  """
    Generate a random example based on the provided TensorFlow example schema.

    Args:
        tf_example_schema (Dict[str, tf.io.FixedLenFeature]): A dictionary defining the schema of the TensorFlow example.

    Returns:
        Dict[str, tf.Tensor]: A dictionary containing random data for each feature defined in the schema.
    """

  example = {}
  for feature_name, feature_spec in tf_example_schema.items():
    dtype = feature_spec.dtype
    if (dtype == tf.int64) or (dtype == tf.int32):
      x = tf.experimental.numpy.random.randint(0, high=10, size=feature_spec.shape, dtype=dtype)
    elif (dtype == tf.float32) or (dtype == tf.float64):
      x = tf.random.uniform(shape=[feature_spec.shape], dtype=dtype)
    else:
      raise NotImplementedError(f"Unknown type {dtype}")

    example[feature_name] = x

  return example


def _float_feature(value):
  """
    Create a TensorFlow float feature.

    Args:
        value: A float or list of floats.

    Returns:
        tf.train.Feature: A TensorFlow feature containing the float value(s).
    """

  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """
    Create a TensorFlow int64 feature.

    Args:
        value: An integer or list of integers.

    Returns:
        tf.train.Feature: A TensorFlow feature containing the int64 value(s).
    """

  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _serialize_example(x: Dict[str, tf.Tensor]) -> bytes:
  """
    Serialize a dictionary of TensorFlow tensors into a binary string.

    Args:
        x (Dict[str, tf.Tensor]): A dictionary of TensorFlow tensors.

    Returns:
        bytes: The serialized binary string.
    """
  feature = {}
  serializers = {tf.float32: _float_feature, tf.int64: _int64_feature}
  for feature_name, tensor in x.items():
    feature[feature_name] = serializers[tensor.dtype](tensor)

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def generate_data(data_path: str, config: recap_config_mod.RecapConfig):
  """
    Generate random data based on the provided configuration and save it as a TFRecord file.

    Args:
        data_path (str): The path where the TFRecord file will be saved.
        config (recap_config_mod.RecapConfig): The configuration for generating the random data.
    """


  with tf.io.gfile.GFile(config.train_data.seg_dense_schema.schema_path, "r") as f:
    seg_dense_schema = json.load(f)["schema"]

  tf_example_schema = tfe_parsing.create_tf_example_schema(
    config.train_data,
    seg_dense_schema,
  )

  record_filename = os.path.join(data_path, "random.tfrecord.gz")

  with tf.io.TFRecordWriter(record_filename, "GZIP") as writer:
    random_example = _generate_random_example(tf_example_schema)
    serialized_example = _serialize_example(random_example)
    writer.write(serialized_example)


def _generate_data_main(unused_argv):
  """
    Main function to generate random data according to the provided configuration.

    Args:
        unused_argv: Unused command-line arguments.
    """
  config = tml_config_mod.load_config_from_yaml(recap_config_mod.RecapConfig, FLAGS.config_path)

  # Find the path where to put the data
  data_path = os.path.dirname(config.train_data.inputs)
  logging.info("Putting random data in %s", data_path)

  generate_data(data_path, config)


if __name__ == "__main__":
  app.run(_generate_data_main)
