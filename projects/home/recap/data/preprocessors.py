"""
Preprocessors applied on DDS workers in order to modify the dataset on the fly.
Some of these preprocessors are also applied to the model at serving time.
"""
from tml.projects.home.recap import config as config_mod
from absl import logging
import tensorflow as tf
import numpy as np


class TruncateAndSlice(tf.keras.Model):
  """Class for truncating and slicing."""

  def __init__(self, truncate_and_slice_config):
    super().__init__()
    self._truncate_and_slice_config = truncate_and_slice_config

    if self._truncate_and_slice_config.continuous_feature_mask_path:
      with tf.io.gfile.GFile(
        self._truncate_and_slice_config.continuous_feature_mask_path, "rb"
      ) as f:
        self._continuous_mask = np.load(f).nonzero()[0]
      logging.info(f"Slicing {np.sum(self._continuous_mask)} continuous features.")
    else:
      self._continuous_mask = None

    if self._truncate_and_slice_config.binary_feature_mask_path:
      with tf.io.gfile.GFile(self._truncate_and_slice_config.binary_feature_mask_path, "rb") as f:
        self._binary_mask = np.load(f).nonzero()[0]
      logging.info(f"Slicing {np.sum(self._binary_mask)} binary features.")
    else:
      self._binary_mask = None

  def call(self, inputs, training=None, mask=None):
    outputs = tf.nest.pack_sequence_as(inputs, tf.nest.flatten(inputs))
    if self._truncate_and_slice_config.continuous_feature_truncation:
      logging.info("Truncating continuous")
      outputs["continuous"] = outputs["continuous"][
        :, : self._truncate_and_slice_config.continuous_feature_truncation
      ]
    if self._truncate_and_slice_config.binary_feature_truncation:
      logging.info("Truncating binary")
      outputs["binary"] = outputs["binary"][
        :, : self._truncate_and_slice_config.binary_feature_truncation
      ]
    if self._continuous_mask is not None:
      outputs["continuous"] = tf.gather(outputs["continuous"], self._continuous_mask, axis=1)
    if self._binary_mask is not None:
      outputs["binary"] = tf.gather(outputs["binary"], self._binary_mask, axis=1)
    return outputs


class DownCast(tf.keras.Model):
  """Class for Down casting dataset before serialization and transferring to training host.
  Depends on the data type and the actual data range, the down casting can be lossless or not.
  It is strongly recommended to compare the metrics before and after down casting.
  """

  def __init__(self, downcast_config):
    super().__init__()
    self.config = downcast_config
    self._type_map = {
      "bfloat16": tf.bfloat16,
      "bool": tf.bool,
    }

  def call(self, inputs, training=None, mask=None):
    outputs = tf.nest.pack_sequence_as(inputs, tf.nest.flatten(inputs))
    for feature, type_str in self.config.features.items():
      assert type_str in self._type_map
      if type_str == "bfloat16":
        logging.warning(
          "Although bfloat16 and float32 have the same number of exponent bits, this down casting is not 100% lossless. Please double check metrics."
        )
      down_cast_data_type = self._type_map[type_str]
      outputs[feature] = tf.cast(outputs[feature], dtype=down_cast_data_type)
    return outputs


class RectifyLabels(tf.keras.Model):
  """Class for rectifying labels"""

  def __init__(self, rectify_label_config):
    super().__init__()
    self._config = rectify_label_config
    self._window = int(self._config.label_rectification_window_in_hours * 60 * 60 * 1000)

  def call(self, inputs, training=None, mask=None):
    served_ts_field = self._config.served_timestamp_field
    impressed_ts_field = self._config.impressed_timestamp_field

    for label, engaged_ts_field in self._config.label_to_engaged_timestamp_field.items():
      impressed = inputs[impressed_ts_field]
      served = inputs[served_ts_field]
      engaged = inputs[engaged_ts_field]

      keep = tf.math.logical_and(inputs[label] > 0, impressed - served < self._window)
      keep = tf.math.logical_and(keep, engaged - served < self._window)
      inputs[label] = tf.where(keep, inputs[label], tf.zeros_like(inputs[label]))

    return inputs


class ExtractFeatures(tf.keras.Model):
  """Class for extracting individual features from dense tensors by their index."""

  def __init__(self, extract_features_config):
    super().__init__()
    self._config = extract_features_config

  def call(self, inputs, training=None, mask=None):

    for row in self._config.extract_feature_table:
      inputs[row.name] = inputs[row.source_tensor][:, row.index]

    return inputs


class DownsampleNegatives(tf.keras.Model):
  """Class for down-sampling/dropping negatives and updating the weights.

  If inputs['fav'] = [1, 0, 0, 0] and inputs['weights'] = [1.0, 1.0, 1.0, 1.0]
  inputs are transformed to inputs['fav'] = [1, 0] and inputs['weights'] = [1.0, 3.0]
  when batch_multiplier=2 and engagements_list=['fav']

  It supports multiple engagements (union/logical_or is used to aggregate engagements), so we don't
  drop positives for any engagement.
  """

  def __init__(self, downsample_negatives_config):
    super().__init__()
    self.config = downsample_negatives_config

  def call(self, inputs, training=None, mask=None):
    labels = self.config.engagements_list
    # union of engagements
    mask = tf.squeeze(tf.reduce_any(tf.stack([inputs[label] == 1 for label in labels], 1), 1))
    n_positives = tf.reduce_sum(tf.cast(mask, tf.int32))
    batch_size = tf.cast(tf.shape(inputs[labels[0]])[0] / self.config.batch_multiplier, tf.int32)
    negative_weights = tf.math.divide_no_nan(
      tf.cast(self.config.batch_multiplier * batch_size - n_positives, tf.float32),
      tf.cast(batch_size - n_positives, tf.float32),
    )
    new_weights = tf.cast(mask, tf.float32) + (1 - tf.cast(mask, tf.float32)) * negative_weights

    def _split_by_label_concatenate_and_truncate(input_tensor):
      # takes positive examples and concatenate with negative examples and truncate
      # DANGER: if n_positives > batch_size down-sampling is incorrect (do not use pb_50)
      return tf.concat(
        [
          input_tensor[mask],
          input_tensor[tf.math.logical_not(mask)],
        ],
        0,
      )[:batch_size]

    if "weights" not in inputs:
      # add placeholder so logic below applies even if weights aren't present in inputs
      inputs["weights"] = tf.ones([tf.shape(inputs[labels[0]])[0], self.config.num_engagements])

    for tensor in inputs:
      if tensor == "weights":
        inputs[tensor] = inputs[tensor] * tf.reshape(new_weights, [-1, 1])

      inputs[tensor] = _split_by_label_concatenate_and_truncate(inputs[tensor])

    return inputs


def build_preprocess(preprocess_config, mode=config_mod.JobMode.TRAIN):
  """Builds a preprocess model to apply all preprocessing stages."""
  if mode == config_mod.JobMode.INFERENCE:
    logging.info("Not building preprocessors for dataloading since we are in Inference mode.")
    return None

  preprocess_models = []
  if preprocess_config.downsample_negatives:
    preprocess_models.append(DownsampleNegatives(preprocess_config.downsample_negatives))
  if preprocess_config.truncate_and_slice:
    preprocess_models.append(TruncateAndSlice(preprocess_config.truncate_and_slice))
  if preprocess_config.downcast:
    preprocess_models.append(DownCast(preprocess_config.downcast))
  if preprocess_config.rectify_labels:
    preprocess_models.append(RectifyLabels(preprocess_config.rectify_labels))
  if preprocess_config.extract_features:
    preprocess_models.append(ExtractFeatures(preprocess_config.extract_features))

  if len(preprocess_models) == 0:
    raise ValueError("No known preprocessor.")

  class PreprocessModel(tf.keras.Model):
    def __init__(self, preprocess_models):
      super().__init__()
      self.preprocess_models = preprocess_models

    def call(self, inputs, training=None, mask=None):
      outputs = inputs
      for model in self.preprocess_models:
        outputs = model(outputs, training, mask)
      return outputs

  if len(preprocess_models) > 1:
    logging.warning(
      "With multiple preprocessing models, we apply these models in a predefined order. Future works may introduce customized models and orders."
    )
  return PreprocessModel(preprocess_models)
