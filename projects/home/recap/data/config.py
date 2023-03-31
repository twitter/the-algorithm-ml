import typing
from enum import Enum


from tml.core import config as base_config

import pydantic


class ExplicitDateInputs(base_config.BaseConfig):
  """Arguments to select train/validation data using end_date and days of data."""

  data_root: str = pydantic.Field(..., description="Data path prefix.")
  end_date: str = pydantic.Field(..., description="Data end date, inclusive.")
  days: int = pydantic.Field(..., description="Number of days of data for dataset.")
  num_missing_days_tol: int = pydantic.Field(
    0, description="We tolerate <= num_missing_days_tol days of missing data."
  )


class ExplicitDatetimeInputs(base_config.BaseConfig):
  """Arguments to select train/validation data using end_datetime and hours of data."""

  data_root: str = pydantic.Field(..., description="Data path prefix.")
  end_datetime: str = pydantic.Field(..., description="Data end datetime, inclusive.")
  hours: int = pydantic.Field(..., description="Number of hours of data for dataset.")
  num_missing_hours_tol: int = pydantic.Field(
    0, description="We tolerate <= num_missing_hours_tol hours of missing data."
  )


class DdsCompressionOption(str, Enum):
  """The only valid compression option is 'AUTO'"""

  AUTO = "AUTO"


class DatasetConfig(base_config.BaseConfig):
  inputs: str = pydantic.Field(
    None, description="A glob for selecting data.", one_of="date_inputs_format"
  )
  explicit_datetime_inputs: ExplicitDatetimeInputs = pydantic.Field(
    None, one_of="date_inputs_format"
  )
  explicit_date_inputs: ExplicitDateInputs = pydantic.Field(None, one_of="date_inputs_format")

  global_batch_size: pydantic.PositiveInt

  num_files_to_keep: pydantic.PositiveInt = pydantic.Field(
    None, description="Number of shards to keep."
  )
  repeat_files: bool = pydantic.Field(
    True, description="DEPRICATED. Files are repeated no matter what this is set to."
  )
  file_batch_size: pydantic.PositiveInt = pydantic.Field(16, description="File batch size")

  cache: bool = pydantic.Field(
    False,
    description="Cache dataset in memory. Careful to only use this when you"
    " have enough memory to fit entire dataset.",
  )

  data_service_dispatcher: str = pydantic.Field(None)
  ignore_data_errors: bool = pydantic.Field(
    False, description="Whether to ignore tf.data errors. DANGER DANGER, may wedge jobs."
  )
  dataset_service_compression: DdsCompressionOption = pydantic.Field(
    None,
    description="Compress the dataset for DDS worker -> training host. Disabled by default and the only valid option is 'AUTO'",
  )

  # tf.data.Dataset options
  examples_shuffle_buffer_size: int = pydantic.Field(1024, description="Size of shuffle buffers.")
  map_num_parallel_calls: pydantic.PositiveInt = pydantic.Field(
    None, description="Number of parallel calls."
  )
  interleave_num_parallel_calls: pydantic.PositiveInt = pydantic.Field(
    None, description="Number of shards to interleave."
  )


class TruncateAndSlice(base_config.BaseConfig):
  # Apply truncation and then slice.
  continuous_feature_truncation: pydantic.PositiveInt = pydantic.Field(
    None, description="Experimental. Truncates continuous features to this amount for efficiency."
  )
  binary_feature_truncation: pydantic.PositiveInt = pydantic.Field(
    None, description="Experimental. Truncates binary features to this amount for efficiency."
  )

  continuous_feature_mask_path: str = pydantic.Field(
    None, description="Path of mask used to slice input continuous features."
  )
  binary_feature_mask_path: str = pydantic.Field(
    None, description="Path of mask used to slice input binary features."
  )


class DataType(str, Enum):
  BFLOAT16 = "bfloat16"
  BOOL = "bool"

  FLOAT32 = "float32"
  FLOAT16 = "float16"

  UINT8 = "uint8"


class DownCast(base_config.BaseConfig):
  # Apply down casting to selected features.
  features: typing.Dict[str, DataType] = pydantic.Field(
    None, description="Map features to down cast data types."
  )


class TaskData(base_config.BaseConfig):
  pos_downsampling_rate: float = pydantic.Field(
    1.0,
    description="Downsampling rate of positives used to generate dataset.",
  )
  neg_downsampling_rate: float = pydantic.Field(
    1.0,
    description="Downsampling rate of negatives used to generate dataset.",
  )


class SegDenseSchema(base_config.BaseConfig):
  schema_path: str = pydantic.Field(..., description="Path to feature config json.")
  features: typing.List[str] = pydantic.Field(
    [],
    description="List of features (in addition to the renamed features) to read from schema path above.",
  )
  renamed_features: typing.Dict[str, str] = pydantic.Field(
    {}, description="Dictionary of renamed features."
  )
  mask_mantissa_features: typing.Dict[str, int] = pydantic.Field(
    {},
    description="(experimental) Number of mantissa bits to mask to simulate lower precision data.",
  )


class RectifyLabels(base_config.BaseConfig):
  label_rectification_window_in_hours: float = pydantic.Field(
    3.0, description="overlap time in hours for which to flip labels"
  )
  served_timestamp_field: str = pydantic.Field(
    ..., description="input field corresponding to served time"
  )
  impressed_timestamp_field: str = pydantic.Field(
    ..., description="input field corresponding to impressed time"
  )
  label_to_engaged_timestamp_field: typing.Dict[str, str] = pydantic.Field(
    ..., description="label to the input field corresponding to engagement time"
  )


class ExtractFeaturesRow(base_config.BaseConfig):
  name: str = pydantic.Field(
    ...,
    description="name of the new field name to be created",
  )
  source_tensor: str = pydantic.Field(
    ...,
    description="name of the dense tensor to look for the feature",
  )
  index: int = pydantic.Field(
    ...,
    description="index of the feature in the dense tensor",
  )


class ExtractFeatures(base_config.BaseConfig):
  extract_feature_table: typing.List[ExtractFeaturesRow] = pydantic.Field(
    [],
    description="list of features to be extracted with their name, source tensor and index",
  )


class DownsampleNegatives(base_config.BaseConfig):
  batch_multiplier: int = pydantic.Field(
    None,
    description="batch multiplier",
  )
  engagements_list: typing.List[str] = pydantic.Field(
    [],
    description="engagements with kept positives",
  )
  num_engagements: int = pydantic.Field(
    ...,
    description="number engagements used in the model, including ones excluded in engagements_list",
  )


class Preprocess(base_config.BaseConfig):
  truncate_and_slice: TruncateAndSlice = pydantic.Field(None, description="Truncation and slicing.")
  downcast: DownCast = pydantic.Field(None, description="Down cast to features.")
  rectify_labels: RectifyLabels = pydantic.Field(
    None, description="Rectify labels for a given overlap window"
  )
  extract_features: ExtractFeatures = pydantic.Field(
    None, description="Extract features from dense tensors."
  )
  downsample_negatives: DownsampleNegatives = pydantic.Field(
    None, description="Downsample negatives."
  )


class Sampler(base_config.BaseConfig):
  """Assumes function is defined in data/samplers.py.

  Only use this for quick experimentation.
  If samplers are useful, we should sample from upstream data generation.

  DEPRICATED, DO NOT USE.
  """

  name: str
  kwargs: typing.Dict


class RecapDataConfig(DatasetConfig):
  seg_dense_schema: SegDenseSchema

  tasks: typing.Dict[str, TaskData] = pydantic.Field(
    description="Description of individual tasks in this dataset."
  )
  evaluation_tasks: typing.List[str] = pydantic.Field(
    [], description="If specified, lists the tasks we're generating metrics for."
  )

  preprocess: Preprocess = pydantic.Field(
    None, description="Function run in tf.data.Dataset at train/eval, in-graph at inference."
  )

  sampler: Sampler = pydantic.Field(
    None,
    description="""DEPRICATED, DO NOT USE. Sampling function for offline experiments.""",
  )

  @pydantic.root_validator()
  def _validate_evaluation_tasks(cls, values):
    if values.get("evaluation_tasks") is not None:
      for task in values["evaluation_tasks"]:
        if task not in values["tasks"]:
          raise KeyError(f"Evaluation task {task} must be in tasks. Received {values['tasks']}")
    return values
