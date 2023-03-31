"""
Taken from https://raw.githubusercontent.com/pytorch/torchrec/v0.3.2/torchrec/distributed/train_pipeline.py
with TrainPipelineSparseDist.progress modified to support gradient accumulation.

"""
import abc
from dataclasses import dataclass, field
import logging
from typing import (
  Any,
  cast,
  Dict,
  Generic,
  Iterator,
  List,
  Optional,
  Set,
  Tuple,
  TypeVar,
)

import torch
from torch.autograd.profiler import record_function
from torch.fx.node import Node
from torchrec.distributed.model_parallel import (
  DistributedModelParallel,
  ShardedModule,
)
from torchrec.distributed.types import Awaitable
from torchrec.modules.feature_processor import BaseGroupedFeatureProcessor
from torchrec.streamable import Multistreamable, Pipelineable


logger: logging.Logger = logging.getLogger(__name__)


In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")


class TrainPipeline(abc.ABC, Generic[In, Out]):
  @abc.abstractmethod
  def progress(self, dataloader_iter: Iterator[In]) -> Out:
    pass


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
  assert isinstance(
    batch, (torch.Tensor, Pipelineable)
  ), f"{type(batch)} must implement Pipelineable interface"
  return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.cuda.streams.Stream]) -> None:
  if stream is None:
    return
  torch.cuda.current_stream().wait_stream(stream)
  # As mentioned in https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html,
  # PyTorch uses the "caching allocator" for memory allocation for tensors. When a tensor is
  # freed, its memory is likely to be reused by newly constructed tenosrs.  By default,
  # this allocator traces whether a tensor is still in use by only the CUDA stream where it
  # was created.   When a tensor is used by additional CUDA streams, we need to call record_stream
  # to tell the allocator about all these streams.  Otherwise, the allocator might free the
  # underlying memory of the tensor once it is no longer used by the creator stream.  This is
  # a notable programming trick when we write programs using multi CUDA streams.
  cur_stream = torch.cuda.current_stream()
  assert isinstance(
    batch, (torch.Tensor, Multistreamable)
  ), f"{type(batch)} must implement Multistreamable interface"
  batch.record_stream(cur_stream)


class TrainPipelineBase(TrainPipeline[In, Out]):
  """
  This class runs training iterations using a pipeline of two stages, each as a CUDA
  stream, namely, the current (default) stream and `self._memcpy_stream`. For each
  iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
  memory, and the default stream runs forward, backward, and optimization.
  """

  def __init__(
    self,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
  ) -> None:
    self._model = model
    self._optimizer = optimizer
    self._device = device
    self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
      torch.cuda.Stream() if device.type == "cuda" else None
    )
    self._cur_batch: Optional[In] = None
    self._connected = False

  def _connect(self, dataloader_iter: Iterator[In]) -> None:
    cur_batch = next(dataloader_iter)
    self._cur_batch = cur_batch
    with torch.cuda.stream(self._memcpy_stream):
      self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
    self._connected = True

  def progress(self, dataloader_iter: Iterator[In]) -> Out:
    if not self._connected:
      self._connect(dataloader_iter)

    # Fetch next batch
    with record_function("## next_batch ##"):
      next_batch = next(dataloader_iter)
    cur_batch = self._cur_batch
    assert cur_batch is not None

    if self._model.training:
      with record_function("## zero_grad ##"):
        self._optimizer.zero_grad()

    with record_function("## wait_for_batch ##"):
      _wait_for_batch(cur_batch, self._memcpy_stream)

    with record_function("## forward ##"):
      (losses, output) = self._model(cur_batch)

    if self._model.training:
      with record_function("## backward ##"):
        torch.sum(losses, dim=0).backward()

    # Copy the next batch to GPU
    self._cur_batch = cur_batch = next_batch
    with record_function("## copy_batch_to_gpu ##"):
      with torch.cuda.stream(self._memcpy_stream):
        self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)

    # Update
    if self._model.training:
      with record_function("## optimizer ##"):
        self._optimizer.step()

    return output


class Tracer(torch.fx.Tracer):
  # Disable proxying buffers during tracing. Ideally, proxying buffers would
  # be disabled, but some models are currently mutating buffer values, which
  # causes errors during tracing. If those models can be rewritten to not do
  # that, we can likely remove this line
  proxy_buffer_attributes = False

  def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
    super().__init__()
    self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

  def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
    if isinstance(m, ShardedModule) or module_qualified_name in self._leaf_modules:
      return True
    return super().is_leaf_module(m, module_qualified_name)


@dataclass
class TrainPipelineContext:
  # pyre-ignore [4]
  input_dist_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
  module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
  # pyre-ignore [4]
  feature_processor_forwards: List[Any] = field(default_factory=list)


@dataclass
class ArgInfo:
  # attributes of input batch, e.g. batch.attr1.attr2 call
  # will produce ["attr1", "attr2"]
  input_attrs: List[str]
  # batch[attr1].attr2 will produce [True, False]
  is_getitems: List[bool]
  # name for kwarg of pipelined forward() call or None
  # for a positional arg
  name: Optional[str]


class PipelinedForward:
  def __init__(
    self,
    name: str,
    args: List[ArgInfo],
    module: ShardedModule,
    context: TrainPipelineContext,
    dist_stream: Optional[torch.cuda.streams.Stream],
  ) -> None:
    self._name = name
    self._args = args
    self._module = module
    self._context = context
    self._dist_stream = dist_stream

  # pyre-ignore [2, 24]
  def __call__(self, *input, **kwargs) -> Awaitable:
    assert self._name in self._context.input_dist_requests
    request = self._context.input_dist_requests[self._name]
    assert isinstance(request, Awaitable)
    with record_function("## wait_sparse_data_dist ##"):
      # Finish waiting on the dist_stream,
      # in case some delayed stream scheduling happens during the wait() call.
      with torch.cuda.stream(self._dist_stream):
        data = request.wait()

    # Make sure that both result of input_dist and context
    # are properly transferred to the current stream.
    if self._dist_stream is not None:
      torch.cuda.current_stream().wait_stream(self._dist_stream)
      cur_stream = torch.cuda.current_stream()

      assert isinstance(
        data, (torch.Tensor, Multistreamable)
      ), f"{type(data)} must implement Multistreamable interface"
      # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
      data.record_stream(cur_stream)

      ctx = self._context.module_contexts[self._name]
      ctx.record_stream(cur_stream)

    if len(self._context.feature_processor_forwards) > 0:
      with record_function("## feature_processor ##"):
        for sparse_feature in data:
          if sparse_feature.id_score_list_features is not None:
            for fp_forward in self._context.feature_processor_forwards:
              sparse_feature.id_score_list_features = fp_forward(
                sparse_feature.id_score_list_features
              )

    return self._module.compute_and_output_dist(self._context.module_contexts[self._name], data)

  @property
  def name(self) -> str:
    return self._name

  @property
  def args(self) -> List[ArgInfo]:
    return self._args


def _start_data_dist(
  pipelined_modules: List[ShardedModule],
  batch: In,
  context: TrainPipelineContext,
) -> None:
  context.input_dist_requests.clear()
  context.module_contexts.clear()
  for module in pipelined_modules:
    forward = module.forward
    assert isinstance(forward, PipelinedForward)

    # Retrieve argument for the input_dist of EBC
    # is_getitem True means this argument could be retrieved by a list
    # False means this argument is getting while getattr
    # and this info was done in the _rewrite_model by tracing the
    # entire model to get the arg_info_list
    args = []
    kwargs = {}
    for arg_info in forward.args:
      if arg_info.input_attrs:
        arg = batch
        for attr, is_getitem in zip(arg_info.input_attrs, arg_info.is_getitems):
          if is_getitem:
            arg = arg[attr]
          else:
            arg = getattr(arg, attr)
        if arg_info.name:
          kwargs[arg_info.name] = arg
        else:
          args.append(arg)
      else:
        args.append(None)
    # Start input distribution.
    module_ctx = module.create_context()
    context.module_contexts[forward.name] = module_ctx
    context.input_dist_requests[forward.name] = module.input_dist(module_ctx, *args, **kwargs)

  # Call wait on the first awaitable in the input dist for the tensor splits
  for key, awaitable in context.input_dist_requests.items():
    context.input_dist_requests[key] = awaitable.wait()


def _get_node_args_helper(
  # pyre-ignore
  arguments,
  num_found: int,
  feature_processor_arguments: Optional[List[Node]] = None,
) -> Tuple[List[ArgInfo], int]:
  """
  Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
  It also counts the number of (args + kwargs) found.
  """

  arg_info_list = [ArgInfo([], [], None) for _ in range(len(arguments))]
  for arg, arg_info in zip(arguments, arg_info_list):
    if arg is None:
      num_found += 1
      continue
    while True:
      if not isinstance(arg, torch.fx.Node):
        break
      child_node = arg

      if child_node.op == "placeholder":
        num_found += 1
        break
      # skip this fp node
      elif feature_processor_arguments is not None and child_node in feature_processor_arguments:
        arg = child_node.args[0]
      elif (
        child_node.op == "call_function"
        and child_node.target.__module__ == "builtins"
        # pyre-ignore[16]
        and child_node.target.__name__ == "getattr"
      ):
        arg_info.input_attrs.insert(0, child_node.args[1])
        arg_info.is_getitems.insert(0, False)
        arg = child_node.args[0]
      elif (
        child_node.op == "call_function"
        and child_node.target.__module__ == "_operator"
        # pyre-ignore[16]
        and child_node.target.__name__ == "getitem"
      ):
        arg_info.input_attrs.insert(0, child_node.args[1])
        arg_info.is_getitems.insert(0, True)
        arg = child_node.args[0]
      else:
        break
  return arg_info_list, num_found


def _get_node_args(
  node: Node, feature_processor_nodes: Optional[List[Node]] = None
) -> Tuple[List[ArgInfo], int]:
  num_found = 0
  pos_arg_info_list, num_found = _get_node_args_helper(
    node.args, num_found, feature_processor_nodes
  )
  kwargs_arg_info_list, num_found = _get_node_args_helper(node.kwargs.values(), num_found)

  # Replace with proper names for kwargs
  for name, arg_info_list in zip(node.kwargs, kwargs_arg_info_list):
    arg_info_list.name = name

  arg_info_list = pos_arg_info_list + kwargs_arg_info_list
  return arg_info_list, num_found


def _get_unsharded_module_names_helper(
  model: torch.nn.Module,
  path: str,
  unsharded_module_names: Set[str],
) -> bool:
  sharded_children = set()
  for name, child in model.named_children():
    curr_path = path + name
    if isinstance(child, ShardedModule):
      sharded_children.add(name)
    else:
      child_sharded = _get_unsharded_module_names_helper(
        child,
        curr_path + ".",
        unsharded_module_names,
      )
      if child_sharded:
        sharded_children.add(name)

  if len(sharded_children) > 0:
    for name, _ in model.named_children():
      if name not in sharded_children:
        unsharded_module_names.add(path + name)

  return len(sharded_children) > 0


def _get_unsharded_module_names(model: torch.nn.Module) -> List[str]:
  """
  Returns a list of top level modules do not contain any sharded sub modules.
  """

  unsharded_module_names: Set[str] = set()
  _get_unsharded_module_names_helper(
    model,
    "",
    unsharded_module_names,
  )
  return list(unsharded_module_names)


def _rewrite_model(  # noqa C901
  model: torch.nn.Module,
  context: TrainPipelineContext,
  dist_stream: Optional[torch.cuda.streams.Stream],
) -> List[ShardedModule]:

  # Get underlying nn.Module
  if isinstance(model, DistributedModelParallel):
    model = model.module

  # Collect a list of sharded modules.
  sharded_modules = {}
  fp_modules = {}
  for name, m in model.named_modules():
    if isinstance(m, ShardedModule):
      sharded_modules[name] = m
    if isinstance(m, BaseGroupedFeatureProcessor):
      fp_modules[name] = m

  # Trace a model.
  tracer = Tracer(leaf_modules=_get_unsharded_module_names(model))
  graph = tracer.trace(model)

  feature_processor_nodes = []
  # find the fp node
  for node in graph.nodes:
    if node.op == "call_module" and node.target in fp_modules:
      feature_processor_nodes.append(node)
  # Select sharded modules, which are top-level in the forward call graph,
  # i.e. which don't have input transformations, i.e.
  # rely only on 'builtins.getattr'.
  ret = []
  for node in graph.nodes:
    if node.op == "call_module" and node.target in sharded_modules:
      total_num_args = len(node.args) + len(node.kwargs)
      if total_num_args == 0:
        continue
      arg_info_list, num_found = _get_node_args(node, feature_processor_nodes)
      if num_found == total_num_args:
        logger.info(f"Module '{node.target}'' will be pipelined")
        child = sharded_modules[node.target]
        child.forward = PipelinedForward(
          node.target,
          arg_info_list,
          child,
          context,
          dist_stream,
        )
        ret.append(child)
  return ret


class TrainPipelineSparseDist(TrainPipeline[In, Out]):
  """
  This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
  forward and backward. This helps hide the all2all latency while preserving the
  training forward / backward ordering.

  stage 3: forward, backward - uses default CUDA stream
  stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
  stage 1: device transfer - uses memcpy CUDA stream

  `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
  To be considered a top-level module, a module can only depend on 'getattr' calls on
  input.

  Input model must be symbolically traceable with the exception of `ShardedModule` and
  `DistributedDataParallel` modules.
  """

  synced_pipeline_id: Dict[int, int] = {}

  def __init__(
    self,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    enable_amp: bool = False,
    enable_grad_scaling: bool = True,
    grad_accum: Optional[int] = None,
  ) -> None:
    self._model = model
    self._optimizer = optimizer
    self._device = device
    self._enable_amp = enable_amp
    # NOTE: Pending upstream feedback, but two flags because we can run AMP without CUDA but cannot scale gradients without CUDA.
    # Background on gradient/loss scaling
    # https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    self._enable_grad_scaling = enable_grad_scaling
    self._grad_scaler = torch.cuda.amp.GradScaler(
      enabled=self._enable_amp and self._enable_grad_scaling
    )
    logging.info(f"Amp is enabled: {self._enable_amp}")

    # use two data streams to support two concurrent batches
    if device.type == "cuda":
      self._memcpy_stream: Optional[torch.cuda.streams.Stream] = torch.cuda.Stream()
      self._data_dist_stream: Optional[torch.cuda.streams.Stream] = torch.cuda.Stream()
    else:
      if self._enable_amp:
        logging.warning("Amp is enabled, but no CUDA available")
      self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
      self._data_dist_stream: Optional[torch.cuda.streams.Stream] = None
    self._batch_i: Optional[In] = None
    self._batch_ip1: Optional[In] = None
    self._batch_ip2: Optional[In] = None
    self._connected = False
    self._context = TrainPipelineContext()
    self._pipelined_modules: List[ShardedModule] = []

    self._progress_calls = 0
    if grad_accum is not None:
      assert isinstance(grad_accum, int) and grad_accum > 0
    self._grad_accum = grad_accum

  def _connect(self, dataloader_iter: Iterator[In]) -> None:
    # batch 1
    with torch.cuda.stream(self._memcpy_stream):
      batch_i = next(dataloader_iter)
      self._batch_i = batch_i = _to_device(batch_i, self._device, non_blocking=True)
      # Try to pipeline input data dist.
      self._pipelined_modules = _rewrite_model(self._model, self._context, self._data_dist_stream)

    with torch.cuda.stream(self._data_dist_stream):
      _wait_for_batch(batch_i, self._memcpy_stream)
      _start_data_dist(self._pipelined_modules, batch_i, self._context)

    # batch 2
    with torch.cuda.stream(self._memcpy_stream):
      batch_ip1 = next(dataloader_iter)
      self._batch_ip1 = batch_ip1 = _to_device(batch_ip1, self._device, non_blocking=True)
    self._connected = True
    self.__class__.synced_pipeline_id[id(self._model)] = id(self)

  def progress(self, dataloader_iter: Iterator[In]) -> Out:
    """
    NOTE: This method has been updated to perform gradient accumulation.
    If `_grad_accum` is set, then loss values are scaled by this amount and
    optimizer update/reset is skipped for `_grad_accum` calls of `progress`
    (congruent to training steps), and then update/reset on every `_grad_accum`th
    step.

    """
    should_step_optimizer = (
      self._grad_accum is not None
      and self._progress_calls > 0
      and (self._progress_calls + 1) % self._grad_accum == 0
    ) or self._grad_accum is None
    should_reset_optimizer = (
      self._grad_accum is not None
      and self._progress_calls > 0
      and (self._progress_calls + 2) % self._grad_accum == 0
    ) or self._grad_accum is None

    if not self._connected:
      self._connect(dataloader_iter)
    elif self.__class__.synced_pipeline_id.get(id(self._model), None) != id(self):
      self._sync_pipeline()
      self.__class__.synced_pipeline_id[id(self._model)] = id(self)

    if self._model.training and should_reset_optimizer:
      with record_function("## zero_grad ##"):
        self._optimizer.zero_grad()

    with record_function("## copy_batch_to_gpu ##"):
      with torch.cuda.stream(self._memcpy_stream):
        batch_ip2 = next(dataloader_iter)
        self._batch_ip2 = batch_ip2 = _to_device(batch_ip2, self._device, non_blocking=True)
    batch_i = cast(In, self._batch_i)
    batch_ip1 = cast(In, self._batch_ip1)

    with record_function("## wait_for_batch ##"):
      _wait_for_batch(batch_i, self._data_dist_stream)

    # Forward
    with record_function("## forward ##"):
      # if using multiple streams (ie. CUDA), create an event in default stream
      # before starting forward pass
      if self._data_dist_stream:
        event = torch.cuda.current_stream().record_event()
      if self._enable_amp:
        # conditionally apply the model to the batch in the autocast context
        # it appears that `enabled=self._enable_amp` should handle this,
        # but it does not.
        with torch.autocast(
          device_type=self._device.type,
          dtype=torch.bfloat16,
          enabled=self._enable_amp,
        ):
          (losses, output) = cast(Tuple[torch.Tensor, Out], self._model(batch_i))
      else:
        (losses, output) = cast(Tuple[torch.Tensor, Out], self._model(batch_i))

    # Data Distribution
    with record_function("## sparse_data_dist ##"):
      with torch.cuda.stream(self._data_dist_stream):
        _wait_for_batch(batch_ip1, self._memcpy_stream)
        # Ensure event in default stream has been called before
        # starting data dist
        if self._data_dist_stream:
          # pyre-ignore [61]: Local variable `event` is undefined, or not always defined
          self._data_dist_stream.wait_event(event)
        _start_data_dist(self._pipelined_modules, batch_ip1, self._context)

    if self._model.training:
      # Backward
      with record_function("## backward ##"):
        # Loss is normalize by number of accumulation steps.
        # The reported loss in `output['loss']` remains the unnormalized value.
        if self._grad_accum is not None:
          losses = losses / self._grad_accum
        self._grad_scaler.scale(torch.sum(losses, dim=0)).backward()

      if should_step_optimizer:
        # Update
        with record_function("## optimizer ##"):
          self._grad_scaler.step(self._optimizer)
          self._grad_scaler.update()

    self._batch_i = batch_ip1
    self._batch_ip1 = batch_ip2

    if self._model.training:
      self._progress_calls += 1

    return output

  def _sync_pipeline(self) -> None:
    """
    Syncs `PipelinedForward` for sharded modules with context and dist stream of the
    current train pipeline. Used when switching between train pipelines for the same
    model.
    """
    for module in self._pipelined_modules:
      module.forward._context = self._context
      module.forward._dist_stream = self._data_dist_stream
