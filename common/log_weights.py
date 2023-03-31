"""For logging model weights."""
import itertools
from typing import Callable, Dict, List, Optional, Union

from tml.ml_logging.torch_logging import logging  # type: ignore[attr-defined]
import torch
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel


def weights_to_log(
  model: torch.nn.Module,
  how_to_log: Optional[Union[Callable, Dict[str, Callable]]] = None,
):
  """Creates dict of reduced weights to log to give sense of training.

  Args:
    model: model to traverse.
    how_to_log: if a function, then applies this to every parameter, if a dict
      then only applies and logs specified parameters.

  """
  if not how_to_log:
    return

  to_log = dict()
  named_parameters = model.named_parameters()
  logging.info(f"Using DMP: {isinstance(model, DistributedModelParallel)}")
  if isinstance(model, DistributedModelParallel):
    named_parameters = itertools.chain(
      named_parameters, model._dmp_wrapped_module.named_parameters()
    )
  logging.info(
    f"Using dmp parameters: {list(name for name, _ in model._dmp_wrapped_module.named_parameters())}"
  )
  for param_name, params in named_parameters:
    if callable(how_to_log):
      how = how_to_log
    else:
      how = how_to_log.get(param_name)  # type: ignore[assignment]
    if not how:
      continue  # type: ignore
    to_log[f"model/{how.__name__}/{param_name}"] = how(params.detach()).cpu().numpy()
  return to_log


def log_ebc_norms(
  model_state_dict,
  ebc_keys: List[str],
  sample_size: int = 4_000_000,
) -> Dict[str, torch.Tensor]:
  """Logs the norms of the embedding tables as specified by ebc_keys.
  As of now, log average norm per rank.

  Args:
      model_state_dict: model.state_dict()
      ebc_keys: list of embedding keys from state_dict to log. Must contain full name,
      i.e. model.embeddings.ebc.embedding_bags.meta__user_id.weight
      sample_size: Limits number of rows per rank to compute average on to avoid OOM.
  """
  norm_logs = dict()
  for emb_key in ebc_keys:
    norms = (torch.ones(1, dtype=torch.float32) * -1).to(torch.device(f"cuda:{dist.get_rank()}"))
    if emb_key in model_state_dict:
      emb_weight = model_state_dict[emb_key]
      try:
        emb_weight_tensor = emb_weight.local_tensor()
      except AttributeError as e:
        logging.info(e)
        emb_weight_tensor = emb_weight
      logging.info("Running Tensor.detach()")
      emb_weight_tensor = emb_weight_tensor.detach()
      sample_mask = torch.randperm(emb_weight_tensor.shape[0])[
        : min(sample_size, emb_weight_tensor.shape[0])
      ]
      # WARNING: .cpu() transfer executes malloc that may be the cause of memory leaks
      # Change sample_size if the you observe frequent OOM errors or remove weight logging.
      norms = emb_weight_tensor[sample_mask].cpu().norm(dim=1).to(torch.float32)
      logging.info(f"Norm shape before reduction: {norms.shape}", rank=-1)
      norms = norms.mean().to(torch.device(f"cuda:{dist.get_rank()}"))

    all_norms = [
      torch.zeros(1, dtype=norms.dtype).to(norms.device) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_norms, norms)
    for idx, norm in enumerate(all_norms):
      if norm != -1.0:
        norm_logs[f"{emb_key}-norm-{idx}"] = norm
  logging.info(f"Norm Logs are {norm_logs}")
  return norm_logs
