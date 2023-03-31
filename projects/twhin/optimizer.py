import functools

from tml.projects.twhin.models.config import TwhinModelConfig
from tml.projects.twhin.models.models import TwhinModel
from tml.optimizers.optimizer import get_optimizer_class, LRShim
from tml.optimizers.config import get_optimizer_algorithm_config, LearningRate
from tml.ml_logging.torch_logging import logging

from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim import keyed


FUSED_OPT_KEY = "fused_opt"
TRANSLATION_OPT_KEY = "operator_opt"


def _lr_from_config(optimizer_config):
  if optimizer_config.learning_rate is not None:
    return optimizer_config.learning_rate
  else:
    # treat None as constant lr
    lr_value = get_optimizer_algorithm_config(optimizer_config).lr
    return LearningRate(constant=lr_value)


def build_optimizer(model: TwhinModel, config: TwhinModelConfig):
  """Builds an optimizer for a Twhin model combining the embeddings optimizer with an optimizer for per-relation translations.

  Args:
    model: TwhinModel to build optimizer for.
    config: TwhinConfig for model.

  Returns:
    Optimizer for model.
  """
  translation_optimizer_fn = functools.partial(
    get_optimizer_class(config.translation_optimizer),
    **get_optimizer_algorithm_config(config.translation_optimizer).dict(),
  )

  translation_optimizer = keyed.KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(model.named_parameters())),
    optim_factory=translation_optimizer_fn,
  )

  lr_dict = {}
  for table in config.embeddings.tables:
    lr_dict[table.name] = _lr_from_config(table.optimizer)
  lr_dict[TRANSLATION_OPT_KEY] = _lr_from_config(config.translation_optimizer)

  logging.info(f"***** LR dict: {lr_dict} *****")

  logging.info(
    f"***** Combining fused optimizer {model.fused_optimizer} with operator optimizer: {translation_optimizer} *****"
  )
  optimizer = keyed.CombinedOptimizer(
    [
      (FUSED_OPT_KEY, model.fused_optimizer),
      (TRANSLATION_OPT_KEY, translation_optimizer),
    ]
  )

  # scheduler = LRShim(optimizer, lr_dict)
  scheduler = None

  logging.info(f"***** Combined optimizer after init: {optimizer} *****")

  return optimizer, scheduler
