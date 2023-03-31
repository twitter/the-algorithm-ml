"""Loss type enums."""
from enum import Enum


class LossType(str, Enum):
  CROSS_ENTROPY = "cross_entropy"
  BCE_WITH_LOGITS = "bce_with_logits"
