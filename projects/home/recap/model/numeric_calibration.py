import torch


class NumericCalibration(torch.nn.Module):
  def __init__(
    self,
    pos_downsampling_rate: float,
    neg_downsampling_rate: float,
  ):
    super().__init__()

    # Using buffer to make sure they are on correct device (and not moved every time).
    # Will also be part of state_dict.
    self.register_buffer(
      "ratio", torch.as_tensor(neg_downsampling_rate / pos_downsampling_rate), persistent=True
    )

  def forward(self, probs: torch.Tensor):
    return probs * self.ratio / (1.0 - probs + (self.ratio * probs))
