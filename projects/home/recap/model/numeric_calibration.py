import torch


class NumericCalibration(torch.nn.Module):
  """
    Numeric calibration module for adjusting probability scores.

    This module scales probability scores to correct for imbalanced datasets, where positive and negative samples
    may be underrepresented or have different ratios. It is designed to be used as a component in a neural network
    for tasks such as binary classification.

    Args:
        pos_downsampling_rate (float): The downsampling rate for positive samples.
        neg_downsampling_rate (float): The downsampling rate for negative samples.

    Example:
        To use `NumericCalibration` in a PyTorch model, you can create an instance of it and apply it to probability
        scores like this:

        ```python
        # Create a NumericCalibration instance with downsampling rates
        calibration = NumericCalibration(pos_downsampling_rate=0.1, neg_downsampling_rate=0.2)

        # Generate probability scores (e.g., from a neural network)
        raw_probs = torch.tensor([0.8, 0.6, 0.2, 0.9])

        # Apply numeric calibration to adjust the probabilities
        calibrated_probs = calibration(raw_probs)

        # The `calibrated_probs` now contains the adjusted probability scores
        ```

    Note:
        The `NumericCalibration` module is used to adjust probability scores to account for differences in
        the number of positive and negative samples in a dataset. It can help improve the calibration of
        probability estimates in imbalanced classification problems.

    Warning:
        This class is intended for internal use within neural network architectures and should not be
        directly accessed or modified by external code.
    """
  def __init__(
    self,
    pos_downsampling_rate: float,
    neg_downsampling_rate: float,
  ):
    """
        Apply numeric calibration to probability scores.

        Args:
            probs (torch.Tensor): Probability scores to be calibrated.

        Returns:
            torch.Tensor: Calibrated probability scores.
        """
    super().__init__()

    # Using buffer to make sure they are on correct device (and not moved every time).
    # Will also be part of state_dict.
    self.register_buffer(
      "ratio", torch.as_tensor(neg_downsampling_rate / pos_downsampling_rate), persistent=True
    )

  def forward(self, probs: torch.Tensor):
    return probs * self.ratio / (1.0 - probs + (self.ratio * probs))
