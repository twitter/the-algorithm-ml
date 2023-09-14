"""MLP feed forward stack in torch."""

from tml.projects.home.recap.model.config import MlpConfig

import torch
from absl import logging


def _init_weights(module):
  """Initializes weights
  
  Example
  -------
    ```python
    import torch
    import torch.nn as nn

    # Define a simple linear layer
    linear_layer = nn.Linear(64, 32)

    # Initialize the weights and biases using _init_weights
    _init_weights(linear_layer)
    ```
  
  """
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(module.weight)
    torch.nn.init.constant_(module.bias, 0)


class Mlp(torch.nn.Module):
  """
    Multi-Layer Perceptron (MLP) feedforward neural network module in PyTorch.

    This module defines an MLP with customizable layers and activation functions. It is suitable for various
    applications such as deep learning for tabular data, feature extraction, and more.

    Args:
        in_features (int): The number of input features or input dimensions.
        mlp_config (MlpConfig): Configuration object specifying the MLP's architecture.

    Example:
        To create an instance of the `Mlp` module and use it for forward passes, you can follow these steps:

        ```python
        # Define the configuration for the MLP
        mlp_config = MlpConfig(
            layer_sizes=[128, 64],  # Specify the sizes of hidden layers
            batch_norm=True,        # Enable batch normalization
            dropout=0.2,            # Apply dropout with a rate of 0.2
            final_layer_activation=True  # Apply ReLU activation to the final layer
        )

        # Create an instance of the MLP module
        mlp_model = Mlp(in_features=input_dim, mlp_config=mlp_config)

        # Generate an input tensor
        input_tensor = torch.randn(batch_size, input_dim)

        # Perform a forward pass through the MLP
        outputs = mlp_model(input_tensor)

        # Access the output and shared layer
        output = outputs["output"]
        shared_layer = outputs["shared_layer"]
        ```

    Note:
        The `Mlp` class allows you to create customizable MLP architectures by specifying the layer sizes,
        enabling batch normalization and dropout, and choosing the activation function for the final layer.

    Warning:
        This class is intended for internal use within neural network architectures and should not be
        directly accessed or modified by external code.
    """
  def __init__(self, in_features: int, mlp_config: MlpConfig):
    """
        Initializes the Mlp module.

        Args:
            in_features (int): The number of input features or input dimensions.
            mlp_config (MlpConfig): Configuration object specifying the MLP's architecture.

        Returns:
            None
        """
    super().__init__()
    self._mlp_config = mlp_config
    input_size = in_features
    layer_sizes = mlp_config.layer_sizes
    modules = []
    for layer_size in layer_sizes[:-1]:
      modules.append(torch.nn.Linear(input_size, layer_size, bias=True))

      if mlp_config.batch_norm:
        modules.append(
          torch.nn.BatchNorm1d(
            layer_size, affine=mlp_config.batch_norm.affine, momentum=mlp_config.batch_norm.momentum
          )
        )

      modules.append(torch.nn.ReLU())

      if mlp_config.dropout:
        modules.append(torch.nn.Dropout(mlp_config.dropout.rate))

      input_size = layer_size
    modules.append(torch.nn.Linear(input_size, layer_sizes[-1], bias=True))
    if mlp_config.final_layer_activation:
      modules.append(torch.nn.ReLU())
    self.layers = torch.nn.ModuleList(modules)
    self.layers.apply(_init_weights)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
        Performs a forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of the MLP.
        """
    net = x
    for i, layer in enumerate(self.layers):
      net = layer(net)
      if i == 1:  # Share the first (widest?) set of activations for other applications.
        shared_layer = net
    return {"output": net, "shared_layer": shared_layer}

  @property
  def shared_size(self):
    """
        Returns the size of the shared layer in the MLP.

        Returns:
            int: Size of the shared layer.
        """
    return self._mlp_config.layer_sizes[-1]

  @property
  def out_features(self):
    """
        Returns the number of output features from the MLP.

        Returns:
            int: Number of output features.
        """

    return self._mlp_config.layer_sizes[-1]
