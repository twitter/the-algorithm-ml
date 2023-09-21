"""MaskNet: Wang et al. (https://arxiv.org/abs/2102.07619)."""

from tml.projects.home.recap.model import config, mlp

import torch


def _init_weights(module):
    """Initializes weights

    Example

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


class MaskBlock(torch.nn.Module):
    """
      MaskBlock module in a mask-based neural network.

      This module represents a MaskBlock, which applies a masking operation to the input data and then
      passes it through a hidden layer. It is typically used as a building block within a MaskNet.

      Args:
          mask_block_config (config.MaskBlockConfig): Configuration for the MaskBlock.
          input_dim (int): Dimensionality of the input data.
          mask_input_dim (int): Dimensionality of the mask input.

      Example:
          To create and use a MaskBlock within a MaskNet, follow these steps:

          ```python
          # Define the configuration for the MaskBlock
          mask_block_config = MaskBlockConfig(
              input_layer_norm=True,  # Apply input layer normalization
              reduction_factor=0.5     # Reduce input dimensionality by 50%
          )

          # Create an instance of the MaskBlock
          mask_block = MaskBlock(mask_block_config, input_dim=64, mask_input_dim=32)

          # Generate input tensors
          input_data = torch.randn(batch_size, 64)
          mask_input = torch.randn(batch_size, 32)

          # Perform a forward pass through the MaskBlock
          output = mask_block(input_data, mask_input)
          ```

      Note:
          The `MaskBlock` module applies layer normalization to the input if specified, followed by a masking
          operation that combines the input and mask input. Then, it passes the result through a hidden layer
          with optional dimensionality reduction.

      Warning:
          This class is intended for internal use within neural network architectures and should not be
          directly accessed or modified by external code.
      """

    def __init__(
        self, mask_block_config: config.MaskBlockConfig, input_dim: int, mask_input_dim: int
    ) -> None:
        """
            Initializes the MaskBlock module.

            Args:
                mask_block_config (config.MaskBlockConfig): Configuration for the MaskBlock.
                input_dim (int): Dimensionality of the input data.
                mask_input_dim (int): Dimensionality of the mask input.

            Returns:
                None
            """

        super(MaskBlock, self).__init__()
        self.mask_block_config = mask_block_config
        output_size = mask_block_config.output_size

        if mask_block_config.input_layer_norm:
            self._input_layer_norm = torch.nn.LayerNorm(input_dim)
        else:
            self._input_layer_norm = None

        if mask_block_config.reduction_factor:
            aggregation_size = int(
                mask_input_dim * mask_block_config.reduction_factor)
        elif mask_block_config.aggregation_size is not None:
            aggregation_size = mask_block_config.aggregation_size
        else:
            raise ValueError(
                "Need one of reduction factor or aggregation size.")

        self._mask_layer = torch.nn.Sequential(
            torch.nn.Linear(mask_input_dim, aggregation_size),
            torch.nn.ReLU(),
            torch.nn.Linear(aggregation_size, input_dim),
        )
        self._mask_layer.apply(_init_weights)
        self._hidden_layer = torch.nn.Linear(input_dim, output_size)
        self._hidden_layer.apply(_init_weights)
        self._layer_norm = torch.nn.LayerNorm(output_size)

    def forward(self, net: torch.Tensor, mask_input: torch.Tensor):
        """
            Performs a forward pass through the MaskBlock.

            Args:
                net (torch.Tensor): Input data tensor.
                mask_input (torch.Tensor): Mask input tensor.

            Returns:
                torch.Tensor: Output tensor of the MaskBlock.
            """
        if self._input_layer_norm:
            net = self._input_layer_norm(net)
        hidden_layer_output = self._hidden_layer(
            net * self._mask_layer(mask_input))
        return self._layer_norm(hidden_layer_output)


class MaskNet(torch.nn.Module):
    """
      MaskNet module in a mask-based neural network.

      This module represents a MaskNet, which consists of multiple MaskBlocks. It can be used to
      create mask-based neural networks with parallel or stacked MaskBlocks.

      Args:
          mask_net_config (config.MaskNetConfig): Configuration for the MaskNet.
          in_features (int): Dimensionality of the input data.

      Example:
          To create and use a MaskNet, you can follow these steps:

          ```python
          # Define the configuration for the MaskNet
          mask_net_config = MaskNetConfig(
              use_parallel=True,  # Use parallel MaskBlocks
              mlp=MlpConfig(layer_sizes=[128, 64])  # Optional MLP on the outputs
          )

          # Create an instance of the MaskNet
          mask_net = MaskNet(mask_net_config, in_features=64)

          # Generate input tensors
          input_data = torch.randn(batch_size, 64)

          # Perform a forward pass through the MaskNet
          outputs = mask_net(input_data)

          # Access the output and shared layer
          output = outputs["output"]
          shared_layer = outputs["shared_layer"]
          ```

      Note:
          The `MaskNet` module allows you to create mask-based neural networks with parallel or stacked
          MaskBlocks. You can also optionally apply an MLP to the outputs for further processing.

      Warning:
          This class is intended for internal use within neural network architectures and should not be
          directly accessed or modified by external code.
      """

    def __init__(self, mask_net_config: config.MaskNetConfig, in_features: int):
        """
            Initializes the MaskNet module.

            Args:
                mask_net_config (config.MaskNetConfig): Configuration for the MaskNet.
                in_features (int): Dimensionality of the input data.

            Returns:
                None
            """

        super().__init__()
        self.mask_net_config = mask_net_config
        mask_blocks = []

        if mask_net_config.use_parallel:
            total_output_mask_blocks = 0
            for mask_block_config in mask_net_config.mask_blocks:
                mask_blocks.append(
                    MaskBlock(mask_block_config, in_features, in_features))
                total_output_mask_blocks += mask_block_config.output_size
            self._mask_blocks = torch.nn.ModuleList(mask_blocks)
        else:
            input_size = in_features
            for mask_block_config in mask_net_config.mask_blocks:
                mask_blocks.append(
                    MaskBlock(mask_block_config, input_size, in_features))
                input_size = mask_block_config.output_size

            self._mask_blocks = torch.nn.ModuleList(mask_blocks)
            total_output_mask_blocks = mask_block_config.output_size

        if mask_net_config.mlp:
            self._dense_layers = mlp.Mlp(
                total_output_mask_blocks, mask_net_config.mlp)
            self.out_features = mask_net_config.mlp.layer_sizes[-1]
        else:
            self.out_features = total_output_mask_blocks
        self.shared_size = total_output_mask_blocks

    def forward(self, inputs: torch.Tensor):
        """
            Performs a forward pass through the MaskNet.

            Args:
                inputs (torch.Tensor): Input data tensor.

            Returns:
                torch.Tensor: Output tensor of the MaskNet.
            """
        if self.mask_net_config.use_parallel:
            mask_outputs = []
            for mask_layer in self._mask_blocks:
                mask_outputs.append(mask_layer(mask_input=inputs, net=inputs))
            # Share the outputs of the MaskBlocks.
            all_mask_outputs = torch.cat(mask_outputs, dim=1)
            output = (
                all_mask_outputs
                if self.mask_net_config.mlp is None
                else self._dense_layers(all_mask_outputs)["output"]
            )
            return {"output": output, "shared_layer": all_mask_outputs}
        else:
            net = inputs
            for mask_layer in self._mask_blocks:
                net = mask_layer(net=net, mask_input=inputs)
            # Share the output of the stacked MaskBlocks.
            output = net if self.mask_net_config.mlp is None else self._dense_layers[
                net]["output"]
            return {"output": output, "shared_layer": net}
