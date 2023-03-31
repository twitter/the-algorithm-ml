"""MaskNet: Wang et al. (https://arxiv.org/abs/2102.07619)."""

from tml.projects.home.recap.model import config, mlp

import torch


def _init_weights(module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(module.weight)
    torch.nn.init.constant_(module.bias, 0)


class MaskBlock(torch.nn.Module):
  def __init__(
    self, mask_block_config: config.MaskBlockConfig, input_dim: int, mask_input_dim: int
  ) -> None:
    super(MaskBlock, self).__init__()
    self.mask_block_config = mask_block_config
    output_size = mask_block_config.output_size

    if mask_block_config.input_layer_norm:
      self._input_layer_norm = torch.nn.LayerNorm(input_dim)
    else:
      self._input_layer_norm = None

    if mask_block_config.reduction_factor:
      aggregation_size = int(mask_input_dim * mask_block_config.reduction_factor)
    elif mask_block_config.aggregation_size is not None:
      aggregation_size = mask_block_config.aggregation_size
    else:
      raise ValueError("Need one of reduction factor or aggregation size.")

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
    if self._input_layer_norm:
      net = self._input_layer_norm(net)
    hidden_layer_output = self._hidden_layer(net * self._mask_layer(mask_input))
    return self._layer_norm(hidden_layer_output)


class MaskNet(torch.nn.Module):
  def __init__(self, mask_net_config: config.MaskNetConfig, in_features: int):
    super().__init__()
    self.mask_net_config = mask_net_config
    mask_blocks = []

    if mask_net_config.use_parallel:
      total_output_mask_blocks = 0
      for mask_block_config in mask_net_config.mask_blocks:
        mask_blocks.append(MaskBlock(mask_block_config, in_features, in_features))
        total_output_mask_blocks += mask_block_config.output_size
      self._mask_blocks = torch.nn.ModuleList(mask_blocks)
    else:
      input_size = in_features
      for mask_block_config in mask_net_config.mask_blocks:
        mask_blocks.append(MaskBlock(mask_block_config, input_size, in_features))
        input_size = mask_block_config.output_size

      self._mask_blocks = torch.nn.ModuleList(mask_blocks)
      total_output_mask_blocks = mask_block_config.output_size

    if mask_net_config.mlp:
      self._dense_layers = mlp.Mlp(total_output_mask_blocks, mask_net_config.mlp)
      self.out_features = mask_net_config.mlp.layer_sizes[-1]
    else:
      self.out_features = total_output_mask_blocks
    self.shared_size = total_output_mask_blocks

  def forward(self, inputs: torch.Tensor):
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
      output = net if self.mask_net_config.mlp is None else self._dense_layers[net]["output"]
      return {"output": output, "shared_layer": net}
