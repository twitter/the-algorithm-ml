"""MLP feed forward stack in torch."""

from tml.projects.home.recap.model.config import MlpConfig

import torch
from absl import logging


def _init_weights(module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(module.weight)
    torch.nn.init.constant_(module.bias, 0)


class Mlp(torch.nn.Module):
  def __init__(self, in_features: int, mlp_config: MlpConfig):
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
    net = x
    for i, layer in enumerate(self.layers):
      net = layer(net)
      if i == 1:  # Share the first (widest?) set of activations for other applications.
        shared_layer = net
    return {"output": net, "shared_layer": shared_layer}

  @property
  def shared_size(self):
    return self._mlp_config.layer_sizes[-1]

  @property
  def out_features(self):
    return self._mlp_config.layer_sizes[-1]
