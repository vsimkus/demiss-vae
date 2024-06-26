from typing import Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'tanh': torch.tanh,
    'sigmoid': F.sigmoid
}

class FullyConnectedNetwork(nn.Module):
    """
    Fully-connected neural network

    Args:
        layer_dims: Dimensions of the model.
        activation: Callable activation function.
    """
    def __init__(self, layer_dims: List[int],
                 *,
                 activation: Callable = F.relu):
        super().__init__()

        self.layer_dims = layer_dims
        self.activation = activation

        # Create the linear layers
        linear_layers = []
        for i in range(len(layer_dims)-1):
            linear_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        # Create the neural model
        self.linear_layers = nn.ModuleList(linear_layers)

    def forward(self, inputs: torch.Tensor):
        out = inputs
        for l, layer in enumerate(self.linear_layers):
            out = layer(out)
            # No activation after final linear layer
            if l < len(self.linear_layers)-1:
                out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features: int,
        *,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.activation = activation

        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs: torch.Tensor):
        temps = self.linear_layers[0](inputs)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps

    def reset_params(self):
        for layer in self.linear_layers:
            layer.reset_parameters()


class ResidualFCNetwork(nn.Module):
    """
    Residual fully-connected neural network.

    Args:
        input_dim:              The dimensionality of the inputs.
        output_dim:             The dimensionality of the outputs.
        num_residual_blocks:    The number of full residual blocks in the model.
        residual_block_dim:     The residual block dimensionality.
        activation:             Callable activation function.
        dropout_probability:    Dropout probability in residual blocks.

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 *,
                 activation: Union[Callable, str] = F.relu,
                 dropout_probability: float = 0.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_residual_blocks = num_residual_blocks
        self.residual_block_dim = residual_block_dim
        if isinstance(activation, Callable):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            raise NotImplementedError()
        self.dropout_probability = dropout_probability

        # Add initial layer
        self.initial_layer = nn.Linear(input_dim, residual_block_dim)

        # Create residual blocks
        blocks = [ResidualBlock(residual_block_dim,
                                activation=self.activation,
                                dropout_probability=self.dropout_probability)
                  for _ in range(num_residual_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # Add final layer
        self.final_layer = nn.Linear(residual_block_dim, output_dim)

    def forward(self, inputs: torch.Tensor):
        out = self.initial_layer(inputs)
        out = self.activation(out)
        for block in self.blocks:
            out = block(out)
            out = self.activation(out)
        return self.final_layer(out)

    # def reset_first_and_last_layers(self):
    #     self.initial_layer.reset_parameters()
    #     self.final_layer.reset_parameters()
    #     for block in self.blocks:
    #         block.reset_params()
