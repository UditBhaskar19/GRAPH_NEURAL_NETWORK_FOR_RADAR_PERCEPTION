# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : model backbone
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from modules.neural_net.common import conv_nxn, get_conv_nxn_block, channel_normalization

# ---------------------------------------------------------------------------------------------------------------
class base(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: Union[Tuple[int], List[int]],
        kernel_sizes: Union[Tuple[int], List[int]],
        activation: str,
        conv_type: str):
        super().__init__()

        block = []
        for i, (out_channels, kernel_size) in enumerate(zip(stem_channels, kernel_sizes)):
            if i == 0: mode = 'downsample_resolution'
            else: mode = 'maintain_resolution'
            conv2d_block = get_conv_nxn_block(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                conv_type = conv_type,
                mode = mode,
                activation = activation )
            in_channels = out_channels
            block.append(conv2d_block)
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor):
        return self.block(x)

# ---------------------------------------------------------------------------------------------------------------
class bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width_channels: int,
        kernel_size: int,
        mode: str,
        activation: str,
        conv_type: str):
        super().__init__()

        conv2d_1x1_block1 = get_conv_nxn_block(
                in_channels = in_channels,
                out_channels = width_channels,
                kernel_size = 1,
                conv_type = conv_type,
                mode = 'maintain_resolution',
                activation = activation )
        
        conv2d_1x1_block2 = get_conv_nxn_block(
                in_channels = width_channels,
                out_channels = width_channels,
                kernel_size = kernel_size,
                conv_type = conv_type,
                mode = mode,
                activation = activation )
        
        conv2d_1x1_block3 = get_conv_nxn_block(
                in_channels = width_channels,
                out_channels = out_channels,
                kernel_size = 1,
                conv_type = conv_type,
                mode = 'maintain_resolution',
                activation = activation )

        self.downsample = False
        if in_channels != out_channels or mode == 'downsample_resolution':
            self.downsample = True
            conv_ = conv_nxn(
                in_channels = in_channels, out_channels = out_channels, 
                kernel_size = 1, mode = mode)
            norm_ = channel_normalization()
            self.downsample_blk = nn.Sequential(conv_, norm_)

        self.layers = nn.Sequential(conv2d_1x1_block1, conv2d_1x1_block2, conv2d_1x1_block3 )
        
    def forward(self, x: torch.Tensor):
        if self.downsample:
            identity = self.downsample_blk(x)
        else: identity = x
        out = self.layers(x)
        out += identity
        return out

# ---------------------------------------------------------------------------------------------------------------
class bottleneck_block(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        width_channels: int,
        kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()

        layers = []
        blk_ = bottleneck(
            in_channels = in_channels,
            out_channels = out_channels,
            width_channels = width_channels,
            kernel_size = kernel_size,
            mode = 'downsample_resolution',
            activation = activation,
            conv_type = conv_type)
        layers.append(blk_)
        
        for _ in range(1, num_blocks):
            blk_ = bottleneck(
                in_channels = out_channels,
                out_channels = out_channels,
                width_channels = width_channels,
                kernel_size = kernel_size,
                mode = 'maintain_resolution',
                activation = activation,
                conv_type = conv_type)
            layers.append(blk_)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
# ---------------------------------------------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        base_stem_channels: Union[Tuple[int], List[int]],
        base_kernel_sizes: Union[Tuple[int], List[int]],
        bottleneck_number_of_blocks: Union[Tuple[int], List[int]],
        bottleneck_stem_channels: Union[Tuple[int], List[int]],
        bottleneck_width_channels: int,
        bottleneck_kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()

        layers = {}
        layers["c0"] = base(
            in_channels = in_channels,
            stem_channels = base_stem_channels,
            kernel_sizes = base_kernel_sizes,
            activation = activation,
            conv_type = conv_type)
        
        in_channels = base_stem_channels[-1]
        for i, (num_blocks, out_channels) in enumerate(zip(bottleneck_number_of_blocks, bottleneck_stem_channels)):
            key = f"c{i + 1}"
            layers[key] = bottleneck_block(
                num_blocks = num_blocks,
                in_channels = in_channels,
                out_channels = out_channels,
                width_channels = bottleneck_width_channels,
                kernel_size = bottleneck_kernel_size,
                activation = activation,
                conv_type = conv_type)
            in_channels = out_channels
        self.layers = nn.ModuleDict(layers)
        
    def forward(self, x: torch.Tensor):
        x_out = {}
        for key, layer in self.layers.items():
            x = layer(x)
            x_out[key] = x
        return x_out