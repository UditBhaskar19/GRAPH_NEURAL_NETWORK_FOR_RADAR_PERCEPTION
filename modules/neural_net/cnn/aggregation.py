# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : feature aggregation (neck)
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict
from modules.neural_net.common import get_conv_nxn_block
from modules.neural_net.constants import _INTERP_MODE_

# ---------------------------------------------------------------------------------------------------------------
class Neck(nn.Module):
    def __init__(
        self, 
        input_image_dimension: int,
        layer_channels: Union[List[int], Tuple[int]],
        out_channels: int,
        kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()

        # save the keys in reverse order that is used in the forward method
        self.num_layers = len(layer_channels)
        keys_reversed = []
        for i in range(self.num_layers):
            key = f"c{self.num_layers - i - 1}"
            keys_reversed.append(key)
        self.keys_reversed = keys_reversed

        # blocks to change the feature dimension
        dim_reduction_blocks = {}
        for i, in_channels in enumerate(layer_channels):
            key = f"c{i}"
            dim_reduction_blocks[key] \
                = get_conv_nxn_block(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    conv_type = conv_type,
                    mode = 'maintain_resolution',
                    activation = activation )
        self.dim_reduction_blocks = nn.ModuleDict(dim_reduction_blocks)
        
        # blocks to transform features after feature resizing
        feat_concat_blocks = {}
        for i in range(self.num_layers):
            key = f"c{i}"
            if i == self.num_layers - 1: in_channels = out_channels
            else: in_channels = 2 * out_channels
            feat_concat_blocks[key] \
                = get_conv_nxn_block(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    conv_type = conv_type,
                    mode = 'maintain_resolution',
                    activation = activation )
        self.feat_concat_blocks = nn.ModuleDict(feat_concat_blocks)

        # blocks to transform features after concatenation with input image data
        self.feat_image_data \
            = get_conv_nxn_block(
                in_channels = out_channels + input_image_dimension,
                out_channels = out_channels,
                kernel_size = kernel_size,
                conv_type = conv_type,
                mode = 'maintain_resolution',
                activation = activation )


    def forward(self, x: Dict[str, torch.Tensor], image: torch.Tensor):
        # change feature dimension
        x_temp = {}
        for key, value in x.items():
            x_temp[key] = self.dim_reduction_blocks[key](value)

        # resize and feature transformation for the top most layer
        key_curr = self.keys_reversed[0]
        key_next = self.keys_reversed[1]
        x_out = F.interpolate(
            input = x_temp[key_curr], 
            size = ( x_temp[key_next].shape[2], x_temp[key_next].shape[3] ), 
            mode = _INTERP_MODE_)
        x_out = self.feat_concat_blocks[key_curr](x_out)

        # resize and feature transformation for all the layers except the input image layer
        for i in range(1, self.num_layers-1):
            key_curr = self.keys_reversed[i]
            key_next = self.keys_reversed[i + 1]
            x_out = torch.concatenate((x_out, x_temp[key_curr]), axis = 1)
            x_out = F.interpolate(
                input = x_out, 
                size = (x_temp[key_next].shape[2], x_temp[key_next].shape[3]), 
                mode = _INTERP_MODE_)
            x_out = self.feat_concat_blocks[key_curr](x_out)

        # resize to input image resolution and feature transformation
        key_curr = self.keys_reversed[-1]    
        x_out = torch.concatenate((x_out, x_temp[key_curr]), axis = 1)
        x_out = F.interpolate(
            input = x_out, 
            size = (image.shape[2], image.shape[3]), 
            mode = _INTERP_MODE_)
        x_out = self.feat_concat_blocks[key_curr](x_out)

        # apply the final transformation that concatenates the image data and tranaforms the features
        x_out = torch.concatenate((x_out, image), axis = 1)
        x_out = self.feat_image_data(x_out)
        return x_out