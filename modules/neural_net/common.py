# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Common NN building blocks
# ---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from modules.neural_net.constants import _EPS_, _LEAKY_RELU_NEG_SLOPE_, _NUM_GROUPS_

# ---------------------------------------------------------------------------------------------------------------
class ws_conv_nxn(nn.Conv2d):
    """ Weight standardized convolution kernel """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]],
        mode: str = 'maintain_resolution' ,
        eps: float = _EPS_):

        if isinstance(kernel_size, int): 
            if kernel_size % 2 != 1:
                raise ValueError("argument 'kernal_size' should be an odd number")
            padding = kernel_size // 2

        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 2: 
                raise ValueError("argument 'kernal_size' should be a tuple with 2 elements")
            kernel_rows = kernel_size[0]
            kernel_cols = kernel_size[1]
            if kernel_rows % 2 != 1 or kernel_cols % 2 != 1:
                raise ValueError("argument 'kernal_size' should have odd numbers")
            padding = ( kernel_rows // 2, kernel_cols // 2 )

        else: raise ValueError("argument 'kernal_size' should be of type 'int' or 'tuple'")

        if mode == 'maintain_resolution': stride = 1
        elif mode == 'downsample_resolution': stride = 2
        else: raise Exception("argument 'mode' should be either 'maintain_resolution' or 'downsample_resolution'")

        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride, padding=padding, 
            dilation=1, 
            groups=1, 
            bias=True)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        out_channels, in_channels, H, W = self.weight.shape
        weight = self.weight.reshape(out_channels, -1)
        mean = torch.mean(weight, dim=1, keepdim=True)
        std = torch.std(weight, dim=1, keepdim=True)
        weight = (weight - mean) / (std + self.eps)
        weight = weight.reshape(out_channels, in_channels, H, W)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
# --------------------------------------------------------------------------------------------------------------
class GroupNorm(nn.Module):
    def __init__(
        self, 
        num_groups: int, 
        num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=_EPS_, affine=True)

    def forward(self, x: torch.Tensor):
        return self.gn(x)
    
# ---------------------------------------------------------------------------------------------------------------
class ws_conv_nxn_block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]],
        mode: str,
        activation: str):
        super().__init__()

        conv2d = ws_conv_nxn(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size,
            mode = mode)
        norm = GroupNorm(num_groups=_NUM_GROUPS_, num_channels=out_channels)
        act = Activation(activation)
        self.block = nn.Sequential(conv2d, norm, act)

    def forward(self, x:torch.Tensor):
        return self.block(x)

# ---------------------------------------------------------------------------------------------------------------
class conv_nxn(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]],
        mode: str = 'maintain_resolution'):
        super().__init__()

        if isinstance(kernel_size, int): 
            if kernel_size % 2 != 1:
                raise ValueError("argument 'kernal_size' should be an odd number")
            padding = kernel_size // 2

        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 2: 
                raise ValueError("argument 'kernal_size' should be a tuple with 2 elements")
            kernel_rows = kernel_size[0]
            kernel_cols = kernel_size[1]
            if kernel_rows % 2 != 1 or kernel_cols % 2 != 1:
                raise ValueError("argument 'kernal_size' should have odd numbers")
            padding = ( kernel_rows // 2, kernel_cols // 2 )

        else: raise ValueError("argument 'kernal_size' should be of type 'int' or 'tuple'")

        if mode == 'maintain_resolution': stride = 1
        elif mode == 'downsample_resolution': stride = 2
        else: raise Exception("argument 'mode' should be either 'maintain_resolution' or 'downsample_resolution'")
        self.conv2d = nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size,
                stride = stride,
                padding = padding)
        
    def forward(self, x: torch.Tensor):
        return self.conv2d(x)
    
# ---------------------------------------------------------------------------------------------------------------
class conv_nxn_block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]],
        mode: str,
        activation: str,
        norm_layer: str = 'layer_normalization'):
        super().__init__()

        conv2d = conv_nxn(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size,
            mode = mode)
        if norm_layer == 'layer_normalization': norm = layer_normalization()
        elif norm_layer == 'channel_normalization': norm = channel_normalization()
        act = Activation(activation)
        self.block = nn.Sequential(conv2d, norm, act)

    def forward(self, x:torch.Tensor):
        return self.block(x)
    
# ---------------------------------------------------------------------------------------------------------------    
def get_conv_nxn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int]],
    conv_type: str,
    mode: str, 
    activation: str):
    if conv_type == 'weight_standardized':
        conv2d_block = ws_conv_nxn_block(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            mode = mode,
            activation = activation )
    else:
        conv2d_block = conv_nxn_block(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            mode = mode,
            activation = activation )
    return conv2d_block
    
# ---------------------------------------------------------------------------------------------------------------
class ffn_block(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        activation: str,
        norm_layer: Optional[str] = None,
        num_groups: Optional[int] = None):
        super().__init__()
        
        ffn = nn.Linear(in_features = in_channels,  out_features = out_channels, bias = True)
        act = Activation(activation)
        if norm_layer != None:
            if norm_layer == 'layer_normalization': norm = layer_normalization()
            elif norm_layer == 'channel_normalization': norm = channel_normalization()
            elif norm_layer == 'group_normalization': norm = group_normalization(num_groups)
            self.block = nn.Sequential(ffn, norm, act)
        else: self.block = nn.Sequential(ffn, act)

    def forward(self, x:torch.Tensor):
        return self.block(x)

# ---------------------------------------------------------------------------------------------------------------
class channel_normalization(nn.Module):
    def __init__(self, eps: float = _EPS_):
        super().__init__()
        self.eps = eps
        self.mu = nn.Parameter(torch.zeros(1))
        self.std = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=1, keepdims=True)
        std = torch.std(x, dim=1, keepdims=True)
        x  = (x - mean) / (std + self.eps)
        x = self.std * x + self.mu
        return x

# ---------------------------------------------------------------------------------------------------------------    
class layer_normalization(nn.Module):
    def __init__(self, eps: float = _EPS_):
        super().__init__()
        self.eps = eps
        self.mu = nn.Parameter(torch.zeros(1))
        self.std = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor):
        x  = (x - torch.mean(x)) / (torch.std(x) + self.eps)
        x = self.std * x + self.mu
        return x

# ---------------------------------------------------------------------------------------------------------------    
class group_normalization(nn.Module):
    def __init__(self, num_groups: int, eps: float = _EPS_):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.mu = nn.Parameter(torch.zeros(1))
        self.std = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor):
        N, D = x.shape
        dim_per_group = D // self.num_groups
        x = x.reshape(N, self.num_groups, dim_per_group).contiguous()
        mean = torch.mean(x, dim=(0,2), keepdims=True)
        std = torch.std(x, dim=(0,2), keepdims=True)
        x  = (x - mean) / (std + self.eps)
        x = self.std * x + self.mu
        x = x.reshape(N, -1).contiguous()
        return x
    
# ---------------------------------------------------------------------------------------------------------------
class Activation(nn.Module):
    """ Activation Layer """
    def __init__(self, activation: str = 'relu'):
        super().__init__()
        if activation == 'relu': layer = nn.ReLU(inplace=False)
        elif activation == 'leakyrelu': layer = nn.LeakyReLU(negative_slope=_LEAKY_RELU_NEG_SLOPE_, inplace=False)
        elif activation == 'swish': layer = torch.nn.SiLU(inplace=False)
        else : layer = nn.ReLU(inplace=False)
        self.activation_layer = layer

    def forward(self, x: torch.Tensor):
        return self.activation_layer(x)

