# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : task specific heads
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict
from modules.neural_net.common import get_conv_nxn_block, conv_nxn, ffn_block
from modules.data_utils.labels import  _INVALID_NUM_
from modules.neural_net.constants import (
    _CLS_CONV_MEAN_INIT_, _CLS_CONV_STD_INIT_, _CLS_CONV_BIAS_INIT_,
    _REG_CONV_MEAN_INIT_, _REG_CONV_STD_INIT_, _REG_CONV_BIAS_INIT_ )

# ---------------------------------------------------------------------------------------------------------------------------
class ConvStemBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: Union[List[int], Tuple[int]],
        kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()
        
        stem_layers = []
        for stem_channel in stem_channels:
            stem_layer = get_conv_nxn_block(
                in_channels = in_channels,
                out_channels = stem_channel,
                kernel_size = kernel_size,
                conv_type = conv_type,
                mode = 'maintain_resolution',
                activation = activation )
            stem_layers += [ stem_layer ]
            in_channels = stem_channel
        self.stem_layers = nn.Sequential(*stem_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_layers(x)
        return x
    
# ---------------------------------------------------------------------------------------------------------------------------
class FFNStemBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: Union[List[int], Tuple[int]],
        activation: str):
        super().__init__()
        
        stem_layers = []
        for stem_channel in stem_channels:
            stem_layer = ffn_block(
                in_channels = in_channels,
                out_channels = stem_channel,
                activation = activation )
            stem_layers += [ stem_layer ]
            in_channels = stem_channel
        self.stem_layers = nn.Sequential(*stem_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_layers(x)
        return x
    
# ---------------------------------------------------------------------------------------------------------------------------
class Conv_TaskSpecificHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        conv_type: str,
        init_weight_mu: torch.Tensor,
        init_weight_sigma: torch.Tensor,
        init_bias: torch.Tensor):
        super().__init__()

        _conv2dblk = get_conv_nxn_block(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = kernel_size,
                conv_type = conv_type,
                mode = 'maintain_resolution',
                activation = activation )
        
        _conv2d = conv_nxn(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                mode = 'maintain_resolution')

        torch.nn.init.normal_(_conv2d.conv2d.weight, mean=init_weight_mu, std=init_weight_sigma)
        torch.nn.init.constant_(_conv2d.conv2d.bias, init_bias) 
        self.head = nn.Sequential(_conv2dblk, _conv2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.head(x)
        x = torch.reshape(x, (B, -1, H*W)).contiguous()
        x = torch.permute(x, (0, 2, 1)).contiguous()
        return x
    
# ---------------------------------------------------------------------------------------------------------------------------
class Head_v1(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        stem_channels: Union[List[int], Tuple[int]],
        num_classes: int,
        kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()

        # stem blocks
        self.stem_blk = ConvStemBlock(
            in_channels = in_channels, 
            stem_channels = stem_channels,
            kernel_size = kernel_size,
            activation = activation,
            conv_type = conv_type)

        # Class prediction head
        self.pred_cls = Conv_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = num_classes,
            kernel_size = kernel_size,
            activation = activation,
            conv_type = conv_type,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
        # Box regression head
        self.pred_offsets = Conv_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = 2,
            kernel_size = kernel_size,
            activation = activation,
            conv_type = conv_type,
            init_weight_mu = _REG_CONV_MEAN_INIT_,
            init_weight_sigma = _REG_CONV_STD_INIT_,
            init_bias = _REG_CONV_BIAS_INIT_)

    def forward(self, x: torch.Tensor):
        featmap = self.stem_blk(x) 
        class_logits = self.pred_cls(featmap)            # Class prediction
        reg_deltas = self.pred_offsets(featmap)          # Offset regression
        return class_logits, reg_deltas
    
# ---------------------------------------------------------------------------------------------------------------------------
class FFN_TaskSpecificHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        init_weight_mu: torch.Tensor,
        init_weight_sigma: torch.Tensor,
        init_bias: torch.Tensor):
        super().__init__()

        _ffn_block = ffn_block(
            in_channels = in_channels,
            out_channels = in_channels,
            activation = activation)
        
        _ffn = nn.Linear(
            in_features = in_channels, 
            out_features = out_channels, 
            bias = True)
        
        torch.nn.init.normal_(_ffn.weight, mean=init_weight_mu, std=init_weight_sigma)
        torch.nn.init.constant_(_ffn.bias, init_bias) 
        self.head = nn.Sequential(_ffn_block, _ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

# ---------------------------------------------------------------------------------------------------------------------------
class Head_v2(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        stem_channels: Union[List[int], Tuple[int]],
        ffn_channels: Union[List[int], Tuple[int]],
        num_augmented_features: int,
        reg_offset_dim: int,
        num_classes: int,
        kernel_size: int,
        activation: str,
        conv_type: str):
        super().__init__()

        # stem blocks
        self.stem_blk = ConvStemBlock(
            in_channels = in_channels, 
            stem_channels = stem_channels,
            kernel_size = kernel_size,
            activation = activation,
            conv_type = conv_type)
        
        self.ffn_blk = FFNStemBlock(
            in_channels = in_channels + num_augmented_features,
            stem_channels = ffn_channels,
            activation = activation)

        # Object Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels = ffn_channels[-1],
            out_channels = num_classes,
            activation = activation,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
        # Box regression head
        self.pred_offsets = FFN_TaskSpecificHead(
            in_channels = ffn_channels[-1],
            out_channels = reg_offset_dim,
            activation = activation,
            init_weight_mu = _REG_CONV_MEAN_INIT_,
            init_weight_sigma = _REG_CONV_STD_INIT_,
            init_bias = _REG_CONV_BIAS_INIT_)

    def forward(
        self, 
        x: torch.Tensor,
        grid_cell_labels: torch.Tensor,
        meas_vr: torch.Tensor,
        meas_rcs: torch.Tensor):

        featmap = self.stem_blk(x).permute(0,2,3,1).contiguous()
        meas_vr, meas_rcs = normalize_vr_rcs(meas_vr, meas_rcs)
        new_features =  torch.stack((meas_vr, meas_rcs), axis=-1)
        augmented_features = torch.concatenate((featmap, new_features), axis=-1)

        _, _, _, D = augmented_features.shape
        augmented_features = augmented_features.reshape(-1, D)
        grid_cell_labels = grid_cell_labels.reshape(-1)
        valid_condition = grid_cell_labels != _INVALID_NUM_
        feature_vector = augmented_features[valid_condition]

        feature_vector = self.ffn_blk(feature_vector)
        class_logits = self.pred_cls(feature_vector)            # Class prediction
        reg_deltas = self.pred_offsets(feature_vector)          # Offset regression
        return class_logits, reg_deltas
    
# ---------------------------------------------------------------------------------------------------------------------------
def normalize_vr_rcs(meas_vr, meas_rcs):
    max_vr = 113; min_vr = -107
    max_rcs = 48; min_rcs = -31
    meas_vr = (meas_vr - min_vr) / (max_vr - min_vr)
    meas_rcs = (meas_rcs - min_rcs) / (max_rcs - min_rcs)
    return meas_vr, meas_rcs

