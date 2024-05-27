import torch
import torch.nn as nn
from typing import List
from torch_geometric.nn.conv import MessagePassing
from modules.neural_net.common import ffn_block, channel_normalization
from modules.neural_net.constants import _CLS_CONV_MEAN_INIT_, _CLS_CONV_STD_INIT_, _CLS_CONV_BIAS_INIT_

# ---------------------------------------------------------------------------------------------------------------
class graph_feature_encoding(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: List[int],
        activation: str):
        super().__init__()

        encoder = []
        for stem_channel in stem_channels:
            blk = ffn_block(in_channels=in_channels, out_channels=stem_channel, activation=activation)
            in_channels = stem_channel
            encoder.append(blk)
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

# ---------------------------------------------------------------------------------------------------------------
class residual_graph_conv_block(MessagePassing):
    def __init__(
        self, 
        in_node_channels: int,
        mlp_stem_channels_msg: List[int],
        mlp_stem_channels_upd: List[int],
        aggregation: str,
        activation: str):
        super().__init__(aggr=aggregation, flow="source_to_target")

        msg = []
        in_channels = 2 * in_node_channels
        for stem_channel in mlp_stem_channels_msg:
            blk = ffn_block(in_channels=in_channels, out_channels=stem_channel, activation=activation)
            in_channels = stem_channel
            msg.append(blk)
        self.msg = nn.Sequential(*msg)

        in_channels = in_node_channels + mlp_stem_channels_msg[-1]

        upd = []
        for stem_channel in mlp_stem_channels_upd:
            blk = ffn_block(in_channels=in_channels, out_channels=stem_channel, activation=activation)
            in_channels = stem_channel
            upd.append(blk)
        self.upd = nn.Sequential(*upd)

        # residual connection
        self.match_channels = False
        if in_node_channels != mlp_stem_channels_upd[-1]:
            self.match_channels = True
        
        self.residual_connection = None
        if self.match_channels:
            ffn_ = nn.Linear(in_features = in_node_channels,  out_features = mlp_stem_channels_upd[-1], bias = True)
            norm_ = channel_normalization()
            self.residual_connection = nn.Sequential(ffn_, norm_)
        
    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_index: torch.Tensor):      # dimension: (2 x |E|)

        if self.match_channels: identity = self.residual_connection(node_features)
        else: identity = node_features

        x = self.propagate(edge_index, x=node_features)
        x = torch.concat((node_features, x), dim=-1)
        x = identity + self.upd(x)
        return x
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor):
        return self.msg(torch.concat((x_i, x_j), dim=-1))
    
# ---------------------------------------------------------------------------------------------------------------
class graph_convolution(nn.Module):
    def __init__(
        self, 
        in_node_channels: int,
        stem_channels: List[int],
        msg_mlp_hidden_dim: int,
        activation: str,
        aggregation: str):
        super().__init__()

        self.conv_blk = nn.ModuleList()
        for _, stem_channel in enumerate(stem_channels):

            self.conv_blk.append( residual_graph_conv_block(
                in_node_channels = in_node_channels,
                mlp_stem_channels_msg = [msg_mlp_hidden_dim, stem_channel],
                mlp_stem_channels_upd = [stem_channel],
                aggregation = aggregation,
                activation = activation) )
            in_node_channels = stem_channel
            
    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_index: torch.Tensor):      # dimension: (2 x |E|)

        x = node_features
        for conv_blk in self.conv_blk:
            x = conv_blk(
                node_features = x,
                edge_index = edge_index)
        return x
    
# ---------------------------------------------------------------------------------------------------------------
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
    
# ---------------------------------------------------------------------------------------------------------------
class object_class_prediction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        num_classes: int,
        activation: str):
        super().__init__()

        stem_blks = []
        for stem_channel in stem_channels:
            blk = ffn_block(in_channels=in_channels, out_channels=stem_channel, activation=activation)
            in_channels = stem_channel
            stem_blks.append(blk)
        self.stem = nn.Sequential(*stem_blks)

        # Object Class prediction head
        self.pred_cls = FFN_TaskSpecificHead(
            in_channels = stem_channels[-1],
            out_channels = num_classes,
            activation = activation,
            init_weight_mu = _CLS_CONV_MEAN_INIT_,
            init_weight_sigma = _CLS_CONV_STD_INIT_,
            init_bias = _CLS_CONV_BIAS_INIT_)
        
    def forward(self, x: torch.Tensor):
        x, _ = torch.max(x, keepdim=True, dim=0)
        x = self.stem(x)
        pred_cls = self.pred_cls(x)
        return pred_cls

