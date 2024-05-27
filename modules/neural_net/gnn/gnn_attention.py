# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Graph attention network building blocks
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn.conv import GATv2Conv
from modules.neural_net.common import ffn_block, layer_normalization

# ---------------------------------------------------------------------------------------------------------------
class residual_graph_attn_block(nn.Module):
    def __init__(
        self,
        in_node_channels: int,
        hidden_node_channels: int,
        in_edge_channels: int,
        num_heads: int,
        mlp_stem_channels_upd: List[int],
        activation: str,
        in_extra_feature_dim: Optional[int] = None):
        super().__init__()

        # graph attention block (node aggregation)
        self.GATblk = GATv2Conv(
            in_channels = in_node_channels,
            out_channels = hidden_node_channels // num_heads,
            heads = num_heads,
            edge_dim = in_edge_channels,
            concat = True,
            negative_slope = 0.2,
            dropout = 0.0,
            add_self_loops = False,
            share_weights = False,
            bias = True)
        
        # node update block
        self.in_extra_feature_dim = in_extra_feature_dim 
        in_channels = in_node_channels + hidden_node_channels
        if in_extra_feature_dim != None:
            in_channels = in_channels + in_extra_feature_dim 
        
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
            norm_ = layer_normalization()
            self.residual_connection = nn.Sequential(ffn_, norm_)

    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_features: torch.Tensor,    # dimension: (|E| x d_edge)
        edge_index: torch.Tensor,       # dimension: (2 x |E|)
        extra_features: Optional[torch.Tensor] = None):  # dimension: (|V| x d_aug)

        if self.match_channels: identity = self.residual_connection(node_features)
        else: identity = node_features

        x = self.GATblk(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        if self.in_extra_feature_dim != None: x = torch.concat((node_features, extra_features, x), dim=-1)
        else: x = torch.concat((node_features, x), dim=-1)
        x = identity + self.upd(x)
        return x
    
# ---------------------------------------------------------------------------------------------------------------
class graph_attention(nn.Module):
    def __init__(
        self, 
        in_node_channels: int,
        in_edge_channels: int,
        stem_channels: List[int],
        hidden_node_channels: int,
        num_heads: int,
        activation: str,
        append_extra_features: Optional[List[bool]] = None,
        in_extra_feature_dim: Optional[int] = None):
        super().__init__()

        self.conv_blk = nn.ModuleList()
        for i, stem_channel in enumerate(stem_channels):
            in_extra_feature = None
            if append_extra_features != None:
                if append_extra_features[i] == True and in_extra_feature_dim != None:
                    in_extra_feature = in_extra_feature_dim

            self.conv_blk.append( residual_graph_attn_block(
                in_node_channels = in_node_channels,
                hidden_node_channels = hidden_node_channels,
                in_edge_channels = in_edge_channels,
                num_heads = num_heads,
                mlp_stem_channels_upd = [hidden_node_channels // 2, hidden_node_channels // 4, stem_channel],
                activation = activation,
                in_extra_feature_dim = in_extra_feature ) )
            in_node_channels = stem_channel
            
    def forward(
        self, 
        node_features: torch.Tensor,    # dimension: (|V| x d_node)
        edge_features: torch.Tensor,    # dimension: (|E| x d_edge)
        edge_index: torch.Tensor,       # dimension: (2 x |E|)
        extra_features: Optional[torch.Tensor]):  # dimension: (|V| x d_aug)

        x = node_features
        for conv_blk in self.conv_blk:
            x = conv_blk(
                node_features = x,
                edge_features = edge_features,
                edge_index = edge_index,
                extra_features = extra_features)
        return x