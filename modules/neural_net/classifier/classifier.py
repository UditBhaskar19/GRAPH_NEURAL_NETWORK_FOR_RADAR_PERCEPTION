import torch
from torch import nn
from typing import List
from modules.neural_net.classifier.blocks import graph_feature_encoding, graph_convolution, object_class_prediction
from modules.neural_net.classifier.loss import Loss

# --------------------------------------------------------------------------------------------------------------
class Model_Inference(nn.Module):
    def __init__(self, net_config):
        super().__init__()

        activation = net_config.classifier_activation
        aggregation = net_config.classifier_aggregation

        input_node_feat_dim = net_config.classifier_input_node_feat_dim
        node_feat_enc_stem_channels = net_config.classifier_node_feat_enc_stem_channels
        graph_convolution_stem_channels = net_config.classifier_graph_convolution_stem_channels

        msg_mlp_hidden_dim = net_config.classifier_msg_mlp_hidden_dim
        node_pred_stem_channels = net_config.classifier_node_pred_stem_channels 
        num_classes = net_config.num_classes

        self.encode_node_feat = graph_feature_encoding(
            in_channels = input_node_feat_dim, 
            stem_channels = node_feat_enc_stem_channels,
            activation = activation)
        
        self.pass_messages = graph_convolution(
            in_node_channels = node_feat_enc_stem_channels[-1],
            stem_channels = graph_convolution_stem_channels,
            msg_mlp_hidden_dim = msg_mlp_hidden_dim,
            activation = activation,
            aggregation = aggregation)
        
        self.predict_node = object_class_prediction(
            in_channels = graph_convolution_stem_channels[-1],
            stem_channels = node_pred_stem_channels,
            num_classes = num_classes,
            activation = activation)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        object_size: torch.Tensor):   # number of measurements of a cluster

        node_cls_predictions = []
        node_features = self.encode_node_feat(node_features)
        node_features = self.pass_messages(node_features, edge_index)

        startidx = torch.zeros_like(object_size)
        startidx[1:] = object_size[:-1]
        endidx =  torch.cumsum(object_size, dim=0)

        for i in range(object_size.shape[0]):
            node_feat = node_features[startidx[i]:endidx[i]]
            node_cls_pred = self.predict_node(node_feat)
            node_cls_predictions.append(node_cls_pred)

        node_cls_predictions = torch.concat(node_cls_predictions, dim=0)
        return node_cls_predictions
    
# ---------------------------------------------------------------------------------------------------------------
class Model_Training(nn.Module):
    def __init__(self, net_config):
        super().__init__()   
        self.pred = Model_Inference(net_config)
        self.loss = Loss(net_config)

    def forward(
        self,
        node_features: List[torch.Tensor],
        edge_index: List[torch.Tensor],
        object_size: List[torch.Tensor],
        groundtruths: List[torch.Tensor]):

        # compute predictions
        object_cls_pred_list = []
        for node_feat, edge_idx, obj_size in zip(node_features, edge_index, object_size):
            obj_cls_pred = self.pred(node_feat, edge_idx, obj_size)
            object_cls_pred_list.append(obj_cls_pred)
        predictions = torch.concat(object_cls_pred_list, dim=0)

        # compute loss
        groundtruths = torch.concat(groundtruths, dim=0)
        object_class_loss = self.loss(predictions, groundtruths)
        return object_class_loss 