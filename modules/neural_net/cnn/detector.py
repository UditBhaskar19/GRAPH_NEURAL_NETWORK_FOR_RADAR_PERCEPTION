# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Integrated model for inference and training
# NOTE: not used in the current version
# ---------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
from typing import List
from collections import namedtuple
from modules.neural_net.cnn.backbone import Backbone
from modules.neural_net.cnn.aggregation import Neck
from modules.neural_net.cnn.head import Head_v2 as Head
from modules.neural_net.cnn.loss import Loss_Grid
from modules.compute_groundtruth.compute_grid_labels import compute_training_groundtruth
det_named_tuple = namedtuple('det_named_tuple', ['class_logits', 'reg_deltas'])

# --------------------------------------------------------------------------------------------------------------
class Model_Inference(nn.Module):
    def __init__(self, net_config):
        super().__init__()
       
        in_channels = net_config.input_image_dimension
        base_stem_channels = net_config.base_stem_channels
        base_kernel_sizes = net_config.base_kernel_sizes

        bottleneck_number_of_blocks = net_config.bottleneck_number_of_blocks
        bottleneck_stem_channels = net_config.bottleneck_stem_channels
        bottleneck_width_channels = net_config.bottleneck_width_channels
        bottleneck_kernel_size = net_config.bottleneck_kernel_size

        neck_out_channels = net_config.neck_out_channels
        neck_kernel_size = net_config.neck_kernel_size

        head_stem_channels = net_config.head_stem_channels
        head_ffn_channels = net_config.head_ffn_channels
        head_kernel_size = net_config.head_kernel_size

        activation = net_config.activation
        conv_type = net_config.conv_type

        object_classes = net_config.object_classes
        augmented_features = net_config.augmented_features
        reg_offset = net_config.reg_offset

        self.backbone = Backbone(
            in_channels = in_channels,
            base_stem_channels = base_stem_channels,
            base_kernel_sizes = base_kernel_sizes,
            bottleneck_number_of_blocks = bottleneck_number_of_blocks,
            bottleneck_stem_channels = bottleneck_stem_channels,
            bottleneck_width_channels = bottleneck_width_channels,
            bottleneck_kernel_size = bottleneck_kernel_size,
            activation = activation,
            conv_type = conv_type)

        self.neck = Neck(
            input_image_dimension = in_channels,
            layer_channels = base_stem_channels[-1:] + bottleneck_stem_channels,
            out_channels = neck_out_channels,
            kernel_size = neck_kernel_size,
            activation = activation,
            conv_type = conv_type)

        self.head = Head(
            in_channels = neck_out_channels, 
            stem_channels = head_stem_channels,
            ffn_channels = head_ffn_channels,
            num_augmented_features = len(augmented_features),
            reg_offset_dim = len(reg_offset),
            num_classes = len(object_classes),
            kernel_size = head_kernel_size,
            activation = activation,
            conv_type = conv_type)
        
    def forward(
        self, 
        x_img: torch.Tensor, 
        grid_cell_labels: torch.Tensor,
        meas_vr_img: torch.Tensor, 
        meas_rcs_img: torch.Tensor):

        x = self.backbone(x_img) 
        x = self.neck(x, x_img)
        class_logits, reg_deltas = self.head(x, grid_cell_labels, meas_vr_img, meas_rcs_img)
        return class_logits, reg_deltas
    
# --------------------------------------------------------------------------------------------------------------
class Model_Train(nn.Module):
    def __init__(self, net_config, device):
        super().__init__()
        self.device = device
        self.offset_mu = net_config.offset_mu
        self.offset_sigma = net_config.offset_sigma

        self.pred = Model_Inference(net_config)
        self.loss = Loss_Grid(net_config, device)
        
    def forward(
        self, 
        x: torch.Tensor, 
        gt_labels_img: torch.Tensor, 
        gt_offsets_img: torch.Tensor,
        meas_vr_img: torch.Tensor, 
        meas_rcs_img: torch.Tensor):

        # compute predictions
        class_logits, reg_deltas = self.pred(x, gt_labels_img, meas_vr_img, meas_rcs_img)
        predictions = det_named_tuple(class_logits, reg_deltas)

        # compute ground-truths
        gt_labels_img, gt_offsets_img = compute_training_groundtruth(
            gt_labels_img, gt_offsets_img, self.offset_mu, self.offset_sigma)
        groundtruths = det_named_tuple(gt_labels_img, gt_offsets_img)
    
        # compute loss
        loss = self.loss(predictions, groundtruths)
        return {
            'loss_cls': loss['loss_cls'],
            'loss_reg': loss['loss_reg'],
        }