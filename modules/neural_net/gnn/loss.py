# ------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : common loss functions
# ------------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
from modules.neural_net.lossfunc import CE_loss, MSE_loss, Focal_Loss

# --------------------------------------------------------------------------------------------------------------
class Loss_Graph(nn.Module):
    def __init__(
        self,
        net_config, 
        device: str):
        super().__init__()

        self.node_cls_loss_weight = net_config.node_cls_loss_weight
        self.edge_cls_loss_weight = net_config.edge_cls_loss_weight
        self.node_reg_loss_weight = net_config.node_reg_loss_weight
        self.obj_cls_loss_weight = net_config.obj_cls_loss_weight
        self.new_labels_to_id_dict = net_config.new_labels_to_id_dict_dyn
        
        self.num_classes_edge = net_config.num_edge_classes
        self.num_classes = net_config.num_classes
        self.class_weights = torch.tensor(net_config.class_weights_dyn, dtype=torch.float32, device=device)
        self.device = device

        self.edge_cls_loss = Focal_Loss()
        self.obj_cls_loss = CE_loss()
        self.node_cls_loss = CE_loss()
        self.node_reg_loss = MSE_loss() 

    def compute_valid_object_mask(self, gt_class_logits):
        condition_object = gt_class_logits != self.new_labels_to_id_dict['FALSE']
        return condition_object
        
    def forward(self, pred, gt):

        # extract gt data (note: the gt class labels are not in one-hot form)
        gt_node_class_logits = gt.node_class_logits
        gt_node_reg_deltas = gt.node_reg_deltas
        gt_edge_class_logits = gt.edge_class_logits
        gt_obj_class_logits = gt.obj_class_logits

        # extract predictions
        pred_node_class_logits = pred.node_class_logits
        pred_node_reg_deltas = pred.node_reg_deltas
        pred_edge_class_logits = pred.edge_class_logits
        pred_obj_class_logits = pred.obj_class_logits

        # compute node class (one-hot) and edge class
        gt_node_class_logits = torch.nn.functional.one_hot(gt_node_class_logits, self.num_classes).to(torch.float32)
        gt_edge_class_logits = torch.nn.functional.one_hot(gt_edge_class_logits, self.num_classes_edge).to(torch.float32)
        gt_obj_class_logits = torch.nn.functional.one_hot(gt_obj_class_logits, self.num_classes).to(torch.float32)

        # compute edge class loss
        edge_cls_loss = self.edge_cls_loss(pred_edge_class_logits, gt_edge_class_logits).sum(-1)
        edge_cls_loss = edge_cls_loss.sum() / edge_cls_loss.shape[0]

        # compute node class loss
        node_cls_loss = self.node_cls_loss(pred_node_class_logits, gt_node_class_logits, self.class_weights)
        node_cls_loss = node_cls_loss.sum() / node_cls_loss.shape[0]

        # compute regression loss
        node_reg_loss = 0.5 * self.node_reg_loss(pred_node_reg_deltas, gt_node_reg_deltas).sum(-1) 
        node_reg_loss = node_reg_loss.sum() / node_reg_loss.shape[0]

        # compute object class loss
        obj_cls_loss = self.obj_cls_loss(pred_obj_class_logits, gt_obj_class_logits)
        obj_cls_loss = obj_cls_loss.sum() / obj_cls_loss.shape[0]
        
        return {
            'loss_node_cls': node_cls_loss * self.node_cls_loss_weight,
            'loss_node_reg': node_reg_loss * self.node_reg_loss_weight,
            'loss_edge_cls': edge_cls_loss * self.edge_cls_loss_weight,
            'loss_obj_cls': obj_cls_loss * self.obj_cls_loss_weight }    
