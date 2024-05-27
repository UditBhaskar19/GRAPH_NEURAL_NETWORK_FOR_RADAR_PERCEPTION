# ------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : common loss functions
# ------------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
from modules.data_utils.labels import _INVALID_NUM_
from modules.neural_net.lossfunc import CE_loss, MSE_loss

# --------------------------------------------------------------------------------------------------------------
class Loss_Grid(nn.Module):
    def __init__(
        self,
        net_config, 
        device: str):
        super().__init__()

        self.cls_loss_weight = net_config.cls_loss_weight
        self.reg_loss_weight = net_config.reg_loss_weight
        self.new_labels_to_id_dict = net_config.new_labels_to_id_dict

        self.num_classes = len(net_config.object_classes)
        self.class_weights = torch.tensor(net_config.class_weights, dtype=torch.float32, device=device)
        self.device = device

        self.cls_loss = CE_loss()
        self.reg_loss = MSE_loss() 

    def compute_valid_object_mask(self, gt_class_logits):
        condition_object = torch.logical_and(gt_class_logits != _INVALID_NUM_,    
                                    torch.logical_and(gt_class_logits != self.new_labels_to_id_dict['STATIC'], 
                                                      gt_class_logits != self.new_labels_to_id_dict['FALSE']))
        return condition_object
        
    def forward(self, pred, gt):
        # extract and flatten gt data (note: the gt class labels are not in one-hot form)
        gt_class_logits = gt.class_logits
        gt_reg_deltas = gt.reg_deltas

        # extract and flatten predictions
        pred_class_logits = pred.class_logits
        pred_reg_deltas = pred.reg_deltas

        # compute empty and valid_object mask
        valid_cell_mask = gt_class_logits != _INVALID_NUM_
        valid_obj_mask = self.compute_valid_object_mask(gt_class_logits)
        invalid_cell_mask = ~valid_cell_mask
        
        # compute class labels and object type one-hot
        gt_class_logits[invalid_cell_mask] = 0
        gt_class_logits = torch.nn.functional.one_hot(gt_class_logits.to(int), self.num_classes).to(torch.float32)
        
        # compute loss
        cls_loss = self.cls_loss(pred_class_logits, gt_class_logits, self.class_weights)
        reg_loss = 0.5 * self.reg_loss(pred_reg_deltas, gt_reg_deltas).sum(-1) 

        # set class and reg losses to 0 if no +ve samples are present
        N = valid_cell_mask.sum().item()
        if N == 0: cls_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        else: cls_loss = torch.where(valid_cell_mask, cls_loss, 0.0).sum() / N
        
        N = valid_obj_mask.sum().item()
        if N == 0: reg_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        else: reg_loss = torch.where(valid_obj_mask, reg_loss, 0.0).sum() / N

        return {
            'loss_cls': cls_loss * self.cls_loss_weight,
            'loss_reg': reg_loss * self.reg_loss_weight }    
