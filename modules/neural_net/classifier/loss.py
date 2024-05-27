import torch
from torch import nn
from modules.neural_net.lossfunc import Focal_Loss

class Loss(nn.Module):
    def __init__(self, net_config):
        super().__init__()
        self.num_classes = net_config.num_classes
        self.obj_cls_loss = Focal_Loss(alpha=-1)

    def forward(self, pred, gt):
        gt = torch.nn.functional.one_hot(gt, self.num_classes).to(torch.float32)
        obj_cls_loss = self.obj_cls_loss(pred, gt).sum(-1)
        obj_cls_loss = obj_cls_loss.sum() / obj_cls_loss.shape[0]
        return obj_cls_loss