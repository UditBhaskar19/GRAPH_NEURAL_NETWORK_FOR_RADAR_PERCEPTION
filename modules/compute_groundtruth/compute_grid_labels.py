# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : 
# ---------------------------------------------------------------------------------------------------------------
import torch
from typing import List
from modules.data_utils.labels import _INVALID_NUM_
from modules.compute_groundtruth.compute_offsets import normalize_gt_offsets

# --------------------------------------------------------------------------------------------------------------
def compute_training_groundtruth(
    gt_labels_img: torch.Tensor, 
    gt_offsets_img: torch.Tensor,
    offset_mu: List[float],
    offset_sigma: List[float]):

    gt_labels_img = gt_labels_img.reshape(-1)
    gt_offsets_img = gt_offsets_img.reshape(-1, 2)

    valid_condition = gt_labels_img != _INVALID_NUM_
    gt_labels_img = gt_labels_img[valid_condition]
    gt_offsets_img = gt_offsets_img[valid_condition]
    gt_offsets_img = normalize_gt_offsets(gt_offsets_img, offset_mu, offset_sigma)
    return gt_labels_img, gt_offsets_img

