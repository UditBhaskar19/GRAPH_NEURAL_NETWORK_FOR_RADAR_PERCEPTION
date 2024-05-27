# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : data generation functions for cnn based architecture
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from modules.data_utils.read_data import get_data_for_datagen
from modules.data_utils.labels import compute_new_labels_to_id_dict, compute_old_to_new_label_id_map
from modules.data_utils.constants import _REJECT_OUTLIER_BY_RANSAC_
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_features.grid_features import compute_encodings

# ---------------------------------------------------------------------------------------------------------------
class RadarScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        metadatset, 
        data_rootdir, 
        data_path, 
        grid_obj,
        device):

        self.device = device
        self.metadatset = metadatset
        self.data_rootdir = data_rootdir
        self.data_path = data_path
        self.grid_obj = grid_obj
        self.labels_to_id_dict = compute_new_labels_to_id_dict()
        self.old_to_new_label_id_map = compute_old_to_new_label_id_map()
        

    def __len__(self):
        return len(self.metadatset)

    def __getitem__(self, idx: int):

        # get data and ground-truths and select only those that are within a defined grid area
        data_dict = get_data_for_datagen(self.data_rootdir, self.data_path, self.metadatset[idx], _REJECT_OUTLIER_BY_RANSAC_)
        gt_dict_node = compute_ground_truth_node(data_dict, self.labels_to_id_dict, self.old_to_new_label_id_map)
        data_dict, gt_dict_node = self.grid_obj.select_meas_within_the_grid(data_dict, gt_dict_node)

        # select data assuming only one measurement can lie within the grid cell
        meas_xy_coord_grid, meas_vr_grid, meas_rcs_grid, gt_labels_grid, gt_offsets_grid \
            = self.grid_obj.gridify_measurements_and_labels(
                np.stack((data_dict['meas_px'], data_dict['meas_py']), axis=-1), 
                data_dict['meas_vr'], data_dict['meas_rcs'], 
                gt_dict_node['class_labels'], 
                np.stack((gt_dict_node['offsetx'], gt_dict_node['offsety']), axis=-1))
        
        # compute input feature maps
        meas_cov = self.grid_obj.compute_covariance_linearly_varying(meas_xy_coord_grid[:,0], meas_xy_coord_grid[:,1])
        meas_likelihood_map, range_map, azimuth_map \
            = compute_encodings(self.grid_obj.xy_coord, meas_xy_coord_grid, meas_cov, self.device)
        image_grid = torch.stack((meas_likelihood_map, range_map, azimuth_map), axis=0)
        meas_vr_grid = torch.from_numpy(meas_vr_grid).to(self.device).to(torch.float32)
        meas_rcs_grid = torch.from_numpy(meas_rcs_grid).to(self.device).to(torch.float32)

        # label dictionary
        labels = {}
        labels['class'] = torch.from_numpy(gt_labels_grid).to(self.device).to(torch.float32)
        labels['offsets'] = torch.from_numpy(gt_offsets_grid).to(self.device).to(torch.float32)
        return image_grid, meas_vr_grid, meas_rcs_grid, labels
    
    def collate_fn(self, sample_batch):
        img_batch, img_vr_batch, img_rcs_batch, labels_batch = self.bdd_collate_fn(sample_batch)
        return img_batch, img_vr_batch, img_rcs_batch, labels_batch
    
    @ staticmethod
    def bdd_collate_fn(sample_batch):
        """ Generate a batch of data """
        img_batch = []
        img_vr_batch = []
        img_rcs_batch = []
        gt_labels_grid_batch = []
        gt_offsets_grid_batch = []
        labels_batch = {}
        
        for i in range(len(sample_batch)):
            img, img_vr, img_rcs, labels = sample_batch[i]
            img_batch.append(img)
            img_vr_batch.append(img_vr)
            img_rcs_batch.append(img_rcs)
            gt_labels_grid_batch.append(labels['class'])
            gt_offsets_grid_batch.append(labels['offsets'])
            
        img_batch = torch.stack(img_batch, dim=0)
        img_vr_batch = torch.stack(img_vr_batch, dim=0)
        img_rcs_batch = torch.stack(img_rcs_batch, dim=0)
        labels_batch['class'] = torch.stack(gt_labels_grid_batch, dim=0) 
        labels_batch['offsets'] = torch.stack(gt_offsets_grid_batch, dim=0) 
        return img_batch, img_vr_batch, img_rcs_batch, labels_batch