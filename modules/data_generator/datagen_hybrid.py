# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : data generation functions for hybrid (cnn + gnn) based architecture
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from modules.data_utils.read_data import get_data_for_datagen
from modules.data_utils.labels import compute_new_labels_to_id_dict, compute_old_to_new_label_id_map
from modules.data_utils.constants import _REJECT_OUTLIER_BY_RANSAC_
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_groundtruth.compute_edge_labels import compute_ground_truth as compute_ground_truth_edge
from modules.compute_features.graph_features import (
    compute_adjacency_information, compute_node_features, compute_edge_features, select_moving_data)
from modules.compute_features.grid_features import compute_encodings

# ---------------------------------------------------------------------------------------------------------------
class RadarScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        metadatset, 
        data_rootdir, 
        data_path, 
        grid_obj,
        config_obj,
        device):

        self.metadatset = metadatset
        self.data_rootdir = data_rootdir
        self.data_path = data_path
        self.grid_obj = grid_obj
        self.labels_to_id_dict = compute_new_labels_to_id_dict()
        self.old_to_new_label_id_map = compute_old_to_new_label_id_map()
        self.ball_query_thresh = config_obj.ball_query_eps_square
        self.k_number_nearest_points = config_obj.k_number_nearest_points
        self.device = device

    def __len__(self):
        return len(self.metadatset)

    def __getitem__(self, idx: int):

        # Data Selection
        # ===============================================================================================================
        # get data and ground-truths and select only those that are within a defined grid area
        data_dict = get_data_for_datagen(self.data_rootdir, self.data_path, self.metadatset[idx], _REJECT_OUTLIER_BY_RANSAC_)
        gt_dict_node = compute_ground_truth_node(data_dict, self.labels_to_id_dict, self.old_to_new_label_id_map)
        data_dict, gt_dict_node = self.grid_obj.select_meas_within_the_grid(data_dict, gt_dict_node)

        # For graph NN we consider those measurements that are dynamic
        data_dict_dyn, node_labels_dict_dyn = select_moving_data(data_dict, gt_dict_node, self.labels_to_id_dict)
        adj_dict_dyn = compute_adjacency_information(data_dict_dyn, self.ball_query_thresh, self.k_number_nearest_points)

        # Grid Features and Grid Labels Computation
        # ===============================================================================================================
        # select data and labels assuming only one item can lie within the grid cell
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

        grid_features = {}
        grid_features['image_grid'] = torch.stack((meas_likelihood_map, range_map, azimuth_map), axis=-1).permute(2,0,1).contiguous()
        grid_features['meas_vr_grid'] = torch.from_numpy(meas_vr_grid).to(self.device).to(torch.float32)
        grid_features['meas_rcs_grid'] = torch.from_numpy(meas_rcs_grid).to(self.device).to(torch.float32)

        # Node,Edge Feature and Label Computation
        # ===============================================================================================================
        other_features_dyn = np.stack((data_dict_dyn['meas_px'], data_dict_dyn['meas_py'], 
                                       data_dict_dyn['meas_vx'], data_dict_dyn['meas_vy']), axis=-1)
        edge_features_dyn = compute_edge_features(data_dict_dyn, adj_dict_dyn['adj_list'])
        node_features_dyn = compute_node_features(data_dict_dyn, adj_dict_dyn['degree'])

        graph_features = {}
        graph_features['other_features_dyn'] = torch.from_numpy(other_features_dyn).to(self.device).to(torch.float32)
        graph_features['edge_features_dyn'] = torch.from_numpy(edge_features_dyn).to(self.device).to(torch.float32)
        graph_features['node_features_dyn'] = torch.from_numpy(node_features_dyn).to(self.device).to(torch.float32)
        graph_features['edge_index_dyn'] = torch.from_numpy(adj_dict_dyn['adj_list'] ).to(self.device).to(torch.int64)
        graph_features['adj_matrix_dyn'] = torch.from_numpy(adj_dict_dyn['adj_matrix'] ).to(self.device).to(torch.bool)

        edge_labels_dyn = compute_ground_truth_edge(data_dict_dyn, adj_dict_dyn['adj_list'], adj_dict_dyn['adj_matrix'])
        gt_labels_dyn = node_labels_dict_dyn['class_labels']
        gt_offsets_dyn = np.stack([node_labels_dict_dyn['offsetx'], node_labels_dict_dyn['offsety']], axis=-1)

        # label dictionary
        labels = {}
        labels['class'] = torch.from_numpy(gt_labels_grid).to(self.device).to(torch.float32)
        labels['offsets'] = torch.from_numpy(gt_offsets_grid).to(self.device).to(torch.float32)
        labels['edge_class'] = torch.from_numpy(edge_labels_dyn).to(self.device).to(torch.float32)
        labels['node_class'] = torch.from_numpy(gt_labels_dyn).to(self.device).to(torch.float32)
        labels['node_offsets'] = torch.from_numpy(gt_offsets_dyn).to(self.device).to(torch.float32)
        return grid_features, graph_features, labels
    
    def collate_fn(self, sample_batch):
        grid_features_batch, graph_features_batch, labels_batch = self.bdd_collate_fn(sample_batch)
        return grid_features_batch, graph_features_batch, labels_batch
    
    @ staticmethod
    def bdd_collate_fn(sample_batch):
        """ Generate a batch of data """
        img_batch = []
        img_vr_batch = []
        img_rcs_batch = []
        gt_labels_grid_batch = []
        gt_offsets_grid_batch = []

        node_features_batch = []
        edge_features_batch = []
        other_features_batch = []
        edge_idx_batch = []
        adj_matrix_batch = []
        edge_class_batch = []
        node_class_batch = []
        node_offsets_batch = []

        grid_features_batch = {}
        graph_features_batch = {}
        labels_batch = {}

        for i in range(len(sample_batch)):
            grid_features, graph_features, labels = sample_batch[i]

            img_batch.append(grid_features['image_grid'])
            img_vr_batch.append(grid_features['meas_vr_grid'])
            img_rcs_batch.append(grid_features['meas_rcs_grid'])
            gt_labels_grid_batch.append(labels['class'])
            gt_offsets_grid_batch.append(labels['offsets'])

            node_features_batch.append(graph_features['node_features_dyn'])
            edge_features_batch.append(graph_features['edge_features_dyn'])
            other_features_batch.append(graph_features['other_features_dyn'])
            edge_idx_batch.append(graph_features['edge_index_dyn'])
            adj_matrix_batch.append(graph_features['adj_matrix_dyn'])
            edge_class_batch.append(labels['edge_class'])
            node_class_batch.append(labels['node_class'])
            node_offsets_batch.append(labels['node_offsets'])
            
        grid_features_batch['image_grid'] = torch.stack(img_batch, dim=0)
        grid_features_batch['meas_vr_grid'] = torch.stack(img_vr_batch, dim=0)
        grid_features_batch['meas_rcs_grid'] = torch.stack(img_rcs_batch, dim=0)

        graph_features_batch['node_features_dyn'] = node_features_batch
        graph_features_batch['edge_features_dyn'] = edge_features_batch
        graph_features_batch['other_features_dyn'] = other_features_batch
        graph_features_batch['edge_index_dyn'] = edge_idx_batch
        graph_features_batch['adj_matrix_dyn'] = adj_matrix_batch

        labels_batch['class'] = torch.stack(gt_labels_grid_batch, dim=0) 
        labels_batch['offsets'] = torch.stack(gt_offsets_grid_batch, dim=0) 
        labels_batch['edge_class'] = edge_class_batch
        labels_batch['node_class'] = node_class_batch
        labels_batch['node_offsets'] = node_offsets_batch

        return grid_features_batch, graph_features_batch, labels_batch