# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : data generation functions for graph NN
# ---------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from modules.data_utils.read_data import get_data_for_datagen
from modules.data_utils.labels import compute_new_labels_to_id_dict, compute_old_to_new_label_id_map
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_groundtruth.compute_edge_labels import compute_ground_truth as compute_ground_truth_edge
from modules.compute_features.graph_features import (
    compute_adjacency_information, compute_node_features, compute_edge_features, select_moving_data)

# ---------------------------------------------------------------------------------------------------------------
def compute_node_idx_for_each_cluster(meas_trackid, gt_labels_dyn, device):
    """ Extract the measurement ids and ground-truth object labels for each cluster
    Inputs: meas_trackid - gt track ids for each measurement
          : gt_labels_dyn - gt class for each measurement 
          : device - cpu or gpu
    Outputs: cluster_node_idx_list - a list of arrays of cluster measurement idx
           : cluster_gt_labels_list - array of object class for each cluster
    """
    cluster_node_idx_list = []
    cluster_gt_labels_list = []
    unique_ids = np.unique(meas_trackid)
    unique_ids = unique_ids[unique_ids != b''] 

    for i in range(unique_ids.shape[0]):
        trackid = unique_ids[i]
        flag = meas_trackid == trackid
        track_meas_ids = np.nonzero(flag)[0]
        gt_labels = gt_labels_dyn[flag][0]

        track_meas_ids = torch.from_numpy(track_meas_ids).to(device).to(int)
        cluster_node_idx_list.append(track_meas_ids)
        cluster_gt_labels_list.append(gt_labels)

    for i in range(meas_trackid.shape[0]):
        if meas_trackid[i] == b'':
            meas_id = torch.tensor([i]).to(device).to(int)
            cluster_node_idx_list.append(meas_id)
            cluster_gt_labels_list.append(gt_labels_dyn[i])

    cluster_gt_labels_list = torch.tensor(cluster_gt_labels_list).to(device).to(int)
    return cluster_node_idx_list, cluster_gt_labels_list

# ---------------------------------------------------------------------------------------------------------------
class RadarScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        metadatset, 
        data_rootdir,
        grid_obj,
        config_obj,
        device):

        self.device = device
        self.metadatset = metadatset
        self.data_rootdir = data_rootdir
        self.grid_obj = grid_obj
        self.labels_to_id_dict = compute_new_labels_to_id_dict()
        self.old_to_new_label_id_map = compute_old_to_new_label_id_map()

        self.data_path = config_obj.dataset_path
        self.ball_query_thresh = config_obj.ball_query_eps_square
        self.k_number_nearest_points = config_obj.k_number_nearest_points
        self.reject_static_meas_by_ransac = config_obj.reject_static_meas_by_ransac
        self.dataset_augmentation = config_obj.dataset_augmentation
        self.include_region_confidence = config_obj.include_region_confidence

        if self.dataset_augmentation == True:
            print('NOTE: augmentation enabled...')

        self.grid_min_th = 0
        self.grid_max_th = np.pi * 0.5
        self.grid_min_r = 0
        self.grid_max_r = np.sqrt(self.grid_obj.max_x ** 2 + self.grid_obj.max_y ** 2)
        
    def __len__(self):
        return len(self.metadatset)

    def __getitem__(self, idx: int):

        graph_features = None
        labels = None

        # Data Selection
        # ===============================================================================================================
        # get data and ground-truths and select only those that are within a defined grid area

        # data augmentation : flip measurements along x axis
        flip_along_x = False
        if self.dataset_augmentation == True:
            if np.random.rand() >= 0.5: flip_along_x = True

        data_dict = get_data_for_datagen(self.data_rootdir, self.data_path, self.metadatset[idx], 
                                         self.reject_static_meas_by_ransac, flip_along_x)
        gt_dict_node = compute_ground_truth_node(data_dict, self.labels_to_id_dict, self.old_to_new_label_id_map)
        data_dict, gt_dict_node = self.grid_obj.select_meas_within_the_grid(data_dict, gt_dict_node)

        # For graph NN we consider those measurements that are dynamic
        data_dict_dyn, node_labels_dict_dyn = select_moving_data(data_dict, gt_dict_node, self.labels_to_id_dict)

        if data_dict_dyn['meas_px'].shape[0] > 1:

            adj_dict_dyn = compute_adjacency_information(data_dict_dyn, self.ball_query_thresh, self.k_number_nearest_points)

            # Node,Edge Feature and Label Computation
            # ===============================================================================================================
            other_features_dyn = np.stack((data_dict_dyn['meas_px'], data_dict_dyn['meas_py'], 
                                           data_dict_dyn['meas_vx'], data_dict_dyn['meas_vy']), axis=-1)
            edge_features_dyn = compute_edge_features(data_dict_dyn, adj_dict_dyn['adj_list'])
            node_features_dyn = compute_node_features(
                data_dict_dyn, adj_dict_dyn['degree'], 
                include_region_confidence = self.include_region_confidence, 
                min_range = self.grid_min_r, max_range = self.grid_max_r, 
                min_azimuth = self.grid_min_th, max_azimuth = self.grid_max_th)

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
            labels['edge_class'] = torch.from_numpy(edge_labels_dyn).to(self.device).to(int)
            labels['node_class'] = torch.from_numpy(gt_labels_dyn).to(self.device).to(int)
            labels['node_offsets'] = torch.from_numpy(gt_offsets_dyn).to(self.device).to(torch.float32)

            cluster_node_idx, cluster_labels = compute_node_idx_for_each_cluster(
                data_dict_dyn['meas_trackid'], gt_labels_dyn, self.device)
            labels['cluster_node_idx'] = cluster_node_idx
            labels['cluster_labels'] = cluster_labels

        return graph_features, labels
    
    @ staticmethod
    def collate_fn(sample_batch):
        """ Generate a batch of data """

        other_features_batch = []
        node_features_batch = []
        edge_features_batch = []
        edge_idx_batch = []
        adj_matrix_batch = []
        edge_class_batch = []
        node_class_batch = []
        node_offsets_batch = []
        cluster_node_idx_batch = []
        cluster_labels_batch = []

        graph_features_batch = None
        labels_batch = None

        for i in range(len(sample_batch)):
            graph_features, labels = sample_batch[i]
            if graph_features != None:
                node_features_batch.append(graph_features['node_features_dyn'])
                edge_features_batch.append(graph_features['edge_features_dyn'])
                other_features_batch.append(graph_features['other_features_dyn'])
                edge_idx_batch.append(graph_features['edge_index_dyn'])
                adj_matrix_batch.append(graph_features['adj_matrix_dyn'])
                edge_class_batch.append(labels['edge_class'])
                node_class_batch.append(labels['node_class'])
                node_offsets_batch.append(labels['node_offsets'])
                cluster_node_idx_batch.append(labels['cluster_node_idx'])
                cluster_labels_batch.append(labels['cluster_labels'])

        if len(node_features_batch) > 0:
            graph_features_batch = {}
            graph_features_batch['other_features_dyn'] = other_features_batch
            graph_features_batch['node_features_dyn'] = node_features_batch
            graph_features_batch['edge_features_dyn'] = edge_features_batch
            graph_features_batch['edge_index_dyn'] = edge_idx_batch
            graph_features_batch['adj_matrix_dyn'] = adj_matrix_batch
        
            labels_batch = {}
            labels_batch['edge_class'] = edge_class_batch
            labels_batch['node_class'] = node_class_batch
            labels_batch['node_offsets'] = node_offsets_batch
            labels_batch['cluster_node_idx'] = cluster_node_idx_batch
            labels_batch['cluster_labels'] = cluster_labels_batch

        return graph_features_batch, labels_batch