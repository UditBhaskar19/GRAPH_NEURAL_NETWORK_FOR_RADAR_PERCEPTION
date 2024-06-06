# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Segmentation performance evaluation
# ---------------------------------------------------------------------------------------------------------------
import torch, os
import torch.nn.functional as F
import numpy as np

from modules.data_utils.read_data import extract_frame
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_features.graph_features import select_moving_data
from modules.compute_features.graph_features import compute_adjacency_information
from modules.compute_features.graph_features import compute_edge_features
from modules.compute_features.graph_features import compute_node_features

# ---------------------------------------------------------------------------------------------------------------
def compute_gt_and_pred_semantic_class(
    idx,
    scene_metadata,
    radar_mount_data,
    radar_data_all_scenes,
    odometry_data_all_scenes,
    labels_to_id_dict,
    old_to_new_label_id_map,
    grid, 
    config_obj, 
    device,
    detector):

    # print(f'frame number: {idx}')

    data_dict = extract_frame(
        idx = idx, 
        windowed_data_list = scene_metadata,
        radar_mount_data = radar_mount_data,
        radar_data_all_scenes = radar_data_all_scenes,
        odometry_data_all_scenes = odometry_data_all_scenes,
        reject_outlier = False)
    
    gt_dict_node = compute_ground_truth_node(data_dict, labels_to_id_dict, old_to_new_label_id_map)
    data_dict, gt_dict_node = grid.select_meas_within_the_grid(data_dict, gt_dict_node)
    data_dict_dyn, node_labels_dict_dyn = select_moving_data(data_dict, gt_dict_node, labels_to_id_dict)
    
    num_meas = data_dict_dyn['meas_px'].shape[0]
    if num_meas > 1:

        adj_dict_dyn = compute_adjacency_information(data_dict_dyn, config_obj.ball_query_eps_square, config_obj.k_number_nearest_points)

        other_features_dyn = np.stack((
            data_dict_dyn['meas_px'], data_dict_dyn['meas_py'], 
            data_dict_dyn['meas_vx'], data_dict_dyn['meas_vy']), axis=-1)
        edge_features_dyn = compute_edge_features(data_dict_dyn, adj_dict_dyn['adj_list'])
        node_features_dyn = compute_node_features(
            data_dict_dyn, adj_dict_dyn['degree'], 
            include_region_confidence = config_obj.include_region_confidence, 
            min_range = config_obj.grid_min_r, max_range = config_obj.grid_max_r, 
            min_azimuth = config_obj.grid_min_th, max_azimuth = config_obj.grid_max_th)

        graph_features = {}
        graph_features['other_features_dyn'] = torch.from_numpy(other_features_dyn).to(device).to(torch.float32)
        graph_features['edge_features_dyn'] = torch.from_numpy(edge_features_dyn).to(device).to(torch.float32)
        graph_features['node_features_dyn'] = torch.from_numpy(node_features_dyn).to(device).to(torch.float32)
        graph_features['edge_index_dyn'] = torch.from_numpy(adj_dict_dyn['adj_list'] ).to(device).to(torch.int64)
        graph_features['adj_matrix_dyn'] = torch.from_numpy(adj_dict_dyn['adj_matrix'] ).to(device).to(torch.bool)

        node_cls_predictions, node_offsets_predictions, edge_cls_predictions, \
        obj_cls_predictions, cluster_members_list = detector(
            node_features = graph_features['node_features_dyn'],
            edge_features = graph_features['edge_features_dyn'],
            other_features = graph_features['other_features_dyn'],
            edge_index = graph_features['edge_index_dyn'],
            adj_matrix = graph_features['adj_matrix_dyn'])
        
        # predict node class
        cls_prob = F.softmax(node_cls_predictions, dim=-1)
        cls_score, cls_idx = torch.max(cls_prob, dim=-1)

        gt_labels_dyn = node_labels_dict_dyn['class_labels']
        pred_class = cls_idx.detach().cpu().numpy()

    else: 
        gt_labels_dyn = np.zeros(shape=(0, ))
        pred_class = np.zeros(shape=(0, ))

    return {
        'gt_semantic_class': gt_labels_dyn,
        'pred_semantic_class': pred_class }