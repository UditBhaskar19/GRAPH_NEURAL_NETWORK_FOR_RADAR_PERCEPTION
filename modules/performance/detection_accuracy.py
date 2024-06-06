# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Detection performance evaluation
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

from modules.inference.inference import compute_proposals
from modules.data_generator.datagen_gnn import compute_node_idx_for_each_cluster

VERY_LARGE_NUM = 9999999

# ---------------------------------------------------------------------------------------------------------------
def compute_gt_and_pred_objects(
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
    detector, 
    cluster_size_threshold):

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

    gt_labels_dyn = node_labels_dict_dyn['class_labels']
    cluster_node_idx, cluster_labels = compute_node_idx_for_each_cluster(data_dict_dyn['meas_trackid'], gt_labels_dyn, device)

    node_cls_predictions, node_offsets_predictions, edge_cls_predictions, \
    obj_cls_predictions, cluster_members_list = detector(
        node_features = graph_features['node_features_dyn'],
        edge_features = graph_features['edge_features_dyn'],
        other_features = graph_features['other_features_dyn'],
        edge_index = graph_features['edge_index_dyn'],
        adj_matrix = graph_features['adj_matrix_dyn'])

    cluster_members_list_pred = cluster_members_list
    cluster_members_list_gt = cluster_node_idx

    # predict object class
    obj_cls_prob = F.softmax(obj_cls_predictions, dim=-1)
    obj_cls_score, obj_cls_idx = torch.max(obj_cls_prob, dim=-1)

    px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
    py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()

    # -----------------------------------------------------------------------------------------------------
    obj_class_pred = obj_cls_idx.detach().cpu().numpy().tolist()
    obj_class_gt = cluster_labels.detach().cpu().numpy().tolist()

    cluster_mean_list_pred, cluster_cov_list_pred, cluster_size_pred \
        = compute_proposals(cluster_members_list_pred, px, py, detector.meas_noise_cov)

    cluster_mean_list_gt, cluster_cov_list_gt, cluster_size_gt \
        = compute_proposals(cluster_members_list_gt, px, py, detector.meas_noise_cov)
    
    # -----------------------------------------------------------------------------------------------------
    # keep only the valid objects
    filtered_cluster_mean_list_pred = []
    filtered_cluster_cov_list_pred = []
    filtered_cluster_size_pred = []
    filtered_obj_class_pred = []
    for cluster_mean, cluster_cov, cluster_size, obj_cls in \
        zip(cluster_mean_list_pred, cluster_cov_list_pred, cluster_size_pred, obj_class_pred):
        if cluster_size > cluster_size_threshold:
            filtered_cluster_mean_list_pred.append(cluster_mean)
            filtered_cluster_cov_list_pred.append(cluster_cov)
            filtered_cluster_size_pred.append(cluster_size)
            filtered_obj_class_pred.append(obj_cls)

    # -----------------------------------------------------------------------------------------------------
    # keep only the valid objects
    filtered_cluster_mean_list_gt = []
    filtered_cluster_cov_list_gt = []
    filtered_cluster_size_gt = []
    filtered_obj_class_gt = []
    for cluster_mean, cluster_cov, cluster_size, obj_cls in \
        zip(cluster_mean_list_gt, cluster_cov_list_gt, cluster_size_gt, obj_class_gt):
        if cluster_size > 1:
            filtered_cluster_mean_list_gt.append(cluster_mean)
            filtered_cluster_cov_list_gt.append(cluster_cov)
            filtered_cluster_size_gt.append(cluster_size)
            filtered_obj_class_gt.append(obj_cls)
     
    return {
        'obj_class_pred': filtered_obj_class_pred,
        'cluster_mean_list_pred': filtered_cluster_mean_list_pred,
        'cluster_cov_list_pred': filtered_cluster_cov_list_pred,
        'cluster_size_pred': filtered_cluster_size_pred,
        'obj_class_gt': filtered_obj_class_gt,
        'cluster_mean_list_gt': filtered_cluster_mean_list_gt,
        'cluster_cov_list_gt': filtered_cluster_cov_list_gt,
        'cluster_size_gt': filtered_cluster_size_gt,
    }

# ---------------------------------------------------------------------------------------------------------------
def compute_gt_and_pred_associations(det_and_gt_clusters, eps):

    condition1 = len(det_and_gt_clusters['cluster_mean_list_pred']) > 0 and len(det_and_gt_clusters['cluster_mean_list_gt']) > 0 
    condition2 = len(det_and_gt_clusters['cluster_mean_list_pred']) == 0 and len(det_and_gt_clusters['cluster_mean_list_gt']) > 0
    condition3 = len(det_and_gt_clusters['cluster_mean_list_pred']) > 0 and len(det_and_gt_clusters['cluster_mean_list_gt']) == 0  
    condition4 = len(det_and_gt_clusters['cluster_mean_list_pred']) == 0 and len(det_and_gt_clusters['cluster_mean_list_gt']) == 0
    empty_array = np.zeros(shape=(0, ))

    if condition1 == True:

        cluster_mean_pred = np.stack(det_and_gt_clusters['cluster_mean_list_pred'], axis=0)
        cluster_mean_gt = np.stack(det_and_gt_clusters['cluster_mean_list_gt'], axis=0)

        obj_class_pred = np.array(det_and_gt_clusters['obj_class_pred'])
        obj_class_gt = np.array(det_and_gt_clusters['obj_class_gt'])

        cluster_mean_gt = np.expand_dims(cluster_mean_gt, axis=1)
        cluster_mean_pred = np.expand_dims(cluster_mean_pred, axis=0)
        l2_dist = np.linalg.norm(cluster_mean_gt - cluster_mean_pred, axis=-1)

        associations = []
        distances = []

        min_num_objects = np.min(l2_dist.shape)
        for i in range(min_num_objects):
            coordinates = np.stack(np.nonzero(l2_dist == np.min(l2_dist)), axis=-1)[0]
            associations.append(coordinates)
            distances.append(l2_dist[coordinates[0], coordinates[1]])
            l2_dist[coordinates[0], :] = VERY_LARGE_NUM
            l2_dist[:, coordinates[1]] = VERY_LARGE_NUM

        associations = np.stack(associations, axis=0)
        associations = associations[np.array(distances) <= eps]

        obj_class_gt_associated = obj_class_gt[associations[:, 0]]
        obj_class_pred_associated = obj_class_pred[associations[:, 1]]

    elif condition2 == True:
        obj_class_gt_associated = empty_array
        obj_class_pred_associated =empty_array
        obj_class_pred = empty_array
        obj_class_gt = np.array(det_and_gt_clusters['obj_class_gt'])

    elif condition3 == True:
        obj_class_gt_associated = empty_array
        obj_class_pred_associated = empty_array
        obj_class_pred = np.array(det_and_gt_clusters['obj_class_pred'])
        obj_class_gt = empty_array
        
    elif condition4 == True:
        obj_class_gt_associated = empty_array
        obj_class_pred_associated = empty_array
        obj_class_pred = empty_array
        obj_class_gt = empty_array

    return {
        'obj_class_gt_associated': obj_class_gt_associated,
        'obj_class_pred_associated': obj_class_pred_associated,
        'obj_class_gt': obj_class_gt,
        'obj_class_pred': obj_class_pred,
    }