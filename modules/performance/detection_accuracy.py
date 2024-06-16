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
    cluster_size_threshold,
    detect_object_by_segmentation_output=True,
    remove_meas_with_invalid_labels=True):

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

    if remove_meas_with_invalid_labels == True:
        flag = node_labels_dict_dyn['class_labels'] != 5
        for key, value in node_labels_dict_dyn.items():
            node_labels_dict_dyn[key] = value[flag]
        for key, value in data_dict_dyn.items():
            data_dict_dyn[key] = value[flag]

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

        if detect_object_by_segmentation_output == True:
            # predicted segmentation class
            node_cls_prob = F.softmax(node_cls_predictions, dim=-1)
            node_cls_score, node_pred_class = torch.max(node_cls_prob, dim=-1)

            obj_pred_cls_list = []
            for cluster_mem in cluster_members_list_pred:
                cluster_meas_labels = node_pred_class[cluster_mem]
                obj_pred_cls_list.append(torch.argmax(torch.bincount(cluster_meas_labels)))
            obj_cls_idx = torch.stack(obj_pred_cls_list, dim=0)

        else:
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
        
        meas_idx = np.arange(px.shape[0])
        
        for i, cluster_members in enumerate(cluster_members_list_pred):
            cluster_meas_idx = meas_idx[cluster_members.detach().cpu().numpy()]
            cluster_members_list_pred[i] = set(cluster_meas_idx)

        for i, cluster_members in enumerate(cluster_members_list_gt):
            cluster_meas_idx = meas_idx[cluster_members.detach().cpu().numpy()]
            cluster_members_list_gt[i] = set(cluster_meas_idx)
        
        # -----------------------------------------------------------------------------------------------------
        # keep only the valid objects
        filtered_cluster_member_list_pred = []
        filtered_cluster_mean_list_pred = []
        filtered_cluster_cov_list_pred = []
        filtered_cluster_size_pred = []
        filtered_obj_class_pred = []
        for cluster_mem, cluster_mean, cluster_cov, cluster_size, obj_cls in \
            zip(cluster_members_list_pred, cluster_mean_list_pred, cluster_cov_list_pred, cluster_size_pred, obj_class_pred):
            if cluster_size > cluster_size_threshold:
                filtered_cluster_member_list_pred.append(cluster_mem)
                filtered_cluster_mean_list_pred.append(cluster_mean)
                filtered_cluster_cov_list_pred.append(cluster_cov)
                filtered_cluster_size_pred.append(cluster_size)
                filtered_obj_class_pred.append(obj_cls)

        # -----------------------------------------------------------------------------------------------------
        # keep only the valid objects
        filtered_cluster_member_list_gt = []
        filtered_cluster_mean_list_gt = []
        filtered_cluster_cov_list_gt = []
        filtered_cluster_size_gt = []
        filtered_obj_class_gt = []
        for cluster_mem, cluster_mean, cluster_cov, cluster_size, obj_cls in \
            zip(cluster_members_list_gt, cluster_mean_list_gt, cluster_cov_list_gt, cluster_size_gt, obj_class_gt):
            if cluster_size > cluster_size_threshold:
                filtered_cluster_member_list_gt.append(cluster_mem)
                filtered_cluster_mean_list_gt.append(cluster_mean)
                filtered_cluster_cov_list_gt.append(cluster_cov)
                filtered_cluster_size_gt.append(cluster_size)
                filtered_obj_class_gt.append(obj_cls)

    else: 
        filtered_cluster_member_list_pred = []
        filtered_obj_class_pred = []
        filtered_cluster_mean_list_pred = []
        filtered_cluster_cov_list_pred = []
        filtered_cluster_size_pred = []
        filtered_cluster_member_list_gt = []
        filtered_obj_class_gt = []
        filtered_cluster_mean_list_gt = []
        filtered_cluster_cov_list_gt = []
        filtered_cluster_size_gt = []
     
    return {
        'obj_class_pred': filtered_obj_class_pred,
        'cluster_member_list_pred': filtered_cluster_member_list_pred,
        'cluster_mean_list_pred': filtered_cluster_mean_list_pred,
        'cluster_cov_list_pred': filtered_cluster_cov_list_pred,
        'cluster_size_pred': filtered_cluster_size_pred,
        'cluster_member_list_gt': filtered_cluster_member_list_gt,
        'obj_class_gt': filtered_obj_class_gt,
        'cluster_mean_list_gt': filtered_cluster_mean_list_gt,
        'cluster_cov_list_gt': filtered_cluster_cov_list_gt,
        'cluster_size_gt': filtered_cluster_size_gt
    }

# ---------------------------------------------------------------------------------------------------------------
def compute_gt_and_pred_associations(
    det_and_gt_clusters, eps, 
    association_criteria='inv_iou', 
    false_class_label = 6):

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

        cluster_mem_gt = det_and_gt_clusters['cluster_member_list_gt']
        cluster_mem_pred = det_and_gt_clusters['cluster_member_list_pred']
        inv_ious_mat = np.zeros((len(cluster_mem_gt), len(cluster_mem_pred)), dtype=np.float32)
        for row_idx in range(len(cluster_mem_gt)):
            for col_idx in range(len(cluster_mem_pred)):
                iou = len(cluster_mem_gt[row_idx] & cluster_mem_pred[col_idx]) \
                    / len(cluster_mem_gt[row_idx] | cluster_mem_pred[col_idx])
                inv_ious_mat[row_idx, col_idx] = 1 - iou

        if association_criteria == 'inv_iou': dist_mat = inv_ious_mat
        elif association_criteria == 'l2_norm': dist_mat = l2_dist
        associations = []
        distances = []
        
        min_num_objects = np.min(dist_mat.shape)
        for i in range(min_num_objects):
            coordinates = np.stack(np.nonzero(dist_mat == np.min(dist_mat)), axis=-1)[0]
            associations.append(coordinates)
            distances.append(dist_mat[coordinates[0], coordinates[1]])
            dist_mat[coordinates[0], :] = VERY_LARGE_NUM
            dist_mat[:, coordinates[1]] = VERY_LARGE_NUM

        associations = np.stack(associations, axis=0)
        associations_pos = associations[np.array(distances) <= eps]
        associations_neg = associations[np.array(distances) >  eps]

        pos_obj_class_gt_associated = obj_class_gt[associations_pos[:, 0]]
        pos_obj_class_pred_associated = obj_class_pred[associations_pos[:, 1]]
        neg_obj_class_pred_associated = obj_class_pred[associations_neg[:, 1]]
        neg_obj_class_gt_associated = np.repeat(false_class_label, repeats=neg_obj_class_pred_associated.shape[0])

        obj_class_gt_associated = np.concatenate(
            (pos_obj_class_gt_associated, neg_obj_class_gt_associated), axis=0)
        obj_class_pred_associated = np.concatenate(
            (pos_obj_class_pred_associated, neg_obj_class_pred_associated), axis=0)

    elif condition2 == True:
        obj_class_gt_associated = empty_array
        obj_class_pred_associated = empty_array
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
        'obj_class_pred': obj_class_pred }