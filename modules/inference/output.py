# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : inference functions
# ---------------------------------------------------------------------------------------------------------------
import torch, os
import torch.nn.functional as F
import numpy as np

from modules.data_utils.read_data import extract_frame
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_groundtruth.compute_edge_labels import compute_ground_truth as compute_ground_truth_edge
from modules.compute_features.graph_features import select_moving_data
from modules.compute_features.graph_features import compute_adjacency_information
from modules.compute_features.graph_features import compute_edge_features
from modules.compute_features.graph_features import compute_node_features
from modules.compute_groundtruth.compute_offsets import unnormalize_gt_offsets

from modules.inference.inference import compute_proposals
from modules.inference.ellipse import compute_cov_ellipse
from modules.data_generator.datagen_gnn import compute_node_idx_for_each_cluster

from modules.plot_utils.show_outputs import plot_all_outputs
from modules.plot_utils.compare_plots import compare_pred_gt_object_classes

# ---------------------------------------------------------------------------------------------------------------
def process_frame(
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
    sequence_name,
    cluster_size_threshold,
    detect_object_by_segmentation_output=True):

    # print('-' * 100)
    print(f'frame number: {idx}')

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

    edge_labels_dyn = compute_ground_truth_edge(data_dict_dyn, adj_dict_dyn['adj_list'], adj_dict_dyn['adj_matrix'])
    gt_labels_dyn = node_labels_dict_dyn['class_labels']
    gt_offsets_dyn = np.stack([node_labels_dict_dyn['offsetx'], node_labels_dict_dyn['offsety']], axis=-1)

    # cluster offsets predictions
    cluster_node_idx, cluster_labels = compute_node_idx_for_each_cluster(data_dict_dyn['meas_trackid'], gt_labels_dyn, device)
    # node_cls_predictions, node_offsets_predictions, edge_cls_predictions, obj_cls_predictions = detector(
    #     node_features = graph_features['node_features_dyn'],
    #     edge_features = graph_features['edge_features_dyn'],
    #     edge_index = graph_features['edge_index_dyn'],
    #     adj_matrix = graph_features['adj_matrix_dyn'],
    #     cluster_node_idx = cluster_node_idx )

    node_cls_predictions, node_offsets_predictions, edge_cls_predictions, \
    obj_cls_predictions, cluster_members_list = detector(
        node_features = graph_features['node_features_dyn'],
        edge_features = graph_features['edge_features_dyn'],
        other_features = graph_features['other_features_dyn'],
        edge_index = graph_features['edge_index_dyn'],
        adj_matrix = graph_features['adj_matrix_dyn'])

    cluster_members_list_pred = cluster_members_list
    cluster_members_list_gt = cluster_node_idx

    # predict cluster centers
    reg_deltas = unnormalize_gt_offsets(node_offsets_predictions, config_obj.reg_mu, config_obj.reg_sigma)
    pred_cluster_centers_xy = graph_features['other_features_dyn'][:, :2] + reg_deltas

    # predict node class
    node_cls_prob = F.softmax(node_cls_predictions, dim=-1)
    node_cls_score, node_pred_class = torch.max(node_cls_prob, dim=-1)

    # predict edge class
    edge_cls_prob= F.softmax(edge_cls_predictions, dim=-1)
    edge_cls_score, edge_cls_idx = torch.max(edge_cls_prob, dim=-1)

    # predict object class
    if detect_object_by_segmentation_output == True:
        obj_pred_cls_list = []
        for cluster_mem in cluster_members_list_pred:
            cluster_meas_labels = node_pred_class[cluster_mem]
            obj_pred_cls_list.append(torch.argmax(torch.bincount(cluster_meas_labels)))
        obj_cls_idx = torch.stack(obj_pred_cls_list, dim=0)

    else:
        obj_cls_prob = F.softmax(obj_cls_predictions, dim=-1)
        obj_cls_score, obj_cls_idx = torch.max(obj_cls_prob, dim=-1)

    flag = obj_cls_idx != 6
    valid_cluster_members_list_pred = []
    for i, members in enumerate(cluster_members_list_pred):
        if flag[i]: valid_cluster_members_list_pred.append(members)
    cluster_members_list_pred = valid_cluster_members_list_pred
    obj_cls_idx = obj_cls_idx[flag] 

    # -----------------------------------------------------------------------------------------------------
    px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
    py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()
    pred_class = node_pred_class.detach().cpu().numpy()
    pred_edge_class = edge_cls_idx.detach().cpu().numpy()

    pred_cluster_centers_xy = pred_cluster_centers_xy.detach().cpu().numpy()
    pred_cluster_centers_x_all = pred_cluster_centers_xy[:,0]
    pred_cluster_centers_y_all = pred_cluster_centers_xy[:,1]
    gt_cluster_centers_x_all = px + node_labels_dict_dyn['offsetx']
    gt_cluster_centers_y_all = py + node_labels_dict_dyn['offsety']
    
    flag = gt_labels_dyn != 6
    pred_cluster_centers_x_valid = pred_cluster_centers_x_all[flag]
    pred_cluster_centers_y_valid = pred_cluster_centers_y_all[flag]
    gt_cluster_centers_x_valid = gt_cluster_centers_x_all[flag]
    gt_cluster_centers_y_valid = gt_cluster_centers_y_all[flag]

    adj_matrix = graph_features['adj_matrix_dyn']
    valid_row_idx, valid_col_idx = torch.nonzero(torch.triu(adj_matrix, diagonal=1), as_tuple=True)
    edge_index = torch.stack((valid_row_idx, valid_col_idx), axis=0)
    edge_index = edge_index.detach().cpu().numpy()

    # -----------------------------------------------------------------------------------------------------
    obj_class_pred = obj_cls_idx.detach().cpu().numpy().tolist()
    obj_class_gt = cluster_labels.detach().cpu().numpy().tolist()

    cluster_mean_list_pred, cluster_cov_list_pred, cluster_size_pred \
        = compute_proposals(cluster_members_list_pred, px, py, detector.meas_noise_cov)

    cluster_mean_list_gt, cluster_cov_list_gt, cluster_size_gt \
        = compute_proposals(cluster_members_list_gt, px, py, detector.meas_noise_cov)

    cluster_boundary_points_list_pred = []
    for mu, cov in zip(cluster_mean_list_pred, cluster_cov_list_pred):
        boundary_points, _ = compute_cov_ellipse(mu, cov, chi_sq = 2, n_points=100)
        cluster_boundary_points_list_pred.append(boundary_points)

    cluster_boundary_points_list_gt = []
    for mu, cov in zip(cluster_mean_list_gt, cluster_cov_list_gt):
        boundary_points, _ = compute_cov_ellipse(mu, cov, chi_sq = 2, n_points=100)
        cluster_boundary_points_list_gt.append(boundary_points)

    # -----------------------------------------------------------------------------------------------------
    out_dir = 'results/outputs/' + sequence_name + '/'
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + str(idx) + '.png'

    plot_all_outputs(
        px, py, config_obj.object_classes_dyn,
        pred_class, 
        pred_cluster_centers_x_valid, pred_cluster_centers_y_valid,
        edge_index, pred_edge_class,
        obj_class_pred,
        cluster_mean_list_pred, 
        cluster_boundary_points_list_pred,
        cluster_size_pred,
        cluster_size_threshold,
        figsize=(11,10),
        save_plot = True,
        out_file = out_file)
    




def compare_pred_and_gt_cluster(
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
    sequence_name,
    cluster_size_threshold,
    detect_object_by_segmentation_output=True):

    # print('-' * 100)
    print(f'frame number: {idx}')

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

    edge_labels_dyn = compute_ground_truth_edge(data_dict_dyn, adj_dict_dyn['adj_list'], adj_dict_dyn['adj_matrix'])
    gt_labels_dyn = node_labels_dict_dyn['class_labels']
    gt_offsets_dyn = np.stack([node_labels_dict_dyn['offsetx'], node_labels_dict_dyn['offsety']], axis=-1)

    # cluster offsets predictions
    cluster_node_idx, cluster_labels = compute_node_idx_for_each_cluster(data_dict_dyn['meas_trackid'], gt_labels_dyn, device)
    # node_cls_predictions, node_offsets_predictions, edge_cls_predictions, obj_cls_predictions = detector(
    #     node_features = graph_features['node_features_dyn'],
    #     edge_features = graph_features['edge_features_dyn'],
    #     edge_index = graph_features['edge_index_dyn'],
    #     adj_matrix = graph_features['adj_matrix_dyn'],
    #     cluster_node_idx = cluster_node_idx )

    node_cls_predictions, node_offsets_predictions, edge_cls_predictions, \
    obj_cls_predictions, cluster_members_list = detector(
        node_features = graph_features['node_features_dyn'],
        edge_features = graph_features['edge_features_dyn'],
        other_features = graph_features['other_features_dyn'],
        edge_index = graph_features['edge_index_dyn'],
        adj_matrix = graph_features['adj_matrix_dyn'])

    cluster_members_list_pred = cluster_members_list
    cluster_members_list_gt = cluster_node_idx

    # predict cluster centers
    reg_deltas = unnormalize_gt_offsets(node_offsets_predictions, config_obj.reg_mu, config_obj.reg_sigma)
    pred_cluster_centers_xy = graph_features['other_features_dyn'][:, :2] + reg_deltas

    # predict node class
    node_cls_prob = F.softmax(node_cls_predictions, dim=-1)
    node_cls_score, node_pred_class = torch.max(node_cls_prob, dim=-1)

    # predict edge class
    edge_cls_prob= F.softmax(edge_cls_predictions, dim=-1)
    edge_cls_score, edge_cls_idx = torch.max(edge_cls_prob, dim=-1)

    # predict object class
    if detect_object_by_segmentation_output == True:
        obj_pred_cls_list = []
        for cluster_mem in cluster_members_list_pred:
            cluster_meas_labels = node_pred_class[cluster_mem]
            obj_pred_cls_list.append(torch.argmax(torch.bincount(cluster_meas_labels)))
        obj_cls_idx = torch.stack(obj_pred_cls_list, dim=0)

    else:
        # predict object class
        obj_cls_prob = F.softmax(obj_cls_predictions, dim=-1)
        obj_cls_score, obj_cls_idx = torch.max(obj_cls_prob, dim=-1)

    flag = obj_cls_idx != 6
    valid_cluster_members_list_pred = []
    for i, members in enumerate(cluster_members_list_pred):
        if flag[i]: valid_cluster_members_list_pred.append(members)
    cluster_members_list_pred = valid_cluster_members_list_pred
    obj_cls_idx = obj_cls_idx[flag]

    # -----------------------------------------------------------------------------------------------------
    px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
    py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()
    pred_class = node_pred_class.detach().cpu().numpy()
    pred_edge_class = edge_cls_idx.detach().cpu().numpy()

    pred_cluster_centers_xy = pred_cluster_centers_xy.detach().cpu().numpy()
    pred_cluster_centers_x_all = pred_cluster_centers_xy[:,0]
    pred_cluster_centers_y_all = pred_cluster_centers_xy[:,1]
    gt_cluster_centers_x_all = px + node_labels_dict_dyn['offsetx']
    gt_cluster_centers_y_all = py + node_labels_dict_dyn['offsety']

    flag = gt_labels_dyn != 6
    pred_cluster_centers_x_valid = pred_cluster_centers_x_all[flag]
    pred_cluster_centers_y_valid = pred_cluster_centers_y_all[flag]
    gt_cluster_centers_x_valid = gt_cluster_centers_x_all[flag]
    gt_cluster_centers_y_valid = gt_cluster_centers_y_all[flag]

    adj_matrix = graph_features['adj_matrix_dyn']
    valid_row_idx, valid_col_idx = torch.nonzero(torch.triu(adj_matrix, diagonal=1), as_tuple=True)
    edge_index = torch.stack((valid_row_idx, valid_col_idx), axis=0)
    edge_index = edge_index.detach().cpu().numpy()

    # -----------------------------------------------------------------------------------------------------
    obj_class_pred = obj_cls_idx.detach().cpu().numpy().tolist()
    obj_class_gt = cluster_labels.detach().cpu().numpy().tolist()

    cluster_mean_list_pred, cluster_cov_list_pred, cluster_size_pred \
        = compute_proposals(cluster_members_list_pred, px, py, detector.meas_noise_cov)

    cluster_mean_list_gt, cluster_cov_list_gt, cluster_size_gt \
        = compute_proposals(cluster_members_list_gt, px, py, detector.meas_noise_cov)

    cluster_boundary_points_list_pred = []
    for mu, cov in zip(cluster_mean_list_pred, cluster_cov_list_pred):
        boundary_points, _ = compute_cov_ellipse(mu, cov, chi_sq = 2, n_points=100)
        cluster_boundary_points_list_pred.append(boundary_points)

    cluster_boundary_points_list_gt = []
    for mu, cov in zip(cluster_mean_list_gt, cluster_cov_list_gt):
        boundary_points, _ = compute_cov_ellipse(mu, cov, chi_sq = 2, n_points=100)
        cluster_boundary_points_list_gt.append(boundary_points)

    # -----------------------------------------------------------------------------------------------------
    out_dir = 'results/compare/' + sequence_name + '/'
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + str(idx) + '.png'

    compare_pred_gt_object_classes(
        px, py, 
        obj_class_pred,
        cluster_mean_list_pred, 
        cluster_boundary_points_list_pred,
        cluster_size_pred,
        obj_class_gt,
        cluster_mean_list_gt, 
        cluster_boundary_points_list_gt,
        cluster_size_gt,
        config_obj.object_classes_dyn,
        cluster_size_threshold,
        figsize=(16, 8),
        save_plot = True,
        out_file = out_file)