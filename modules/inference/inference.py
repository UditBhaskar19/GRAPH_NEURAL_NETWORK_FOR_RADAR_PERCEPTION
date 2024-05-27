# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : inference functions
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from modules.compute_groundtruth.compute_offsets import unnormalize_gt_offsets
from modules.inference.clustering import compute_adjacency_mat_from_predicted_offsets

# ---------------------------------------------------------------------------------------------------------------
def compute_proposals(
    detector, 
    graph_features, 
    graph_meas_labels,
    config_obj, 
    clustering_obj, 
    clustering_eps,
    num_meas_thr):

    _, node_offsets_predictions, _ = detector(
        node_features = graph_features['node_features_dyn'],
        edge_features = graph_features['edge_features_dyn'],
        edge_index = graph_features['edge_index_dyn'],
        adj_matrix = graph_features['adj_matrix_dyn'] )

    # compute offsets
    offset_mu = config_obj.reg_mu
    offset_sigma = config_obj.reg_sigma
    reg_deltas = unnormalize_gt_offsets(node_offsets_predictions, offset_mu, offset_sigma)
    pred_cluster_centers_xy = graph_features['other_features_dyn'][:, :2] + reg_deltas

    # move data to cpu
    px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
    py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()
    rcs = graph_features['node_features_dyn'][:, 1].detach().cpu().numpy()
    pred_cluster_centers_xy = pred_cluster_centers_xy.detach().cpu().numpy()

    # perform clustering
    node_xyrcs = np.stack((px, py, rcs), axis=-1)
    adj_mat = compute_adjacency_mat_from_predicted_offsets(pred_cluster_centers_xy, clustering_eps)
    clustering_obj.cluster_nodes(node_xyrcs[:, :2], adj_mat)

    # aggregate valid clusters : clustera whose number of measurements is >= num_meas_thr
    cluster_mean_list = []
    cluster_cov_list = []
    cluster_measurements_list = []    # px, py, rcs for each measurements
    cluster_measurements_label_list = []
    for i in range(clustering_obj.num_clusters):
        if clustering_obj.num_measurements[i] >= num_meas_thr:
            mu = clustering_obj.cluster_mean[i]
            cov = clustering_obj.cluster_covariance[i]
            flag = clustering_obj.meas_to_cluster_id == i
            meas = node_xyrcs[flag]
            meas_labels = graph_meas_labels[flag]
            cluster_mean_list.append(mu)
            cluster_cov_list.append(cov)
            cluster_measurements_list.append(meas)
            cluster_measurements_label_list.append(meas_labels)
    
    return \
        cluster_mean_list, \
        cluster_cov_list, \
        cluster_measurements_list, \
        cluster_measurements_label_list

# ---------------------------------------------------------------------------------------------------------------
def compute_object_labels(cluster_measurements_label_list):
    labels_list = []
    for labels in cluster_measurements_label_list:
        unique_labels = np.unique(labels)
        label_instances = []
        for idx in range(unique_labels.shape[0]):
            n = np.sum(labels == unique_labels[idx])
            label_instances.append(n)
        # print(label_instances)
        label_instances = np.array(label_instances)
        label = unique_labels[np.argmax(label_instances)]
        labels_list.append(label)
    return labels_list
         
