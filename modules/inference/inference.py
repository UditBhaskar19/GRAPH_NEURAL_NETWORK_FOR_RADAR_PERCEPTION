# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : inference functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from modules.compute_groundtruth.compute_offsets import unnormalize_gt_offsets
from modules.inference.clustering import compute_adjacency_mat_from_predicted_offsets

# ---------------------------------------------------------------------------------------------------------------
def extract_proposals(
    pred_cluster_centers_xy,
    clustering_obj):
    # perform clustering
    clustering_obj.cluster_nodes(pred_cluster_centers_xy)
    # get cluster members
    cluster_members_list = []
    for i in range(clustering_obj.num_clusters):
        cluster_members = np.nonzero(clustering_obj.meas_to_cluster_id == i)
        cluster_members_list.append(cluster_members[0])
    return cluster_members_list

# ---------------------------------------------------------------------------------------------------------------
def compute_cluster_sample_mean_and_cov(meas_vector, meas_noise_cov):
    meas_mu = np.sum(meas_vector, axis=0) / meas_vector.shape[0]
    if meas_vector.shape[0] > 1:
        error = np.expand_dims((meas_mu[:2] - meas_vector[:, :2]), axis=-1)
        sigma = error @ error.transpose(0, 2, 1)
        sigma = np.sum(sigma, axis=0) / ( meas_vector.shape[0] - 1 ) + meas_noise_cov
    else: sigma = meas_noise_cov
    return meas_mu, sigma

# ---------------------------------------------------------------------------------------------------------------
def compute_proposals(cluster_members_list, px, py, meas_noise_cov):
    node_xy = np.stack((px, py), axis=-1)
    cluster_mean_list = []
    cluster_cov_list = []
    cluster_size = []

    for cluster_members in cluster_members_list:
        cluster_members = cluster_members.detach().cpu().numpy().tolist()
        meas_vector = node_xy[cluster_members]
        mu, sigma = compute_cluster_sample_mean_and_cov(meas_vector[:, :2], meas_noise_cov)
        cluster_mean_list.append(mu)
        cluster_cov_list.append(sigma)
        cluster_size.append(len(cluster_members))
    
    return cluster_mean_list, cluster_cov_list, cluster_size

# ---------------------------------------------------------------------------------------------------------------
# def compute_proposals(
#     detector, 
#     graph_features, 
#     graph_meas_labels,
#     config_obj, 
#     clustering_obj, 
#     clustering_eps,
#     num_meas_thr):

#     _, node_offsets_predictions, _ = detector(
#         node_features = graph_features['node_features_dyn'],
#         edge_features = graph_features['edge_features_dyn'],
#         edge_index = graph_features['edge_index_dyn'],
#         adj_matrix = graph_features['adj_matrix_dyn'] )

#     # compute offsets
#     offset_mu = config_obj.reg_mu
#     offset_sigma = config_obj.reg_sigma
#     reg_deltas = unnormalize_gt_offsets(node_offsets_predictions, offset_mu, offset_sigma)
#     pred_cluster_centers_xy = graph_features['other_features_dyn'][:, :2] + reg_deltas

#     # move data to cpu
#     px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
#     py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()
#     rcs = graph_features['node_features_dyn'][:, 1].detach().cpu().numpy()
#     pred_cluster_centers_xy = pred_cluster_centers_xy.detach().cpu().numpy()

#     # perform clustering
#     node_xyrcs = np.stack((px, py, rcs), axis=-1)
#     adj_mat = compute_adjacency_mat_from_predicted_offsets(pred_cluster_centers_xy, clustering_eps)
#     clustering_obj.cluster_nodes(node_xyrcs[:, :2], adj_mat)

#     # aggregate valid clusters : clustera whose number of measurements is >= num_meas_thr
#     cluster_mean_list = []
#     cluster_cov_list = []
#     cluster_measurements_list = []    # px, py, rcs for each measurements
#     cluster_measurements_label_list = []
#     for i in range(clustering_obj.num_clusters):
#         if clustering_obj.num_measurements[i] >= num_meas_thr:
#             mu = clustering_obj.cluster_mean[i]
#             cov = clustering_obj.cluster_covariance[i]
#             flag = clustering_obj.meas_to_cluster_id == i
#             meas = node_xyrcs[flag]
#             meas_labels = graph_meas_labels[flag]
#             cluster_mean_list.append(mu)
#             cluster_cov_list.append(cov)
#             cluster_measurements_list.append(meas)
#             cluster_measurements_label_list.append(meas_labels)
    
#     return \
#         cluster_mean_list, \
#         cluster_cov_list, \
#         cluster_measurements_list, \
#         cluster_measurements_label_list

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
         
