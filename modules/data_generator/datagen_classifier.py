# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : data generation functions classification module
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from scipy.linalg import block_diag

from modules.data_utils.read_data import get_data_for_datagen
from modules.data_utils.labels import compute_new_labels_to_id_dict, compute_old_to_new_label_id_map
from modules.compute_groundtruth.compute_node_labels import compute_ground_truth as compute_ground_truth_node
from modules.compute_features.graph_features import (
    compute_adjacency_information, compute_node_features, compute_edge_features, select_moving_data)

from modules.data_generator.datagen_gnn import compute_node_idx_for_each_cluster
from modules.compute_groundtruth.compute_offsets import unnormalize_gt_offsets
from modules.set_configurations.set_param_for_inference_gnn import set_parameters_for_inference
from modules.inference.clustering import Simple_DBSCAN

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

def shift_and_rotate_points(samples_xy,  mu, sigma):
    eigen_values, eigen_vectors = np.linalg.eig(sigma) 
    samples_xy_transformed = (samples_xy - mu) @ eigen_vectors
    return samples_xy_transformed, eigen_values

# ---------------------------------------------------------------------------------------------------------------
def compute_object_labels(cluster_meas_labels):
    unique_labels = np.unique(cluster_meas_labels)
    label_instances = []
    for idx in range(unique_labels.shape[0]):
        n = np.sum(cluster_meas_labels == unique_labels[idx])
        label_instances.append(n)
    # print(label_instances)
    label_instances = np.array(label_instances)
    label = unique_labels[np.argmax(label_instances)]
    return label

# ---------------------------------------------------------------------------------------------------------------
def extract_and_compute_features_and_labels(
    cluster_members_list,
    px, py, rcs, 
    graph_meas_labels,
    meas_noise_cov):
    
    node_xyrcs = np.stack((px, py, rcs), axis=-1)
    object_features_list = []
    object_class_list = []
    object_mu_list = []
    object_sigma_list = []
    object_num_meas_list = []

    for cluster_members in cluster_members_list:

        # compute normalized features
        meas_vector = node_xyrcs[cluster_members]
        mu, sigma = compute_cluster_sample_mean_and_cov(meas_vector[:, :2], meas_noise_cov)
        meas_vector_xy, _ = shift_and_rotate_points(meas_vector[:, :2],  mu, sigma)
        meas_vector_r = np.sqrt(meas_vector_xy[:, 0] ** 2 + meas_vector_xy[:, 1] ** 2) 
        meas_vector_th = np.arctan2(meas_vector_xy[:, 1], meas_vector_xy[:, 0])
        cluster_feature = np.stack((meas_vector_xy[:, 0], meas_vector_xy[:, 1], meas_vector_r, meas_vector_th, meas_vector[:, -1]), axis=-1)

        # class labels of the object
        meas_labels = graph_meas_labels[cluster_members]
        unique_label = compute_object_labels(meas_labels)

        # append everything in a list
        object_features_list.append(cluster_feature)
        object_mu_list.append(mu)
        object_sigma_list.append(sigma)
        object_num_meas_list.append(cluster_members.shape[0])
        object_class_list.append(unique_label)

    return \
        object_features_list, object_class_list, \
        object_mu_list, object_sigma_list, \
        object_num_meas_list

# ---------------------------------------------------------------------------------------------------------------
def compute_edge_index(object_num_meas_list):
    adj_mat = []
    for num_meas in object_num_meas_list:
        mat = np.ones(shape=(num_meas, num_meas), dtype=np.bool_)
        adj_mat.append(mat)
    adj_mat = block_diag(*adj_mat)
    num_meas = adj_mat.shape[0]
    idx = np.arange(num_meas)
    adj_mat[idx, idx] = False
    edge_idx = np.stack(np.nonzero(adj_mat), axis=0)
    return edge_idx


def compute_graph(object_features_list, object_class_list, object_num_meas_list, device):
    edge_idx = compute_edge_index(object_num_meas_list)
    edge_idx = torch.from_numpy(edge_idx).to(device).to(torch.int64)

    object_features = np.concatenate(object_features_list, axis=0)
    object_features = torch.from_numpy(object_features).to(device).to(torch.float32)

    object_class = torch.tensor(object_class_list, device=device).to(torch.int64)
    object_num_meas = torch.tensor(object_num_meas_list, dtype=torch.int64, device=device)
    return object_features, object_class, object_num_meas, edge_idx

# ---------------------------------------------------------------------------------------------------------------
class RadarScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        metadatset, 
        data_rootdir,
        grid_obj,
        config_obj,
        device,
        predictor_eval):

        self.device = device
        self.grid_obj = grid_obj
        self.predictor_eval = predictor_eval

        self.metadatset = metadatset
        self.data_rootdir = data_rootdir
        self.labels_to_id_dict = compute_new_labels_to_id_dict()
        self.old_to_new_label_id_map = compute_old_to_new_label_id_map()

        self.data_path = config_obj.dataset_path
        self.ball_query_thresh = config_obj.ball_query_eps_square
        self.k_number_nearest_points = config_obj.k_number_nearest_points

        self.offset_mu = config_obj.reg_mu
        self.offset_sigma = config_obj.reg_sigma
        self.reject_static_meas_by_ransac = config_obj.reject_static_meas_by_ransac
        self.dataset_augmentation = config_obj.dataset_augmentation
        self.include_region_confidence = config_obj.include_region_confidence

        self.clustering_eps = config_obj.clustering_eps # 1.4
        self.valid_cluster_num_meas_thr = config_obj.valid_cluster_num_meas_thr  # 2
        self.meas_noise_cov = config_obj.meas_noise_cov
        self.clustering_obj = Simple_DBSCAN(self.clustering_eps)

        self.grid_min_th = 0
        self.grid_min_r = 0
        self.grid_max_th = np.pi * 0.5
        self.grid_max_r = np.sqrt(self.grid_obj.max_x ** 2 + self.grid_obj.max_y ** 2)

    def remove_low_quality_proposals(
        self, 
        object_features_list, 
        object_class_list,
        object_mu_list, 
        object_sigma_list, 
        object_num_meas_list):

        object_features_list_upd = []
        object_class_list_upd = []
        object_mu_list_upd = [] 
        object_sigma_list_upd = [] 
        object_num_meas_list_upd = []

        for i, num_meas in enumerate(object_num_meas_list):
            if num_meas >= self.valid_cluster_num_meas_thr:
                object_features_list_upd.append(object_features_list[i])
                object_class_list_upd.append(object_class_list[i])
                object_mu_list_upd.append(object_mu_list[i])
                object_sigma_list_upd.append(object_sigma_list[i])
                object_num_meas_list_upd.append(object_num_meas_list[i])
        
        return \
            object_features_list_upd, object_class_list_upd, \
            object_mu_list_upd, object_sigma_list_upd, object_num_meas_list_upd

    def __len__(self):
        return len(self.metadatset)
    
    def __getitem__(self, idx: int):

        training_data = None

        # data augmentation : flip measurements along x axis
        flip_along_x = False
        if self.dataset_augmentation == True:
            if np.random.rand() >= 0.5: flip_along_x = True

        # get data and ground-truths and select only those that are within a defined grid area
        data_dict = get_data_for_datagen(self.data_rootdir, self.data_path, self.metadatset[idx], 
                                         self.reject_static_meas_by_ransac, flip_along_x)
        gt_dict_node = compute_ground_truth_node(data_dict, self.labels_to_id_dict, self.old_to_new_label_id_map)
        data_dict, gt_dict_node = self.grid_obj.select_meas_within_the_grid(data_dict, gt_dict_node)

        # For graph NN we consider those measurements that are dynamic
        data_dict_dyn, node_labels_dict_dyn = select_moving_data(data_dict, gt_dict_node, self.labels_to_id_dict)
        num_dyn_meas = data_dict_dyn['meas_px'].shape[0]

        if num_dyn_meas > 0:

            adj_dict_dyn = compute_adjacency_information(data_dict_dyn, self.ball_query_thresh, self.k_number_nearest_points)
            other_features_dyn = np.stack((
                    data_dict_dyn['meas_px'], data_dict_dyn['meas_py'], 
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

            # cluster offsets predictions
            gt_labels_dyn = node_labels_dict_dyn['class_labels']
            cluster_node_idx, cluster_labels \
                = compute_node_idx_for_each_cluster(data_dict_dyn['meas_trackid'], gt_labels_dyn, self.device)
            
            _, node_offsets_predictions, _, _ = self.predictor_eval(
                    node_features = graph_features['node_features_dyn'],
                    edge_features = graph_features['edge_features_dyn'],
                    edge_index = graph_features['edge_index_dyn'],
                    adj_matrix = graph_features['adj_matrix_dyn'],
                    cluster_node_idx = cluster_node_idx  )
            
            reg_deltas = unnormalize_gt_offsets(node_offsets_predictions, self.offset_mu, self.offset_sigma)
            pred_cluster_centers_xy = graph_features['other_features_dyn'][:, :2] + reg_deltas
            
            # extract proposals
            pred_cluster_centers_xy = pred_cluster_centers_xy.detach().cpu().numpy()
            proposal_list = extract_proposals(pred_cluster_centers_xy, self.clustering_obj)

            # compute aggregate features and compute labels for each proposal
            px = graph_features['other_features_dyn'][:, 0].detach().cpu().numpy()
            py = graph_features['other_features_dyn'][:, 1].detach().cpu().numpy()
            rcs = graph_features['node_features_dyn'][:, 1].detach().cpu().numpy()

            object_features_list, object_class_list, \
            object_mu_list, object_sigma_list, object_num_meas_list \
                = extract_and_compute_features_and_labels(
                        proposal_list, px, py, rcs, 
                        node_labels_dict_dyn['class_labels'], 
                        self.meas_noise_cov)
            
            # remove bad quality proposals
            # a proposal is considered bad quality of num_meas of each clustered proposal < self.num_meas_thr
            object_features_list, object_class_list, \
            object_mu_list, object_sigma_list, object_num_meas_list \
                = self.remove_low_quality_proposals(
                        object_features_list, object_class_list, object_mu_list, object_sigma_list, object_num_meas_list)
            
            # create training data: node_features, graph_structure, cluster_sizes
            if len(object_mu_list) != 0 :
                object_features, object_class, object_num_meas, edge_idx = \
                    compute_graph(object_features_list, object_class_list, object_num_meas_list, self.device)
                training_data = {}
                training_data['object_features'] = object_features
                training_data['object_class'] = object_class
                training_data['object_num_meas'] = object_num_meas
                training_data['edge_idx'] = edge_idx

        return training_data
    
    @ staticmethod
    def collate_fn(sample_batch):
        """ Generate a batch of data """
        object_features_batch = []
        object_class_batch = []
        object_num_meas_batch = []
        edge_idx_batch = []
        training_data_batch = None

        for i in range(len(sample_batch)):
            training_data = sample_batch[i]
            if training_data != None:
                object_features_batch.append(training_data['object_features'])
                object_class_batch.append(training_data['object_class'])
                object_num_meas_batch.append(training_data['object_num_meas'])
                edge_idx_batch.append(training_data['edge_idx'])

        if len(object_features_batch) > 0:
            training_data_batch = {}
            training_data_batch['object_features'] = object_features_batch
            training_data_batch['object_class'] = object_class_batch
            training_data_batch['object_num_meas'] = object_num_meas_batch
            training_data_batch['edge_idx'] = edge_idx_batch
            
        return training_data_batch


def remove_low_quality_proposals(
    num_meas_thr, 
    object_features_list, 
    object_class_list,
    object_mu_list, 
    object_sigma_list, 
    object_num_meas_list):

    object_features_list_upd = []
    object_class_list_upd = []
    object_mu_list_upd = [] 
    object_sigma_list_upd = [] 
    object_num_meas_list_upd = []

    for i, num_meas in enumerate(object_num_meas_list):
        if num_meas >= num_meas_thr:
            object_features_list_upd.append(object_features_list[i])
            object_class_list_upd.append(object_class_list[i])
            object_mu_list_upd.append(object_mu_list[i])
            object_sigma_list_upd.append(object_sigma_list[i])
            object_num_meas_list_upd.append(object_num_meas_list[i])
    
    return \
        object_features_list_upd, object_class_list_upd, \
        object_mu_list_upd, object_sigma_list_upd, object_num_meas_list_upd