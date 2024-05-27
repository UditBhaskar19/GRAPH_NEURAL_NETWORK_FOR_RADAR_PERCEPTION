# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : ground truth computation of the graph node
# ---------------------------------------------------------------------------------------------------------------
import torch
from typing import List
import numpy as np
from modules.data_utils.labels import reassign_label_ids
from modules.compute_groundtruth.compute_grid_labels import normalize_gt_offsets

# ---------------------------------------------------------------------------------------------------------------
def compute_cluster_sample_mean_and_cov(meas_vector, meas_noise_cov):
    """ Compute cluster mean and covariance 
    Input: meas_vector - measurement array
           meas_noise_cov -  measurement noise covariance
    Output: meas_mu - sample mean 
            sigma - sample covariance
    """
    meas_mu = np.sum(meas_vector, axis=0) / meas_vector.shape[0]
    if meas_vector.shape[0] > 1:
        error = np.expand_dims((meas_mu[:2] - meas_vector[:, :2]), axis=-1)
        sigma = error @ error.transpose(0, 2, 1)
        sigma = np.sum(sigma, axis=0) / ( meas_vector.shape[0] - 1 ) + meas_noise_cov
    else: sigma = meas_noise_cov
    return meas_mu, sigma

def compute_gt_clusters(meas_px, meas_py, meas_trackid, meas_noise_cov):
    """ Compute ground-truth clusters 
    Input: meas_px, meas_py - measurement array
         : meas_trackid -  measurement ground-truth track ids
         : meas_noise_cov -  measurement noise covariance
    Output: clusters_mu - array of cluster means
          : clusters_sigma - array of cluster covariance
    """
    clusters_mu = []
    clusters_sigma = []
    unique_ids = np.unique(meas_trackid)
    unique_ids = unique_ids[unique_ids != b'']
    meas_pxpy = np.stack((meas_px, meas_py), axis=-1)
    for i in range(unique_ids.shape[0]):
        trackid = unique_ids[i]
        track_meas_flag = meas_trackid == trackid
        track_meas_pxpy = meas_pxpy[track_meas_flag]
        meas_mu, sigma = compute_cluster_sample_mean_and_cov(track_meas_pxpy, meas_noise_cov)
        clusters_mu.append(meas_mu)
        clusters_sigma.append(sigma)
    return clusters_mu, clusters_sigma

# ---------------------------------------------------------------------------------------------------------------
def generate_gt_offset(meas_px, meas_py, meas_trackid):
    """ Compute ground-truth regression offsets
    Input: meas_px, meas_py - measurement array
         : meas_trackid -  measurement ground-truth track ids
    Output: offsetx, offsety - regression offsets
    """
    offsetx = np.zeros(shape=(meas_px.shape[0], ), dtype=np.float32)
    offsety = np.zeros(shape=(meas_px.shape[0], ), dtype=np.float32)
    unique_ids = np.unique(meas_trackid)
    unique_ids = unique_ids[unique_ids != b'']
    for i in range(unique_ids.shape[0]):
        trackid = unique_ids[i]
        track_meas_flag = meas_trackid == trackid
        track_meas_px = meas_px[track_meas_flag]
        track_meas_py = meas_py[track_meas_flag]
        offsetx[track_meas_flag] = np.mean(track_meas_px) - track_meas_px
        offsety[track_meas_flag] = np.mean(track_meas_py) - track_meas_py
    return offsetx, offsety

# ---------------------------------------------------------------------------------------------------------------
def generate_gt_labels(
    meas_trackid, 
    meas_stationary_flag, 
    meas_class_id, 
    labels_to_id_dict, 
    old_to_new_label_id_map):

    class_labels = np.zeros(shape=(meas_class_id.shape[0], ), dtype=np.float32)
    clutter_flag = (meas_trackid == b'') & (~meas_stationary_flag)
    static_env_flag = (meas_trackid == b'') & meas_stationary_flag
    valid_object_flag = meas_trackid != b''

    reassigned_id = reassign_label_ids(meas_class_id, old_to_new_label_id_map)
    class_labels[valid_object_flag] = reassigned_id[valid_object_flag]
    class_labels[clutter_flag] = labels_to_id_dict['FALSE']
    class_labels[static_env_flag] = labels_to_id_dict['STATIC']
    return class_labels

# ---------------------------------------------------------------------------------------------------------------
def compute_ground_truth(
    data_dict, 
    labels_to_id_dict, 
    old_to_new_label_id_map):
    
    meas_px = data_dict['meas_px']
    meas_py = data_dict['meas_py']
    meas_trackid = data_dict['meas_trackid']
    stationary_meas_flag = data_dict['stationary_meas_flag']
    meas_label_id =  data_dict['meas_label_id']

    class_labels = generate_gt_labels(meas_trackid, stationary_meas_flag, meas_label_id, labels_to_id_dict, old_to_new_label_id_map)
    offsetx, offsety = generate_gt_offset(meas_px, meas_py, meas_trackid)
    return {
        'offsetx': offsetx,
        'offsety': offsety,
        'class_labels': class_labels }

# ---------------------------------------------------------------------------------------------------------------
def compute_training_groundtruth(
    gt_labels: torch.Tensor, 
    gt_offsets: torch.Tensor,
    offset_mu: List[float],
    offset_sigma: List[float]):

    gt_offsets_img = normalize_gt_offsets(gt_offsets, offset_mu, offset_sigma)
    return gt_labels, gt_offsets_img