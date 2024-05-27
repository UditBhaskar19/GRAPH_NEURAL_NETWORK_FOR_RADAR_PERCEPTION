# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : make x and y coordinates of a cluster rotation invarient 
# NOTE: not used in the current implementation
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def compute_sample_mean_covariance(samples_xy):
    mu = np.sum(samples_xy, axis=0) / samples_xy.shape[0]
    error = np.expand_dims((mu[:2] - samples_xy[:, :2]), axis=-1)
    sigma = error @ error.transpose(0, 2, 1)
    sigma = np.sum(sigma, axis=0) / ( samples_xy.shape[0] - 1 )
    return mu, sigma

# ---------------------------------------------------------------------------------------------------------------
def shift_and_rotate_points(samples_xy):
    mu, sigma = compute_sample_mean_covariance(samples_xy)
    eigen_values, eigen_vectors = np.linalg.eig(sigma) 
    samples_xy_transformed = (samples_xy - mu) @ eigen_vectors
    return samples_xy_transformed, eigen_values

# ---------------------------------------------------------------------------------------------------------------
def convert_cartesian_to_polar(sample_xy):
    r = np.sqrt(sample_xy[:,0] ** 2 + sample_xy[:,1] ** 2)
    th = np.arctan2(sample_xy[:,1], sample_xy[:,0])
    sample_polar = np.stack((r, th), axis=-1)
    return sample_polar