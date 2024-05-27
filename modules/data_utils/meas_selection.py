# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : measurement selection functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from modules.data_utils.constants import (
    _RANSAC_MIN_NUM_SAMPLES_, _RANSAC_ERROR_MARGIN_, _RANSAC_NUM_ITERATIONS_,
    _INLIER_RATIO_THRESHOLD_, _MIN_NUM_MEAS_, _GAMMA_STATIONARY_ )

# ---------------------------------------------------------------------------------------------------------------
def coordinate_transform_vx_vy(vx, vy, theta):
    """ Coordinate transformation of the velocity vector from sensor frame to vehicle frame
    Input: ( vx, vy ) - velocity vector
         : theta - sensor mount azimuth angle
    Output: ( vx_cts, vy_cts ) - coordinate transformed velocity vector
    """
    vx_cts = vx * np.cos(theta) - vy * np.sin(theta)
    vy_cts = vx * np.sin(theta) + vy * np.cos(theta)
    return vx_cts, vy_cts

# ---------------------------------------------------------------------------------------------------------------
def generate_meas_sensor_frame(
    vx_ego, vy_ego, yawrate_ego, 
    tx, ty, theta):
    """ compute velocity vector in the sensor frame at the sensor location given the ego motion
    Input: vx_ego, vy_ego, yawrate_ego - vehicle ego motion
         : ( tx, ty ) - radar mounting position vector
         : theta - radar mount azimuth angle
    Output: ( vx_sensor, vy_sensor ) - velocity vector in the sensor frame at the sensor location
    """
    vx_sensor = vx_ego - yawrate_ego * ty
    vy_sensor = vy_ego + yawrate_ego * tx
    vx_sensor, vy_sensor = coordinate_transform_vx_vy(vx_sensor, vy_sensor, -theta)
    return vx_sensor, vy_sensor

# ---------------------------------------------------------------------------------------------------------------
def predict_range_rate_at_meas_locations(
    vx_ego, vy_ego, yawrate_ego, 
    meas_theta, 
    tx, ty, theta):
    """ compute range rate at different locations in the radar FOV give the vehicle ego motion and sensor mount info
    Input: vx_ego, vy_ego, yawrate_ego - vehicle ego motion
         : meas_theta - an array of measurement azimuths
         : ( tx, ty ) - radar mounting position vector
         : theta - radar mount azimuth angle
    Output: vr_pred - computed range rates at the measurement locations 
    """
    vx_sensor, vy_sensor = generate_meas_sensor_frame(vx_ego, vy_ego, yawrate_ego, tx, ty, theta)
    vr_pred = -( vx_sensor * np.cos(meas_theta) + vy_sensor * np.sin(meas_theta) )
    return vr_pred

# ---------------------------------------------------------------------------------------------------------------
def gate_stationary_meas_flag(
    vx_ego, vy_ego, yawrate_ego, \
    z_azimuth, z_vr, \
    tx, ty, theta):
    """ given a prior estimate of the vehicle ego motion which can be either from vehicle odometry 
        or the predicted motion estimate from the previous state ego motion select measurements which are likeliy to be stationary
        Input: ego_motion_prior - vx_ego, vy_ego, yawrate_ego 
             : z_azimuth, z_vr - 
             : tx, ty, theta - radar mounting parameters
        Output: z -  a vector of selected measurement
              : error - a vector of errors
              : gamma_stationary - threshold to determine if the measurements are stationary
    """
    vr_pred = predict_range_rate_at_meas_locations(vx_ego, vy_ego, yawrate_ego, z_azimuth, tx, ty, theta)
    error = vr_pred - z_vr
    sel_z_flag = ( np.abs(error) <= _GAMMA_STATIONARY_ )
    return sel_z_flag, error

# ---------------------------------------------------------------------------------------------------------------
def estimate_sensor_vx_vy(meas_theta, meas_vr):
    """ estimate radar ego-motion from stationary measurements
    Input: meas_theta - an array of measurement azimuths
         : meas_vr - an array of measurement range rate azimuths
    Output: ( vx, vy ) - radar translational ego motion
    """
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    n = meas_theta.shape[0]
    for i in range(n):
        cos_t  = np.cos(meas_theta[i])
        sin_t  = np.sin(meas_theta[i])
        sin_2t = np.sin(2 * meas_theta[i])
        A[0,0] += ( cos_t ** 2 )
        A[0,1] += sin_2t
        b[0,0] -= ( cos_t * meas_vr[i] )
        b[1,0] -= ( sin_t * meas_vr[i] )
    A[0,1] = 0.5 * A[0,1]
    A[1,0] = A[0,1]
    A[1,1] = n - A[0,0]
    x = np.linalg.inv(A) @ b
    return x[0,0], x[1,0]

# ---------------------------------------------------------------------------------------------------------------
def ransac(z):
    """ use RANSAC to select stationary measurements and remove clutter
    Input: z_azimuth, z_vr ( range, azimuth, range_rate )
    Output: inliers_flag - an array if boolean flag indicating if the corrosponding meas is an inlier
          : is_valid - a boolean flag indicating if the measurement can be trusted
          : in_ratio - ratio of the number of inliers / total number of measurement
    """
    num_meas = z.shape[0]
    is_valid = False
    in_ratio = 0
    inliers_flag = np.zeros(z.shape[0], dtype=bool)

    if num_meas > _MIN_NUM_MEAS_:

        # ransac parameters
        num_iter = _RANSAC_NUM_ITERATIONS_
        error_margin = _RANSAC_ERROR_MARGIN_
        min_num_samples = _RANSAC_MIN_NUM_SAMPLES_
        in_ratio_thresh = _INLIER_RATIO_THRESHOLD_

        # for maintaining outputs of each ransac iteratiions
        vx = np.zeros((num_iter, ))
        vy = np.zeros((num_iter, ))
        num_inliers = np.zeros((num_iter, ))
        inlier_ratio = np.zeros((num_iter, ))
    
        # for randomly selecting measurements
        meas_idx = np.arange(z.shape[0])
        for i in range(num_iter):
        
            # randomly select measurements for the consensus set and create test set
            np.random.shuffle(meas_idx)
            z_consensus_set = z[meas_idx[:min_num_samples]]
            z_test_set = z[meas_idx[min_num_samples:]]
        
            # compute the radar ego motion using the consensus set
            meas_theta = z_consensus_set[:, 0]
            meas_vr = z_consensus_set[:, 1]
            vx_est, vy_est = estimate_sensor_vx_vy(meas_theta, meas_vr)
        
            # compute the predicted vr using the computes vx, vy
            meas_theta_test = z_test_set[:, 0]
            meas_vr_test = z_test_set[:, 1] 
            pred_vr_test = -( vx_est * np.cos(meas_theta_test) + vy_est * np.sin(meas_theta_test) )
        
            # compute the errors and select the inliers
            error = np.abs(meas_vr_test - pred_vr_test)
            inliers_flag = ( error <= error_margin )
        
            # compute the output
            vx[i] = vx_est
            vy[i] = vy_est
            num_inliers[i] = np.sum(inliers_flag)
            inlier_ratio[i] = ( num_inliers[i] + min_num_samples ) / num_meas
        
        # find the index corrorspondin to the maximum number of inliers and extract vals
        max_inlier_idx = np.argmax(num_inliers)
        vx_est = vx[max_inlier_idx]
        vy_est = vy[max_inlier_idx]
        in_ratio = inlier_ratio[max_inlier_idx]
    
        # recompute the inliers  
        meas_theta = z[:, 0]
        meas_vr = z[:, 1]
        pred_vr = -( vx_est * np.cos(meas_theta) + vy_est * np.sin(meas_theta) )
        error = np.abs(meas_vr - pred_vr)
        inliers_flag = ( error <= error_margin )

        # compute the validity flag based on the in_lier ratio
        if in_ratio >= in_ratio_thresh : is_valid = True
    return inliers_flag, is_valid, in_ratio

# ---------------------------------------------------------------------------------------------------------------
def identify_stationary_measurements(
    z_azimuth, z_vr,
    tx, ty, theta,
    vx_odom, yawrate_odom,
    reject_outlier_by_ransac: bool = False):

    """ Stationary measurement selection
    Inputs: rad_meas - radar measurements ( all measurements ) 
          : rad_mount_param - radar mount parameters (px, py, yaw)
          : vx_odom - temporally closest odometry vx to the radar frame
          : yaw_rate_odom - temporally closest odometry yaw_rate to the radar frame
    Outputs: rad_meas - an array of selected stationary measurements
    """
    # perform a preliminary selection of stationary measurements
    vy_odom = 0.0
    rad_meas_stationary_flag, _ \
        = gate_stationary_meas_flag(
            vx_odom, vy_odom, yawrate_odom, \
            z_azimuth, z_vr, \
            tx, ty, theta)
    
    if reject_outlier_by_ransac:
        # identify inliers by ransac
        z = np.stack((z_azimuth, z_vr), axis=1)
        inliers_flag, _, _ = ransac(z[rad_meas_stationary_flag])

        # recompute the flag
        gated_idx = np.arange(z.shape[0])
        flag = np.zeros((z.shape[0],), dtype=np.bool8)
        flag[gated_idx[rad_meas_stationary_flag]] = inliers_flag
        return flag
    
    else: return rad_meas_stationary_flag