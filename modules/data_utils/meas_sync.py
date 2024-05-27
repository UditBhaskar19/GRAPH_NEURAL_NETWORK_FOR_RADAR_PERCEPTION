# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : measurement synchronozation functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def vel_polar_to_cart(vr, px, py):
    azi_angle = np.arctan2(py, px)
    vx = vr * np.cos(azi_angle)
    vy = vr * np.sin(azi_angle)
    return vx, vy

# ---------------------------------------------------------------------------------------------------------------
def vr_cartesian_vf(vr, azi_angle, mount_yaw):
    """ represent range-rate (compensated or raw) in cartesian coordinates in the vehicle frame """
    angle = azi_angle + mount_yaw
    vx = vr * np.cos(angle)
    vy = vr * np.sin(angle)
    return vx, vy

# ---------------------------------------------------------------------------------------------------------------
def construct_SE2_group_element(px, py, theta):
    """ given a pose construct a trasformation matrix T """
    T = np.eye(3)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = np.array([[px],[py]])
    T[:2, :2] = R
    T[:2, 2:] = t
    return T

# ---------------------------------------------------------------------------------------------------------------
def inverse_SE2(T):
    """ the inverse of the transformation matrix T """
    R = T[:2, :2]
    t = T[:2, 2:]
    T = np.eye(3)
    T[:2, :2] = R.transpose()
    T[:2, 2:] = -R.transpose() @ t
    return T

# ---------------------------------------------------------------------------------------------------------------
def coordinate_transform_vxvy(meas_vx, meas_vy, T):
    """ coordinate transform measurements given the matrix T """
    R = T[:2, :2]
    meas = np.stack([meas_vx, meas_vy], axis=0)
    meas = R @ meas
    return meas[0,:], meas[1,:]

# ---------------------------------------------------------------------------------------------------------------
def ego_compensate_prev_meas_vehicle_frame(
    meas_px_prev, 
    meas_py_prev,
    meas_vx_prev, 
    meas_vy_prev,
    T_pose_curr, 
    T_pose_prev):
    """ spatially align prev and curr sensor acans in the current ego-vehicle frame.
    Assuming that the measurements are in the ego vehicle frame  """
    T =  np.linalg.inv( T_pose_curr ) @ T_pose_prev
    R = T[:2, :2]
    t = T[:2, 2:]
    meas_pos_prev = np.stack([meas_px_prev, meas_py_prev], axis=0)
    meas_pos_ego_comp = R @ meas_pos_prev + t
    meas_vel_prev = np.stack([meas_vx_prev, meas_vy_prev], axis=0)
    # meas_vel_ego_comp = R @ meas_vel_prev
    meas_vel_ego_comp = meas_vel_prev
    return \
        meas_pos_ego_comp[0,:], meas_pos_ego_comp[1,:], \
        meas_vel_ego_comp[0,:], meas_vel_ego_comp[1,:]

# ---------------------------------------------------------------------------------------------------------------
def ego_compensate_radar_frames_list(
    meas_px_list,
    meas_py_list,
    meas_vx_list,
    meas_vy_list,
    ego_px_gloal,
    ego_py_gloal,
    ego_yaw_gloal):

    # create a list of pose matrix list from ego loc data
    T_list = []
    for ego_px, ego_py, ego_yaw in zip(ego_px_gloal, ego_py_gloal, ego_yaw_gloal):
        T = construct_SE2_group_element(ego_px, ego_py, ego_yaw)
        T_list.append(T)

    T_curr = T_list[-1]
    meas_px_sync_list = []
    meas_py_sync_list = []
    meas_vx_sync_list = []
    meas_vy_sync_list = []
    for px, py, vx, vy, T_prev in zip(meas_px_list, meas_py_list, meas_vx_list, meas_vy_list, T_list):
        px, py, vx, vy = ego_compensate_prev_meas_vehicle_frame(px, py, vx, vy, T_curr, T_prev)
        meas_px_sync_list.append(px)
        meas_py_sync_list.append(py)
        meas_vx_sync_list.append(vx)
        meas_vy_sync_list.append(vy)

    return \
        meas_px_sync_list, meas_py_sync_list, \
        meas_vx_sync_list, meas_vy_sync_list