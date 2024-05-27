# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : grid map feature computation functions
# NOTE: these functions are currently not used
# ---------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from modules.data_utils.labels import _INVALID_NUM_
from modules.data_utils.labels import compute_new_labels_to_id_dict

# ---------------------------------------------------------------------------------------------------------------
class grid_properties:
    """ Create a grid and the relevant helper functions """
    def __init__(
        self, 
        min_x, max_x, 
        min_y, max_y, 
        min_sigma_x, max_sigma_x,
        min_sigma_y, max_sigma_y,
        dx, dy):

        self.min_x = min_x
        self.max_x = max_x
        
        self.min_y = min_y
        self.max_y = max_y
        
        self.dx = dx
        self.dy = dy

        self.min_sigma_x = min_sigma_x
        self.max_sigma_x = max_sigma_x

        self.min_sigma_y = min_sigma_y
        self.max_sigma_y = max_sigma_y

        self.max_range = np.sqrt(self.max_x ** 2 + self.max_y ** 2)
        self.num_digits = np.floor(np.log10(self.max_range) + 1)

        self.num_cells_x, self.num_cells_y = self.compute_number_of_grid_cells()
        self.xy_coord = self.compute_grid_coord()
        self.grid_temp_ = np.full((self.num_cells_x, self.num_cells_y, 7), _INVALID_NUM_, dtype=np.float32)

        self.labels_to_id_dict = compute_new_labels_to_id_dict()

    
    def reinit_grid_temp_(self):
        self.grid_temp_[:,:,:] = _INVALID_NUM_


    def compute_number_of_grid_cells(self):
        eps = 1e-6
        num_cells_x = int( np.ceil( ( self.max_x + eps - self.min_x ) / self.dx ) )
        num_cells_y = int( np.ceil( ( self.max_y + eps - self.min_y ) / self.dy ) )
        return num_cells_x, num_cells_y
    

    def compute_grid_coord(self):
        x_ids = np.arange(0, self.num_cells_x, dtype=np.int32)
        y_ids = np.arange(0, self.num_cells_y, dtype=np.int32)
        x_coord = ( x_ids.astype(np.float32) + 0.5 ) * self.dx + self.min_x
        y_coord = ( y_ids.astype(np.float32) + 0.5 ) * self.dy + self.min_y
        x_coord = np.repeat(np.expand_dims(x_coord, axis=-1), self.num_cells_y, axis=-1)
        y_coord = np.repeat(np.expand_dims(y_coord, axis= 0), self.num_cells_x, axis= 0)
        xy_coord = np.stack((x_coord, y_coord), axis=-1)
        return xy_coord


    def compute_grid_idx_from_xy_coord(self, meas_x, meas_y):
        cell_x_idx = np.floor( ( meas_x - self.min_x ) / self.dx )
        cell_y_idx = np.floor( ( meas_y - self.min_y ) / self.dy )
        xy_idx = np.stack((cell_x_idx.astype(np.uint32), cell_y_idx.astype(np.uint32)), axis=-1)
        return xy_idx


    def compute_grid_coord_from_xy_coord(self, meas_x, meas_y):
        cell_x_idx = np.floor( ( meas_x - self.min_x ) / self.dx )
        cell_y_idx = np.floor( ( meas_y - self.min_y ) / self.dy )
        x_coord = ( cell_x_idx + 0.5 ) * self.dx + self.min_x
        y_coord = ( cell_y_idx + 0.5 ) * self.dy + self.min_y
        xy_coord = np.stack((x_coord, y_coord), axis=-1)
        return xy_coord


    @ staticmethod
    def compute_sigma_linear(x, min_x, max_x, min_sigma, max_sigma):
        sigma = ( x - min_x ) * ( max_sigma - min_sigma ) / ( max_x - min_x ) + min_sigma
        sigma = np.clip(sigma, min_sigma, max_sigma)
        return sigma


    @ staticmethod
    def compute_rotation_matrix(measx, measy):
        r = np.sqrt(measx ** 2 + measy ** 2)
        cos_theta = measx / r
        sin_theta = measy / r
        R = np.stack((cos_theta, -sin_theta, sin_theta, cos_theta), axis=-1).reshape(measx.shape[0], 2, 2)
        return R


    def compute_covariance_linearly_varying(self, measx, measy):
        rot_matrix = self.compute_rotation_matrix(measx, measy)
        sigmax = self.compute_sigma_linear(np.abs(measx), self.min_x, self.max_x, self.min_sigma_x, self.max_sigma_x)
        sigmay = self.compute_sigma_linear(np.abs(measy), self.min_y, self.max_y, self.min_sigma_y, self.max_sigma_y)
        meas_sig = np.stack([sigmax, sigmax, sigmay, sigmay], axis=-1).reshape(sigmax.shape[0], 2, 2)
        meas_cov = np.repeat(np.expand_dims(np.eye(2), axis=0), sigmax.shape[0], axis=0)
        meas_cov = meas_cov * meas_sig
        meas_cov = rot_matrix @ meas_cov @ rot_matrix.transpose(0,2,1)
        return meas_cov
    

    def compute_meas_priority(self, meas_xy_coord, stationary_meas_flag):
        meas_range = np.sqrt(meas_xy_coord[:, 0] ** 2 + meas_xy_coord[:, 1] ** 2)
        num_digits = np.floor(np.log10(self.max_range) + 1)
        meas_score = np.where(stationary_meas_flag, 0.0, 1.0) * np.power(10, num_digits) + self.max_range - meas_range
        return meas_score
    

    def gridify_measurements(self, meas_xy_coord, meas_vr, meas_rcs, meas_labels):
        stationary_meas_flag = meas_labels == self.labels_to_id_dict['STATIC']
        meas_score = self.compute_meas_priority(meas_xy_coord, stationary_meas_flag)
        meas_sorted_idx = np.argsort(meas_score)

        meas_vr_rcs = np.stack((meas_vr, meas_rcs), axis=-1)
        meas_info = np.concatenate((meas_xy_coord, meas_vr_rcs, np.expand_dims(meas_labels, axis=-1)), axis=-1)
        meas_info_sorted = meas_info[meas_sorted_idx]
         
        xy_idx = self.compute_grid_idx_from_xy_coord(meas_info_sorted[:,0], meas_info_sorted[:,1])
        self.reinit_grid_temp_()
        self.grid_temp_[[xy_idx[:, 0]], [xy_idx[:, 1]], :5] = meas_info_sorted

        meas_labels = self.grid_temp_[:, :, 4]
        meas_vr = self.grid_temp_[:, :, 2]
        meas_rcs = self.grid_temp_[:, :, 3]
        valid_id_x, valid_id_y = np.where(meas_labels != _INVALID_NUM_)
        meas_xy_coord = self.grid_temp_[valid_id_x, valid_id_y, :2]
        return meas_xy_coord, meas_vr, meas_rcs, meas_labels
    

    def gridify_measurements_and_labels(self, meas_xy_coord, meas_vr, meas_rcs, gt_labels, gt_offset):
        stationary_meas_flag = gt_labels == self.labels_to_id_dict['STATIC']
        meas_score = self.compute_meas_priority(meas_xy_coord, stationary_meas_flag)
        meas_sorted_idx = np.argsort(meas_score)

        meas_vr_rcs = np.stack((meas_vr, meas_rcs), axis=-1)
        meas_info = np.concatenate((meas_xy_coord, meas_vr_rcs, gt_offset, np.expand_dims(gt_labels, axis=-1)), axis=-1)
        meas_info_sorted = meas_info[meas_sorted_idx]

        xy_idx = self.compute_grid_idx_from_xy_coord(meas_info_sorted[:,0], meas_info_sorted[:,1])
        self.reinit_grid_temp_()
        self.grid_temp_[[xy_idx[:, 0]], [xy_idx[:, 1]], :7] = meas_info_sorted

        gt_labels = self.grid_temp_[:, :, -1]
        gt_offsets = self.grid_temp_[:, :, -3:-1]
        meas_vr = self.grid_temp_[:, :, 2]
        meas_rcs = self.grid_temp_[:, :, 3]
        valid_id_x, valid_id_y = np.nonzero(gt_labels != _INVALID_NUM_)
        meas_xy_coord = self.grid_temp_[valid_id_x, valid_id_y, :2]
        return meas_xy_coord, meas_vr, meas_rcs, gt_labels, gt_offsets
    

    def select_meas_within_the_grid(self, meas_dict, label_dict):

        flag = np.logical_and(
            meas_dict['meas_px'] >= self.min_x, np.logical_and(
                meas_dict['meas_px'] < self.max_x, np.logical_and(
                    meas_dict['meas_py'] >= self.min_y, meas_dict['meas_py'] < self.max_y)))

        for key, value in meas_dict.items():
            meas_dict[key] = value[flag]
        for key, value in label_dict.items():
            label_dict[key] = value[flag]
        return meas_dict, label_dict

# ---------------------------------------------------------------------------------------------------------------    
def move_to_device(grid_xy_coord, meas_xy_coord, meas_cov, device):
    grid_xy_coord = torch.from_numpy(grid_xy_coord).to(device)
    meas_xy_coord = torch.from_numpy(meas_xy_coord).to(device)
    meas_cov = torch.from_numpy(meas_cov).to(device)
    return grid_xy_coord, meas_xy_coord, meas_cov
    
# --------------------------------------------------------------------------------------------------------------- 
def compute_range_azimuth_encoding(grid_xy_coord):
    num_rows, num_cols, _ = grid_xy_coord.shape
    grid_xy_coord = grid_xy_coord.reshape(num_rows*num_cols, -1)
    x_coord, y_coord = grid_xy_coord[:, 0], grid_xy_coord[:, 1]
    
    grid_range = torch.sqrt(x_coord ** 2 + y_coord ** 2)
    max_range = torch.max(grid_range); min_range = torch.min(grid_range)
    range_map = ( grid_range - max_range ) / ( min_range - max_range )
    
    grid_azimuth = torch.abs(torch.atan2(y_coord, x_coord))
    max_azimuth = torch.max(grid_azimuth); min_azimuth = torch.min(grid_azimuth)
    azimuth_map = ( grid_azimuth - max_azimuth ) / ( min_azimuth - max_azimuth )
    
    range_map = range_map.reshape(num_rows, num_cols)
    azimuth_map = azimuth_map.reshape(num_rows, num_cols)
    return range_map, azimuth_map
    
# ---------------------------------------------------------------------------------------------------------------
def compute_meas_likelihood(grid_xy_coord, meas_xy_coord, meas_cov):
    num_rows, num_cols, _ = grid_xy_coord.shape
    grid_xy_coord = grid_xy_coord.reshape(num_rows*num_cols, -1)
    
    meas_cov_inv = torch.linalg.inv(meas_cov).to(torch.float32)
    dx = (grid_xy_coord.unsqueeze(1) - meas_xy_coord.unsqueeze(0)).unsqueeze(-1)
    dist = (dx.permute(0,1,3,2) @ dx).squeeze(2, -1)
    
    meas_idx = torch.argmin(input=dist, dim=-1, keepdims=False)
    dist = (grid_xy_coord - meas_xy_coord[meas_idx]).unsqueeze(-1)
    prob = torch.exp(-0.5 * dist.permute(0,2,1) @ meas_cov_inv[meas_idx] @ dist).reshape(num_rows, num_cols)
    return prob

# ---------------------------------------------------------------------------------------------------------------
def compute_encodings(grid_xy_coord, meas_xy_coord, meas_cov, device):
    grid_xy_coord, meas_xy_coord, meas_cov = move_to_device(grid_xy_coord, meas_xy_coord, meas_cov, device)
    range_map, azimuth_map = compute_range_azimuth_encoding(grid_xy_coord)
    if meas_xy_coord.shape[0] > 0:
        meas_likelihood_map = compute_meas_likelihood(grid_xy_coord, meas_xy_coord, meas_cov)
    else: meas_likelihood_map = torch.zeros(shape=grid_xy_coord.shape, dtype=torch.float32, device=device)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return meas_likelihood_map, range_map, azimuth_map