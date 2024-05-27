# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : ground truth computation of the graph edge
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

def compute_ground_truth(data_dict, edge_list, adj_matrix):
    meas_trackid = data_dict['meas_trackid']
    source_node_meas_trackid = meas_trackid[edge_list[0]]
    target_node_meas_trackid = meas_trackid[edge_list[1]]
    valid_link_flag = \
        ( source_node_meas_trackid == target_node_meas_trackid ) & \
        ( source_node_meas_trackid != b'' )

    mat_link_temp = np.zeros_like(adj_matrix)
    mat_link_temp[edge_list[0], edge_list[1]] = valid_link_flag

    valid_row_idx, valid_col_idx = np.nonzero(np.triu(adj_matrix, k=1))
    valid_link_flag = mat_link_temp[valid_row_idx, valid_col_idx ]
    return valid_link_flag.astype(np.float32)
