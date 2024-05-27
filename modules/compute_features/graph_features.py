# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : graph feature computation functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

_us2sec_ = 1e-6  # micro second to sec
_us2ms_ = 1e-3   # micro second to millisec

# ---------------------------------------------------------------------------------------------------------------
def compute_ball_query(distance_mat, eps):
    """ Given a NxN matrix of pair-wise L2 distances between all possible pair of measurements. 
    Compute a gating matrix. if meas j is within a L2 distance of eps within meas i, then meas j is gated 
    with meas i, the entry in gated_flag boolean matrix is set as true ( gated_flag[i, j] = True )
    Input: distance_mat - a 2D matrix of size (n x n) of type float32
         : eps - distance threshold of type float32
    Output: gated_flag - a 2D matrix of size (n x n) of type boolean
    """
    gated_flag = distance_mat <= eps
    idx = np.arange(gated_flag.shape[0]).tolist()
    gated_flag[idx, idx] = False
    return gated_flag

# ---------------------------------------------------------------------------------------------------------------
def compute_knn(distance_mat, knn):
    """ Given a NxN matrix of pair-wise L2 distances between all possible pair of measurements. 
    Compute a adjacency matrix such that each of the measurements is connected to 'knn' number of nearest measurement
    Input: distance_mat - a 2D matrix of size (n x n) of type float32
         : knn - number of nearest measurements of type uint32
    Output: gated_flag - a 2D matrix of size (n x n) of type boolean
    """
    num_meas = distance_mat.shape[0]
    idx = np.arange(num_meas).tolist()
    sorted_array = np.argsort(distance_mat, axis=-1)
    knn_ = num_meas if knn >= num_meas else knn + 1
    destination_idx = sorted_array[:, :knn_]
    source_idx = np.repeat(np.expand_dims(idx, axis=-1), repeats=knn_, axis=-1)
    idx_set1 = np.stack((source_idx, destination_idx), axis=-1).reshape(-1, 2)
    idx_set2 = np.stack((idx_set1[:,1], idx_set1[:,0]), axis=-1)
    idx_set = np.concatenate((idx_set1, idx_set2), axis=0)
    gated_flag = np.zeros(distance_mat.shape, dtype=np.bool_)
    gated_flag[idx_set[:,0], idx_set[:,1]] = True
    gated_flag[idx, idx] = False  # set the diagonal entries as zero
    return gated_flag

# ---------------------------------------------------------------------------------------------------------------
def normalize_time(meas_timestamp):
    """ Given a array of measurement time stamps, normalize the time such that the time is in the range [0, 1]
    Input: meas_timestamp - 1D array of size (n, ) of type float32
    Output: meas_timestamp - 1D array of size (n, ) of type float32
    """
    max_time = np.max(meas_timestamp)
    min_time = np.min(meas_timestamp)
    if max_time == min_time: return (meas_timestamp - min_time)
    else: return (meas_timestamp - min_time) / (max_time - min_time)

# ---------------------------------------------------------------------------------------------------------------
def compute_adjacency_information(data_dict, eps, knn):
    """ Given radar measurements, construct the graph adjacency matrix, adjacency list and node degree
    Input: data_dict - radar measurements dictionary, which has several attributes out of which the following are used:
           - meas_px: an array of meas px, - meas_py: an array of meas py
         : eps - distance threshold of type float32
         : knn - number of nearest measurements of type uint32 
    Output: a dictionary if computed graph strucrture
            - adj_matrix: graph adjacency matrix of shape (n x n) and type boolean
            - distance_mat: L2 distance matrix of shape (n x n) and type float32
            - adj_list: graph adjacency list in the form of pairwise graph nodes of shape (2 x m) of type uint32
            - degree: graph node degree array of shape (n) and type float21
    """
    meas_pxpy = np.stack((data_dict['meas_px'], data_dict['meas_py']), axis=-1)
    meas_pxpy1 = np.expand_dims(meas_pxpy, axis=1)
    meas_pxpy2 = np.expand_dims(meas_pxpy, axis=0)
    meas_deltapxpy = np.expand_dims(meas_pxpy1 - meas_pxpy2, axis=-1)
    distance_mat = meas_deltapxpy.transpose(0,1,3,2) @ meas_deltapxpy
    distance_mat = distance_mat.squeeze(-1).squeeze(-1)
    gated_flag1 = compute_ball_query(distance_mat, eps)
    gated_flag2 = compute_knn(distance_mat, knn)
    degree = np.sum(gated_flag1, axis=-1)
    adj_list = np.stack(np.where(gated_flag2), axis=0)
    return {
        'adj_matrix': gated_flag2,
        'distance_mat': distance_mat,
        'adj_list': adj_list,
        'degree': degree }

# ---------------------------------------------------------------------------------------------------------------
def compute_adjacency_information_v2(data_dict, eps, knn):
    """ Given radar measurements, construct the graph adjacency matrix, adjacency list and node degree
    Input: data_dict - radar measurements dictionary, which has several attributes out of which the following are used:
           - meas_px: an array of meas px, - meas_py: an array of meas py
         : eps - distance threshold of type float32
         : knn - number of nearest measurements of type uint32 
    Output: a dictionary if computed graph strucrture
            - adj_matrix: graph adjacency matrix of shape (n x n) and type boolean
            - distance_mat: L2 distance matrix of shape (n x n) and type float32
            - adj_list: graph adjacency list in the form of pairwise graph nodes of shape (2 x m) of type uint32
            - degree: graph node degree array of shape (n) and type float21
    """
    meas_pxpy = np.stack((data_dict['meas_px'], data_dict['meas_py']), axis=-1)
    meas_pxpy1 = np.expand_dims(meas_pxpy, axis=1)
    meas_pxpy2 = np.expand_dims(meas_pxpy, axis=0)
    meas_deltapxpy = np.expand_dims(meas_pxpy1 - meas_pxpy2, axis=-1)
    distance_mat = meas_deltapxpy.transpose(0,1,3,2) @ meas_deltapxpy
    distance_mat = distance_mat.squeeze(-1).squeeze(-1)
    gated_flag1 = compute_ball_query(distance_mat, eps)
    gated_flag2 = compute_knn(distance_mat, knn)
    gated_flag = gated_flag1 | gated_flag2
    degree = np.sum(gated_flag1, axis=-1)
    adj_list = np.stack(np.where(gated_flag), axis=0)
    return {
        'adj_matrix': gated_flag,
        'distance_mat': distance_mat,
        'adj_list': adj_list,
        'degree': degree }

# ---------------------------------------------------------------------------------------------------------------
def compute_node_features(
    data_dict, node_degree, 
    include_region_confidence = False, 
    min_range = None, max_range = None,
    min_azimuth = None, max_azimuth = None):
    """ Compute graph node features
    Input: data_dict - radar measurements dictionary
         : include_region_confidence - flag to indicate if region confidence should be used
         : min_range - min radar fov range
         : max_range - max radar fov range
         : min_azimuth - min radar fov azimuth
         : max_azimuth - max radar fov azimuth
    Output: node_features - an array of graph node features
    """
    meas_vr = data_dict['meas_vr']
    meas_rcs = data_dict['meas_rcs']
    meas_timestamp_norm = normalize_time(data_dict['meas_timestamp'])
    meas_degree = node_degree / 10

    if include_region_confidence == True:
        r = np.sqrt(data_dict['meas_px'] ** 2 + data_dict['meas_py'] ** 2)
        th = np.abs(np.arctan2(data_dict['meas_py'], data_dict['meas_px']))
        range_conf = ( r - max_range ) / ( min_range - max_range )
        azimuth_map = ( th - max_azimuth ) / ( min_azimuth - max_azimuth )
        node_features = np.stack((meas_vr, meas_rcs, meas_timestamp_norm, meas_degree, range_conf, azimuth_map), axis=-1)
    else:
        node_features = np.stack((meas_vr, meas_rcs, meas_timestamp_norm, meas_degree), axis=-1)
    return node_features

# ---------------------------------------------------------------------------------------------------------------
def compute_edge_features(data_dict, adj_list):
    """ Compute graph node features
    Input: data_dict - radar measurements dictionary
         : adj_list - an array of graph adjacency list info in terms of node pairs of shape (2 x m)
    Output: edge_features - an array of graph edge features
    """
    edge_dx = (data_dict['meas_px'][adj_list[0]] - data_dict['meas_px'][adj_list[1]]) / 10
    edge_dy = (data_dict['meas_py'][adj_list[0]] - data_dict['meas_py'][adj_list[1]]) / 10
    edge_dl = (np.sqrt(edge_dx ** 2 + edge_dy ** 2)) / 10

    edge_dvx = data_dict['meas_vx'][adj_list[0]] - data_dict['meas_vx'][adj_list[1]]
    edge_dvy = data_dict['meas_vy'][adj_list[0]] - data_dict['meas_vy'][adj_list[1]]
    edge_dvl = np.sqrt(edge_dvx ** 2 + edge_dvy ** 2)

    edge_dt = (data_dict['meas_timestamp'][adj_list[0]] - data_dict['meas_timestamp'][adj_list[1]]) * _us2sec_
    edge_features = np.stack((edge_dx, edge_dy, edge_dl, edge_dvx, edge_dvy, edge_dvl, edge_dt), axis=-1)
    # edge_features = np.stack((edge_dl, edge_dvl, edge_dt), axis=-1)
    return edge_features

# ---------------------------------------------------------------------------------------------------------------
def select_moving_data(data_dict, gt_dict, new_labels_to_id_dict):
    """ Extract the dynamic measurements only
    Input: data_dict - radar measurements dictionary
         : gt_dict - radar measurements gt dictionary which contains classification information
         : new_labels_to_id_dict - dict which contains mapping from class name to a numeric id
    Output: data_dict_dyn - radar dynamic measurements dictionary
          : gt_dict_dyn - radar dynamic measurements gt dictionary
    """
    data_dict_dyn = {}
    gt_dict_dyn = {}
    flag = gt_dict['class_labels'] != new_labels_to_id_dict['STATIC']
    for key, value in data_dict.items():
        data_dict_dyn[key] = value[flag]
    for key, value in gt_dict.items():
        gt_dict_dyn[key] = value[flag]
    return data_dict_dyn, gt_dict_dyn