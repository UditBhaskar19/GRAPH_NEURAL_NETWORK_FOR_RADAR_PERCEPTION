# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Clustering
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def compute_adjacency_mat_from_predicted_edges(input_graph_adj_matrix, node_xy_coord, pred_edges, eps):
    # extract the valid links
    valid_row_idx, valid_col_idx = np.nonzero(np.triu(input_graph_adj_matrix, k=1))
    # remove highly unlikely edges: i.e if the L2 norm between a node pair is >= eps
    node_pair_dist = np.sqrt((node_xy_coord[valid_row_idx, 0] - node_xy_coord[valid_col_idx, 0]) ** 2 + \
                             (node_xy_coord[valid_row_idx, 1] - node_xy_coord[valid_col_idx, 1]) ** 2)
    invalid_edge = np.logical_and(node_pair_dist >= eps, pred_edges == 1) 
    pred_edges[invalid_edge] = 0

    # compute the adjacency matrix
    row_idx = valid_row_idx[pred_edges == 1]
    col_idx = valid_col_idx[pred_edges == 1]
    adjacency_mat = np.zeros_like(input_graph_adj_matrix)
    adjacency_mat[row_idx, col_idx] = True
    adjacency_mat[col_idx, row_idx] = True
    return adjacency_mat

# ---------------------------------------------------------------------------------------------------------------
def compute_l2_norm(meas_vector_i, meas_vector_js):
    l2 = np.expand_dims(np.expand_dims(meas_vector_i, axis=0) - meas_vector_js, axis=-1)
    l2 = l2.transpose(0,2,1) @ l2
    return l2

def compute_adjacency_mat_from_predicted_offsets(pred_node_offsets, eps):
    num_nodes = pred_node_offsets.shape[0]
    adjacency_mat = np.zeros(shape=(num_nodes, num_nodes), dtype=np.bool_)
    for i in range(num_nodes):
        condition = compute_l2_norm(pred_node_offsets[i], pred_node_offsets[i:]) <= eps
        condition = np.reshape(condition, -1)
        adjacency_mat[i, i:num_nodes] = condition
        adjacency_mat[i:num_nodes, i] = condition
        adjacency_mat[i,  i] = False
    return adjacency_mat

# ---------------------------------------------------------------------------------------------------------------
class Simple_DBSCAN:
    def __init__(self, eps, compute_adj_mat_from_links=False):
        self.eps = eps
        self.compute_adj_mat_from_links = compute_adj_mat_from_links

    def init_datastruct(self, num_nodes):
        self.num_clusters = 0
        self.meas_to_cluster_id = -1 + np.zeros((num_nodes, ), dtype=np.int16)
        self.cluster_member = np.zeros((num_nodes, ), dtype=np.uint16)

    def cluster_nodes(self, meas_xy, pred_edges=None, input_graph_adj_matrix=None):
        cluster_id = 0
        num_clstr_mem = 0
        num_nodes = meas_xy.shape[0]
        self.init_datastruct(num_nodes)
        if self.compute_adj_mat_from_links == True:
            adjacency_mat = compute_adjacency_mat_from_predicted_edges(
                input_graph_adj_matrix, meas_xy, pred_edges, self.eps)
        else: adjacency_mat = compute_adjacency_mat_from_predicted_offsets(meas_xy, self.eps)

        # start clustering (Breadth First Search - BFS)
        for m in range(num_nodes):

            # start with an unclustered node
            if self.meas_to_cluster_id[m] == -1: 
                self.cluster_member[num_clstr_mem] = m
                self.meas_to_cluster_id[m] = cluster_id
                num_clstr_mem += 1
                
                # expand the cluster
                n = 0
                while n < num_clstr_mem:
                    i_s = np.repeat(self.cluster_member[n], num_nodes)
                    j_s =  np.arange(num_nodes)
                    condition1 = self.meas_to_cluster_id[j_s] == -1
                    condition2 = adjacency_mat[i_s, j_s] == True
                    condition = np.logical_and(condition1, condition2)
                    
                    num_new_neighbours = np.sum(condition, dtype=np.int16)
                    valid_j_s = j_s[condition]
                    new_num_clstr_mem = num_clstr_mem + num_new_neighbours
                    self.cluster_member[num_clstr_mem:new_num_clstr_mem] = valid_j_s
                    self.meas_to_cluster_id[valid_j_s] = cluster_id
                    num_clstr_mem = new_num_clstr_mem
                    n += 1

                # update the cluster data
                self.num_clusters = cluster_id + 1
                cluster_id += 1
                num_clstr_mem = 0

