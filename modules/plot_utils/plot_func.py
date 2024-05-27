# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : ploting functions
# ---------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

def plot_measurements(x, y, figsize, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):
    _, ax = plt.subplots(figsize=figsize)
    s = 1
    ax.scatter(x, y, s, color='red', marker='.')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud')
    plt.show()

def plot_track_centers(x_meas, y_meas, xcc, ycc, figsize, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):
    _, ax = plt.subplots(figsize=figsize)
    s = 1
    ax.scatter(x_meas, y_meas, 10, color='red', marker='.')
    ax.scatter(xcc, ycc, 50, color='blue', marker='x')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud')
    plt.show()

def plot_range_rates(x_meas, y_meas, vx_meas, vy_meas, figsize, width=0.002, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):
    _, ax = plt.subplots(figsize=figsize)
    s = 1
    ax.scatter(x_meas, y_meas, 10, color='red', marker='.')
    ax.quiver(x_meas, y_meas, vx_meas, vy_meas, color='k', width=width)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud ego compensated range rates')
    plt.show()

def plot_graph(meas_px, meas_py, edge_coordinates, figsize):
    nodes_x = np.stack((meas_px[edge_coordinates[0]], meas_px[edge_coordinates[1]]), axis=-1)
    nodes_y = np.stack((meas_py[edge_coordinates[0]], meas_py[edge_coordinates[1]]), axis=-1)
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(nodes_x.T, nodes_y.T, color='k', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    ax.scatter(meas_px, meas_py, 30, color='red', marker='o')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    plt.title('graph')
    plt.show()

def plot_edge_labels(meas_px, meas_py, edge_coordinates, edge_labels, figsize):
    positive_edges = edge_labels == 1
    negative_edges = edge_labels == 0
    source_node_idx = edge_coordinates[0]
    target_node_idx = edge_coordinates[1]
    
    _, ax = plt.subplots(figsize=figsize)

    # plot negative edges in red
    source_node_idx = edge_coordinates[0][negative_edges]
    target_node_idx = edge_coordinates[1][negative_edges]
    nodes_x = np.stack((meas_px[source_node_idx], meas_px[target_node_idx]), axis=-1)
    nodes_y = np.stack((meas_py[source_node_idx], meas_py[target_node_idx]), axis=-1)
    ax.plot(nodes_x.T, nodes_y.T, color='r', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    
    # plot positive edges in green
    source_node_idx = edge_coordinates[0][positive_edges]
    target_node_idx = edge_coordinates[1][positive_edges]
    nodes_x = np.stack((meas_px[source_node_idx], meas_px[target_node_idx]), axis=-1)
    nodes_y = np.stack((meas_py[source_node_idx], meas_py[target_node_idx]), axis=-1)
    ax.plot(nodes_x.T, nodes_y.T, color='g', marker='.', markersize=2, markeredgecolor='none', linewidth=0.5)

    # plot nodes
    ax.scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    plt.title('graph')
    plt.show()


def plot_channels(image):
    # image = image.detach().cpu().numpy()
    meas_likelihood_map = np.flip(image[0], axis=0)
    range_map = np.flip(image[1], axis=0)
    azimuth_map = np.flip(image[2], axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].matshow(meas_likelihood_map)
    axs[1].matshow(range_map)
    axs[2].matshow(azimuth_map)
    plt.show()


def plot_meas_and_gt_labels_cluster_centers(
    meas_px, meas_py, cluster_centers_x, cluster_centers_y, gt_labels, all_labels,
    figsize, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):
    
    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    unique_id = np.unique(gt_labels)

    _, ax = plt.subplots(figsize=figsize)
    for i in range(unique_id.shape[0]):
        id = unique_id[i]
        flag = gt_labels==id
        ax.scatter(meas_px[flag], meas_py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])
    ax.scatter(cluster_centers_x, cluster_centers_y, 50, color='black', marker='x')

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud')
    plt.show()


def plot_meas_cluster_centers(
    meas_px, meas_py, cluster_centers_x, cluster_centers_y,
    figsize, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(meas_px, meas_py, 10, color='red', marker='.', label='measurements')
    ax.scatter(cluster_centers_x, cluster_centers_y, 50, color='black', marker='x', label='cluster centers')

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud and cluster center')
    plt.show()


def plot_meas_and_gt_labels(
    meas_px, meas_py, gt_labels, all_labels,
    figsize, xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):
    
    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    unique_id = np.unique(gt_labels)

    _, ax = plt.subplots(figsize=figsize)
    for i in range(unique_id.shape[0]):
        id = unique_id[i]
        flag = gt_labels==id
        ax.scatter(meas_px[flag], meas_py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud')
    plt.show()


def compare_pred_gt_class(
    px, py, 
    pred_node_class,
    gt_node_class,
    all_labels, figsize = (8, 8),
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    pred_unique_class = np.unique(pred_node_class)
    gt_unique_class = np.unique(gt_node_class)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predictions
    for i in range(pred_unique_class.shape[0]):
        id = pred_unique_class[i]
        flag = pred_node_class==id
        ax[0].scatter(px[flag], py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])

    # plot hround-truths
    for i in range(gt_unique_class.shape[0]):
        id = gt_unique_class[i]
        flag = gt_node_class==id
        ax[1].scatter(px[flag], py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])

    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].legend()
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend()
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 

    plt.suptitle('predicted vs ground-truth regression offsets')
    plt.tight_layout()
    plt.show()


def compare_pred_gt_offsets(
    px, py, 
    pred_cluster_centers_x, pred_cluster_centers_y, pred_node_class,
    gt_cluster_centers_x, gt_cluster_centers_y, gt_node_class,
    all_labels, figsize = (8, 8),
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    pred_unique_class = np.unique(pred_node_class)
    gt_unique_class = np.unique(gt_node_class)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predictions
    for i in range(pred_unique_class.shape[0]):
        id = pred_unique_class[i]
        flag = pred_node_class==id
        ax[0].scatter(px[flag], py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])
    ax[0].scatter(pred_cluster_centers_x, pred_cluster_centers_y, 50, color='black', marker='x')

    # plot hround-truths
    for i in range(gt_unique_class.shape[0]):
        id = gt_unique_class[i]
        flag = gt_node_class==id
        ax[1].scatter(px[flag], py[flag], 10, color=colors[id.astype(int)], marker='.', label=all_labels[id.astype(int)])
    ax[1].scatter(gt_cluster_centers_x, gt_cluster_centers_y, 50, color='black', marker='x')

    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].legend()
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend()
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 

    plt.suptitle('predicted vs ground-truth regression offsets')
    plt.tight_layout()
    plt.show()


def compare_pred_gt_offsets_meas(
    px, py, 
    pred_cluster_centers_x, pred_cluster_centers_y,
    gt_cluster_centers_x, gt_cluster_centers_y,
    figsize = (8, 8),
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predictions
    ax[0].scatter(px, py, 10, color='red', marker='.', label='measurements')
    ax[0].scatter(pred_cluster_centers_x, pred_cluster_centers_y, 50, color='black', marker='x', label='cluster centers')

    # plot hround-truths
    ax[1].scatter(px, py, 10, color='red', marker='.', label='measurements')
    ax[1].scatter(gt_cluster_centers_x, gt_cluster_centers_y, 50, color='black', marker='x', label='cluster centers')

    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].legend()
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend()
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 

    plt.suptitle('predicted vs ground-truth regression offsets')
    plt.tight_layout()
    plt.show()


# def compare_pred_gt_class_offsets_meas(
#     px, py, 
#     pred_cluster_centers_x, pred_cluster_centers_y,
#     gt_cluster_centers_x, gt_cluster_centers_y,
#     figsize = (8, 8),
#     xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

#     _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

#     # plot predictions
#     ax[0].scatter(px, py, 10, color='red', marker='.', label='measurements')
#     ax[0].scatter(pred_cluster_centers_x, pred_cluster_centers_y, 50, color='black', marker='x', label='cluster centers')

#     # plot hround-truths
#     ax[1].scatter(px, py, 10, color='red', marker='.', label='measurements')
#     ax[1].scatter(gt_cluster_centers_x, gt_cluster_centers_y, 50, color='black', marker='x', label='cluster centers')

#     ax[0].set_xlabel('x(m)')
#     ax[0].set_ylabel('y(m)')
#     ax[0].legend()
#     ax[0].set_aspect('equal')
#     ax[0].set_xlim(xlim_min, xlim_max) 
#     ax[0].set_ylim(ylim_min, ylim_max) 

#     ax[1].set_xlabel('x(m)')
#     ax[1].set_ylabel('y(m)')
#     ax[1].legend()
#     ax[1].set_aspect('equal')
#     ax[1].set_xlim(xlim_min, xlim_max) 
#     ax[1].set_ylim(ylim_min, ylim_max) 

#     plt.suptitle('predicted vs ground-truth regression offsets')
#     plt.tight_layout()
#     plt.show()


def compute_node_pairs(meas_px, meas_py, edge_labels, edge_coordinates):
    # predicted edge labels
    positive_edges = edge_labels == 1
    negative_edges = edge_labels == 0

    # plot negative edges in red
    source_node_idx = edge_coordinates[0][negative_edges]
    target_node_idx = edge_coordinates[1][negative_edges]
    neg_nodes_x = np.stack((meas_px[source_node_idx], meas_px[target_node_idx]), axis=-1)
    neg_nodes_y = np.stack((meas_py[source_node_idx], meas_py[target_node_idx]), axis=-1)
    # ax[0].plot(nodes_x.T, nodes_y.T, color='r', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    
    # plot positive edges in green
    source_node_idx = edge_coordinates[0][positive_edges]
    target_node_idx = edge_coordinates[1][positive_edges]
    pos_nodes_x = np.stack((meas_px[source_node_idx], meas_px[target_node_idx]), axis=-1)
    pos_nodes_y = np.stack((meas_py[source_node_idx], meas_py[target_node_idx]), axis=-1)
    # ax.plot(nodes_x.T, nodes_y.T, color='g', marker='.', markersize=2, markeredgecolor='none', linewidth=0.5)
    return pos_nodes_x, pos_nodes_y, neg_nodes_x, neg_nodes_y

    
def compare_pred_gt_offsets_edge_labels(
    meas_px, meas_py, edge_coordinates, 
    pred_edge_labels, gt_edge_labels, figsize,
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predicted graph
    pred_pos_nodes_x, pred_pos_nodes_y, pred_neg_nodes_x, pred_neg_nodes_y \
        = compute_node_pairs(meas_px, meas_py, pred_edge_labels, edge_coordinates)
    
    # ax[0].plot(pred_neg_nodes_x.T, pred_neg_nodes_y.T, color='r', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    ax[0].plot(pred_pos_nodes_x.T, pred_pos_nodes_y.T, color='g', marker='.', markersize=2, markeredgecolor='none', linewidth=0.5)

    ax[0].scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].set_aspect('equal')

    # plot gt graph
    gt_pos_nodes_x, gt_pos_nodes_y, gt_neg_nodes_x, gt_neg_nodes_y \
        = compute_node_pairs(meas_px, meas_py, gt_edge_labels, edge_coordinates)
    
    # ax[1].plot(gt_neg_nodes_x.T, gt_neg_nodes_y.T, color='r', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    ax[1].plot(gt_pos_nodes_x.T, gt_pos_nodes_y.T, color='g', marker='.', markersize=2, markeredgecolor='none', linewidth=0.5)

    ax[1].scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].set_aspect('equal')

    plt.suptitle('predicted vs ground-truth graph edge classes')
    plt.tight_layout()
    plt.show()


def plot_clusters(
    cluster_boundary_list, cluster_mean_list, measurements_xy, figsize, 
    xlim_min=-10, xlim_max=100, ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(measurements_xy[:,0], measurements_xy[:,1], 1, color='red', marker='.', label='measurements')
    for cluster_boundary, cluster_mean in zip(cluster_boundary_list, cluster_mean_list):
        ax.scatter(cluster_boundary[:,0], cluster_boundary[:,1], 5, color='blue', marker='.', label='cluster boundary')
        ax.scatter(cluster_mean[0], cluster_mean[1], 5, color='black', marker='*', label='cluster center')

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 

    plt.suptitle('valid object clusters')
    plt.tight_layout()
    plt.show()
    return 0


def compare_pred_gt_clusters(
    px, py, 
    pred_mu_list, pred_cluster_boundary_points_list,
    gt_mu_list, gt_cluster_boundary_points_list,
    xlim_min=-10, xlim_max=100, ylim_min=-50, ylim_max=50,
    boundary_marker_size=2,
    figsize=(16,8)):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predicted clusters
    ax[0].scatter(px, py, 1, color='red', marker='.', label='measurements')
    for cluster_boundary, cluster_mean in zip(pred_cluster_boundary_points_list, pred_mu_list):
        ax[0].scatter(cluster_boundary[:,0], cluster_boundary[:,1], boundary_marker_size, color='blue', marker='.', label='predicted clusters')
        ax[0].scatter(cluster_mean[0], cluster_mean[1], 5, color='black', marker='*')
    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max)

    # plot gt clusters
    ax[1].scatter(px, py, 1, color='red', marker='.', label='measurements')
    for cluster_boundary, cluster_mean in zip(gt_cluster_boundary_points_list, gt_mu_list):
        ax[1].scatter(cluster_boundary[:,0], cluster_boundary[:,1], boundary_marker_size, color='blue', marker='.', label='ground-truth clusters')
        ax[1].scatter(cluster_mean[0], cluster_mean[1], 5, color='black', marker='*')
    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max)

    plt.suptitle('predicted vs ground-truth clusters')
    plt.tight_layout()
    plt.show()
    return 0




def save_compare_pred_gt_clusters(
    out_file,
    px, py, 
    pred_mu_list, pred_cluster_boundary_points_list,
    gt_mu_list, gt_cluster_boundary_points_list,
    xlim_min=-10, xlim_max=100, ylim_min=-50, ylim_max=50,
    boundary_marker_size=2,
    figsize=(16,8)):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predicted clusters
    ax[0].scatter(px, py, 1, color='red', marker='.', label='measurements')
    for cluster_boundary, cluster_mean in zip(pred_cluster_boundary_points_list, pred_mu_list):
        ax[0].scatter(cluster_boundary[:,0], cluster_boundary[:,1], boundary_marker_size, color='blue', marker='.', label='predicted clusters')
        ax[0].scatter(cluster_mean[0], cluster_mean[1], 5, color='black', marker='*')
    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max)

    # plot gt clusters
    ax[1].scatter(px, py, 1, color='red', marker='.', label='measurements')
    for cluster_boundary, cluster_mean in zip(gt_cluster_boundary_points_list, gt_mu_list):
        ax[1].scatter(cluster_boundary[:,0], cluster_boundary[:,1], boundary_marker_size, color='blue', marker='.', label='ground-truth clusters')
        ax[1].scatter(cluster_mean[0], cluster_mean[1], 5, color='black', marker='*')
    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max)

    plt.suptitle('predicted vs ground-truth clusters')
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_file)
    plt.close()
    return 0

    
