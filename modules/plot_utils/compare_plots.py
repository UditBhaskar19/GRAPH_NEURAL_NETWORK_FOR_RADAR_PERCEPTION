# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : ploting functions
# ---------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
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
    ax[0].legend(loc='upper right')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 
    ax[0].set_title("predicted node class")

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend(loc='upper right')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 
    ax[1].set_title("GT node class")

    # plt.suptitle('predicted vs ground-truth node classes')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------
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
    ax[0].legend(loc='upper right')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 
    ax[0].set_title("predicted cluster centers")

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend(loc='upper right')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 
    ax[1].set_title("GT cluster centers")

    # plt.suptitle('predicted vs ground-truth cluster centers')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------
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
    ax[0].legend(loc='upper right')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 
    ax[0].set_title("predicted cluster centers")

    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].legend(loc='upper right')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 
    ax[1].set_title("GT cluster centers")

    # plt.suptitle('predicted vs ground-truth cluster centers')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------    
def compare_pred_gt_edge_class(
    meas_px, meas_py, edge_coordinates, 
    pred_edge_labels, gt_edge_labels, figsize,
    plot_neg_edges = True,
    xlim_min=0, xlim_max=100, 
    ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # plot predicted graph
    pred_pos_nodes_x, pred_pos_nodes_y, pred_neg_nodes_x, pred_neg_nodes_y \
        = compute_node_pairs(meas_px, meas_py, pred_edge_labels, edge_coordinates)
    
    if plot_neg_edges == True:
        ax[0].plot(
            pred_neg_nodes_x.T, pred_neg_nodes_y.T, 
            color='r', marker='.', markersize=1, markeredgecolor='none', 
            linewidth=0.5)
    ax[0].plot(
        pred_pos_nodes_x.T, pred_pos_nodes_y.T, 
        color='g', marker='.', markersize=2, markeredgecolor='none', 
        linewidth=0.5)

    ax[0].scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(xlim_min, xlim_max) 
    ax[0].set_ylim(ylim_min, ylim_max) 
    ax[0].set_title("Predicted graph edge classes")

    # plot gt graph
    gt_pos_nodes_x, gt_pos_nodes_y, gt_neg_nodes_x, gt_neg_nodes_y \
        = compute_node_pairs(meas_px, meas_py, gt_edge_labels, edge_coordinates)
    
    if plot_neg_edges == True:
        ax[1].plot(
            gt_neg_nodes_x.T, gt_neg_nodes_y.T, 
            color='r', marker='.', markersize=1, markeredgecolor='none', 
            linewidth=0.5)
    ax[1].plot(
        gt_pos_nodes_x.T, gt_pos_nodes_y.T, 
        color='g', marker='.', markersize=2, markeredgecolor='none', 
        linewidth=0.5)

    ax[1].scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(xlim_min, xlim_max) 
    ax[1].set_ylim(ylim_min, ylim_max) 
    ax[1].set_title("GT graph edge classes")

    # plt.suptitle('predicted vs ground-truth graph edge classes')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------  
def _plot_clusters_measurements_and_object_class(
    ax,
    px, py, 
    cluster_class_list,
    cluster_mean_list, 
    cluster_boundary_list,
    cluster_size_list,
    all_labels,
    cluster_size_threshold=1,
    xlim_min=-10, xlim_max=100, 
    ylim_min=-50, ylim_max=50,
    boundary_marker_size=2):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    ax.scatter(px, py, 2, color='black', marker='.')

    if len(cluster_size_list) > 0:

        for cluster_boundary, cluster_size, cluster_class \
            in zip(cluster_boundary_list, cluster_size_list, cluster_class_list):
            if cluster_size > cluster_size_threshold:
                ax.scatter(
                    cluster_boundary[:,0], cluster_boundary[:,1], 
                    boundary_marker_size, color=colors[cluster_class], marker='.')
                
        cluster_class_numpy = np.array(cluster_class_list)
        unique_class = np.unique(cluster_class_numpy)
        cluster_mean_numpy = np.stack(cluster_mean_list, axis=0)
        for i in range(unique_class.shape[0]):
            flag = unique_class[i] == cluster_class_numpy
            ax.scatter(
                cluster_mean_numpy[flag, 0], cluster_mean_numpy[flag, 1], 
                2, color=colors[unique_class[i]], marker='.', label=all_labels[unique_class[i]])
        
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max)
    return ax

# ---------------------------------------------------------------------------------------------------------------
def compare_pred_gt_object_classes(
    px, py, 
    cluster_class_list_pred,
    cluster_mean_list_pred, 
    cluster_boundary_list_pred,
    cluster_size_list_pred,
    cluster_class_list_gt,
    cluster_mean_list_gt, 
    cluster_boundary_list_gt,
    cluster_size_list_gt,
    all_labels,
    cluster_size_threshold=1,
    figsize=(16,8),
    save_plot = False,
    out_file = None):

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0] = _plot_clusters_measurements_and_object_class(
        ax[0], px, py, 
        cluster_class_list_pred,
        cluster_mean_list_pred, 
        cluster_boundary_list_pred,
        cluster_size_list_pred,
        all_labels,
        cluster_size_threshold)
    ax[0].set_title("predicted clusters and object type")
    
    ax[1] = _plot_clusters_measurements_and_object_class(
        ax[1], px, py, 
        cluster_class_list_gt,
        cluster_mean_list_gt, 
        cluster_boundary_list_gt,
        cluster_size_list_gt,
        all_labels)
    ax[1].set_title("GT clusters and object type")
    
    # plt.suptitle('predicted vs ground-truth clusters and object type')
    plt.tight_layout()
    if save_plot == True:
        plt.savefig(out_file)
        # fig.clf()
        plt.close()
    else: plt.show()

