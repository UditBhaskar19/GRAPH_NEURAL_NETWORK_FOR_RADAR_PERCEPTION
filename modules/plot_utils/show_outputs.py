# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : ploting functions
# ---------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from modules.plot_utils.compare_plots import compute_node_pairs

# ---------------------------------------------------------------------------------------------------------------
def plot_pred_class(
    px, py, node_class, all_labels,
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50,
    figsize = (8, 8), ax=None):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    unique_node_class = np.unique(node_class)

    plot_here = False
    if ax == None:
        plot_here = True
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    for i in range(unique_node_class.shape[0]):
        id = unique_node_class[i]
        flag = node_class==id
        ax.scatter(px[flag], py[flag], 10, color=colors[id], marker='.', label=all_labels[id])
        
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_title("predicted node class")
    
    if plot_here == True:
        # plt.title('predicted node class')
        plt.tight_layout()
        plt.show()
    else: return ax

# ---------------------------------------------------------------------------------------------------------------
def plot_pred_offsets(
    px, py, cluster_centers_x, cluster_centers_y,
    node_class, all_labels,
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50,
    figsize = (8, 8), ax=None):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver']
    unique_node_class = np.unique(node_class)

    plot_here = False
    if ax == None:
        plot_here = True
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    for i in range(unique_node_class.shape[0]):
        id = unique_node_class[i]
        flag = node_class==id
        # ax.scatter(px[flag], py[flag], 10, color=colors[id], marker='.', label=all_labels[id])
        ax.scatter(px[flag], py[flag], 10, color=colors[id], marker='.')
    ax.scatter(cluster_centers_x, cluster_centers_y, 50, color='black', marker='x')

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    # ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    ax.set_title("predicted cluster centers")

    if plot_here == True:
        # plt.title('predicted cluster centers')
        plt.tight_layout()
        plt.show()
    else: return ax

# ---------------------------------------------------------------------------------------------------------------
def plot_pred_edge_class(
    meas_px, meas_py, edge_coordinates,
    edge_class,
    figsize=(8, 8), ax=None,
    plot_neg_edges = True,
    xlim_min=0, xlim_max=100, 
    ylim_min=-50, ylim_max=50):

    plot_here = False
    if ax == None:
        plot_here = True
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    pos_nodes_x, pos_nodes_y, neg_nodes_x, neg_nodes_y \
        = compute_node_pairs(meas_px, meas_py, edge_class, edge_coordinates)
    
    if plot_neg_edges == True:
        ax.plot(
            neg_nodes_x.T, neg_nodes_y.T, 
            color='r', marker='.', markersize=1, markeredgecolor='none', 
            linewidth=0.5)
    ax.plot(
        pos_nodes_x.T, pos_nodes_y.T, 
        color='g', marker='.', markersize=2, markeredgecolor='none', 
        linewidth=0.5)

    ax.scatter(meas_px, meas_py, 30, color='k', marker='o')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    ax.set_title("predicted graph edge class")

    if plot_here == True:
        # plt.title('predicted graph edge class')
        plt.tight_layout()
        plt.show()
    else: return ax

# ---------------------------------------------------------------------------------------------------------------
def plot_clusters_measurements_and_object_class(
    px, py, 
    cluster_class_list,
    cluster_mean_list, 
    cluster_boundary_list,
    cluster_size_list,
    all_labels,
    cluster_size_threshold = 2, 
    xlim_min=-10, xlim_max=100, 
    ylim_min=-50, ylim_max=50,
    boundary_marker_size=3,
    figsize=(8,8), ax=None):

    colors = ['green', 'darkviolet', 'magenta', 'purple', 'orange', 'cyan', 'red', 'silver'] 

    plot_here = False
    if ax == None:
        plot_here = True
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.scatter(px, py, 2, color='black', marker='.')

    if len(cluster_class_list) > 0:

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
    ax.set_title("predicted clusters and object class")

    if plot_here == True:
        # plt.title('predicted clusters and object class')
        plt.tight_layout()
        plt.show()
    else: return ax

# ---------------------------------------------------------------------------------------------------------------
def plot_all_outputs(
    px, py, all_labels,
    node_class, 
    cluster_centers_x, cluster_centers_y,
    edge_coordinates, edge_class,
    cluster_class_list,
    cluster_mean_list, 
    cluster_boundary_list,
    cluster_size_list,
    cluster_size_threshold,
    figsize=(10,10),
    save_plot = False,
    out_file = None):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    ax[0,0] = plot_pred_class(
        px, py, node_class, all_labels,
        xlim_min=-10, xlim_max=100, 
        ylim_min=-50, ylim_max=50,
        ax=ax[0,0])
    
    ax[0,1] = plot_pred_offsets(
        px, py, cluster_centers_x, cluster_centers_y,
        node_class, all_labels,
        xlim_min=-10, xlim_max=100, ylim_min=-50, ylim_max=50,
        ax=ax[0,1])
    
    ax[1,0] = plot_pred_edge_class(
        px, py, edge_coordinates,
        edge_class,
        figsize=(8, 8), ax=ax[1,0],
        plot_neg_edges = True,
        xlim_min=-10, xlim_max=100, 
        ylim_min=-50, ylim_max=50)
    
    ax[1,1] = plot_clusters_measurements_and_object_class(
        px, py, 
        cluster_class_list,
        cluster_mean_list, 
        cluster_boundary_list,
        cluster_size_list,
        all_labels,
        cluster_size_threshold,
        xlim_min=-10, xlim_max=100, 
        ylim_min=-50, ylim_max=50,
        boundary_marker_size=2,
        figsize=(8,8), ax=ax[1,1])
    
    # plt.suptitle('predictions')
    plt.tight_layout()
    if save_plot == True:
        plt.savefig(out_file)
        # fig.clf()
        plt.close()
    else: plt.show()