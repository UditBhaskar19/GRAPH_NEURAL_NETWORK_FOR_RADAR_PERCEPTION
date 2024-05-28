# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : ploting functions
# ---------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def plot_measurements(
    x, y, size, figsize = (8, 8),
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, size, color='red', marker='o', label='measurements')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('radar point cloud')
    plt.show()

# ---------------------------------------------------------------------------------------------------------------
def plot_graph(
    meas_px, meas_py, edge_coordinates, figsize=(8,8),
    xlim_min=0, xlim_max=100, ylim_min=-50, ylim_max=50):

    nodes_x = np.stack((meas_px[edge_coordinates[0]], meas_px[edge_coordinates[1]]), axis=-1)
    nodes_y = np.stack((meas_py[edge_coordinates[0]], meas_py[edge_coordinates[1]]), axis=-1)
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(nodes_x.T, nodes_y.T, color='k', marker='.', markersize=1, markeredgecolor='none', linewidth=0.5)
    ax.scatter(meas_px, meas_py, 30, color='red', marker='o')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_aspect('equal')
    ax.set_xlim(xlim_min, xlim_max) 
    ax.set_ylim(ylim_min, ylim_max) 
    plt.title('graph edges')
    plt.show()