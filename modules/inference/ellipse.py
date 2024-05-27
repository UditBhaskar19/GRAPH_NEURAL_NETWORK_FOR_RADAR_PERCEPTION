import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def compute_cov_ellipse(mu, cov, chi_sq, n_points):

    # eigenvalue decomposition
    d, v = np.linalg.eig(cov)

    # idx corresponding to largest and smallest eig values
    largest_eigval_idx = np.where(d == np.max(d))[0]
    largest_eigval_idx = np.reshape(largest_eigval_idx, -1)
    if largest_eigval_idx[0] == 0:
        largest_eigval_idx = 0
        smallest_eigval_idx = 1
    elif largest_eigval_idx[0] == 1:
        largest_eigval_idx = 1
        smallest_eigval_idx = 0

    # ellipse axis half lengths
    a = chi_sq * np.sqrt(d[largest_eigval_idx]) 
    b = chi_sq * np.sqrt(d[smallest_eigval_idx])

    # compute ellipse orientation
    if (v[largest_eigval_idx, 0] == 0): theta = np.pi / 2
    else: theta = np.arctan2(v[largest_eigval_idx, 1], v[largest_eigval_idx, 0])

    # genetare theta and points(x,y)
    th = np.linspace(0, 2 * np.pi, n_points)
    px = a * np.cos(th)
    py = b * np.sin(th)
    points = np.stack((px,py), axis=-1)

    # Rotate by theta and translate by mu
    R = np.array([ [ np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)] ], dtype=np.float64)
    T = mu
    points = points @ R.transpose(1,0) + T
    return points, mu