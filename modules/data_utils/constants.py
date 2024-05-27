# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : data utils constants
# ---------------------------------------------------------------------------------------------------------------

# ransac parameters (stationary measurements identification)
_REJECT_OUTLIER_BY_RANSAC_ = False
_RANSAC_MIN_NUM_SAMPLES_ = 2
_RANSAC_ERROR_MARGIN_ = 0.25
_RANSAC_NUM_ITERATIONS_ = 30
_INLIER_RATIO_THRESHOLD_ = 0.6
_MIN_NUM_MEAS_ = 10

# parameter for gating stationary measurements using ego motion
_GAMMA_STATIONARY_ = 1.5