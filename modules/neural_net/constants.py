# ---------------------------------------------------------------------------------------------------------------
# Author Name : Udit Bhaskar
# description : Constants for NN architecture
# ---------------------------------------------------------------------------------------------------------------
import math
from collections import namedtuple
det_named_tuple = namedtuple('det_named_tuple', ['class_logits', 'reg_deltas'])

_EPS_ = 1e-5
_LEAKY_RELU_NEG_SLOPE_ = 0.01
_NUM_GROUPS_ = 16

_INTERP_MODE_ = 'bilinear'

_STEM_CONV_MEAN_INIT_ = 0.0
_STEM_CONV_STD_INIT_ = 0.01
_STEM_CONV_BIAS_INIT_ = 0.0

_CLS_CONV_MEAN_INIT_ = _STEM_CONV_MEAN_INIT_
_CLS_CONV_STD_INIT_ = _STEM_CONV_STD_INIT_
# Use a negative bias in class prediction to improve training. Without this, the training can diverge
_CLS_CONV_BIAS_INIT_ = -math.log(99)    

_REG_CONV_MEAN_INIT_ = _STEM_CONV_MEAN_INIT_
_REG_CONV_STD_INIT_ = _STEM_CONV_STD_INIT_
_REG_CONV_BIAS_INIT_ = _STEM_CONV_BIAS_INIT_