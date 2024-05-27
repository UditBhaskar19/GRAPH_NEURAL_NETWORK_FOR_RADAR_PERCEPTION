# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : create parameter class for gnn based model architecture
# --------------------------------------------------------------------------------------------------------------
import numpy as np
from modules.set_configurations.common import read_yaml
from modules.data_utils.labels import compute_new_labels_to_id_dict, compute_labels_to_id_dict

class config:
    def __init__(self, config_filepath):
        configurations = read_yaml(config_filepath)
        self.seed = configurations['RANDOM']['seed']
        # ---------------------------------------------------------------
        # paths and file names for loading data and saving model weights
        # ---------------------------------------------------------------
        self.dataset_path = configurations['DIRECTORIES']['dataset_dir']
        self.model_weights_dir = configurations['DIRECTORIES']['model_weights_dir']
        self.weights_name = configurations['DIRECTORIES']['weights_name']
        # ----------------------------------------------------------------------------------------------------
        # parameter that specified how much measurement history to accumulate, and how to construct the graph
        # ----------------------------------------------------------------------------------------------------
        self.window_size = configurations['DATA_SELECTION_PARAM']['temporal_window_size']
        self.ball_query_eps_square = configurations['DATA_SELECTION_PARAM']['ball_query_eps_square']
        self.k_number_nearest_points = configurations['DATA_SELECTION_PARAM']['k_number_nearest_points']
        self.reject_static_meas_by_ransac = configurations['DATA_SELECTION_PARAM']['reject_static_meas_by_ransac']
        self.dataset_augmentation = configurations['DATA_SELECTION_PARAM']['dataset_augmentation']
        # --------------------
        # set grid parameters
        # --------------------
        self.min_x = configurations['GRID_LIMITS']['min_x'] 
        self.max_x = configurations['GRID_LIMITS']['max_x'] 
        self.min_y = configurations['GRID_LIMITS']['min_y'] 
        self.max_y = configurations['GRID_LIMITS']['max_y']
        self.min_sigma_x = configurations['GRID_LIMITS']['min_sigma_x'] 
        self.max_sigma_x = configurations['GRID_LIMITS']['max_sigma_x']
        self.min_sigma_y = configurations['GRID_LIMITS']['min_sigma_y'] 
        self.max_sigma_y = configurations['GRID_LIMITS']['max_sigma_y']
        self.dx = configurations['GRID_LIMITS']['dx'] 
        self.dy = configurations['GRID_LIMITS']['dy']
        self.grid_min_th = 0
        self.grid_min_r = 0
        self.grid_max_th = np.pi * 0.5
        self.grid_max_r = np.sqrt(self.max_x ** 2 + self.max_y ** 2)
        # ---------------
        # GNN parameters
        # ---------------
        self.node_features = configurations['GNN_ARCHITECTURE']['node_features']
        self.edge_features = configurations['GNN_ARCHITECTURE']['edge_features']
        self.reg_offset = configurations['GNN_ARCHITECTURE']['reg_offset']
        self.activation = configurations['GNN_ARCHITECTURE']['activation']
        self.norm_layer = configurations['GNN_ARCHITECTURE']['normalization']
        self.num_groups = configurations['GNN_ARCHITECTURE']['num_groups']
        self.reg_mu = configurations['GNN_ARCHITECTURE']['reg_mu']
        self.reg_sigma = configurations['GNN_ARCHITECTURE']['reg_sigma']
        self.aggregation = configurations['GNN_ARCHITECTURE']['aggregation']
        self.node_feat_enc_stem_channels = configurations['GNN_ARCHITECTURE']['node_feat_enc_stem_channels']
        self.edge_feat_enc_stem_channels = configurations['GNN_ARCHITECTURE']['edge_feat_enc_stem_channels']
        self.graph_convolution_stem_channels = configurations['GNN_ARCHITECTURE']['graph_convolution_stem_channels']
        self.msg_mlp_hidden_dim = configurations['GNN_ARCHITECTURE']['msg_mlp_hidden_dim']
        self.num_blocks_to_compute_edge = configurations['GNN_ARCHITECTURE']['num_blocks_to_compute_edge']
        self.hidden_node_channels_GAT = configurations['GNN_ARCHITECTURE']['hidden_node_channels_GAT']
        self.num_heads_GAT = configurations['GNN_ARCHITECTURE']['num_heads_GAT']
        self.link_pred_stem_channels = configurations['GNN_ARCHITECTURE']['link_pred_stem_channels']
        self.node_pred_stem_channels = configurations['GNN_ARCHITECTURE']['node_pred_stem_channels']
        self.input_node_feat_dim = len(configurations['GNN_ARCHITECTURE']['node_features'])
        self.input_edge_feat_dim = len(configurations['GNN_ARCHITECTURE']['edge_features'])
        self.num_classes = len(configurations['OBJECT_CATEGORIES']['OBJECT_CLASS_DYN'])
        self.reg_offset_dim = len(configurations['GNN_ARCHITECTURE']['reg_offset'])
        self.offset_mu = configurations['GNN_ARCHITECTURE']['reg_mu']
        self.offset_sigma = configurations['GNN_ARCHITECTURE']['reg_sigma']
        self.num_edge_classes = configurations['GNN_ARCHITECTURE']['num_edge_classes']
        # ------------------------------
        # Object categories and weights
        # ------------------------------
        self.object_classes = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS']
        self.class_weights = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS_WEIGHTS']
        self.object_classes_dyn = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS_DYN']
        self.class_weights_dyn = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS_WEIGHTS_DYN']
        # -------------
        # loss weights
        # -------------
        self.edge_cls_loss_weight = configurations['LOSS_WEIGHTS']['edge_loss_cls']
        self.node_cls_loss_weight = configurations['LOSS_WEIGHTS']['node_loss_cls']
        self.node_reg_loss_weight = configurations['LOSS_WEIGHTS']['node_loss_reg']
        self.obj_cls_loss_weight = configurations['LOSS_WEIGHTS']['obj_loss_cls']
        # -----------------------
        # Class labels to ID dict
        # -----------------------
        self.new_labels_to_id_dict = compute_new_labels_to_id_dict()
        self.new_labels_to_id_dict_dyn = compute_labels_to_id_dict(self.object_classes_dyn)
        # -----------------------
        # Optimization parameters
        # -----------------------
        self.optim = configurations['OPTIMIZATION']['optim']
        self.max_train_iter = configurations['OPTIMIZATION']['max_training_iterations']
        self.learning_rate = configurations['OPTIMIZATION']['learning_rate']
        self.weight_decay = configurations['OPTIMIZATION']['weight_decay']
        # ----------------------------------------------------------------------------------
        # Set the number of training and validation samples and choose if we want to shuffle
        # ----------------------------------------------------------------------------------
        self.num_training_samples = configurations['DATASET']['num_training_samples']
        self.num_validation_samples = configurations['DATASET']['num_validation_samples']
        self.shuffle_training_samples = configurations['DATASET']['shuffle_training_samples']
        self.shuffle_validation_samples = configurations['DATASET']['shuffle_validation_samples']
        self.include_region_confidence = configurations['DATASET_INFO']['include_region_confidence']