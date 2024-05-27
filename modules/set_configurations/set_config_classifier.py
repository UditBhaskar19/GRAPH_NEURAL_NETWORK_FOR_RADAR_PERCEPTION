# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : create parameter class for object classification
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
import numpy as np
from modules.set_configurations.common import read_yaml
from modules.set_configurations.set_config_gnn import config as config_gnn

class config(config_gnn):
    def __init__(
        self, 
        config_gnn_filepath, 
        config_classifier_filepath):
        super().__init__(config_gnn_filepath)

        configurations = read_yaml(config_classifier_filepath)

        # ---------------------
        # CLUSTERING parameters
        # ---------------------
        self.clustering_eps = configurations['CLUSTERING']['clustering_eps']
        self.valid_cluster_num_meas_thr = configurations['CLUSTERING']['valid_cluster_num_meas_thr']
        self.meas_noise_cov = configurations['CLUSTERING']['meas_noise_var'] * np.eye(2, dtype=np.float32)
        # ---------------
        # GNN parameters
        # ---------------
        self.classifier_node_features = configurations['GNN_ARCHITECTURE']['node_features']
        self.classifier_input_node_feat_dim = len(configurations['GNN_ARCHITECTURE']['node_features'])
        self.classifier_activation = configurations['GNN_ARCHITECTURE']['activation']
        self.classifier_aggregation = configurations['GNN_ARCHITECTURE']['aggregation']
        self.classifier_node_feat_enc_stem_channels = configurations['GNN_ARCHITECTURE']['node_feat_enc_stem_channels']
        self.classifier_graph_convolution_stem_channels = configurations['GNN_ARCHITECTURE']['graph_convolution_stem_channels']
        self.classifier_msg_mlp_hidden_dim = configurations['GNN_ARCHITECTURE']['msg_mlp_hidden_dim']
        self.classifier_node_pred_stem_channels = configurations['GNN_ARCHITECTURE']['node_pred_stem_channels']
        # ---------------------------------------------------------------
        # paths and file names for loading data and saving model weights
        # ---------------------------------------------------------------
        self.model_weights_dir = configurations['DIRECTORIES']['model_weights_dir']
        self.weights_name = configurations['DIRECTORIES']['weights_name']
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
