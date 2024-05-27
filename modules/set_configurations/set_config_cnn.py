# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : create parameter class for cnn based model architecture
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
from modules.set_configurations.common import read_yaml
from modules.data_utils.labels import compute_new_labels_to_id_dict

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
        # ---------------
        # CNN parameters
        # ---------------
        self.input_image_dimension = configurations['CNN_ARCHITECTURE']['input_image_dimension']
        self.base_stem_channels = configurations['CNN_ARCHITECTURE']['base_stem_channels']
        self.base_kernel_sizes = configurations['CNN_ARCHITECTURE']['base_kernel_sizes']
        
        self.bottleneck_number_of_blocks = configurations['CNN_ARCHITECTURE']['bottleneck_number_of_blocks']
        self.bottleneck_stem_channels = configurations['CNN_ARCHITECTURE']['bottleneck_stem_channels']
        self.bottleneck_width_channels = configurations['CNN_ARCHITECTURE']['bottleneck_width_channels']
        self.bottleneck_kernel_size = configurations['CNN_ARCHITECTURE']['bottleneck_kernel_size']
        
        self.neck_out_channels = configurations['CNN_ARCHITECTURE']['neck_out_channels']
        self.neck_kernel_size = configurations['CNN_ARCHITECTURE']['neck_kernel_size']
        
        self.head_stem_channels = configurations['CNN_ARCHITECTURE']['head_stem_channels']
        self.head_ffn_channels = configurations['CNN_ARCHITECTURE']['head_ffn_channels']
        self.head_kernel_size = configurations['CNN_ARCHITECTURE']['head_kernel_size']
        
        self.activation = configurations['CNN_ARCHITECTURE']['activation']
        self.conv_type = configurations['CNN_ARCHITECTURE']['conv_type']

        self.reg_offset = configurations['CNN_ARCHITECTURE']['reg_offset']
        self.offset_mu = configurations['CNN_ARCHITECTURE']['reg_mu']
        self.offset_sigma = configurations['CNN_ARCHITECTURE']['reg_sigma']
        self.augmented_features = configurations['CNN_ARCHITECTURE']['augmented_features_for_feed_forward_block']
        # ------------------------------
        # Object categories and weights
        # ------------------------------
        self.object_classes = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS']
        self.class_weights = configurations['OBJECT_CATEGORIES']['OBJECT_CLASS_WEIGHTS']
        # -------------
        # loss weights
        # -------------
        self.cls_loss_weight = configurations['LOSS_WEIGHTS']['loss_cls']
        self.reg_loss_weight = configurations['LOSS_WEIGHTS']['loss_reg']
        # -----------------------
        # Class labels to ID dict
        # -----------------------
        self.new_labels_to_id_dict = compute_new_labels_to_id_dict()
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