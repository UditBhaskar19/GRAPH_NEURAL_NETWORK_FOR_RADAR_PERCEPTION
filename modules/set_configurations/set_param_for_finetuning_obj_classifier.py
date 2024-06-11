# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : set parameters for gnn based model inference
# --------------------------------------------------------------------------------------------------------------
import torch, sys
import numpy as np
from torch.utils.data import DataLoader
from modules.set_configurations.common import reset_seed, get_device, infinite_loader

# --------------------------------------------------------------------------------------------------------------
def set_parameters_for_finetuning(
    module_rootdir: str,
    dataset_rootdir: str,
    config_obj,
    batch_size: int,
    trained_weights_path: str = ''):

    sys.path.append(module_rootdir)
    from modules.neural_net.gnn.gnn_detector import Model_Object_Classifier_Finetuning
    from modules.compute_features.grid_features import grid_properties
    from modules.data_utils.read_data import create_sequences_info_list, create_train_validation_sets
    from modules.data_generator.datagen_gnn import RadarScenesDataset

    reset_seed(config_obj.seed)
    device = get_device()

    # ================================================> INIT NETWORK STRUCTURE <========================================
    # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
    # set the weights_path
    weights_path = trained_weights_path
    detector_train = Model_Object_Classifier_Finetuning(config_obj)
    detector_train.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector_train = detector_train.to(device)
    detector_train.pred.freeze_layers_except_object_class_predictor()

    # ============================================> SET OPTIMIZATION PARAMETERS <=======================================
    initial_lr = config_obj.learning_rate_finetuning
    wt_decay = config_obj.weight_decay_finetuning
    params = [p for p in detector_train.parameters() if p.requires_grad]
    if config_obj.optim_finetuning == 'sgd': optimizer = torch.optim.SGD(params, momentum=0.9, lr=initial_lr, weight_decay=wt_decay)
    if config_obj.optim_finetuning == 'adamw': optimizer = torch.optim.AdamW(params, lr=initial_lr, weight_decay=wt_decay)
    lr_scheduler = None

    # ==============================================> DATASET & DATALOADER <===================================================
    grid_obj = grid_properties(
        min_x = config_obj.min_x, max_x = config_obj.max_x, 
        min_y = config_obj.min_y, max_y = config_obj.max_y, 
        min_sigma_x = config_obj.min_sigma_x, max_sigma_x = config_obj.max_sigma_x, 
        min_sigma_y = config_obj.min_sigma_y, max_sigma_y = config_obj.max_sigma_y, 
        dx = config_obj.dx, dy = config_obj.dy)
    
    dataset_metadata = create_sequences_info_list(dataset_rootdir, config_obj.dataset_path, config_obj.window_size)
    train_dataset_metadata, validation_dataset_metadata = create_train_validation_sets(dataset_metadata)

    # --------------------------------------------------------------------------------------------------------------
    if config_obj.num_training_samples > 0:
        train_sample_di = int(len(train_dataset_metadata) / config_obj.num_training_samples)
        train_dataset_metadata = train_dataset_metadata[0::train_sample_di]

    if config_obj.shuffle_training_samples:
        random_idx = np.arange(len(train_dataset_metadata))
        np.random.shuffle(random_idx)
        train_dataset_metadata = [train_dataset_metadata[idx] for idx in random_idx]

    dataset_train = RadarScenesDataset(
        metadatset = train_dataset_metadata, 
        data_rootdir = dataset_rootdir, 
        grid_obj = grid_obj,
        config_obj = config_obj,
        device = device)
    
    train_args = dict(batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)
    train_loader = DataLoader(dataset_train, **train_args)
    train_loader = infinite_loader(train_loader)

    # --------------------------------------------------------------------------------------------------------------
    if config_obj.num_validation_samples > 0:
        val_sample_di = int(len(validation_dataset_metadata) / config_obj.num_validation_samples)
        validation_dataset_metadata = validation_dataset_metadata[0::val_sample_di]

    if config_obj.shuffle_validation_samples:
        random_idx = np.arange(len(validation_dataset_metadata))
        np.random.shuffle(random_idx)
        validation_dataset_metadata = [validation_dataset_metadata[idx] for idx in random_idx]

    dataset_val = RadarScenesDataset(
        metadatset = validation_dataset_metadata, 
        data_rootdir = dataset_rootdir,  
        grid_obj = grid_obj,
        config_obj = config_obj,
        device = device)
    
    val_args = dict(batch_size=batch_size, shuffle=False, collate_fn=dataset_train.collate_fn)
    val_loader = DataLoader(dataset_val, **val_args)

    return {
        'detector': detector_train,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'dataloader_train': train_loader,
        'dataloader_val': val_loader }