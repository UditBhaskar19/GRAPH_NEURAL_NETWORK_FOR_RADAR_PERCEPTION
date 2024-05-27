# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : set parameters for object classification model training
# NOTE: not used in the current version
# --------------------------------------------------------------------------------------------------------------
import torch, sys
import numpy as np
from torch.utils.data import DataLoader
from modules.set_configurations.common import reset_seed, infinite_loader

def set_parameters_for_training(
    module_rootdir: str,
    dataset_rootdir: str,
    config_obj,
    batch_size: int,
    trained_region_proposal_weights_path: str,
    starting_iter_num: int = 0, 
    load_model_weights_train: bool = False, 
    trained_weights_path: str = ''):

    sys.path.append(module_rootdir)
    from modules.neural_net.classifier.classifier import Model_Training
    from modules.compute_features.grid_features import grid_properties
    from modules.data_utils.read_data import create_sequences_info_list, create_train_validation_sets
    from modules.data_generator.datagen_classifier import RadarScenesDataset
    from modules.set_configurations.set_param_for_inference_gnn import set_parameters_for_inference
    reset_seed(config_obj.seed)

    param_obj = set_parameters_for_inference(module_rootdir, config_obj, trained_region_proposal_weights_path)
    device = param_obj['device']

    # ================================================> INIT NETWORK STRUCTURE <========================================
    # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
    # set the weights_path
    load_model_weights = load_model_weights_train
    weights_path = trained_weights_path

    detector_train = Model_Training(config_obj)
    if load_model_weights:
        detector_train.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector_train = detector_train.to(device)

    # ============================================> SET OPTIMIZATION PARAMETERS <=======================================
    initial_lr = config_obj.learning_rate
    wt_decay = config_obj.weight_decay
    params = [p for p in detector_train.parameters() if p.requires_grad]
    if config_obj.optim == 'sgd': optimizer = torch.optim.SGD(params, momentum=0.9, lr=initial_lr, weight_decay=wt_decay)
    if config_obj.optim == 'adamw': optimizer = torch.optim.AdamW(params, lr=initial_lr, weight_decay=wt_decay)

    # in case we have to abruptly stop training and resume the training at a later time
    init_start = starting_iter_num # ==> start from this iteration  
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( 
        optimizer, 
        gamma=0.1,
        milestones=[int(0.6 * config_obj.max_train_iter - init_start), 
                    int(0.9 * config_obj.max_train_iter - init_start)])
    
    # ==============================================> DATASET & DATALOADER <===================================================
    dataset_metadata = create_sequences_info_list(dataset_rootdir, config_obj.dataset_path, config_obj.window_size)
    train_dataset_metadata, validation_dataset_metadata = create_train_validation_sets(dataset_metadata)

    grid_obj = param_obj['grid']
    predictor_eval = param_obj['detector']

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
        device = device,
        predictor_eval = predictor_eval)
    
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
        device = device,
        predictor_eval = predictor_eval)
    
    val_args = dict(batch_size=batch_size, shuffle=False, collate_fn=dataset_train.collate_fn)
    val_loader = DataLoader(dataset_val, **val_args)

    return {
        'detector': detector_train,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'dataloader_train': train_loader,
        'dataloader_val': val_loader }