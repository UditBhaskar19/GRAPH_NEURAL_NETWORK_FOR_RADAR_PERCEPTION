# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : set parameters for gnn based model inference
# --------------------------------------------------------------------------------------------------------------
import torch, sys

# --------------------------------------------------------------------------------------------------------------
def set_parameters_for_inference(module_rootdir, config_obj, trained_weights_path):

    sys.path.append(module_rootdir)
    from modules.neural_net.gnn.gnn_detector import Model_Training
    from modules.compute_features.grid_features import grid_properties
    from modules.set_configurations.common import get_device

    device = get_device()

    # ================================================> INIT NETWORK STRUCTURE <========================================
    # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
    # set the weights_path
    weights_path = trained_weights_path
    detector_train = Model_Training(config_obj, device)
    detector_train.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector_train = detector_train.to(device)
    
    # ==============================================> DATASET & DATALOADER <===================================================
    grid_obj = grid_properties(
        min_x = config_obj.min_x, max_x = config_obj.max_x, 
        min_y = config_obj.min_y, max_y = config_obj.max_y, 
        min_sigma_x = config_obj.min_sigma_x, max_sigma_x = config_obj.max_sigma_x, 
        min_sigma_y = config_obj.min_sigma_y, max_sigma_y = config_obj.max_sigma_y, 
        dx = config_obj.dx, dy = config_obj.dy)

    return {
        'device': device,
        'grid': grid_obj,
        'detector': detector_train.pred.eval()}