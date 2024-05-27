# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : common functions for setting configurations
# --------------------------------------------------------------------------------------------------------------
import yaml, torch

# --------------------------------------------------------------------------------------------------------------
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# --------------------------------------------------------------------------------------------------------------
def reset_seed(number):
    """ Reset random seed to the specific number
    Inputs- number: A seed number to use
    """
    import random
    random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    return

# --------------------------------------------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print("GPU is available. Good to go!")
        DEVICE = torch.device("cuda")
    else:
        print("Only CPU is available.")
        DEVICE = torch.device("cpu")
    return DEVICE

# --------------------------------------------------------------------------------------------------------------
def infinite_loader(loader):
    """ Get an infinite stream of batches from a data loader """
    while True:
        yield from loader