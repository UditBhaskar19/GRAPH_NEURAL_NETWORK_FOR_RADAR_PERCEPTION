# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : measurement labels
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

_INVALID_NUM_ = -99999.0

_ALL_LABELS_ = [
    'CAR', 
    'LARGE_VEHICLE', 'TRUCK', 'BUS', 'TRAIN', 
    'BICYCLE', 'MOTORIZED_TWO_WHEELER', 
    'PEDESTRIAN', 
    'PEDESTRIAN_GROUP', 
    'ANIMAL', 'OTHER', 
    'STATIC']

_NEW_LABELS_ = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE', 'NONE', 'FALSE', 'STATIC']

_LABELS_OLD_TO_NEW_ = {}
_LABELS_OLD_TO_NEW_['CAR'] = 'CAR'
_LABELS_OLD_TO_NEW_['LARGE_VEHICLE'] = 'LARGE_VEHICLE'
_LABELS_OLD_TO_NEW_['TRUCK'] = 'LARGE_VEHICLE'
_LABELS_OLD_TO_NEW_['BUS'] = 'LARGE_VEHICLE'
_LABELS_OLD_TO_NEW_['TRAIN'] = 'LARGE_VEHICLE'
_LABELS_OLD_TO_NEW_['BICYCLE'] = 'TWO_WHEELER'
_LABELS_OLD_TO_NEW_['MOTORIZED_TWO_WHEELER'] = 'TWO_WHEELER'
_LABELS_OLD_TO_NEW_['PEDESTRIAN'] = 'PEDESTRIAN'
_LABELS_OLD_TO_NEW_['PEDESTRIAN_GROUP'] = 'PEDESTRIAN_GROUP'
_LABELS_OLD_TO_NEW_['ANIMAL'] = 'NONE'
_LABELS_OLD_TO_NEW_['OTHER'] = 'NONE'
_LABELS_OLD_TO_NEW_['STATIC'] = 'STATIC'

_LABELS_NEW_TO_DYN_ = {}
_LABELS_NEW_TO_DYN_['CAR'] = 'CAR'
_LABELS_NEW_TO_DYN_['PEDESTRIAN'] = 'PEDESTRIAN'
_LABELS_NEW_TO_DYN_['PEDESTRIAN_GROUP'] = 'PEDESTRIAN_GROUP'
_LABELS_NEW_TO_DYN_['TWO_WHEELER'] = 'TWO_WHEELER'
_LABELS_NEW_TO_DYN_['LARGE_VEHICLE'] = 'LARGE_VEHICLE'
_LABELS_NEW_TO_DYN_['NONE'] = 'NONE'
_LABELS_NEW_TO_DYN_['CAR'] = 'CAR'

# ---------------------------------------------------------------------------------------------------------------
def compute_old_labels_to_id_dict():
    labels_to_id_dict = {}
    labels_to_id_dict['CAR'] = 0
    labels_to_id_dict['LARGE_VEHICLE'] = 1
    labels_to_id_dict['TRUCK'] = 2
    labels_to_id_dict['BUS'] = 3
    labels_to_id_dict['TRAIN'] = 4
    labels_to_id_dict['BICYCLE'] = 5
    labels_to_id_dict['MOTORIZED_TWO_WHEELER'] = 6
    labels_to_id_dict['PEDESTRIAN'] = 7
    labels_to_id_dict['PEDESTRIAN_GROUP'] = 8
    labels_to_id_dict['ANIMAL'] = 9
    labels_to_id_dict['OTHER'] = 10
    labels_to_id_dict['STATIC'] = 11
    return labels_to_id_dict

def compute_new_labels_to_id_dict():
    labels_to_id_dict = {}
    labels_to_id_dict['CAR'] = 0
    labels_to_id_dict['PEDESTRIAN'] = 1
    labels_to_id_dict['PEDESTRIAN_GROUP'] = 2
    labels_to_id_dict['TWO_WHEELER'] = 3
    labels_to_id_dict['LARGE_VEHICLE'] = 4
    labels_to_id_dict['NONE'] = 5        # the object class in not well defined.
    labels_to_id_dict['FALSE'] = 6       # non-zero ego motion compensated doppler but no gt class assigned is 'FALSE'
    labels_to_id_dict['STATIC'] = 7
    return labels_to_id_dict

def compute_dyn_labels_to_id_dict():
    labels_to_id_dict = {}
    labels_to_id_dict['CAR'] = 0
    labels_to_id_dict['PEDESTRIAN'] = 1
    labels_to_id_dict['PEDESTRIAN_GROUP'] = 2
    labels_to_id_dict['TWO_WHEELER'] = 3
    labels_to_id_dict['LARGE_VEHICLE'] = 4
    labels_to_id_dict['NONE'] = 5        # the object class in not well defined.
    labels_to_id_dict['FALSE'] = 6       # non-zero ego motion compensated doppler but no gt class assigned is 'FALSE'
    return labels_to_id_dict

def compute_labels_to_id_dict(labels):
    labels_to_id_dict = {}
    for i, label in enumerate(labels):
        labels_to_id_dict[label] = i
    return labels_to_id_dict

# ---------------------------------------------------------------------------------------------------------------
def compute_old_to_new_label_id_map():
    old_labels_to_id_dict = compute_old_labels_to_id_dict()
    new_labels_to_id_dict = compute_new_labels_to_id_dict()
    n = len(list(old_labels_to_id_dict.values()))

    old_to_new_label_id_map = np.full((n, ), fill_value=-1, dtype=np.int32)
    for old_label, new_label in _LABELS_OLD_TO_NEW_.items():
        old_label_id = old_labels_to_id_dict[old_label]
        new_label_id = new_labels_to_id_dict[new_label]
        old_to_new_label_id_map[old_label_id] = new_label_id
    return old_to_new_label_id_map

# ---------------------------------------------------------------------------------------------------------------
def reassign_label_ids(old_ids, old_to_new_label_id_map):
    return old_to_new_label_id_map[old_ids]