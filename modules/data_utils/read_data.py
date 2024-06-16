# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : functions for reading sensor data from files
# ---------------------------------------------------------------------------------------------------------------
import os, json, h5py
from tqdm import tqdm
import numpy as np
from modules.data_utils.meas_sync import ego_compensate_radar_frames_list, vr_cartesian_vf
from modules.data_utils.meas_selection import identify_stationary_measurements

# ---------------------------------------------------------------------------------------------------------------
def get_train_val_sequence_names(dataset_rootdir, dataset_path):
    train_sequence_names = []
    validation_sequence_names = []
    seq_info_file = os.path.join(dataset_rootdir, dataset_path, 'sequences.json')
    with open(seq_info_file, "r") as f:
        seq_info_data = json.load(f)
    n_sequences = seq_info_data['n_sequences']
    for i in range(n_sequences):
        sequence_name = 'sequence_' + str(i+1)
        category = seq_info_data['sequences'][sequence_name]['category']
        if category == 'train': train_sequence_names.append(sequence_name)
        if category == 'validation': validation_sequence_names.append(sequence_name)
    return train_sequence_names, validation_sequence_names


def get_train_val_sequence_names_v2(dataset_rootdir, dataset_path):
    # get total number of sequences in the dataset
    seq_info_file = os.path.join(dataset_rootdir, dataset_path, 'sequences.json')
    with open(seq_info_file, "r") as f:
        seq_info_data = json.load(f)
    n_sequences = seq_info_data['n_sequences']

    # get the sequences that corrospond to 'training' and 'validation'
    train_test_sequence_names = []
    validation_sequence_names = []
    for i in range(n_sequences):
        sequence_name = 'sequence_' + str(i+1)
        category = seq_info_data['sequences'][sequence_name]['category']
        if category == 'train': train_test_sequence_names.append(sequence_name)
        if category == 'validation': validation_sequence_names.append(sequence_name)

    # sequence indexes for test and train
    idx_all = {i for i in range(len(train_test_sequence_names))}
    idx_test = {4, 6, 11, 16, 18, 24, 33, 34, 36, 37, 42, 44, 48, 52, 
                53, 60, 63, 67, 73, 84, 86, 92, 94, 100, 108, 119, 124, 126}
    idx_train = idx_all - idx_test

    train_sequence_names = [train_test_sequence_names[i] for i in idx_train]
    test_sequence_names = [train_test_sequence_names[i] for i in idx_test]

    return train_sequence_names, validation_sequence_names, test_sequence_names

# ---------------------------------------------------------------------------------------------------------------
def extract_radar_mount_data(dataset_rootdir, dataset_path):
    mount_file = os.path.join(dataset_rootdir, dataset_path, 'sensors.json')
    with open(mount_file, "r") as f:
        mount_data = json.load(f)
    return mount_data

# ---------------------------------------------------------------------------------------------------------------
def extract_sensor_data_all_scenes(sequence_name: str, dataset_rootdir: str, dataset_path: str):
    sensor_data_file = os.path.join(dataset_rootdir, dataset_path, sequence_name, 'radar_data.h5')
    with h5py.File(sensor_data_file, "r") as f:           # radar and odom data for a sequence
        radar_data_all_scenes = f["radar_data"][:]
        odometry_data_all_scenes = f["odometry"][:]
    return radar_data_all_scenes, odometry_data_all_scenes

# ---------------------------------------------------------------------------------------------------------------
def extract_all_radar_data(radar_data_all_scenes, startidx, endidx):
    radar_data = radar_data_all_scenes[startidx:endidx]
    timestamp = radar_data['timestamp']        # in micro seconds relative to some arbitrary origin
    rcs = radar_data['rcs']                    # in dBsm, RCS value of the detection
    vr = radar_data['vr']                            # in m/s. Radial velocity measured for this detection
    vr_compensated = radar_data['vr_compensated']    # in m/s: Radial velocity for this detection but compensated for the ego-motion
    x_cc = radar_data['x_cc']          # in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
    y_cc = radar_data['y_cc']          # in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
    track_id = radar_data['track_id']  # id of the dynamic object this detection belongs to. Empty, if it does not belong to any.
    # semantic class id of the object to which this detection belongs. 
    # passenger cars (0), 
    # large vehicles (like agricultural or construction vehicles) (1), 
    # trucks (2),
    # buses (3), 
    # trains (4), 
    # bicycles (5), 
    # motorized two-wheeler (6), 
    # pedestrians (7), 
    # groups of pedestrian (8), 
    # animals (9), 
    # all other dynamic objects encountered while driving (10), 
    # and the static environment (11)
    label_id = radar_data['label_id']  
    return {
        'timestamp': timestamp,
        'rcs': rcs,
        'vr': vr,
        'vr_compensated': vr_compensated,
        'x_cc': x_cc,
        'y_cc': y_cc,
        'track_id': track_id,
        'label_id': label_id }

def extract_all_radar_data_v2(radar_data_all_scenes, startidx, endidx):
    radar_data = radar_data_all_scenes[startidx:endidx]
    timestamp = radar_data['timestamp']        # in micro seconds relative to some arbitrary origin
    range_sc = radar_data['range_sc']          # in meters, radial distance to the detection, sensor coordinate system
    azimuth_sc = radar_data['azimuth_sc']      # in radians, azimuth angle to the detection, sensor coordinate system
    rcs = radar_data['rcs']                    # in dBsm, RCS value of the detection
    vr = radar_data['vr']                            # in m/s. Radial velocity measured for this detection
    vr_compensated = radar_data['vr_compensated']    # in m/s: Radial velocity for this detection but compensated for the ego-motion
    x_cc = radar_data['x_cc']          # in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
    y_cc = radar_data['y_cc']          # in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
    x_seq = radar_data['x_seq']        # in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
    y_seq = radar_data['y_seq']        # in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
    uuid = radar_data['uuid']          # unique identifier for the detection. Can be used for association with predicted labels and debugging
    track_id = radar_data['track_id']  # id of the dynamic object this detection belongs to. Empty, if it does not belong to any.
    # semantic class id of the object to which this detection belongs. 
    # passenger cars (0), 
    # large vehicles (like agricultural or construction vehicles) (1), 
    # trucks (2),
    # buses (3), 
    # trains (4), 
    # bicycles (5), 
    # motorized two-wheeler (6), 
    # pedestrians (7), 
    # groups of pedestrian (8), 
    # animals (9), 
    # all other dynamic objects encountered while driving (10), 
    # and the static environment (11)
    label_id = radar_data['label_id']  
    return {
        'timestamp': timestamp,
        'range_sc': range_sc,
        'azimuth_sc': azimuth_sc,
        'rcs': rcs,
        'vr': vr,
        'vr_compensated': vr_compensated,
        'x_cc': x_cc,
        'y_cc': y_cc,
        'x_seq': x_seq,
        'y_seq': y_seq,
        'uuid': uuid,
        'track_id': track_id,
        'label_id': label_id }

# ---------------------------------------------------------------------------------------------------------------
def extract_odometry_data(odometry_data_all_scenes, index):
    odometry_data = odometry_data_all_scenes[index]
    timestamp = odometry_data['timestamp']
    x_seq = odometry_data['x_seq']       # px position of the ego-vehicle relative to some global origin
    y_seq = odometry_data['y_seq']       # py position of the ego-vehicle relative to some global origin
    yaw_seq = odometry_data['yaw_seq']   # orientation of the ego-vehicle relative to some global origin
    vx = odometry_data['vx']
    yaw_rate = odometry_data['yaw_rate']
    return {
        'timestamp': timestamp,
        'x_seq': x_seq,
        'y_seq': y_seq,
        'yaw_seq': yaw_seq,
        'vx': vx,
        'yaw_rate': yaw_rate }

# ---------------------------------------------------------------------------------------------------------------
def aggregate_dataset(sequence_name, dataset_rootdir, dataset_path):
    scenes_file = os.path.join(dataset_rootdir, dataset_path, sequence_name, 'scenes.json')
    with open(scenes_file, "r") as f:        # specific sequence info
        scenes_data = json.load(f)

    current_timestamp_list = []
    radar_id_list = []
    odometry_timestamp_list = []
    odometry_index_list = []
    radar_indices_list = []

    scenes = scenes_data['scenes']
    first_scene = scenes[str(scenes_data['first_timestamp'])]
    current_timestamp_list.append(scenes_data['first_timestamp'])
    radar_id_list.append(first_scene['sensor_id'])
    odometry_timestamp_list.append(first_scene['odometry_timestamp'])
    odometry_index_list.append(first_scene['odometry_index'])
    radar_indices_list.append(first_scene['radar_indices'])

    next_timestamp = first_scene['next_timestamp']
    while next_timestamp is not None:
        curr_timestamp = next_timestamp
        scene = scenes[str(curr_timestamp)]
        current_timestamp_list.append(curr_timestamp)
        radar_id_list.append(scene['sensor_id'])
        odometry_timestamp_list.append(scene['odometry_timestamp'])
        odometry_index_list.append(scene['odometry_index'])
        radar_indices_list.append(scene['radar_indices'])
        next_timestamp = scene['next_timestamp']

    return \
        current_timestamp_list,\
        radar_id_list,\
        odometry_timestamp_list,\
        odometry_index_list,\
        radar_indices_list,\
        scenes

# ---------------------------------------------------------------------------------------------------------------
def create_dataset_sliding_window(
    window_size,
    current_timestamp_list,
    radar_id_list,
    odometry_timestamp_list,
    odometry_index_list,
    radar_data_indices_list):

    data_list = []
    sequence_length = len(current_timestamp_list)
    num_sliding_positions = sequence_length - window_size + 1
    for i in range(num_sliding_positions):
        startidx = i
        endidx = i + window_size
        data = {}
        data['current_timestamps'] = current_timestamp_list[startidx:endidx]
        data['radar_id'] = radar_id_list[startidx:endidx]
        data['odometry_timestamp'] = odometry_timestamp_list[startidx:endidx]
        data['odometry_index'] = odometry_index_list[startidx:endidx]
        data['radar_data_indices'] = radar_data_indices_list[startidx:endidx]
        data_list.append(data)
    return data_list

# ---------------------------------------------------------------------------------------------------------------
def extract_and_sync_radar_data(
    radar_mount_data,
    radar_data_all_scenes,
    odometry_data_all_scenes,
    windowed_data,
    reject_outlier_by_ransac):

    radar_data_indices_list = windowed_data['radar_data_indices']
    odometry_index_list = windowed_data['odometry_index']
    radar_id_list = windowed_data['radar_id']

    meas_rcs_list = []
    meas_timestamp_list = []
    meas_track_id_list = []
    meas_sensor_id_list = []
    stationary_meas_flag_list = []
    meas_label_id_list = []

    meas_px_list = []; meas_py_list = []
    meas_vx_list = []; meas_vy_list = []; meas_vr_list = []
    ego_px_gloal = []; ego_py_gloal = []
    ego_yaw_gloal = []

    for radar_data_indices, odometry_index, radar_id in zip(radar_data_indices_list, odometry_index_list, radar_id_list):

        # extract radar, odom and radar mount data
        radar_data = radar_data_all_scenes[radar_data_indices[0]:radar_data_indices[1]]
        odometry_data = odometry_data_all_scenes[odometry_index] 
        radar_mount = radar_mount_data['radar_' + str(radar_id) ]

        # extract radar id and radar mount parameters
        mount_tx = radar_mount['x']
        mount_ty = radar_mount['y']
        mount_yaw = radar_mount['yaw']

        # identify stationary measurements (by gating and ransac)
        stationary_meas_flag = identify_stationary_measurements(
            radar_data['azimuth_sc'], radar_data['vr'], \
            mount_tx, mount_ty, mount_yaw, \
            odometry_data['vx'], odometry_data['yaw_rate'],
            reject_outlier_by_ransac)
        stationary_meas_flag_list.append(stationary_meas_flag)

        # extract odometry data
        ego_px_gloal.append(odometry_data['x_seq'] )
        ego_py_gloal.append(odometry_data['y_seq']   )
        ego_yaw_gloal.append(odometry_data['yaw_seq']  )

        # extract radar data
        meas_px_list.append(radar_data['x_cc'])
        meas_py_list.append(radar_data['y_cc'])
        vx, vy = vr_cartesian_vf(
            radar_data['vr_compensated'], 
            radar_data['azimuth_sc'], 
            mount_yaw)
        meas_vx_list.append(vx)
        meas_vy_list.append(vy)

        # extract other radar data
        meas_vr_list.append(radar_data['vr_compensated'])
        meas_rcs_list.append(radar_data['rcs'])
        meas_timestamp_list.append(radar_data['timestamp'])
        meas_track_id_list.append(radar_data['track_id'])
        meas_sensor_id_list.append(radar_data['sensor_id'])
        meas_label_id_list.append(radar_data['label_id'])

    meas_px_list, meas_py_list, meas_vx_list, meas_vy_list,\
        = ego_compensate_radar_frames_list(
                meas_px_list, meas_py_list, meas_vx_list, meas_vy_list, \
                ego_px_gloal, ego_py_gloal, ego_yaw_gloal)
    return \
        meas_px_list, meas_py_list, \
        meas_vx_list, meas_vy_list, meas_vr_list, \
        meas_rcs_list, meas_timestamp_list, \
        meas_track_id_list, meas_sensor_id_list, \
        stationary_meas_flag_list, \
        meas_label_id_list

# ---------------------------------------------------------------------------------------------------------------
def convert_list_ndarry_to_ndarray(
    meas_px_list, meas_py_list, \
    meas_vx_list, meas_vy_list, meas_vr_list, \
    meas_rcs_list, meas_timestamp_list, \
    meas_trackid_list, meas_sensorid_list, \
    stationary_meas_flag_list, meas_label_id_list):

    meas_px = np.concatenate(meas_px_list, axis=0)
    meas_py = np.concatenate(meas_py_list, axis=0)
    meas_vx = np.concatenate(meas_vx_list, axis=0)
    meas_vy = np.concatenate(meas_vy_list, axis=0)
    meas_vr = np.concatenate(meas_vr_list, axis=0)
    meas_rcs = np.concatenate(meas_rcs_list, axis=0)
    meas_timestamp = np.concatenate(meas_timestamp_list, axis=0)
    meas_trackid = np.concatenate(meas_trackid_list, axis=0)
    meas_sensorid = np.concatenate(meas_sensorid_list, axis=0)
    stationary_meas_flag = np.concatenate(stationary_meas_flag_list, axis=0)
    meas_label_id = np.concatenate(meas_label_id_list, axis=0)
    return \
        meas_px, meas_py, meas_vx, meas_vy, meas_vr, meas_rcs, \
        meas_timestamp, meas_trackid, meas_sensorid, \
        stationary_meas_flag, meas_label_id

# ---------------------------------------------------------------------------------------------------------------
def create_sequences_info_list_v2(
    dataset_rootdir, dataset_path, 
    temporal_window_size, 
    sequence_names):

    dataset_metadata = []
    for sequence_name in sequence_names:

        current_timestamp_list,\
        radar_id_list,\
        odometry_timestamp_list,\
        odometry_index_list,\
        radar_data_indices_list, _ \
            = aggregate_dataset(sequence_name, dataset_rootdir, dataset_path)

        windowed_data_list \
            = create_dataset_sliding_window(
                temporal_window_size,
                current_timestamp_list,
                radar_id_list,
                odometry_timestamp_list,
                odometry_index_list,
                radar_data_indices_list)
        
        for windowed_data in windowed_data_list:
            data = {}
            data['sequence_name'] = sequence_name
            data['data'] = windowed_data
            dataset_metadata.append(data)

    return dataset_metadata


def create_sequences_info_list(dataset_rootdir, dataset_path, temporal_window_size):
    sequences_file = os.path.join(dataset_rootdir, dataset_path, 'sequences.json')
    with open(sequences_file, "r") as f:
        sequences_data = json.load(f)
    sequences_data = sequences_data['sequences']

    dataset_metadata = []
    for sequence_name in tqdm(sequences_data.keys()):
        value = sequences_data[sequence_name]

        current_timestamp_list,\
        radar_id_list,\
        odometry_timestamp_list,\
        odometry_index_list,\
        radar_data_indices_list, _ \
            = aggregate_dataset(sequence_name, dataset_rootdir, dataset_path)

        windowed_data_list \
            = create_dataset_sliding_window(
                temporal_window_size,
                current_timestamp_list,
                radar_id_list,
                odometry_timestamp_list,
                odometry_index_list,
                radar_data_indices_list)
        
        for windowed_data in windowed_data_list:
            data = {}
            data['sequence_name'] = sequence_name
            data['category'] = value['category']  # train or validation
            data['data'] = windowed_data
            dataset_metadata.append(data)

    return dataset_metadata

# ---------------------------------------------------------------------------------------------------------------
def create_train_validation_sets(dataset_metadata):
    train_dataset_metadata = []
    validation_dataset_metadata = []
    for item in dataset_metadata:
        if item['category'] == 'train':
            train_dataset_metadata.append(item)
        elif item['category'] == 'validation':
            validation_dataset_metadata.append(item)
    return train_dataset_metadata, validation_dataset_metadata

# ---------------------------------------------------------------------------------------------------------------
def save_metadata(metadata, file_out_path):
    with open(file_out_path, 'w') as json_file: 
        json.dump(metadata, json_file, indent=4)
    print(f'Labels saved in : {file_out_path}')

# ---------------------------------------------------------------------------------------------------------------
def get_sequence_data(
    dataset_rootdir,
    dataset_path,
    sequence_name, 
    window_size):

    radar_mount_data = extract_radar_mount_data(dataset_rootdir, dataset_path)
    radar_data_all_scenes, odometry_data_all_scenes = extract_sensor_data_all_scenes(sequence_name, dataset_rootdir, dataset_path)

    current_timestamp_list,\
    radar_id_list,\
    odometry_timestamp_list,\
    odometry_index_list,\
    radar_data_indices_list, \
    scenes = aggregate_dataset(sequence_name, dataset_rootdir, dataset_path)

    windowed_data_list = create_dataset_sliding_window(
        window_size,
        current_timestamp_list,
        radar_id_list,
        odometry_timestamp_list,
        odometry_index_list,
        radar_data_indices_list)
    return windowed_data_list, radar_mount_data, radar_data_all_scenes, odometry_data_all_scenes

# ---------------------------------------------------------------------------------------------------------------
def extract_frame(
    idx, 
    windowed_data_list,
    radar_mount_data,
    radar_data_all_scenes,
    odometry_data_all_scenes,
    reject_outlier):

    meas_px_sync, meas_py_sync, \
    meas_vx_sync, meas_vy_sync, meas_vr_sync, \
    meas_rcs_sync, meas_timestamp_sync, \
    meas_trackid_sync, meas_sensorid_sync, \
    stationary_flag, meas_label_id \
        = extract_and_sync_radar_data(
            radar_mount_data,
            radar_data_all_scenes,
            odometry_data_all_scenes,
            windowed_data_list[idx],
            reject_outlier)

    meas_px_all, meas_py_all, \
    meas_vx_all, meas_vy_all, meas_vr_all, \
    meas_rcs_all, meas_timestamp_all, \
    meas_trackid_all, meas_sensorid_all, \
    stationary_flag, meas_label_id \
        = convert_list_ndarry_to_ndarray(
            meas_px_sync, meas_py_sync, \
            meas_vx_sync, meas_vy_sync, meas_vr_sync, \
            meas_rcs_sync, meas_timestamp_sync, \
            meas_trackid_sync, meas_sensorid_sync,
            stationary_flag, meas_label_id )
    
    data_dict = {}
    data_dict['meas_px'] = meas_px_all.astype(np.float32)
    data_dict['meas_py'] = meas_py_all.astype(np.float32)
    data_dict['meas_vx'] = meas_vx_all.astype(np.float32)
    data_dict['meas_vy'] = meas_vy_all.astype(np.float32)
    data_dict['meas_vr'] = meas_vr_all.astype(np.float32)
    data_dict['meas_rcs'] = meas_rcs_all.astype(np.float32)
    data_dict['meas_timestamp'] = meas_timestamp_all
    data_dict['meas_trackid'] = meas_trackid_all
    data_dict['meas_sensorid'] = meas_sensorid_all
    data_dict['stationary_meas_flag'] = stationary_flag
    data_dict['meas_label_id'] = meas_label_id
    return data_dict

# ---------------------------------------------------------------------------------------------------------------
def get_data_for_datagen(dataset_rootdir, dataset_path, metadata, reject_outlier, flip_along_x=False):
    sequence_name = metadata['sequence_name']
    windowed_data = metadata['data']

    radar_mount_data = extract_radar_mount_data(dataset_rootdir, dataset_path)
    radar_data_all_scenes, odometry_data_all_scenes = extract_sensor_data_all_scenes(sequence_name, dataset_rootdir, dataset_path)

    meas_px_list, meas_py_list, \
    meas_vx_list, meas_vy_list, meas_vr_list, \
    meas_rcs_list, meas_timestamp_list, \
    meas_trackid_list, meas_sensorid_list, \
    stationary_meas_flag_list, \
    meas_label_id_list \
        = extract_and_sync_radar_data(
                radar_mount_data,
                radar_data_all_scenes,
                odometry_data_all_scenes,
                windowed_data,
                reject_outlier)

    meas_px, meas_py, \
    meas_vx, meas_vy, meas_vr, \
    meas_rcs, meas_timestamp, \
    meas_trackid, meas_sensorid, \
    stationary_meas_flag, \
    meas_label_id \
        = convert_list_ndarry_to_ndarray(
                meas_px_list, meas_py_list, \
                meas_vx_list, meas_vy_list, meas_vr_list, \
                meas_rcs_list, meas_timestamp_list, \
                meas_trackid_list, meas_sensorid_list, \
                stationary_meas_flag_list, meas_label_id_list)
    
    if flip_along_x == True:
        meas_py = -meas_py
        meas_vy = -meas_vy
    
    return {
        'meas_px': meas_px.astype(np.float32), 'meas_py': meas_py.astype(np.float32),
        'meas_vx': meas_vx.astype(np.float32), 'meas_vy': meas_vy.astype(np.float32), 'meas_vr': meas_vr.astype(np.float32),
        'meas_rcs': meas_rcs.astype(np.float32), 'meas_timestamp': meas_timestamp,
        'meas_trackid': meas_trackid, 'meas_sensorid': meas_sensorid,
        'stationary_meas_flag': stationary_meas_flag,
        'meas_label_id': meas_label_id }