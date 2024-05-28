# RadarScenes: A Real-World Radar Point Cloud Data Set for Automotive Applications
The RadarScenes data set (“data set”) contains recordings from four automotive radar sensors, which were mounted on one measurement-vehicle. Images from one front-facing documentary camera are added. 

The data set has a length of over 4h and in addition to the point cloud data from the radar sensors, semantic annotations on a point-wise level from 12 different classes are provided. 

In addition to point-wise class labels, a track-id is attached to each individual detection of a dynamic object, so that individual objects can be tracked over time.

## Structure of the Data Set
The data set consist of 158 individual sequences. For each sequence, the recorded data from radar and odometry sensors are stored in one hdf5 file. Each of these files is accompanied by a json file called “scenes.json” in which meta-information are stored. In a subfolder, the camera images are stored as jpg files. 

Two additional json files give further meta-information: in the "sensor.json" file, the sensor mounting position and rotation angles are defined. In the file "sequences.json", all recorded sequences are listed with additional information, e.g. about the recording duration.

### sensors.json
This file describes the position and orientation of the four radar sensors. Each sensor is attributed with an integer id. The mounting position is given relative to the center of the rear axle of the vehicle. This allows for an easier calculation of the ego-motion at the position of the sensors. Only the x and y position is given, since no elevation information is provided by the sensors. Similarly, only the yaw-angle for the rotation is needed.

### sequences.json
This file contains one entry for each recorded sequence. Each entry is built from the following information: the category (training or validation of machine learning algorithms), the number of individual scenes within the sequence, the duration in seconds and the names of the sensors which performed measurements within this sequence.

### scenes.json
In this file, meta-information for a specific sequence and the scenes within this sequence are stored.

The name of the sequence is listed within the top-level dictionary, the group of this sequence (training or validation) as well as the timestamps of the first and last time a radar sensor performed a measurement in this sequence.

A scene is defined as one measurement of one of the four radar sensors. For each scene, the sensor id of the respective radar sensor is listed. Each scene has one unique timestamp, namely the time at which the radar sensor performed the measurement. Four timestamps of different radar measurement are given for each scene: the next and previous timestamp of a measurement of the same sensor and the next and previous timestamp of a measurement of any radar sensor. This allows to quickly iterate over measurements from all sensors or over all measurements of a single sensor. For the association with the odometry information, the timestamp of the closest odometry measurement and additionally the index in the odometry table in the hdf5 file where this measurement can be found are given. Furthermore, the filename of the camera image whose timestamp is closest to the radar measurement is given. Finally, the start and end indices of this scene’s radar detections in the hdf5 data set “radar_data” is given. The first index corresponds to the row in the hdf5 data set in which the first detection of this scene can be found. The second index corresponds to the row in the hdf5 data set in which the next scene starts. That is, the detection in this row is the first one that does not belong to the scene anymore. This convention allows to use the common python indexing into lists and arrays, where the second index is exclusive: arr[start:end].

### radar_data.h5
In this file, both the radar and the odometry data are stored. Two data sets exists within this file: “odometry” and “radar_data”.

The “odometry” data has six columns: timestamp, x_seq, y_seq, yaw_seq, vx, yaw_rate. Each row corresponds to one measurement of the driving state. The columns x_seq, y_seq and yaw_seq describe the position and orientation of the ego-vehicle relative to some global origin. Hence, the pose in a global (sequence) coordinate system is defined. The column “vx” contains the velocity of the ego-vehicle in x-direction and the yaw_rate column contains the current yaw rate of the car.

The hdf5 data set “radar_data” is composed of the individual detections. Each row in the data set corresponds to one detection. A detection is defined by the following signals, each being listed in one column:

* timestamp: in micro seconds relative to some arbitrary origin
* sensor_id: integer value, id of the sensor that recorded the detection
* range_sc: in meters, radial distance to the detection, sensor coordinate system
* azimuth_sc: in radians, azimuth angle to the detection, sensor coordinate system
* rcs: in dBsm, radar cross section (RCS value) of the detection
* vr: in m/s. Radial velocity measured for this detection
* vr_compensated in m/s: Radial velocity for this detection but compensated for the ego-motion
* x_cc and y_cc: in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
* x_seq and y_seq in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
* uuid: unique identifier for the detection. Can be used for association with predicted labels and for debugging
* track_id: id of the dynamic object this detection belongs to. Empty, if it does not belong to any.
* label_id: semantic class id of the object to which this detection belongs. passenger cars (0), large vehicles (like agricultural or construction vehicles) (1), trucks (2), busses (3), trains (4), bicycles (5), motorized two-wheeler (6), pedestrians (7), groups of pedestrian (8), animals (9), all other dynamic objects encountered while driving (10), and the static environment (11)


## Camera Images
The images of the documentary camera are located in the subfolder “camera” of each sequence. The filename of each image corresponds to the timestamp at which the image was recorded.

The data set is a radar data set. Camera images are only included so that users of the data set get a better understanding of the recorded scenes. However, due to GDPR requirements, personal information was removed from these images via re-painting of regions proposed by a semantic instance segmentation network and manual correction. The networks were optimized for high recall values so that false-negatives were suppressed at the cost of having false positive markings. As the camera images are only meant to be used as guidance to the recorded radar scenes, this shortcoming has no negative effect on the actual data.


## License
The data set is licensed under Creative Commons Attribution Non Commercial Share Alike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Hence, the data set must not be used for any commercial use cases.


##Disclaimer

That the data set comes "AS IS", without express or implied warranty and/or any liability exceeding mandatory statutory obligations.
This especially applies to any obligations of care or indemnification in connection with the data set.
The annotations were created for our research purposes only and no quality assessment was done for the usage in products of any kind. 
We can therefore not guarantee for the correctness, completeness or reliability of the provided data set.