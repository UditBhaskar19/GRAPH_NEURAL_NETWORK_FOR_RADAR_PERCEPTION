

# Multi-task Learning using Message Passing Graph Neural Network for Radar based Perception Functions

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0.1+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

<br>

[Project Document](doc) <br>
[Output Videos](results) <br>

## Introduction
Radar is increasingly recognized as a crucial sensor for Advanced Driver Assistance Systems (ADAS) and autonomous driving (AD) perception tasks. Its ability to **penetrate occlusions**, **withstand adverse weather conditions**, and **operate independently of external lighting sources** positions it as one of the primary sensors for vehicle autonomy. Advancements in automotive radar technology, encompassing both hardware and digital signal processing (DSP) pipelines, have led to the availability of Synthetic Aperture Radar (SAR) in compact form factors suitable for installation on various vehicle types, including cars, trucks, buses and construction vehicles. Such radars have a high measurement resolution which is beneficial for deep learning based techniques to improve components in radar perception pipelines like measurement clustering, object tracking and track prediction.
<br><br>
One of the critical step in radar based object tracking for perception functions is the initialization of track hypothyesis. Typically, track hypotheses are established by clustering radar measurements using techniques such as DBSCAN, followed by assigning a unique track ID to each unassociated cluster. The density of the point cloud varies within the radar FOV as it depends on various factors like the object type, object shape, location of the object from the sensor, scene geometry and varous internal properties of the sensor. Traditional clustering algorithms like DBSCAN, employing constant threshold parameters, exhibit suboptimal performance, especially in cluttered scenes.
<br><br>
Thus, this project proposes a deep-learning-based transformation of radar point cloud to enable the use of DBSCAN-like clustering techniques with constant threshold parameters for object identification. Additionally, various other tasks are addressed, including link prediction, node segmentation, and object classification.
<br><br>
In summary, **given as input the radar measurements**. The **following task are performed** using deep learning:
   - **Measurement offset prediction for clustering**
   - **Link prediction for clustering**
   - **Measurement classification / Node segmentation**
   - **Object classification**

<br>

<div align="center">

![](modules/readme_artifacts/1_high_level_concept.PNG)

*GNN Inputs and Outputs.*

<br>

![](modules/readme_artifacts/2_model_architecture.PNG)

*Model Architecture.*

<br>

![](modules/readme_artifacts/10_tensorboard_plots.PNG)

*Tensorboard Plots.*

<br>

![](modules/readme_artifacts/9_confusion_mat_test.PNG)

*Normalized Confusion Matrix.*

<br>

![](modules/readme_artifacts/3_result_gif.gif)

*Predictions.*

<br>

</div>

<br>






<details>
<summary>

## Table of Contents <a name="t0"></a>

</summary>

<ul>

<li><a href="#About-The-Project">About The Project</a></li>
   <ol>
      <li><a href="#Requirements">Requirements</a></li>
      <li><a href="#How-to-run-the-project">How to run the project</a></li>
      <li><a href="#Project-Folder-Structure">Project Folder Structure</a></li>
   </ol>
</li>
<li><a href="#Dataset-and-Sensor-Setup">Dataset and Sensor Setup</a></li>
<li><a href="#Model-Architecture">Model Architecture</a> 
   <ol>
       <li><a href="#Concept-Level-Architecture">Concept Level Architecture</a></li> 
       <li><a href="#Node-and-Edge-Embedding">Node and Edge Embedding</a></li> 
       <li><a href="#Graph-Convolution">Graph Convolution</a></li> 
       <li><a href="#Graph-Link-Prediction">Graph Link Prediction</a></li> 
       <li><a href="#Node-Offset-Prediction">Node Offset Prediction</a></li> 
       <li><a href="#Node-Segmentation">Node Segmentation</a></li> 
       <li><a href="#Object-Classification">Object Classification</a></li> 
   </ol> 
</li>
<li><a href="#Predicted-vs-GT-Clusters-Visualization">Predicted vs GT Clusters Visualization</a></li>
<li><a href="#References">References</a></li>

</ul>
</details>

<br>





## About The Project

### Requirements
```bash
imageio>=2.34.0
matplotlib>=3.7.2
numpy>=1.25.0
torch>=2.0.1
torchvision>=0.15.2
torch_geometric>=2.5.0
tqdm>=4.66.1
```

### How to run the project
```bash
git clone https://github.com/UditBhaskar19/GRAPH_NEURAL_NETWORK_FOR_AUTOMOTIVE_RADAR_PERCEPTION
cd GRAPH_NEURAL_NETWORK_FOR_AUTOMOTIVE_RADAR_PERCEPTION

# to change the configurations modify the following file
configuration_radarscenes_gnn.yml

# to test the model if it can overfit use the follwing notebook
script_overfit_gnn.ipynb

# to train the model use the following notebook
script_train_model_gnn.ipynb

# to save outputs as a sequence of images use the following notebook
save_predictions.ipynb

# to save the ground-truth vs prediction comparison results use the following notebook
save_predictions_and_gt.ipynb

# to create GIF use the following notebook
create_gif.ipynb
```

### Project Folder Structure
```bash
doc               # project documents
dataset           # radarscenes dataset folder 
model_weights     # Model weights folder
tensorboard       # data folder for loss visualization in tensorboard
modules
│───compute_features         # module to compute input graph features for GNN
│───compute_groundtruth      # Compute ground-truths for model training
│───data_generator           # dataset generator module
│───data_utils               # dataset utilities for reading and arranging the input data from files
│───inference                # model inference modules
│───neural_net               # neural net modules.
│───plot_utils               # plotting and visualization.
│───set_configurations       # create and set configuration class.
│───readme_artifacts         # readme files.
script_overfit_gnn.ipynb         # overfit on a very small dataset
script_train_model_gnn.ipynb     # train GNN model
script_train_model_gnn_continue_train.ipynb    # continue training
save_predictions.ipynb           # save predictions inside the folder 'results'
save_predictions_and_gt.ipynb    # save predictions and gt comparizon plots inside the folder 'results'
create_gif.ipynb                 # create a gif video from a sequence of saved images               
```
[TOC](#t0)

<br>





## Dataset and Sensor Setup
[RadarScenes](https://radar-scenes.com/) dataset is used in this project 

<br>

<div align="center">

![](modules/readme_artifacts/0_sensor_setup.PNG)

*Sensor Setup.*

<br>

![](modules/readme_artifacts/0_input_data_processing.PNG)

*Input Data Processing. For more details refer the [Project Document](doc)*

</div>

[TOC](#t0)

<br>






## Model Architecture

### Concept Level Architecture

<br>

![](modules/readme_artifacts/2_model_architecture.PNG)

<br>

### Node and Edge Embedding

<br>

![](modules/readme_artifacts/3_feature_embed.PNG)

<br>

### Graph Convolution

<br>

![](modules/readme_artifacts/4_graph_conv.PNG)

<br>

### Graph Link Prediction

<br>

![](modules/readme_artifacts/5_link_pred.PNG)

<br>

### Node Offset Prediction

<br>

![](modules/readme_artifacts/6_offset_pred.PNG)

<br>

### Node Segmentation

<br>

![](modules/readme_artifacts/7_node_class.PNG)

<br>

### Object Classification

<br>

![](modules/readme_artifacts/8_object_class.PNG)

<br>

[TOC](#t0)

<br>




## Predicted vs GT Clusters Visualization

#### Sequence 108

![](modules/readme_artifacts/comp_sequence_108.gif)

<br>

#### Sequence 138

![](modules/readme_artifacts/comp_sequence_138.gif)

<br>


#### Sequence 147

![](modules/readme_artifacts/comp_sequence_147.gif)

<br>


#### Sequence 148

![](modules/readme_artifacts/comp_sequence_148.gif)

<br>



[TOC](#t0)

<br>

## References
<ul>
   <li>

   [RadarScenes: A Real-World Radar Point Cloud Data Set for Automotive Applications](https://arxiv.org/abs/2104.02493)</li>
   <li>
   
   [RadarGNN: Transformation Invariant Graph Neural Network for Radar-based Perception](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Fent_RadarGNN_Transformation_Invariant_Graph_Neural_Network_for_Radar-Based_Perception_CVPRW_2023_paper.pdf)</li>
   <li>
   
   [CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)</li>
   <li>
   https://radar-scenes.com/</li>
</ul>

<br>

[TOC](#t0)



