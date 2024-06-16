

# Anchor Free Object Detection 

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0.1+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

[Code](AnchorFree2DObjectDetection) <br>
[Project Documents](AnchorFree2DObjectDetection/doc) <br>
[Output Videos](AnchorFree2DObjectDetection/video_inference) <br>

## Introduction
This project is about the development of an **Anchor free 2D object detection** model using **PyTorch**, 
that aims to provide a comprehensive guide for enthusiasts, researchers, and practitioners. 
Here the object detection model is trained from scratch, incorporating a **ImageNet pre-trained backbone from PyTorch**. The model is trained using a modest system configuration ( NVIDIA RTX A2000 4 GB Laptop GPU ), thus enabling users with low computational resources to train object detection models that give resonably good performance.
An easy to understand and extend codebase is developed in this project.
The following are the key highlights:
   - Training a 2D object detection Model in PyTorch from scratch by utilizing 
     Imagenet dataset pre-trained backbone from PyTorch.
   - Development of an easy to understand and well documented codebase.
   - Implementation of a method for tuning the detection threshold parameters.
   - Utilizing training samples from two publicly available datasets: [KITTI](https://www.cvlibs.net/datasets/kitti/) and [BDD](https://bdd-data.berkeley.edu/), 
     so as to provide a technique to merge samples from multiple training datasets,
     enabling users to utilize a diverse range of training data for model generalization.

<br>

<div align="center">

![](AnchorFree2DObjectDetection/_readme_artifacts/1_model_archi.PNG)

*Anchor Free Network Architecture.*

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/2_detections.PNG)

*Detected Bounding Boxes (BDD).*

<br>

![](AnchorFree2DObjectDetection/video_inference/kitti/gif/0007.gif)

*Detections in video (KITTI).*

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
<li><a href="#Exploratory-Data-Analysis">Exploratory Data Analysis</a>
   <ol>
      <li><a href="#Scene-and-Label-Instance">Scene and Label Instance</a></li>
      <li><a href="#Bounding-box-distribution">Bounding box distribution</a></li>
      <li><a href="#Wrong-annotations">Wrong annotations</a></li>
      <li><a href="#Dataset-Modification">Dataset Modification</a></li>
   </ol>
</li>
<li><a href="#Model-Architecture">Model Architecture</a> 
   <ol>
       <li><a href="#Concept-Level-Architecture">Concept Level Architecture</a></li> 
       <li><a href="#Backbone-for-Feature-Computation">Backbone for Feature Computation</a></li> 
       <li><a href="#Neck-for-Feature-Aggregation">Neck for Feature Aggregation</a></li> 
       <li><a href="#Head-for-Dense-Object Detection">Head for Dense Object Detection</a></li> 
   </ol> 
</li>
<li><a href="#Ground-Truth-Generation">Ground Truth Generation</a>
   <ol>
       <li><a href="#Bounding-Box-Offsets">Bounding Box Offsets</a></li> 
       <li><a href="#Centerness-Score">Centerness Score</a></li> 
       <li><a href="#Objectness-and-Object-Class">Objectness and Object Class</a></li>
   </ol> 
</li>
<li><a href="#Training">Training</a>  
   <ol>
       <li><a href="#Augmentation">Augmentation</a></li>
       <li><a href="#Loss-Functions">Loss Functions</a></li> 
       <li><a href="#Optimization-method">Optimization method</a></li>
   </ol> 
</li>
<li><a href="#Performance-Evaluation">Performance Evaluation</a>
   <ol>
       <li><a href="#BDD-Dataset">BDD Dataset</a></li>
       <li><a href="#KITTI-Dataset">KITTI Dataset</a></li>
   </ol>
</li>
<li><a href="#Conclusion">Conclusion</a></li>
<li><a href="#Reference">Reference</a></li>

</ul>
</details>

<br>


## About The Project

### Requirements
```bash
opencv_python>=4.8.0.74
imageio>=2.34.0
matplotlib>=3.7.2
numpy>=1.25.0
torch>=2.0.1
torchvision>=0.15.2
tqdm>=4.66.1
```

### How to run the project
```bash
git clone https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA
cd ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/AnchorFree2DObjectDetection

# to run inference on bdd video frames
python video_inference_bdd.py

# to run inference on kitti video frames
python video_inference_kitti.py

# to create the labels file
python script1_create_dataset.py

# to train the model use script3_train_model.ipynb 

# to write detections to video
cd write_detections_to_video
python write_detection_to_video_bdd.py
```

### Project Folder Structure
```bash
AnchorFree2DObjectDetection
│───doc                          # Project documents
│───hyperparam                   # Statistical data of the Bounding Box offsets
│───labels                       # aggregated GT labels data of KITTI and BDD dataset
│───mAP                          # module to compute mAP ( https://github.com/Cartucho/mAP.git )
│───model_weights                # model weights data after training
│───tensorboard                  # data folder for loss visualization in tensorboard.
│───modules                      # main modules 
      │───augmentation           # scripts for image augmentation functions            
      │───dataset_utils          # scripts for data analysis and dataset generation
      │───evaluation             # scripts for detector evaluation and threshold determination   
      │───first_stage            # scripts for defining the model and ground truth generation function for dense object detection
      │───hyperparam             # scripts for computing the bounding box offsets statistics from training data    
      │───loss                   # loss functions
      │───neural_net             # scripts for defining various neural net blocks             
            │───backbone               # model backbone blocks
            │───bifpn                  # BIFPN blocks for model neck            
            │───fpn                    # FPN blocks for model neck
            │───head                   # blocks for model head            
            │   common.py              # common model building blocks
            │   constants.py           # constants for model construction  
      │───plot                   # contains plotting functions
      │───pretrained             # scripts for loading the pre-trained backbone from pytorch            
      │───proposal               # scripts for proposal generation
      │───second-stage           # <work under progress> scripts for defining the model and ground truth generation function for second stage object detection              
│───tests                                    # folder for testing and validation scripts
│───video_inference                          # detection results saved as video
│───write_detections_to_video                # scripts to save detections as video, results are saved in 'video_inference' folder
│   config_dataset.py                        # parameters and constants for dataset 
│   config_neuralnet_stage1.py               # model design parameters
│   script1_create_datasets.py               # aggregate gt labels and save it inside the 'labels' folder
│   script2_gen_hyperparam.py                # aggregate and save the box offsets and its statistics inside the 'hyperparam' folder
│   script3_train_model.ipynb                # notebook to train the model 
│   script4_inference_bdd.ipynb              # run inference on the bdd dataset images
│   script4_inference_kitti.ipynb            # run inference on the kitti dataset images      
│   script5_compute_mAP_bdd.ipynb            # compute mean average precison (mAP) on the bdd dataset   
│   script5_compute_mAP_kitti.ipynb          # compute mean average precison (mAP) on the kitti dataset
│   video_inference_bdd.py                   # run inference on the bdd dataset video
│   video_inference_kitti.py                 # run inference on the kitti dataset frame sequence video               
```
[TOC](#t0)

<br>




## Exploratory Data Analysis
To have good performance from a trained object detection model, the training dataset needs to be large, diverse, balanced and the annotation has to be correct. BDD dataset is adequately large to train a resonably good performing model. Below are the data analysis conducted to get an insight about the quality of the dataset where good quality means that the training dataset has to be diverse and balanced.

### Scene and Label Instance
![](AnchorFree2DObjectDetection/_readme_artifacts/4_eda_class_count.PNG)

<div align="center">

*Number of instances of different classes and scenes.* 
</div>

<br>

**Observations**
<ul>
   <li>There is a huge intra-class as well as inter-clss imbalance in the dataset (depends on how we are considering the intra and inter class).</li>
   <li>The intra-class imbalance is present in the number of instances of traffic light, where there are much less number of yellow traffic lights. The red and green instances are resonably balanced.</li>
   <li>The intra-class imbalance is also observed in the number of instances of road vehicles, where the car class has huge number of instances than other classes like truck and bus.</li>
   <li>The inter-class imbalance can be seen in the number of instances of vehicles and non-vehicles, where the car class has huge number of instances than other classes like person, rider, train etc.</li>
</ul>

[TOC](#t0)
<br>

### Bounding box distribution
![](AnchorFree2DObjectDetection/_readme_artifacts/5_box_distribution.png)

<div align="center">

*Annotated bounding box dimension scatter plot.*
</div>

<br>

**Observations**
<ul>
   <li>From the plot we can observe that there are some boxes that are probably incorrect annotations. These either have extreme aspect ratio or the area is too small</li>
</ul>

[TOC](#t0)
<br>

### Wrong annotations
If we select those boxes from the previous scatter plot that has some **extreme aspect ratio** or the **area is very small**, we would be able to identfy annotation errors. Some of them can be categorized as follows.
<ul>
<li> 

**Box area too small**

![](AnchorFree2DObjectDetection/_readme_artifacts/6_box_area_too_small.PNG) 

</li>
<li> 

**Extreme Box Aspect Ratio**

![](AnchorFree2DObjectDetection/_readme_artifacts/6_box_aspect_ratio_extreme.PNG) 

</li>
<li> 

**Incorrect Class**

![](AnchorFree2DObjectDetection/_readme_artifacts/6_incorrect_class.PNG) 

</li>


[TOC](#t0)
<br>

### Dataset Modification
Based on the above analysis the training samples and the dataset annotations are modified to 
<ul>
   <li>Simplify the development of object detection model in version 1 by reducing the number of classes and removing the highly imbalanced and irrelevant classes.</li> 
   <li>Reduce the number of wrong and low quality annotations. </li>
</ul>

<br>

The modifications are as follows:
<ul>
<li>

**Car**, **bus**, **truck** are merged as **vehicle**; **person** and **rider** are merged as **person**. The remaining classes are part of negative class.</li>
<li>Select boxes that satisfy the below conditions:
<ul>
<li> Box width &ge; 5 pixels </li>
<li> Box heighth &ge; 5 pixels </li>
<li> 0.1 &le; Box aspect ratio &le; 10 </li>
</ul></li>
</ul>

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/6_dataset_modifications.PNG)

<br>

<div align="center">

**Relevant Scripts (BDD)**

<table>
<tr><td>

|                         SCRIPT                       |               LINK                 |
|:----------------------------------------------------:|:------------------------------------------:|
|    1_1_eda_vis_anno_data.ipynb                       |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/1_1_eda_vis_anno_data.ipynb) | 
|    1_2_eda_plot_label_count_distrib.ipynb            |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/1_2_eda_plot_label_count_distrib.ipynb)                                |
|    1_3_eda_bbox_distrib.ipynb                        |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/1_3_eda_bbox_distrib.ipynb)                           |
|    1_4_eda_vis_different_obj_categories.ipynb        |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/1_4_eda_vis_different_obj_categories.ipynb)                |
|    1_5_eda_identifying_anno_errors.ipynb             |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/1_5_eda_identifying_anno_errors.ipynb)       | 
|    2_1_eda_vis_remapped_anno_data.ipynb              |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/2_1_eda_vis_remapped_anno_data.ipynb)                              |
|    2_2_eda_plot_remapped_label_count_distrib.ipynb   |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/2_2_eda_plot_remapped_label_count_distrib.ipynb)                           |
|    2_3_eda_remapped_bbox_distrib.ipynb               |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/2_3_eda_remapped_bbox_distrib.ipynb)               |
|    2_4_eda_vis_remapped_obj_categories.ipynb         |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/2_4_eda_vis_remapped_obj_categories.ipynb)        | 
|    2_5_eda_identifying_outliers.ipynb                |  [Link](AnchorFree2DObjectDetection/tests/bdd/dataset_utils/2_5_eda_identifying_outliers.ipynb)                               |

</td></tr> 
</table>

<br>

**Relevant Scripts (KITTI)**

<table>
<tr><td>

|                         SCRIPT                       |               LINK                 |
|:----------------------------------------------------:|:------------------------------------------:|
|    eda_identifying_outliers.ipynb                    |  [Link](AnchorFree2DObjectDetection/tests/kitti/dataset_utils/eda_identifying_outliers.ipynb) | 
|    eda_plot_remapped_label_count_distrib.ipynb       |  [Link](AnchorFree2DObjectDetection/tests/kitti/dataset_utils/eda_plot_remapped_label_count_distrib.ipynb)                                |
|    eda_remapped_bbox_distrib.ipynb                   |  [Link](AnchorFree2DObjectDetection/tests/kitti/dataset_utils/eda_remapped_bbox_distrib.ipynb)                           |

</td></tr> 
</table>

</div>


[TOC](#t0)
<br>






## Model Architecture

### Concept Level Architecture

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/7_high_level_archi.PNG)

### Backbone for Feature Computation

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/7_backbone_archi.PNG)

### Neck for Feature Aggregation

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/7_bifpn.PNG)

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/7_bifpn_formulas.PNG)

### Head for Dense Object Detection

![](AnchorFree2DObjectDetection/_readme_artifacts/7_head_archi.PNG)


### Architecture Summary

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/7_summary.PNG)


[TOC](#t0)

<br>






## Ground Truth Generation
Each of the anchors corrospond to an object hypothesis where the network shall learn to predict 4 values : **box offsets**, **centerness score**, **objectness score**, and **classification score** from the image. The groundtruth for training is computed as follows.

### Bounding Box Offsets

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/8_box_offsets.PNG)

### Centerness Score

![](AnchorFree2DObjectDetection/_readme_artifacts/8_centerness.PNG)

### Objectness and Object Class

![](AnchorFree2DObjectDetection/_readme_artifacts/8_one_hot.PNG)


[TOC](#t0)

<br>





## Training

### Augmentation 
Augmentation is performed during training. The augmentation process is depicted as follows

![](AnchorFree2DObjectDetection/_readme_artifacts/9_augment1.PNG)

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/9_augment2.PNG)

<br>

### Loss Functions

<div align="center">

<table>
<tr><td>

|                 TASK                 |    LOSS FUNCTION                           |
|:------------------------------------:|:------------------------------------------:|
|    Class Prediction                  |      Class Weighted Cross Entrophy Loss    | 
|    Objectness Prediction             |      Focal Loss                            |
|    Box Offset Regression             |      Smooth L1 Loss                        |
|    Centerness Score Regression       |      Binary Cross Entrophy Loss            |

</td></tr> 
</table>

</div>

<br>

### Optimization Method
Either **SGD with momentum** or **AdamW** oprimization method can be used. Refer to these scripts for more details:

<div align="center">

<table>
<tr><td>

|                         SCRIPT                       |               LINK                         |
|:----------------------------------------------------:|:------------------------------------------:|
|    set_parameters_for_training.py                    |  [Link](AnchorFree2DObjectDetection/modules/first_stage/set_parameters_for_training.py) | 
|    script3_train_model.ipynb                         |  [Link](AnchorFree2DObjectDetection/script3_train_model.ipynb) |

</td></tr> 
</table>

</div>

<br>

[TOC](#t0)

<br>

## Performance Evaluation

### BDD Dataset

<br>

<div align="center">

![](AnchorFree2DObjectDetection/_readme_artifacts/10_roc_bdd.PNG)

*Detection Rate vs False Positives per image (ROC Curve)*

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/10_pr_bdd.PNG)

*Recall vs Precision (PR Curve)*


<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/10_comp_bdd.PNG)

*Comparing performance for Vehicle and Person class*

<br>

<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|              Vehicle Detection Threshold         |       Precision (%)      |      Recall (%)   |       mAP@0.5 (%)   |
|:------------------------------------------------:|:------------------------:|:-----------------:|:-------------------:|
|                              0.4                 |             62.74%       |          79.77%   |           76.50%    |
|                              0.5                 |             80.15%       |          75.06%   |           73.11%    |
|                              **0.6**             |             90%          |          69.13%   |           68.06%    |
|                              0.7                 |             95.58%       |          61.21%   |           60.70%    |
|          **Person Detection Threshold**          |   **Precision (%)**      |  **Recall (%)**   |   **mAP@0.5 (%)**   |
|                              0.3                 |             44.7%        |          65.42%   |           56.41%    |
|                              0.4                 |             63.48%       |          59.52%   |           53.18%    |
|                              **0.5**             |             77.08%       |          50.68%   |           46.92%    |
|                              0.6                 |             86.46%       |          40.49%   |           38.54%    |


</td><td>

<img src="AnchorFree2DObjectDetection/_readme_artifacts/10_bdd_viz3.PNG"/>

</td></tr> </table>

*mAP at different detection threshold ( computed using [Link](https://github.com/Cartucho/mAP) )*



<br>

<table>
<tr><td>

|                         SCRIPT        |               LINK                 |
|:-------------------------------------:|:----------------------------------:|
|    bdd_score_tuning.ipynb             |  [Link](AnchorFree2DObjectDetection/modules/evaluation/bdd_score_tuning.ipynb) | 
|    bdd_nms_tuning.ipynb               |  [Link](AnchorFree2DObjectDetection/modules/evaluation/bdd_nms_tuning.ipynb)   |
|    script5_compute_mAP_bdd.ipynb      |  [Link](AnchorFree2DObjectDetection/script5_compute_mAP_bdd.ipynb) |

</td></tr> 
</table>

*Relevant Scripts*

</div>

[TOC](#t0)

<br>





### KITTI Dataset

<br>

<div align="center">

![](AnchorFree2DObjectDetection/_readme_artifacts/10_roc_kitti.PNG)

*Detection Rate vs False Positives per image (ROC Curve)*

<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/10_pr_kitti.PNG)

*Recall vs Precision (PR Curve)*


<br>

![](AnchorFree2DObjectDetection/_readme_artifacts/10_comp_kitti.PNG)

*Comparing performance for Vehicle and Person class*

<br>

<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|              Vehicle Detection Threshold         |       Precision (%)    |      Recall (%)  |       mAP@0.5 (%)   |
|:------------------------------------------------:|:----------------------:|:----------------:|:-------------------:|
|                              0.5                 |             79.24%     |          89.71%  |           88.03%    |
|                              0.6                 |             85.77%     |          87.92%  |           86.60%    |
|                          **0.7**                 |             91.15%     |          85.62%  |           84.56%    |
|                              0.8                 |             95.18%     |          80.20%  |           79.50%    |
|          **Person Detection Threshold**          |   **Precision (%)**    |  **Recall (%)**  |   **mAP@0.5 (%)**   |
|                              0.4                 |             45.69%     |          79.73%  |           70.60%    |
|                              0.5                 |             57.61%     |          75.63%  |           68.62%    |
|                          **0.6**                 |             69.73%     |          70.44%  |           65.38%    |
|                              0.7                 |             81.84%     |          62.53%  |           59.50%    |


</td><td>

<img src="AnchorFree2DObjectDetection/_readme_artifacts/10_kitti_viz3.PNG"/>

</td></tr> </table>

*mAP at different detection threshold ( computed using [Link](https://github.com/Cartucho/mAP) )*



<br>

<table>
<tr><td>

|                         SCRIPT          |               LINK                 |
|:---------------------------------------:|:----------------------------------:|
|    kitti_score_tuning.ipynb             |  [Link](AnchorFree2DObjectDetection/modules/evaluation/kitti_score_tuning.ipynb) | 
|    kitti_nms_tuning.ipynb               |  [Link](AnchorFree2DObjectDetection/modules/evaluation/kitti_nms_tuning.ipynb)   |
|    script5_compute_mAP_kitti.ipynb      |  [Link](AnchorFree2DObjectDetection/script5_compute_mAP_kitti.ipynb) |

</td></tr> 
</table>

*Relevant Scripts*

</div>

[TOC](#t0)

<br>

## Conclusion
<ul>
<li> Person class suffers from low recall due to much less number of training samples </li>
<li> The basic building block of the model is weight standardized conv2d followed by group norm and a non-linear activation. This helped in setting the batch size small (6 in this case) so that it fits in the gpu memory. It also helps in keeping the training stable (no NaNs). </li>
<li>There are ways to improve the performance. Some of them are: fine-tuning the backbone, utilizing several other open source datasets, taking a second stage to improve recall, training the model end to end for different tasks such as segmentation and tracking. These shall be part of future releases </li>
</ul>

<br>

[TOC](#t0)

<br>

## Reference
<ul>
   <li>

   [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://arxiv.org/pdf/1805.04687.pdf)</li>
   <li>
   
   [FCOS: A simple and strong anchor-free object detector](https://arxiv.org/pdf/2006.09214.pdf)</li>
   <li>
   
   [HybridNets: End-to-End Perception Network](https://arxiv.org/ftp/arxiv/papers/2203/2203.09035.pdf)</li>
   <li>
   https://www.cvlibs.net/datasets/kitti/</li>
</ul>

<br>

[TOC](#t0)



