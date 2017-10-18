# Facial-Keypoint-Detector

### List of Content
- data: folder to keep training and testing data file to train neural network to recognize facial keypoints
- detector_architectures: folder to keep OpenCV pre-trained model to recognize faces and eyes
- AIND_Capstone.ipynb: python notebook with all code
- cnn_tuning.xlsx: many experiments to tune the global parameters of the neural network, which help finalize the best set of parameters used in the notebook

### Environment
Python 3.5 with Keras, which should be set to use Tensorflow as backend

Refer to notebook for all packages needed

### Data
Download data from [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data)

Unzip training and testing as two .csv file in data folder

### Code
All Python code reside in AIND_Capstone.ipynb, and it has four major sections
1. Libraries and utility functions
2. Face and eye detections using OpenCV's pre-trained model
3. Training a neural network to identify facial keypoints
4. Running detection functions on both images and camera video feed

### Reference
This is a consolidated version of my capstone project in Udacity AI Nanodegree program, and some utility functions are direct reuse of [this GitHub repo](https://github.com/udacity/AIND-CV-FacialKeypoints), and other parts of my code followed its templates and instructions. 
