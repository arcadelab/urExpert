# Code for kinematics prediction on the segmented JIGSAW dataset.

# Introduction
  This is the code repo for a transformer kinematics predictor. The model takes kinematics of several previous frames as input (task, state, and video context info as optional info) and predicts following kinematics for several frames. 

# Component
  - ./datasets : contains the code to read, parse the segmented JIGSAW dataset, and how to generate the batched data for training. ( detailed introduction is in the readme of this folder)
  - ./model: contains the code of the formation of the model. ( detailed introduction is in the readme of this folder)
  - train.py: use this file to train a model on regular epoch training or one-sample overfitting.
