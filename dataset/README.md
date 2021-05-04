# The code for read parse and load the paired JIGSAW dataset.

Use the provided dataset(https://www.icloud.com/iclouddrive/0ezkRNPVptNd0vXWlhpJ1GzrQ#Knot_Tying_Aligned)

The code in reference is the reference code from the Aniruddha's code repo.

All useful codes are in util.py and dataset.py.

- util.py it defines util functions for reading video frames and kinematics annotations.

- dataset.py defines two main class for read and generate data:
  - JIGSAWSegmentsDataset takes the path of the dataset as input and load the raw video segments, kinematics, task, and state as item.
  - JIGSAWSegmentsDataloader takes the JIGSAWSegmentsDataset as input and generate batched source/target videos segments, kinematics segments, 
  task segments and state segments as well as the mask for sourece and target as input and ground truth of the transformer model.

For detailed implementation, please refer to the comment of the code.
