import numpy as np
import torch
import os
import os.path as osp
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import cv2
from .util import getVideoFrames, readKinematics
import torchvision.transforms as T
import random

total_states = 5
state_id = {'s':0, 'f':1, 'a':2, 'm':3, 'n':4}
total_gesture = 15


class JIGSAWSegmentsDataset(data.Dataset):
    def __init__(self, folder_paths, tasks, transforms=None, kinematicsObj=['sl_tool_xyz', 'sr_tool_xyz']):
        assert len(folder_paths) == len(tasks)
        self.folder_paths = folder_paths
        self.tasks = tasks
        self.transforms = transforms
        self.kinematicsObj = kinematicsObj
        self.allvideoPath = []
        self.allAnnotations = []
        self.allKinematicPath = []
        self.allvideoTask = []
        for i in range(len(folder_paths)):
            path = osp.join(folder_paths[i], 'videoSegments')
            allTrials = os.listdir(path)
            for trial in allTrials:
                if '.' == trial[0]:
                    continue
                videoPath = osp.join(path, trial, 'videos')
                kinematicPath = osp.join(path, trial, 'kinematics')
                videos = os.listdir(videoPath)
                kinematics = []
                for video in videos:
                    if '.avi' == video[-4:] and '.' != video[0]:
                        name = video[:-4]
                        self.allvideoTask.append(tasks[i])
                        self.allvideoPath.append(osp.join(videoPath, video))
                        self.allKinematicPath.append(osp.join(kinematicPath, name+'.txt'))
                        anno = name.split('_')
                        annotation = {}
                        annotation['gesture_id'] = int(anno[1][1:]) - 1
                        annotation['state'] = anno[3]
                        self.allAnnotations.append(annotation)

    def __len__(self):
        return len(self.allvideoPath)

    def __getitem__(self, idx: int):
        frames = getVideoFrames(self.allvideoPath[idx])[0]
        rawKinematics = readKinematics(self.allKinematicPath[idx])
        annotation = self.allAnnotations[idx]

        #an one-hot encoding for gesture
        gesture = torch.zeros(len(frames)).to(torch.int)
        gesture[:] = int(annotation['gesture_id'])
        #an one-hot encoding for state
        state = torch.zeros(len(frames), total_states).to(torch.int)
        state[:,:] = int(state_id[annotation['state']])

        #an N x 3 x W x H tensor for video frames
        frames = [T.ToTensor()(f) for f in frames]
        segments = torch.stack(frames)

        #an N x m tensor for kinematics
        kinematics = []
        for r in rawKinematics:
            outKinematics = torch.cat([torch.tensor(r[o]) for o in self.kinematicsObj])
            kinematics.append(outKinematics)
        kinematics = torch.stack(kinematics)
        return segments, kinematics, gesture, state

class JIGSAWSegmentsDataloader(data.Dataset):
    def __init__(self, batch_size, input_size, output_size, dataset, scale=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.minimum_size = output_size+batch_size+input_size
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        segments, kinematics, gesture, state = self.dataset[idx]
        #not enough size for sample, pad it with still image/kinematics
        if segments.size(0) <= self.minimum_size:
            return None
            new_segments = segments.new_zeros(self.minimum_size, *(segments.shape[1:]))
            new_segments[:segments.size(0)] = segments
            new_segments[segments.size(0):] = segments[-1]
            segments = new_segments
            new_kinematics = kinematics.new_zeros(self.minimum_size, *(kinematics.shape[1:]))
            new_kinematics[:kinematics.size(0)] = kinematics
            new_kinematics[kinematics.size(0):] = kinematics[-1]
            kinematics = new_kinematics
            new_gesture = gesture.new_zeros(self.minimum_size, *(gesture.shape[1:]))
            new_gesture[:gesture.size(0)] = gesture
            new_gesture[gesture.size(0):] = gesture[-1]
            gesture = new_gesture
            new_state = state.new_zeros(self.minimum_size, *(state.shape[1:]))
            new_state[:state.size(0)] = state
            new_state[state.size(0):] = state[-1]
            state = new_state

        kinematics = kinematics * self.scale

        #randomly sample batch_size samples from current series
        output_size = self.output_size
        input_size = self.input_size
        sample_thresh = list(range(len(kinematics) - output_size - input_size))
        start_idns = random.sample(sample_thresh, self.batch_size)
        #start_idns = sample_thresh[:self.batch_size]
        batched_src_segments = []
        batched_tgt_segments = []
        batched_src_kinematics = []
        batched_tgt_kinematics = []
        batched_src_gesture = []
        batched_tgt_gesture = []
        batched_src_state = []
        batched_tgt_state = []

        for i in start_idns:
            batched_src_segments.append(segments[i:i+input_size])
            batched_tgt_segments.append(segments[i+input_size-1:i+input_size+output_size])
            batched_src_kinematics.append(kinematics[i:i+input_size])
            batched_tgt_kinematics.append(kinematics[i+input_size-1:i+input_size+output_size])        
            batched_src_gesture.append(gesture[i:i+input_size])
            batched_tgt_gesture.append(gesture[i+input_size-1:i+input_size+output_size])      
            batched_src_state.append(state[i:i+input_size])
            batched_tgt_state.append(state[i+input_size-1:i+input_size+output_size])
        batched_src_segments = torch.stack(batched_src_segments)
        batched_tgt_segments = torch.stack(batched_tgt_segments)
        batched_src_kinematics = torch.stack(batched_src_kinematics)
        batched_tgt_kinematics = torch.stack(batched_tgt_kinematics)
        batched_src_gesture = torch.stack(batched_src_gesture)
        batched_tgt_gesture = torch.stack(batched_tgt_gesture)
        batched_src_state = torch.stack(batched_src_state)
        batched_tgt_state = torch.stack(batched_tgt_state)
              
        return JIGSAWBatch(batched_src_segments, batched_src_kinematics, batched_src_gesture, batched_src_state,
                    batched_tgt_segments, batched_tgt_kinematics, batched_tgt_gesture, batched_tgt_state)

class JIGSAWBatch:

    "Object for holding a batch of data with mask during training."
    def __init__(self, batched_src_segments, batched_src_kinematics, batched_src_gesture, batched_src_state,
                    batched_tgt_segments=None, batched_tgt_kinematics=None, batched_tgt_gesture=None, batched_tgt_state=None):
        self.batched_src_segments = batched_src_segments
        self.batched_src_kinematics = batched_src_kinematics
        self.batched_src_gesture = batched_src_gesture
        self.batched_src_state = batched_src_state
        self.src_mask = self.batched_src_segments.new_ones(self.batched_src_segments.shape[0], self.batched_src_segments.shape[1]).unsqueeze(-2).to(torch.bool)
        if batched_tgt_kinematics is not None:
            self.batched_tgt_segments = batched_tgt_segments[:, :-1]
            self.batched_tgt_segments_y = batched_tgt_segments[:, 1:]
            self.batched_tgt_kinematics = batched_tgt_kinematics[:, :-1]
            self.batched_tgt_kinematics_y = batched_tgt_kinematics[:, 1:]
            self.batched_tgt_gesture = batched_tgt_gesture[:, :-1]
            self.batched_tgt_gesture_y = batched_tgt_gesture[:, 1:]
            self.batched_tgt_state = batched_tgt_state[:, :-1]
            self.batched_tgt_state_y = batched_tgt_state[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.batched_tgt_segments)
            self.ntokens = (self.batched_tgt_segments_y.shape[0] * self.batched_tgt_segments_y.shape[1])

    @staticmethod
    def make_std_mask(tgt):
        "Create a mask to hide padding and future words."
        tgt_mask = tgt.new_ones(tgt.shape[0], tgt.shape[1]).unsqueeze(-2).to(torch.bool)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).type_as(tgt_mask.data)
        return tgt_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequentMask = np.triu(np.ones(attn_shape), k=1).astype(np.bool_)
    return torch.from_numpy(subsequentMask) == 0


if __name__ == '__main__':
    dataset = JIGSAWSegmentsDataset(['/home/hding15/cis2/data/Knot_Tying'],['Knot_Tying'])
    video, kinematics, gesture, state = dataset[0]
    print("video.shape", video.shape)
    print("kinematics", kinematics.shape)
    print("gesture", gesture.shape)
    print("state", state.shape)
    dataloader = JIGSAWSegmentsDataloader(10, 30, 10, dataset)
    batch = dataloader[0]