from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from skimage.io import imread
from skimage.transform import rescale


class KinematicsDataset(Dataset):
    def __init__(self, kinematics_folder_path, video_capture_path, input_frames: int = 30, output_frames: int = 10,
                 is_zero_mean: bool = False, is_zero_center: bool = False, is_overfit_extreme: bool = False,
                 is_gap: bool = False, is_decoder_only: bool = False, interval_size: int = 10, scale: int = 1000,
                 norm: bool = True):
        self.kinematics_folder_path = kinematics_folder_path
        self.video_capture_path = video_capture_path
        self.kinematics_files = sorted(os.listdir(kinematics_folder_path))
        self.video_capture_files = sorted(os.listdir(video_capture_path))
        self.kinematics = None
        self.kinematics_len = list()
        self.current_file_idx = 0
        self.length_limit = 0
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.batch_frames = input_frames + output_frames
        self.interval_size = interval_size
        self.features = 3
        self.feature_axis = [38, 39, 40]  # 57, 58, 59
        self.norm = norm
        self.scale = scale
        self.is_zero_mean = is_zero_mean
        self.is_zero_center = is_zero_center
        self.is_overfit_extreme = is_overfit_extreme
        self.is_gap = is_gap
        self.is_decoder_only = is_decoder_only
        # read kinematics
        for f in self.kinematics_files:
            path = os.path.join(self.kinematics_folder_path, f)
            length = (np.genfromtxt(path).shape[0] // self.batch_frames) * self.batch_frames
            self.kinematics_len.append(length)
            self.kinematics = np.genfromtxt(path)[:length] \
                if self.kinematics is None else np.concatenate((self.kinematics, np.genfromtxt(path)[:length]), axis=0)
        # normalize feature axes
        if self.norm:
            start_seq = 0
            for i in range(len(self.kinematics_len)):
                self.kinematics[start_seq:start_seq + self.kinematics_len[i], self.feature_axis] = self.kinematics[start_seq:start_seq + self.kinematics_len[i], \
                    self.feature_axis] / np.linalg.norm(self.kinematics[start_seq:start_seq + self.kinematics_len[i], self.feature_axis]) * self.scale
                start_seq = start_seq + self.kinematics_len[i]
        else:
            self.kinematics = self.kinematics * self.scale

    def __len__(self):
        if self.is_overfit_extreme:
            return 1
        else:
            if self.is_gap:
                return np.sum(self.kinematics_len) // self.batch_frames
            else:
                return (np.sum(self.kinematics_len) - len(self.kinematics_len) * self.batch_frames) // self.interval_size

    def __getitem__(self, idx):
        # initialize length limit
        if idx == 0:
            self.length_limit = self.kinematics_len[0]
            self.current_file_idx = 0
        # choose gap mode if required
        if self.is_gap:
            x = idx * self.batch_frames
            if x >= self.length_limit:
                self.current_file_idx += 1
                self.length_limit += self.kinematics_len[self.current_file_idx]
        else:
            fstart = idx * self.interval_size + self.current_file_idx * self.batch_frames
            if fstart + self.batch_frames >= self.length_limit:
                fstart = fstart + self.batch_frames
                self.current_file_idx += 1
                self.length_limit += self.kinematics_len[self.current_file_idx]
            x = fstart

        # extract 200 frames from PSM1 and PSM2
        psm1_pos = np.transpose(self.kinematics[x:x + self.batch_frames, 38:41].copy()) # 38:41
        psm2_pos = np.transpose(self.kinematics[x:x + self.batch_frames, 57:60].copy()) # 57:60

        # preprocessed zero-center data if desired
        if self.is_zero_center:
            for channel in range(psm1_pos.shape[0]):
                psm1_pos[channel] = psm1_pos[channel] - psm1_pos[channel, 0]
                psm2_pos[channel] = psm2_pos[channel] - psm2_pos[channel, 0]

        # preprocessed zero-mean data if desired
        if self.is_zero_mean:
            for channel in range(psm1_pos.shape[0]):
                psm1_pos[channel] = (psm1_pos[channel] - np.mean(psm1_pos[channel])) / np.std(psm1_pos[channel])
                psm2_pos[channel] = (psm2_pos[channel] - np.mean(psm2_pos[channel])) / np.std(psm2_pos[channel])

        # convert to tensors
        psm1_pos, psm2_pos = torch.from_numpy(psm1_pos), torch.from_numpy(psm2_pos)

        # load starting frame of input and prediction
        if self.is_decoder_only is False:
            # print("x = {}".format(x))
            # print(self.length_limit)
            # print(self.kinematics_len[self.current_file_idx])
            # print(self.current_file_idx)
            frame_idx = x - self.length_limit + self.kinematics_len[self.current_file_idx]
            # print("frame idx = {}".format(frame_idx))
            frame_path = os.path.join(self.video_capture_path, self.video_capture_files[self.current_file_idx*2])
            # print(frame_path)
            frame_input = cv2.cvtColor(cv2.imread(os.path.join(frame_path, os.listdir(frame_path)[frame_idx])), cv2.COLOR_BGR2RGB)
            frame_pred = cv2.cvtColor(cv2.imread(os.path.join(frame_path, os.listdir(frame_path)[frame_idx + self.input_frames])), cv2.COLOR_BGR2RGB)
            if frame_input.shape[0] != 480 or frame_input.shape[1] != 640:
                frame_input = rescale(frame_input, (2, 2, 1), anti_aliasing=True)
            if frame_pred.shape[0] != 480 or frame_pred.shape[1] != 640:
                frame_pred = rescale(frame_pred, (2, 2, 1), anti_aliasing=True)
            frame_input, frame_pred = torch.from_numpy(frame_input), torch.from_numpy(frame_pred)
            frames = torch.cat((frame_input.permute(2, 0, 1).unsqueeze(0), frame_pred.permute(2, 0, 1).unsqueeze(0)), dim=0)
            return psm1_pos, psm2_pos, frames
        else:
            return psm1_pos, psm2_pos


if __name__ == "__main__":
    task = "Needle_Passing"
    data_path = "./jigsaw_dataset_colab"
    task_folder = os.path.join(os.path.join(data_path, task), "kinematics")
    video_capture_folder = os.path.join(os.path.join(data_path, task), "video_captures")
    train_data_path = os.path.join(task_folder, "train")
    train_video_capture_path = os.path.join(video_capture_folder, "train")

    train_dataset = KinematicsDataset(train_data_path, train_video_capture_path, input_frames=150, output_frames=30, is_zero_center=True, is_gap=True, is_decoder_only=False)
    loader_train = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=True)

    for i in range(2):
        print("epoch {} started...".format(i))
        for batch_id, data in enumerate(loader_train):
            psm1_pos, psm2_pos, frames = data
            # print(frames.shape)
            # print(batch_id)
            # print(psm1_pos.shape)
            # print(psm2_pos.shape)
            # print(nframes.numpy()[0])
        print("epoch {} ended...".format(i))
