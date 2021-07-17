import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import cv2.cv2 as cv2
from skimage.transform import rescale


class KinematicsDataset(Dataset):
    def __init__(self, kinematics_folder_path, video_capture_path, input_frames, output_frames,
                 is_zero_mean, is_zero_center, is_overfit_extreme, is_gap, interval_size, scale,
                 norm, capture_size, resize_img_height, resize_img_width):
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
        self.capture_size = capture_size
        self.capture_interval = input_frames // capture_size
        self.resize_img_height = resize_img_height
        self.resize_img_width = resize_img_width
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
        psm1_pos = np.transpose(self.kinematics[x:x + self.batch_frames, 38:41].copy())  # 38:41
        psm2_pos = np.transpose(self.kinematics[x:x + self.batch_frames, 57:60].copy())  # 57:60

        # preprocessed zero-center data if desired
        if self.is_zero_center:
            for channel in range(psm1_pos.shape[0]):
                psm1_pos[channel] = psm1_pos[channel] - psm1_pos[channel, 0]
                psm2_pos[channel] = psm2_pos[channel] - psm2_pos[channel, 0]

        # load starting frame of input and prediction
        captured_frames = list()
        frame_idx = x - self.length_limit + self.kinematics_len[self.current_file_idx]
        frame_path = os.path.join(self.video_capture_path, self.video_capture_files[self.current_file_idx*2])
        for i in range(self.capture_size):
            frame = cv2.cvtColor(cv2.imread(os.path.join(frame_path, os.listdir(frame_path)[frame_idx + i * self.capture_interval])), cv2.COLOR_BGR2RGB)
            if frame.shape[0] != self.resize_img_height or frame.shape[1] != self.resize_img_width:
                frame = rescale(frame, (self.resize_img_height/frame.shape[0], self.resize_img_width/frame.shape[1], 1), anti_aliasing=True)
            captured_frames.append(frame)
        frames = np.stack(captured_frames, axis=0)

        # convert to tensors
        psm1_pos, psm2_pos = torch.from_numpy(psm1_pos), torch.from_numpy(psm2_pos)
        psm_pos = torch.cat((psm1_pos, psm2_pos), dim=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return psm_pos, frames


if __name__ == "__main__":
    task = "Needle_Passing"
    data_path = "./jigsaw_dataset_colab"
    task_folder = os.path.join(os.path.join(data_path, task), "kinematics")
    video_capture_folder = os.path.join(os.path.join(data_path, task), "video_captures")
    train_data_path = os.path.join(task_folder, "train")
    train_video_capture_path = os.path.join(video_capture_folder, "train")

    train_dataset = KinematicsDataset(train_data_path, train_video_capture_path, input_frames=150, output_frames=30, is_zero_center=True, is_gap=True, resize_img_height=120, resize_img_width=160,
                                      is_zero_mean=False, is_overfit_extreme=False, interval_size=30, scale=1000, norm=False, capture_size=2)
    loader_train = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=True)

    for i in range(2):
        print("epoch {} started...".format(i))
        for batch_id, data in enumerate(loader_train):
            psm_pos, frames = data
            print(psm_pos.shape)
            print(frames.shape)
        print("epoch {} ended...".format(i))
