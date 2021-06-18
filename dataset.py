from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


class KinematicsDataset(Dataset):
    def __init__(self, kinematics_folder_path, input_frames: int = 30, output_frames: int = 10, is_zero_mean: bool = False, is_zero_center: bool = False, is_overfit: bool = False, is_gap: bool = False):
        self.kinematics_folder_path = kinematics_folder_path
        self.kinematics_files = os.listdir(kinematics_folder_path)
        self.kinematics = None
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.features = 3
        self.feature_axis = [38, 39, 40]  # 57, 58, 59
        self.norm = True
        self.is_zero_mean = is_zero_mean
        self.is_zero_center = is_zero_center
        self.is_overfit = is_overfit
        self.is_gap = is_gap
        for f in self.kinematics_files:
            path = os.path.join(self.kinematics_folder_path, f)
            if self.kinematics is None:
                self.kinematics = np.genfromtxt(path)
            else:
                self.kinematics = np.concatenate((self.kinematics, np.genfromtxt(path)), axis=0)
        if self.norm:
            self.kinematics[:, self.feature_axis] = self.kinematics[:, self.feature_axis] / np.linalg.norm(self.kinematics[:, self.feature_axis]) * 1000

    def __len__(self):
        if self.is_overfit:
            return 1
        else:
            if self.is_gap:
                return int(len(self.kinematics) / self.input_frames) - 2
            else:
                return len(self.kinematics) - 2*self.input_frames

    def __getitem__(self, idx):
        # choose gap mode if required
        if self.is_gap:
            x = idx * self.input_frames
        else:
            x = idx

        # extract 200 frames from PSM1 and PSM2
        psm1_pos = np.transpose(self.kinematics[x:x + self.input_frames + self.output_frames, 38:41].copy()) # 38:41
        psm2_pos = np.transpose(self.kinematics[x:x + self.input_frames + self.output_frames, 57:60].copy()) # 57:60

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

        return psm1_pos, psm2_pos


if __name__ == "__main__":
    train_dataset = KinematicsDataset("./data/Needle_Passing/overfit", input_frames=30, output_frames=10, is_zero_center=True, is_gap=False)
    loader_train = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)

    for batch_id, data in enumerate(loader_train):
        psm1_pos, psm2_pos = data
        print(batch_id)
        # print(psm1_pos.shape)
        # print(psm2_pos.shape)
        # print(nframes.numpy()[0])