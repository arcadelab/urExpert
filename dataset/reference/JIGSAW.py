import numpy as np
import torch
import os
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate
from .util import 

class gestureBlobDataset:
    def __init__(self, blobs_folder_path: str) -> None:
        self.blobs_folder_path = blobs_folder_path
        self.blobs_folder = os.listdir(self.blobs_folder_path)
        self.blobs_folder = list(filter(lambda x: '.DS_Store' not in x, self.blobs_folder))
        self.blobs_folder.sort(key = lambda x: int(x.split('_')[1]))

    def __len__(self) -> int:
        return(len(self.blobs_folder))

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_file_path = self.blobs_folder[idx]
        curr_file_path = os.path.join(self.blobs_folder_path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, 'rb'))
        # print(curr_tensor_tuple[0].size())
        if curr_tensor_tuple[0].size()[0] == 50:
            return(curr_tensor_tuple)
        else:
            return(None)

class gestureBlobBatchDataset:
    def __init__(self, gesture_dataset: gestureBlobDataset, random_tensor: str = 'random') -> None:
        self.gesture_dataset = gesture_dataset
        self.random_tensor = random_tensor
    
    def __len__(self) -> None:
        return(len(self.gesture_dataset))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_tensor = self.gesture_dataset.__getitem__(idx)
        
        if self.random_tensor == 'random':
            rand_idx = np.random.randint(low = 0, high = len(self.gesture_dataset))
        
        elif self.random_tensor == 'next':
            if idx != len(self.gesture_dataset) - 1:
                rand_idx = idx + 1
            else:
                rand_idx = idx - 1
        else:
            raise ValueError('Value of random_tensor should be "random" or "next".')
        
        random_tensor = self.gesture_dataset.__getitem__(rand_idx)

        y_match = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)
        if idx != rand_idx:
            y_rand = torch.tensor([0, 1], dtype = torch.float32).view(1, 2)
        else:
            y_rand = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)

        return((curr_tensor, random_tensor, y_match, y_rand))

class gestureBlobMultiDataset:
    def __init__(self, blobs_folder_paths_list: List[str]) -> None:
        self.blobs_folder_paths_list = blobs_folder_paths_list
        self.blobs_folder_dict = {path: [] for path in self.blobs_folder_paths_list}
        for path in self.blobs_folder_paths_list:
            self.blobs_folder_dict[path] = os.listdir(path)
            self.blobs_folder_dict[path] = list(filter(lambda x: '.DS_Store' not in x, self.blobs_folder_dict[path]))
            self.blobs_folder_dict[path].sort(key = lambda x: int(x.split('_')[1]))
        
        self.dir_lengths = [len(os.listdir(path)) for path in self.blobs_folder_paths_list]
        for i in range(1, len(self.dir_lengths)):
            self.dir_lengths[i] += self.dir_lengths[i - 1]

    def __len__(self) -> int:
        return(self.dir_lengths[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        dir_idx = 0
        while idx >= self.dir_lengths[dir_idx]:
            dir_idx += 1
        adjusted_idx = idx - self.dir_lengths[dir_idx]
        path = self.blobs_folder_paths_list[dir_idx]

        curr_file_path = self.blobs_folder_dict[path][adjusted_idx]
        curr_file_path = os.path.join(path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, 'rb'))
        # print(curr_tensor_tuple[0].size())
        if curr_tensor_tuple[0].size()[0] == 50:
            return(curr_tensor_tuple)
        else:
            return(None)

def size_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return(default_collate(batch))

def main():
    optical_flow_folder_path = '../jigsaw_dataset/Knot_Tying/optical_flow/'
    transcriptions_folder_path = '../jigsaw_dataset/Knot_Tying/transcriptions'
    num_frames_per_blob = 25
    blobs_save_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    spacing = 2
    kinematics_folder_path = '../jigsaw_dataset/Knot_Tying/kinematics/AllGestures/'

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs/', '../jigsaw_dataset/Needle_Passing/blobs/', '../jigsaw_dataset/Suturing/blobs/']
    # dataset = gestureBlobDataset(blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs/')
    dataset = gestureBlobMultiDataset(blobs_folder_paths_list = blobs_folder_paths_list)
    out = dataset.__getitem__(3)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()