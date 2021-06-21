import os
import torch
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataset import KinematicsDataset
from model import DecoderOnlyTransformer
from torch.utils.data import DataLoader


def infer():
    # set evaluation mode
    model.eval()
    infer_loss = list()
    # define target sequence length
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos = data
        gt = psm1_pos.to(device).float()
        psm1_pos, psm2_pos  = psm1_pos.to(device).float()[:, :, :input_frames], psm2_pos.to(device).float()
        for frame in range(output_frames):
            with torch.no_grad():
                output = model(psm1_pos[:, :, :input_frames+frame])
            # extract last element from output as new prediction
            pred = output[:, :, -1].unsqueeze(2)
            # concatenate new prediction to input sequence
            psm1_pos = torch.cat((psm1_pos, pred), dim=2)

        # compute prediction L1 Loss
        loss = loss_function(psm1_pos[:, :, input_frames:], gt[:, :, input_frames:])
        infer_loss.append(loss.item())

        # plot inference result
        print("plot...")
        gt = gt.cpu().detach().data.numpy()
        psm1_pos = psm1_pos.cpu().detach().data.numpy()
        x_ = np.arange(input_frames + output_frames)
        for j in range(psm1_pos.shape[0]):
            for i in range(psm1_pos.shape[1]):
                plt.figure()
                plt.scatter(x_[:input_frames], psm1_pos[j, i, :input_frames], label='input', c="g")
                plt.scatter(x_[input_frames:], psm1_pos[j, i, input_frames:], label='prediction', c="b")
                plt.scatter(x_[input_frames:], gt[j, i, input_frames:], label='ground truth', c="r")
                plt.xlabel('frames')
                plt.ylabel('true value')
                plt.title('frame ' + str(j * 100) + ' to frame ' + str((j + 1) * 100))
                plt.legend()
                pic_name = pos_name[i] + " start frame " + str(j * 100) + " batch " + str(batch_id + 1)
                plt.savefig(os.path.join(inference_image_folder, pic_name), dpi=300, bbox_inches='tight')
                # plt.show()
    print("Average loss is {}.".format(np.mean(infer_loss)))


if __name__ == "__main__":
    # user parameters
    # training specific
    num_epochs = 500
    num_eval_epoch = 50
    lr = 0.0001
    weight_decay = 0.01
    # dataset specific
    is_generalize = False
    task = "Needle_Passing"
    task_folder = os.path.join("./data", task)
    test_data_path = os.path.join(task_folder, "test") if is_generalize else os.path.join(task_folder, "overfit")
    input_frames = 60
    output_frames = 10
    is_zero_center = True
    is_overfit = False
    is_gap = True
    # model specific
    feat_dim = 512
    nhead = 8
    num_attn_layers = 6
    input_channel = 3
    load_checkpoint = False
    use_norm = False
    # plot specific
    suffix = "DecoderOnly-" + task + "-zerocenter-nonorm-nogap-penalall-in60out10"
    pos_name = ["PSM1 tool tip position x", "PSM1 tool tip position y", "PSM1 tool tip position z"]
    time = "18_05"

    # create dataset
    batch_size = 150
    test_dataset = KinematicsDataset(test_data_path, input_frames=input_frames,
                                     output_frames=output_frames, is_zero_center=is_zero_center, is_overfit=is_overfit,
                                     is_gap=is_gap)

    # create dataloaders
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyTransformer(feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                                   channel=input_channel, device=device, use_norm=use_norm)
    model.cuda()

    # load checkpoint
    filename = time + suffix
    date_folder = os.path.join('checkpoints', sorted(os.listdir('checkpoints'))[-1])
    subfolder = os.path.join(date_folder, filename)
    inference_image_folder = os.path.join(subfolder, 'inference_image_folder')
    if not os.path.exists(inference_image_folder):
        os.makedirs(inference_image_folder)
    ckpt = torch.load(os.path.join(subfolder, filename + ".ckpt"), map_location=None)
    model.load_state_dict(ckpt)

    # testing starts here
    loss_function = torch.nn.L1Loss()
    infer()
