import os
import torch
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataset import KinematicsDataset
from model import DecoderOnlyTransformer, EncoderDecoderTransformer
from torch.utils.data import DataLoader


def infer_decoder_only():
    # set evaluation mode
    model.eval()
    infer_loss = list()
    # define target sequence length
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos = data
        gt = psm1_pos.to(device).float()
        psm1_pos, psm2_pos  = psm1_pos.to(device).float()[:, :, :input_frames], psm2_pos.to(device).float()
        if is_mask_enable:
            for frame in range(output_frames):
                with torch.no_grad():
                    output = model(psm1_pos[:, :, :input_frames + output_frames - 1])  # fetch 0-29, feed in 0-28
                # extract last element from output as new prediction
                pred = output[:, :, -1].unsqueeze(2)
                # concatenate new prediction to input sequence
                psm1_pos = torch.cat((psm1_pos, pred), dim=2)
        else:
            output = model(psm1_pos[:, :, :input_frames])

        # compute prediction L1 Loss
        loss = loss_function(psm1_pos[:, :, input_frames:] if is_mask_enable else output, gt[:, :, input_frames:])
        infer_loss.append(loss.item())

        # plot inference result
        print("plot...")
        gt = gt.cpu().detach().data.numpy()
        output = output.cpu().detach().data.numpy()
        psm1_pos = psm1_pos.cpu().detach().data.numpy()
        x_ = np.arange(input_frames + output_frames)
        for j in range(psm1_pos.shape[0]):
            for i in range(psm1_pos.shape[1]):
                plt.figure()
                plt.scatter(x_[:input_frames], psm1_pos[j, i, :input_frames], label='input', c="g")
                plt.scatter(x_[input_frames:], gt[j, i, input_frames:], label='ground truth', c="r")
                plt.scatter(x_[input_frames:], psm1_pos[j, i, input_frames:] if is_mask_enable else output[j, i], label='prediction', c="b")
                plt.xlabel('frames')
                plt.ylabel('true value')
                plt.title('frame ' + str(j * 100) + ' to frame ' + str((j + 1) * 100))
                plt.legend()
                pic_name = pos_name[i] + " start frame " + str(j * 100) + " batch " + str(batch_id + 1)
                plt.savefig(os.path.join(inference_image_folder, pic_name), dpi=300, bbox_inches='tight')
                plt.show()
    print("Average loss is {}.".format(np.mean(infer_loss)))


def infer_encoder_decoder():
    # set evaluation mode
    model.eval()
    infer_loss = list()
    # define target sequence length
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos, frames = data
        psm1_pos, psm2_pos, frames, gt = psm1_pos.to(device).float()[:, :, :input_frames], psm2_pos.to(device).float(), frames.to(device).float(), psm1_pos.to(device).float()
        if is_mask_enable:
            for frame in range(output_frames):
                with torch.no_grad():
                    input = {"kinematics": psm1_pos, "captures": frames}
                    output = model(input)  # fetch 0-29, feed in 0-28
                # extract last element from output as new prediction
                pred = output[:, :, -1].unsqueeze(2)
                # concatenate new prediction to input sequence
                psm1_pos = torch.cat((psm1_pos, pred), dim=2)
        else:
            input = {"kinematics": psm1_pos, "captures": frames}
            output = model(input)

        # compute prediction L1 Loss
        loss = loss_function(psm1_pos[:, :, input_frames:] if is_mask_enable else output, gt[:, :, input_frames:])
        infer_loss.append(loss.item())

        # plot inference result
        print("plot...")
        gt = gt.cpu().detach().data.numpy()
        output = output.cpu().detach().data.numpy()
        psm1_pos = psm1_pos.cpu().detach().data.numpy()
        x_ = np.arange(input_frames + output_frames)
        for j in range(psm1_pos.shape[0]):
            for i in range(psm1_pos.shape[1]):
                plt.figure()
                plt.scatter(x_[:input_frames], psm1_pos[j, i, :input_frames], label='input', c="g")
                plt.scatter(x_[input_frames:], gt[j, i, input_frames:], label='ground truth', c="r")
                plt.scatter(x_[input_frames:], psm1_pos[j, i, input_frames:] if is_mask_enable else output[j, i], label='prediction', c="b")
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
    num_epochs = 1000
    num_eval_epoch = 10
    lr = 0.0001
    weight_decay = 0.0001
    clip_max_norm = 2
    clip_norm_type = 2
    clip_value = 1e06
    save_ckpt = True
    is_penalize_all = True
    is_mask_enable = False
    is_gradient_clip = False


    # model specific
    is_decoder_only = True
    is_encode = True
    load_checkpoint = False
    use_norm = False
    feat_dim = 512
    interm_dim = 2048
    nhead = 8
    num_decoder_layers = 6
    num_encoder_layer = 6
    input_channel = 3

    # dataset specific
    norm = False
    is_generalize = True
    is_extreme = False
    is_overfit = False
    is_zero_center = True
    is_gap = True
    drop_last = False
    input_frames = 150
    output_frames = 30
    interval_size = 10

    scope = "general" if is_generalize else "overfit"
    task = "Needle_Passing"
    data_path = "./jigsaw_dataset_colab"
    set_type = "train"
    task_folder = os.path.join(os.path.join(data_path, task), "kinematics")
    video_capture_folder = os.path.join(os.path.join(data_path, task), "video_captures")
    test_data_path = os.path.join(task_folder, set_type) if is_generalize else os.path.join(task_folder, "overfit_extreme" if is_extreme else "overfit")
    test_video_capture_path = os.path.join(video_capture_folder, set_type if is_generalize else "overfit_extreme" if is_extreme else "overfit") if is_decoder_only is False else None

    # encoder specific
    is_feature_extract = False
    num_conv_layers = 3
    output_channel = 256
    conv_kernel_size = 3
    conv_stride = 1
    pool_kernel_size = 2
    pool_stride = 2
    padding = 1
    img_height = 480
    img_width = 640
    patch_height = 16
    patch_width = 16
    in_dim = 3
    batch_size = 32 if is_generalize else 1 if is_extreme else 2
    capture_size = 2
    dropout = 0.1

    # plot specific
    # suffix = "DecoderOnly-" + task + "-zerocenter-nonorm-penalall-in" + str(input_frames) + "out" + str(
    # output_frames) + "-" + scope + "-numdecode" + str(num_attn_layers) + "-classifier" if is_mask_enable else "-"
    suffix = "decoder-gap-nomask-noprenorm"
    pos_name = ["PSM1 tool tip position x", "PSM1 tool tip position y", "PSM1 tool tip position z"]
    time = "15_35"

    # create dataset
    test_dataset = KinematicsDataset(test_data_path, test_video_capture_path, input_frames=input_frames,
                                     output_frames=output_frames, is_zero_center=is_zero_center, is_overfit_extreme=is_extreme,
                                     is_gap=is_gap, is_decoder_only=is_decoder_only, interval_size=interval_size, norm=norm)

    # create dataloaders
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyTransformer(feat_dim=feat_dim, nhead=nhead, num_decoder_layers=num_decoder_layers,
                                   channel=input_channel, device=device, use_norm=use_norm,
                                   is_mask_enable=is_mask_enable, input_frames=input_frames,
                                   output_frames=output_frames, is_decoder_only=is_decoder_only) if is_decoder_only else \
        EncoderDecoderTransformer(feat_dim=feat_dim, nhead=nhead, num_decoder_layers=num_decoder_layers,
                                  channel=input_channel, device=device, use_norm=use_norm,
                                  is_mask_enable=is_mask_enable, input_frames=input_frames,
                                  output_frames=output_frames, is_feature_extract=is_feature_extract,
                                  num_conv_layers=num_conv_layers, input_channel=input_channel,
                                  output_channel=output_channel, conv_kernel_size=conv_kernel_size,
                                  conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                                  pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width,
                                  patch_height=patch_height, patch_width=patch_width, in_dim=in_dim,
                                  batch_size=batch_size, capture_size=capture_size, dropout=dropout,
                                  interm_dim=interm_dim, num_encoder_layer=num_encoder_layer, is_encode=is_encode)
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
    infer_encoder_decoder() if is_decoder_only is False else infer_decoder_only()
