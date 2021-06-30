import torch
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import KinematicsDataset
from model import DecoderOnlyTransformer, EncoderDecoderTransformer
from torch.utils.data import DataLoader
from datetime import datetime
import time


def visualize_decoder_only():
    # set evaluation mode
    model.eval()
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        with torch.no_grad():
            # forward pass
            if is_mask_enable:
                output = model(psm1_pos[:, :, :input_frames + output_frames - 1])  # fetch 0-29, feed in 0-28
            else:
                output = model(psm1_pos[:, :, :input_frames])

        # plot prediction vs gt
        # separate past from future
        output = output.cpu().detach().data.numpy()
        psm1_pos = psm1_pos.cpu().detach().data.numpy()
        psm1_cur_pos = psm1_pos[:, :, :input_frames]
        psm1_nxt_pos = psm1_pos[:, :, input_frames:]

        x = np.arange(input_frames + output_frames)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                plt.figure()
                plt.scatter(x[:input_frames], psm1_cur_pos[i, j], label='input', c="g")
                plt.scatter(x[input_frames:], psm1_nxt_pos[i, j], label='ground truth', c="r")
                plt.scatter(x[input_frames:], output[i, j], label='prediction', c="b")
                plt.title('frame ' + str(i * 100) + ' to frame ' + str((i + 1) * 100))
                plt.xlabel('frames')
                plt.ylabel('true value')
                plt.legend()
                # pic_name = pos_name[j] + " start frame " + str(i * 100) + " batch " + str(batch_id + 1)
                # plt.savefig(os.path.join(image_folder, pic_name), dpi=300, bbox_inches='tight')
                plt.show()


def visualize_encoder_decoder():
    # set evaluation mode
    model.eval()
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos, frames = data
        psm1_pos, psm2_pos, frames = psm1_pos.to(device).float(), psm2_pos.to(device).float(), frames.to(device).float()
        with torch.no_grad():
            # forward pass, compute, backpropagation, and record loss
            if is_mask_enable:
                input = {"kinematics": psm1_pos[:, :, :input_frames + output_frames - 1], "captures": frames}
                # fetch 0-29, feed in 0-28
                output = model(input)
            else:
                input = {"kinematics": psm1_pos[:, :, :input_frames], "captures": frames}
                output = model(input)

        # plot prediction vs gt
        # separate past from future
        output = output.cpu().detach().data.numpy()
        psm1_pos = psm1_pos.cpu().detach().data.numpy()
        psm1_cur_pos = psm1_pos[:, :, :input_frames]
        psm1_nxt_pos = psm1_pos[:, :, input_frames:]

        x = np.arange(input_frames + output_frames)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                plt.figure()
                plt.scatter(x[:input_frames], psm1_cur_pos[i, j], label='input', c="g")
                plt.scatter(x[input_frames:], psm1_nxt_pos[i, j], label='ground truth', c="r")
                plt.scatter(x[input_frames:], output[i, j], label='prediction', c="b")
                plt.title('frame ' + str(i * 100) + ' to frame ' + str((i + 1) * 100))
                plt.xlabel('frames')
                plt.ylabel('true value')
                plt.legend()
                # pic_name = pos_name[j] + " start frame " + str(i * 100) + " batch " + str(batch_id + 1)
                # plt.savefig(os.path.join(image_folder, pic_name), dpi=300, bbox_inches='tight')
                plt.show()


def train_decoder_only():
    # set training mode
    model.train()
    running_loss = 0
    for batch_id, data in enumerate(loader_train):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        # set optimizer zero gradient
        optimizer.zero_grad()
        # forward pass, compute, backpropagation, and record loss
        if is_mask_enable:
            output = model(psm1_pos[:, :, :input_frames + output_frames - 1])  # fetch 0-29, feed in 0-28
            loss = loss_function(output, psm1_pos[:, :, 1:]) if is_penalize_all else loss_function(
                output[:, :, input_frames - 1:],
                psm1_pos[:, :, input_frames:])  # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
        else:
            output = model(psm1_pos[:, :, :input_frames])
            loss = loss_function(output, psm1_pos[:, :, input_frames:])
        loss.backward()
        running_loss += loss.item()
        # step optimizer
        optimizer.step()
    train_loss.append(running_loss)
    print("Train Loss = {} in epoch {}".format(running_loss, epoch))
    return running_loss


def train_encoder_decoder():
    # set training mode
    model.train()
    running_loss = 0
    for batch_id, data in enumerate(loader_train):
        # fetch data
        psm1_pos, psm2_pos, frames = data
        psm1_pos, psm2_pos, frames = psm1_pos.to(device).float(), psm2_pos.to(device).float(), frames.to(device).float()
        # set optimizer zero gradient
        optimizer.zero_grad()
        # forward pass, compute, backpropagation, and record loss
        if is_mask_enable:
            input = {"kinematics": psm1_pos[:, :, :input_frames + output_frames - 1], "captures": frames}
            # fetch 0-29, feed in 0-28
            output = model(input)
            # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
            loss = loss_function(output, psm1_pos[:, :, 1:]) if is_penalize_all else loss_function(
                output[:, :, input_frames - 1:], psm1_pos[:, :, input_frames:])
        else:
            input = {"kinematics": psm1_pos[:, :, :input_frames], "captures": frames}
            output = model(input)
            loss = loss_function(output, psm1_pos[:, :, input_frames:])
        loss.backward()
        running_loss += loss.item()
        # step optimizer
        optimizer.step()
    train_loss.append(running_loss)
    print("Train Loss = {} in epoch {}".format(running_loss, epoch))
    return running_loss


def eval_decoder_only():
    # set evaluation mode
    model.eval()
    running_loss = 0
    for batch_id, data in enumerate(loader_eval):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        with torch.no_grad():
            # forward pass, compute, backpropagation, and record loss
            if is_mask_enable:
                output = model(psm1_pos[:, :, :input_frames + output_frames - 1])  # fetch 0-29, feed in 0-28
                loss = loss_function(output, psm1_pos[:, :, 1:]) if is_penalize_all else loss_function(
                    output[:, :, input_frames - 1:],
                    psm1_pos[:, :, input_frames:])  # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
            else:
                output = model(psm1_pos[:, :, :input_frames])
                loss = loss_function(output, psm1_pos[:, :, input_frames:])
        running_loss += loss.item()
    scheduler.step(running_loss)
    eval_loss.append(running_loss)
    print("Eval Loss = {} in epoch {}".format(running_loss, epoch + 1))
    return running_loss


def eval_encoder_decoder():
    # set evaluation mode
    model.eval()
    running_loss = 0
    for batch_id, data in enumerate(loader_eval):
        # fetch data
        psm1_pos, psm2_pos, frames = data
        psm1_pos, psm2_pos, frames = psm1_pos.to(device).float(), psm2_pos.to(device).float(), frames.to(device).float()
        with torch.no_grad():
            # forward pass, compute, backpropagation, and record loss
            if is_mask_enable:
                input = {"kinematics": psm1_pos[:, :, :input_frames + output_frames - 1], "captures": frames}
                # fetch 0-29, feed in 0-28
                output = model(input)
                # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
                loss = loss_function(output, psm1_pos[:, :, 1:]) if is_penalize_all else loss_function(
                    output[:, :, input_frames - 1:], psm1_pos[:, :, input_frames:])
            else:
                input = {"kinematics": psm1_pos[:, :, :input_frames], "captures": frames}
                output = model(input)
                loss = loss_function(output, psm1_pos[:, :, input_frames:])
        running_loss += loss.item()
    scheduler.step(running_loss)
    eval_loss.append(running_loss)
    print("Eval Loss = {} in epoch {}".format(running_loss, epoch + 1))
    return running_loss


def save_model():
    file_name = time + '.ckpt'
    file_name = os.path.join(sub_folder, file_name)
    torch.save(model.state_dict(), file_name)


def plot_loss():
    # with open(os.path.join(sub_folder, 'loss.csv'), 'w') as f:
        # write = csv.writer(f)
        # write.writerow(train_loss)
    x = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x, train_loss, label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(sub_folder, "loss"))
    plt.show()


if __name__ == "__main__":
    # user parameters
    # training specific
    num_epochs = 5000
    num_eval_epoch = 100
    lr = 0.0001
    weight_decay = 0.0001
    is_penalize_all = True
    is_mask_enable = False
    save_ckpt = True
    # model specific
    is_decoder_only = False
    feat_dim = 1024
    nhead = 8
    num_attn_layers = 6
    input_channel = 3
    load_checkpoint = False
    use_norm = False
    # dataset specific
    is_generalize = True
    is_extreme = False
    is_overfit = False
    scope = "general" if is_generalize else "overfit"
    task = "Needle_Passing"
    task_folder = os.path.join("./data", task)
    video_capture_folder = os.path.join(os.path.join("E:\Research\Arcade\jigsaw_dataset", task), "video_captures")
    train_data_path = os.path.join(task_folder, "train") if is_generalize else os.path.join(task_folder,
                                                                                            "overfit_extreme" if is_extreme else "overfit")
    eval_data_path = os.path.join(task_folder, "eval") if is_generalize else os.path.join(task_folder,
                                                                                          "overfit_extreme" if is_extreme else "overfit")
    test_data_path = os.path.join(task_folder, "test") if is_generalize else os.path.join(task_folder,
                                                                                          "overfit_extreme" if is_extreme else "overfit")
    train_video_capture_path = os.path.join(video_capture_folder, "train" if is_generalize else "overfit_extreme" if is_extreme else "overfit") if is_decoder_only is False else None
    eval_video_capture_path = os.path.join(video_capture_folder, "eval" if is_generalize else "overfit_extreme" if is_extreme else "overfit") if is_decoder_only is False else None
    test_video_capture_path = os.path.join(video_capture_folder, "test" if is_generalize else "overfit_extreme" if is_extreme else "overfit") if is_decoder_only is False else None
    input_frames = 150
    output_frames = 30
    is_zero_center = True
    is_gap = True
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
    patch_height = 32
    patch_width = 32
    in_dim = 3
    batch_size = 16
    capture_size = 2
    dropout = 0.1
    # plot specific
    # suffix = "DecoderOnly-" + task + "-zerocenter-nonorm-penalall-in" + str(input_frames) + "out" + str(
        # output_frames) + "-" + scope + "-numdecode" + str(num_attn_layers) + "-classifier" if is_mask_enable else "-"
    suffix = "encoderdecoder-encodernorm"
    pos_name = ["PSM1 tool tip position x", "PSM1 tool tip position y", "PSM1 tool tip position z"]

    # create dataset
    train_dataset = KinematicsDataset(train_data_path, train_video_capture_path, input_frames=input_frames, output_frames=output_frames,
                                      is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap, is_decoder_only=is_decoder_only)
    eval_dataset = KinematicsDataset(eval_data_path, eval_video_capture_path, input_frames=input_frames, output_frames=output_frames,
                                     is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap, is_decoder_only=is_decoder_only)
    test_dataset = KinematicsDataset(test_data_path, test_video_capture_path, input_frames=input_frames, output_frames=output_frames,
                                     is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap, is_decoder_only=is_decoder_only)

    # create dataloaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    loader_eval = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyTransformer(feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                                   channel=input_channel, device=device, use_norm=use_norm,
                                   is_mask_enable=is_mask_enable, input_frames=input_frames,
                                   output_frames=output_frames) if is_decoder_only else \
        EncoderDecoderTransformer(feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                                  channel=input_channel, device=device, use_norm=use_norm,
                                  is_mask_enable=is_mask_enable, input_frames=input_frames,
                                  output_frames=output_frames, is_feature_extract=is_feature_extract,
                                  num_conv_layers=num_conv_layers, input_channel=input_channel,
                                  output_channel=output_channel, conv_kernel_size=conv_kernel_size,
                                  conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                                  pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width,
                                  patch_height=patch_height, patch_width=patch_width, in_dim=in_dim,
                                  batch_size=batch_size, capture_size=capture_size, dropout=dropout)
    model.cuda()
    # check numbers of parameters
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # initialize loss function, optimizer, scheduler
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True)

    # load checkpoints if required
    if load_checkpoint:
        file = "./model_checkpoints/2021-01-06/13_43.ckpt"
        ckpt = torch.load(file, map_location=None)
        model.load_state_dict(ckpt)
        print("loading checkpoint succeed!")

    # create checkpoint folder for saving plots and model ckpt
    if save_ckpt:
        now = datetime.now()
        now = (str(now).split('.')[0]).split(' ')
        date = now[0]
        time = now[1].split(':')[0] + '_' + now[1].split(':')[1] + suffix
        folder = os.path.join('./checkpoints', date)
        if not os.path.exists(folder):
            os.makedirs(folder)
        sub_folder = os.path.join(folder, time)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        train_image_folder = os.path.join(sub_folder, 'train_image_folder')
        if not os.path.exists(train_image_folder):
            os.makedirs(train_image_folder)

    # training starts
    train_loss = []
    eval_loss = []
    best_val_loss = 0
    for epoch in tqdm.tqdm(range(num_epochs)):
        if (epoch + 1) % num_eval_epoch != 0:
            print("Train Epoch {}".format(epoch + 1))
            loss_train = train_decoder_only() if is_decoder_only else train_encoder_decoder()
        else:
            print("Validation {}".format((epoch + 1) / num_eval_epoch))
            loss_eval = eval_decoder_only() if is_decoder_only else eval_encoder_decoder()
            if best_val_loss == 0:
                best_val_loss = loss_eval
            if best_val_loss > loss_eval:
                best_val_loss = loss_eval
                print("New validation best, save model...")
                save_model()
    plot_loss()
    visualize_decoder_only() if is_decoder_only else visualize_encoder_decoder()