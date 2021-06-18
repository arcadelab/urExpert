import torch
import os
import tqdm
import matplotlib.pyplot as plt
import csv
import numpy as np
from dataset import KinematicsDataset
from model import DecoderOnlyTransformer
from torch.utils.data import DataLoader
from datetime import datetime
import time


def visualize():
    # set evaluation mode
    model.eval()
    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        with torch.no_grad():
            output = model(psm1_pos[:, :, :input_frames + output_frames - 1])  # fetch 0-29, feed in 0-28

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
                plt.scatter(x[input_frames:], output[i, j, input_frames-1:], label='prediction', c="b")
                plt.title('frame ' + str(i * 100) + ' to frame ' + str((i + 1) * 100))
                plt.xlabel('frames')
                plt.ylabel('true value')
                plt.legend()
                pic_name = pos_name[j] + " start frame " + str(i * 100) + " batch " + str(batch_id + 1)
                plt.savefig(os.path.join(image_folder, pic_name), dpi=300, bbox_inches='tight')
                plt.show()


def train():
    # set training mode
    model.train()
    count = 0
    running_loss = 0
    for batch_id, data in enumerate(loader_train):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        # set optimizer zero gradient
        optimizer.zero_grad()
        # forward pass
        output = model(psm1_pos[:, :, :input_frames + output_frames - 1]) # fetch 0-29, feed in 0-28
        # compute, backpropagation, and record loss
        loss = loss_function(output[:, :, input_frames - 1:], psm1_pos[:, :, input_frames:]) # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
        loss.backward()
        running_loss += loss.item()
        # step optimizer
        optimizer.step()
        count = count + 1
    train_loss.append(running_loss / count)
    print("Train Loss = {} in epoch {}".format(running_loss / count, epoch))
    return running_loss / count


def eval():
    # set evaluation mode
    model.eval()
    count = 0
    running_loss = 0
    for batch_id, data in enumerate(loader_eval):
        # fetch data
        psm1_pos, psm2_pos = data
        psm1_pos, psm2_pos = psm1_pos.to(device).float(), psm2_pos.to(device).float()
        with torch.no_grad():
            output = model(psm1_pos[:, :, :input_frames + output_frames - 1]) # fetch 0-29, feed in 0-28
        loss = loss_function(output[:, :, input_frames - 1:], psm1_pos[:, :, input_frames:]) # out 0-28 (1-29 in practice), take 20-29, compare with gt 20-29
        running_loss += loss.item()
        count += 1
    # scheduler.step(running_loss / count)
    eval_loss.append(running_loss / count)
    print("Eval Loss = {} in epoch {}".format(running_loss / count, epoch + 1))
    return running_loss / count


def save_model():
    file_name = time + '.ckpt'
    file_name = os.path.join(sub_folder, file_name)
    torch.save(model.state_dict(), file_name)


def plot_loss():
    with open(os.path.join(sub_folder, 'loss.csv'), 'w') as f:
        write = csv.writer(f)
        write.writerow(train_loss)
    x = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x, train_loss, label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    name = "loss"
    plt.savefig(os.path.join(sub_folder, name))


if __name__ == "__main__":
    # user parameters
    # training specific
    num_epochs = 500
    num_eval_epoch = 50
    lr = 0.0001
    weight_decay = 0.01
    # dataset specific
    input_frames = 60
    output_frames = 10
    is_zero_center = True
    is_overfit = False
    is_gap = False
    # model specific
    feat_dim = 512
    nhead = 8
    num_attn_layers = 6
    input_channel = 3
    load_checkpoint = False
    use_norm = False
    # plot specific
    suffix = 'DecoderOnly-zerocenter-nonorm-nogap-simpfinal-in60out10'
    pos_name = ["PSM1 tool tip position x", "PSM1 tool tip position y", "PSM1 tool tip position z"]

    # create dataset
    batch_size = 150
    train_dataset = KinematicsDataset("./data/Needle_Passing/overfit", input_frames=input_frames, output_frames=output_frames, is_zero_center=is_zero_center, is_overfit=is_overfit, is_gap=is_gap)
    eval_dataset = KinematicsDataset("./data/Needle_Passing/overfit", input_frames=input_frames, output_frames=output_frames, is_zero_center=is_zero_center, is_overfit=is_overfit, is_gap=is_gap)
    test_dataset = KinematicsDataset("./data/Needle_Passing/overfit", input_frames=input_frames, output_frames=output_frames, is_zero_center=is_zero_center, is_overfit=is_overfit, is_gap=is_gap)

    # create dataloaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader_eval = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyTransformer(feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers, channel=input_channel, device=device, use_norm=use_norm)
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
    image_folder = os.path.join(sub_folder, 'image_folder')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # training starts
    train_loss = []
    eval_loss = []
    best_val_loss = 0
    for epoch in tqdm.tqdm(range(num_epochs)):
        if (epoch + 1) % num_eval_epoch != 0:
            print("Train Epoch {}".format(epoch + 1))
            loss_train = train()
        else:
            print("Validation {}".format((epoch + 1) / num_eval_epoch))
            loss_eval = eval()
            if best_val_loss == 0:
                best_val_loss = loss_eval
            if best_val_loss > loss_eval:
                best_val_loss = loss_eval
                print("New validation best, save model...")
                save_model()
    plot_loss()
    visualize()