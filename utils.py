import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch
import torch.nn as nn


def plot_predict(pos_name, batch_id, folder, output, cur, nxt, input_frames, output_frames, arm_id):
    image_folder = os.path.join(folder, arm_id)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    x = np.arange(input_frames + output_frames)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            plt.figure()
            plt.scatter(x[:input_frames], cur[i, j], label='input', c="g")
            plt.scatter(x[input_frames:], nxt[i, j], label='ground truth', c="r")
            plt.scatter(x[input_frames:], output[i, j], label='prediction', c="b")
            plt.title('frame ' + str(i * 100) + ' to frame ' + str((i + 1) * 100))
            plt.xlabel('frames')
            plt.ylabel('true value')
            plt.legend()
            pic_name = pos_name[j] + " start frame " + str(i * 100) + " batch " + str(batch_id + 1)
            plt.savefig(os.path.join(image_folder, pic_name), dpi=300, bbox_inches='tight')
            plt.show()


def save_model(time, sub_folder, model):
    file_name = time + '.ckpt'
    file_name = os.path.join(sub_folder, file_name)
    torch.save(model.state_dict(), file_name)


def plot_loss(sub_folder, loss, loss_type):
    with open(os.path.join(sub_folder, loss_type + 'loss.csv'), 'w') as f:
        write = csv.writer(f)
        write.writerow(loss)
    x = np.arange(len(loss))
    plt.figure()
    plt.plot(x, loss, label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(sub_folder, loss_type + " loss"))
    plt.show()


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)


def train(epoch, model, loader_train, optimizer, train_loss, device, loss_fn, input_frames, is_gradient_clip, clip_max_norm, clip_norm_type):
    # set training mode
    model.train()
    running_loss = 0
    for batch_id, data in enumerate(loader_train):
        # print(batch_id)
        # fetch data
        psm_pos, frames = data
        psm_pos, frames = psm_pos.to(device).float(), frames.to(device).float()
        # set optimizer zero gradient
        optimizer.zero_grad()
        # forward pass, compute, backpropagation, and record loss
        output = model(psm_pos[:, :, :input_frames], frames)
        loss = loss_fn(output, psm_pos[:, :, input_frames:])
        loss.backward()
        # Gradient Value Clipping if desired
        if is_gradient_clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm, norm_type=clip_norm_type)
        # step optimizer
        optimizer.step()
        running_loss += loss.item()
    train_loss.append(running_loss)
    print("Train Loss = {} in epoch {}".format(running_loss, epoch))
    return running_loss


def evaluate(epoch, model, loader_eval, scheduler, eval_loss, device, loss_fn, input_frames):
    # set evaluation mode
    model.eval()
    running_loss = 0
    for batch_id, data in enumerate(loader_eval):
        # fetch data
        psm_pos, frames = data
        psm_pos, frames = psm_pos.to(device).float(), frames.to(device).float()
        with torch.no_grad():
            # forward pass, compute, backpropagation, and record loss
            output = model(psm_pos[:, :, :input_frames], frames)
            loss = loss_fn(output, psm_pos[:, :, input_frames:])
        running_loss += loss.item()
    scheduler.step(running_loss)
    eval_loss.append(running_loss)
    print("Eval Loss = {} in epoch {}".format(running_loss, epoch + 1))
    return running_loss


def visualize(model, loader_test, device, input_frames, output_frames, folder, set_type):
    # set evaluation mode
    model.eval()

    pos1_name = ["PSM1 tool tip position x", "PSM1 tool tip position y", "PSM1 tool tip position z"]
    pos2_name = ["PSM2 tool tip position x", "PSM2 tool tip position y", "PSM2 tool tip position z"]

    inference_image_folder = os.path.join(folder, set_type + '_image_folder')
    if not os.path.exists(inference_image_folder):
        os.makedirs(inference_image_folder)

    for batch_id, data in enumerate(loader_test):
        # fetch data
        psm_pos, frames = data
        psm_pos, frames = psm_pos.to(device).float(), frames.to(device).float()
        with torch.no_grad():
            # forward pass
            output = model(psm_pos[:, :, :input_frames], frames)

        output = output.cpu().detach().data.numpy()
        psm_pos = psm_pos.cpu().detach().data.numpy()

        plot_predict(pos1_name, batch_id, inference_image_folder, output[:, :3], psm_pos[:, :3, :input_frames],
                     psm_pos[:, :3, input_frames:], input_frames, output_frames, "psm1")
        plot_predict(pos2_name, batch_id, inference_image_folder, output[:, 3:], psm_pos[:, 3:, :input_frames],
                     psm_pos[:, 3:, input_frames:], input_frames, output_frames, "psm2")