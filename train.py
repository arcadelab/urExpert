import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import JIGSAWSegmentsDataset, JIGSAWSegmentsDataloader
from model import EncoderDecoder
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()
    return args

def parse_configs(configs):
    pass

def visulize_results(dataloader, data_id, loss_function, model, use_gpu=False, model_path=None, use_task=False):
    if model_path is not None:
        model = torch.load(model_path)

    data = dataloader[data_id]
    src = data.batched_src_kinematics
    tgt = data.batched_tgt_kinematics
    tgt_y = data.batched_tgt_kinematics_y
    src_mask = data.src_mask
    tgt_mask = data.trg_mask
    loss_plot = []
    running_loss = 0
    running_total = 0
    start = time.time()
    if use_gpu:
        model = model.cuda()
        src = src.cuda()
        tgt = tgt.cuda()
        tgt_y = tgt_y.cuda()
        src_mask = src_mask.cuda()
        tgt_mask = tgt_mask.cuda()
    if use_task:
        src_task = data.batched_src_gesture
        if use_gpu:
            src_task = src_task.cuda()
        tgt_hat = model(src, tgt, src_mask, tgt_mask, src_task)
    else:
        tgt_hat = model(src, tgt, src_mask, tgt_mask)
    loss = loss_function(tgt_hat, tgt_y)
    print(loss.item())
    for j in range(tgt.size(0)):
        plt.figure(figsize=(4*tgt.size(2),5))
        for i in range(tgt.size(2)):
            plt.subplot(1, tgt.size(2), i+1)
            gt_inds = list(range(src.size(1)+tgt.size(1)))
            gt_data = torch.cat([src, tgt_y], dim=1)[j, :, i].detach().cpu().numpy()
            pred_inds = list(range(src.size(1), src.size(1)+tgt.size(1)))
            pred_data = tgt_hat[j, :, i].detach().cpu().numpy()
            plt.plot(gt_inds, gt_data,  color='green')
            plt.plot(pred_inds, pred_data,color='red')
        plt.show()

def train_epochs(dataloader, split,model, loss_function, optimizer, n_epochs=100, use_gpu=False, use_task=False):

    "Standard Training and Logging Function"
    running_loss = 0
    running_total = 0
    running_loss_plot = []
    if use_gpu:
        model = model.cuda()
    for e in range(n_epochs):
        valid = 0
        running_loss = 0
        running_total = 0
        start = time.time()
        for i in range(len(split)):
            data = dataloader[split[i]]
            if data is None:
                continue
            valid += 1
            src = data.batched_src_kinematics
            tgt = data.batched_tgt_kinematics
            tgt_y = data.batched_tgt_kinematics_y
            src_mask = data.src_mask
            tgt_mask = data.trg_mask
            if use_gpu:
                src = src.cuda()
                tgt = tgt.cuda()
                tgt_y = tgt_y.cuda()
                src_mask = src_mask.cuda()
                tgt_mask = tgt_mask.cuda()
            if use_task:
                src_task = data.batched_src_gesture
                if use_gpu:
                    src_task = src_task.cuda()
                pred = model(src, tgt, src_mask, tgt_mask, src_task)
            else:
                pred = model(src, tgt, src_mask, tgt_mask)
            optimizer.zero_grad()
            loss = loss_function(pred, tgt_y)
            loss.backward()
            optimizer.step()
            running_total += 1
            running_loss += loss.item()
            elapsed = time.time() - start
            if i % 50 == 49:
                running_loss_plot.append(running_loss / running_total)
                print("Epoch Step: %d Loss: %f iteration per Sec: %f" %
                        (i, running_loss / running_total, running_total / elapsed))
        print("Epoch_number : %d Loss: %f iteration per Sec: %f, valid data ration: %f" %
                        (e, running_loss / running_total, running_total / elapsed, valid/len(split)))
        if e % 10 == 9:
            torch.save(model, "checkpoint/model"+str(e)+".pth")
        for p in optimizer.param_groups:
            p['lr'] *= 0.95
    return running_loss_plot


def l2_norm(preds, targets):
    loss = ((preds - targets) ** 2).mean()
    return loss

def l1_norm(preds, targets):
    loss = torch.abs(preds - targets).mean()
    return loss

loss_choice = {'l1_norm':l1_norm, 'l2_norm':l2_norm}

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
    args = parse_args()
    if args.configs is not None:
        configs = parse_configs(args.configs)
        dataset_path = configs.dataset_path
        dataset_tasks = configs.dataset_tasks
        batch_size = configs.batch_size
        input_length = configs.input_length
        output_length = configs.output_length
        scale = configs.scale
        src_vocab = configs.src_vocab
        tgt_vocab = configs.tgt_vocab
        num_layers = configs.num_layers
        feature_dim = configs.feature_dim
        hidden_layer = configs.hidden_layer
        num_heads = configs.num_heads
        dropout = configs.dropout
        loss_function = loss_choice[configs.loss_function]
        learning_rate = configs.learning_rate
        betas = configs.betas
        eps = configs.eps
        num_epochs = configs.num_epochs
        use_task = configs.use_task
        task_dim = configs.task_dim
        train_split = configs.train_split
    else:
        dataset_path = ['/home/hding15/cis2/data/Knot_Tying']
        dataset_tasks = ['Knot_Tying']
        batch_size = 10
        input_length = 30
        output_length = 10
        scale = 100
        src_vocab = 6
        tgt_vocab = 6
        num_layers = 20
        feature_dim = 512
        hidden_layer = 2048
        num_heads = 8
        dropout = 0.1
        loss_function = loss_choice['l1_norm']
        learning_rate = 0.01
        betas = (0.9, 0.98)
        eps = 1e-9
        num_epochs = 150
        use_task = False
        task_dim = 15
        train_split = [i for i in range(400)]
        # train_split = [0]
        # num_epochs = 15000


    dataset = JIGSAWSegmentsDataset(dataset_path,dataset_tasks)
    dataloader = JIGSAWSegmentsDataloader(batch_size, input_length, output_length, dataset, scale=scale)
    model = EncoderDecoder(src_vocab, tgt_vocab, N=num_layers, input_size=feature_dim, hidden_layer=hidden_layer, h=num_heads, dropout=dropout, task_dim=task_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps)
    running_loss_plot = train_epochs(dataloader, train_split, model, loss_function, optimizer, n_epochs=num_epochs, use_gpu=use_gpu, use_task=use_task)
    visulize_results(dataloader, 0, loss_function, model, use_gpu=use_gpu, use_task=use_task,model_path='/home/hding15/cis2/urExpert/checkpoint/model149.pth')