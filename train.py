import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import JIGSAWSegmentsDataset, JIGSAWSegmentsDataloader
from model import EncoderDecoder
import matplotlib.pyplot as plt
import numpy as np
import time

def overfit_samples(dataloader, data_id, model, loss_function, optimizer, iteration_number=2000, use_gpu=False, use_task=False):
    loss_plot = []
    running_loss = 0
    running_total = 0
    start = time.time()
    for i in range(iteration_number):
        data = dataloader[data_id]
        src = data.batched_src_kinematics
        tgt = data.batched_tgt_kinematics
        tgt_y = data.batched_tgt_kinematics_y
        src_mask = data.src_mask
        tgt_mask = data.trg_mask
        if use_gpu:
            model = model.cuda()
            src = src.cuda()
            tgt = tgt.cuda()
            tgt_y = tgt_y.cuda()
            src_mask = src_mask.cuda()
            tgt_mask = tgt_mask.cuda()
        optimizer.zero_grad()
        if use_task:
            src_task = data.batched_src_gesture
            if use_gpu:
                src_task = src_task.cuda()
            pred = model(src, tgt, src_mask, tgt_mask, src_task)
        else:
            pred = model(src, tgt, src_mask, tgt_mask)
        loss = loss_function(pred, tgt_y)
        loss.backward()
        optimizer.step()
        running_total+=1
        running_loss+=loss.item()
        if i % 100 == 99:
            loss_plot.append(running_loss / running_total)
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f iteration per Sec: %f" %
                    (i, running_loss / running_total, running_total / elapsed))
        if i % 3000 == 2999:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
    torch.save(model, "checkpoint/model_overfir"+str(data_id)+".pth")
    return loss_plot

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
    start = time.time()
    if use_gpu:
        model = model.cuda()
    for e in range(n_epochs):
        for i in range(len(split)):
            data = dataloader[split[i]]
            if data is None:
                continue
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
            if i % 50 == 49:
                running_loss_plot.append(running_loss / running_total)
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f iteration per Sec: %f" %
                        (i, running_loss / running_total, running_total / elapsed))
        print("Epoch_number : %d Loss: %f iteration per Sec: %f" %
                        (e, running_loss / running_total, running_total / elapsed))
        if e % 10 == 9:
            torch.save(model, "checkpoint/model"+str(e)+".pth")
        for p in optimizer.param_groups:
            p['lr'] *= 0.98
    return running_loss_plot


class NoamOpt:

    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):

        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):

        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):

    return NoamOpt(model.src_embed[0].input_size, 2, 500,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def l2_norm(preds, targets):
    loss = ((preds - targets) ** 2).mean()
    return loss

def l1_norm(preds, targets):
    loss = torch.abs(preds - targets).mean()
    return loss

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use_gpu")
    dataset = JIGSAWSegmentsDataset(['/home/hding15/cis2/data/Knot_Tying'],['Knot_Tying'])
    dataloader = JIGSAWSegmentsDataloader(10, 30, 10, dataset, scale=100)
    model = EncoderDecoder(6, 6, N=10, input_size=512, hidden_layer=2048, h=8, dropout=0.1, task_dim=15)
    loss_function = l1_norm
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    loss_plot = overfit_samples(dataloader, 0, model, loss_function, optimizer, iteration_number=10000 , use_gpu=use_gpu, use_task=False)
    # np.save("loss_plot.npy", np.array(loss_plot))
    # loss_index = list(range(len(loss_plot)))
    visulize_results(dataloader, 0, loss_function, model, use_gpu=use_gpu, model_path=None, use_task=False)
    #visulize_overfit_results(dataloader, 0, loss_function, model, use_gpu=use_gpu, model_path="/home/hding15/cis2/urExpert/checkpoint/model_overfit0_simple.pth")
    # visulize_overfit_results(dataloader, 0, loss_function, model, use_gpu=use_gpu, model_path="/home/hding15/cis2/urExpert/checkpoint/model_overfit0_complex.pth")
    # train_split = [i for i in range(400)]
    # running_loss_plot = train_epochs(dataloader, train_split, model, loss_function, optimizer, n_epochs=300, use_gpu=use_gpu, use_task=True)
    # visulize_results(dataloader, 0, loss_function, model, use_gpu=use_gpu, use_task=True)
    # visulize_results(dataloader, 400, loss_function, model, use_gpu=use_gpu)
    # np.save("running_loss_plot.npy", np.array(running_loss_plot))
    # running_loss_index = list(range(len(running_loss_plot)))
    # plt.figure(figsize=(10,5))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_index, loss_plot)
    # plt.subplot(1, 2, 2)
    # plt.plot(running_loss_index, running_loss_plot)
    # plt.show()
