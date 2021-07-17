import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def generate_attn_mask(length):
    """
    generate self-attention mask
    :param length: length of input kinematics sequence
    :return: upper triangular mask indices
    """
    mask = np.triu(np.ones((1, length, length)), k=1).astype(np.bool_)
    mask_indices = torch.from_numpy(mask) == 0
    return mask_indices


def attention(q, k, v, d_k, mask=None, dropout=None):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # apply mask if needed
    if mask is not None:
        mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -1e9)
    # apply dropout if needed
    score = F.softmax(score, dim=-1)
    if dropout is not None:
        score = dropout(score)
    # return attention
    attn = torch.matmul(score, v)
    return attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        if torch.equal(q, k) and torch.equal(k, v):
            assert q.shape[1] == k.shape[1] and k.shape[1] == v.shape[
                1], "self-attention input doesn't have equal length"
        num_batch = q.shape[0]
        _q = self.w_q(q).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)
        _k = self.w_k(k).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)
        _v = self.w_v(v).view(num_batch, -1, self.nhead, self.d_k).transpose(1, 2)

        attn = attention(_q, _k, _v, self.d_k, mask=mask, dropout=self.dropout)
        attn = attn.transpose(1, 2).contiguous().view(num_batch, -1, self.d_model)
        attn = self.linear(attn)

        return attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, kin):
        x = kin + self.pe[:kin.size(0), :]
        return self.dropout(x)
