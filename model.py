import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        assert q.shape[1] == k.shape[1] and k.shape[1] == v.shape[1], "self-attention input doesn't have equal length"
        nbatch = q.shape[0]
        _q = self.w_q(q).view(nbatch, -1, self.nhead, self.d_k).transpose(1, 2)
        _k = self.w_k(k).view(nbatch, -1, self.nhead, self.d_k).transpose(1, 2)
        _v = self.w_v(v).view(nbatch, -1, self.nhead, self.d_k).transpose(1, 2)

        attn = attention(_q, _k, _v, self.d_k, mask=mask, dropout=self.dropout)
        attn = attn.transpose(1, 2).contiguous().view(nbatch, -1, self.d_model)
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

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """

    def __init__(self, interm_dim: int = 2048, feat_dim: int = 512, nhead: int = 8, dropout: int = 0, device: str = 'cuda', use_norm: bool = False):
        super(DecoderLayer, self).__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(feat_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, feat_dim)

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.device = device
        self.use_norm = use_norm

    def forward(self, x):
        # x.shape: N * L * D
        # generate mask
        mask = generate_attn_mask(x.size(1)).to(self.device)
        # self-attn
        x = self.attn_layer(q=x, k=x, v=x, mask=mask) + x
        # feed forward
        if self.use_norm:
            x = self.norm1(x)
        x = self.linear2(self.activation(self.linear1(x))) + x
        if self.use_norm:
            x = self.norm1(x)
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, interm_dim: int = 2048, feat_dim: int = 256, nhead: int = 8, num_attn_layers: int = 10,
                 dropout: int = 0, device: str = 'cuda', use_norm: bool = False):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model=feat_dim, dropout=0, max_len=5000)
        self.decoder_layer = DecoderLayer(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead, dropout=dropout, device=device, use_norm=use_norm)
        self.decoders = get_clones(self.decoder_layer, num_attn_layers)

    def forward(self, x):
        """

        :param x: CNN feature map, extract from input kinematics sequence [W, N, C1]
        :param y: raw input kinematics sequence [W, N, C2]
        :return:
        """
        x = self.pos_encoder.forward(x)

        for module in self.decoders:
            x = module(x)

        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, feat_dim: int = 512, nhead: int = 8, num_attn_layers: int = 6, channel: int = 3,
                 device: str = 'cuda', use_norm: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

        self.embeddings = nn.Linear(channel, feat_dim)
        self.decoder = Decoder(interm_dim=2048, feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                               dropout=0, device=device, use_norm=use_norm)
        self.final = nn.Linear(feat_dim, channel)

    def forward(self, psm1: torch.Tensor):
        """
        :param psm1_past: feature descriptor of past kinematics, [N,C,W]
        :param psm1_future: feature descriptor of future kinematics, [N,C,W]
        :param pos_enc: relative positional encoding, [N,C,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """

        # kinematics embeddings
        x = self.embeddings(psm1.permute(0, 2, 1))
        # print("embedding shape = {}".format(x.shape))

        # Decode
        decoder_output = self.decoder(x)

        # Final layer
        output = self.final(decoder_output).permute(0, 2, 1)  # input shape [N, W, C]
        # print("fully connected layers output = {}".format(output.shape))

        return output


if __name__ == "__main__":
    # print(generate_attn_mask(10))
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(input)
    print(output)