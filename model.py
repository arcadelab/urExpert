import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import Encoder
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
        if torch.equal(q, k) and torch.equal(k, v):
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
    Transformer Decoder Layer:
    Applies self-attention on embedded kinematics sequences, followed by cross-attention on encoded video captures
    :param x: embedded kinematics sequences
    :param y: encoded video captures
    """

    def __init__(self, interm_dim: int = 2048, feat_dim: int = 512, nhead: int = 8, dropout: int = 0, device: str = 'cuda', use_norm: bool = False, is_mask_enable: bool = False):
        super(DecoderLayer, self).__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(feat_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, feat_dim)

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.device = device
        self.use_norm = use_norm
        self.is_mask_enable = is_mask_enable

    def forward(self, x, y):
        # generate mask
        mask = generate_attn_mask(x.size(1)).to(self.device) if self.is_mask_enable else None
        # self-attn
        x = self.attn_layer(q=x, k=x, v=x, mask=mask) + x
        if self.use_norm:
            x = self.norm1(x)
        # cross-attn
        x = self.attn_layer(q=x, k=y, v=y, mask=None) + x
        # feed forward
        if self.use_norm:
            x = self.norm2(x)
        x = self.linear2(self.activation(self.linear1(x))) + x
        if self.use_norm:
            x = self.norm3(x)
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder:
    Stacked collections of decoder layers
    :param x: embedded kinematics sequences
    :param y: encoded video captures
    """

    def __init__(self, interm_dim: int = 2048, feat_dim: int = 256, nhead: int = 8, num_attn_layers: int = 10,
                 dropout: int = 0, device: str = 'cuda', use_norm: bool = False, is_mask_enable: bool = False):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model=feat_dim, dropout=0, max_len=5000)
        self.decoder_layer = DecoderLayer(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead, dropout=dropout,
                                          device=device, use_norm=use_norm, is_mask_enable=is_mask_enable)
        self.decoders = get_clones(self.decoder_layer, num_attn_layers)

    def forward(self, x, y):
        """
        :param x: embedded kinematics sequences
        :param y: encoded video captures
        """
        x = self.pos_encoder.forward(x)

        for module in self.decoders:
            x = module(x, y)

        return x


class Classifier(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 3):
        super().__init__()
        # bs * (sl * dim) --> bs * (ousl * dim2)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 512),
            nn.ReLU(),
            nn.Linear(in_channels // 512, out_channels)
        )

    def forward(self, x):
        return self.classifier(x)


class DecoderOnlyTransformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, feat_dim: int = 512, nhead: int = 8, num_attn_layers: int = 6, channel: int = 3,
                 device: str = 'cuda', use_norm: bool = False, is_mask_enable: bool = False, input_frames: int = 300, output_frames: int = 30):
        super().__init__()
        self.feat_dim = feat_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers
        self.channel = channel
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.is_mask_enable = is_mask_enable

        self.embeddings = nn.Linear(channel, feat_dim)
        self.decoder = Decoder(interm_dim=2048, feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                               dropout=0, device=device, use_norm=use_norm, is_mask_enable=is_mask_enable)
        self.final = nn.Linear(feat_dim, channel)
        self.final_classifier = Classifier(in_channels=feat_dim*input_frames, out_channels=channel*output_frames)

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
        if self.is_mask_enable:
            output = self.final(decoder_output).permute(0, 2, 1)  # input shape [N, W, C]
        else:
            output = self.final_classifier(decoder_output.reshape(-1, self.feat_dim * self.input_frames)).reshape(-1, self.output_frames, self.channel).permute(0, 2, 1)
        # print("fully connected layers output = {}".format(output.shape))

        return output


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, feat_dim, nhead, num_attn_layers, channel, device, use_norm, is_mask_enable, input_frames,
                 output_frames, is_feature_extract, num_conv_layers, input_channel, output_channel, conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding, img_height, img_width, patch_height, patch_width,
                 in_dim, batch_size, capture_size, dropout):
        super().__init__()
        self.feat_dim = feat_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers
        self.channel = channel
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.batch_size = batch_size
        self.is_mask_enable = is_mask_enable

        self.encoder = Encoder(is_feature_extract=is_feature_extract, num_conv_layers=num_conv_layers, input_channel=input_channel,
                    output_channel=output_channel, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width, patch_height=patch_height,
                    patch_width=patch_width, in_dim=in_dim, feat_dim=feat_dim, batch_size=batch_size, capture_size=capture_size, dropout=dropout, device=device)

        self.embeddings = nn.Linear(channel, feat_dim)
        self.decoder = Decoder(interm_dim=2048, feat_dim=feat_dim, nhead=nhead, num_attn_layers=num_attn_layers,
                               dropout=0, device=device, use_norm=use_norm, is_mask_enable=is_mask_enable)
        self.final = nn.Linear(feat_dim, channel)
        self.final_classifier = Classifier(in_channels=feat_dim * input_frames, out_channels=channel * output_frames)

    def forward(self, input):
        """
        :param psm1_past: feature descriptor of past kinematics, [N,C,W]
        :param psm1_future: feature descriptor of future kinematics, [N,C,W]
        :param pos_enc: relative positional encoding, [N,C,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """
        # encoder video captures
        encoded_cap = self.encoder(input["captures"]) # encoded capture shape [N, capture_size, patch_size, dimension]
        # print("encoding shape = {}".format(encoded_cap.shape))

        # kinematics embeddings
        embedded_kin = self.embeddings(input["kinematics"].permute(0, 2, 1))
        # print("embedding shape = {}".format(embedded_kin.shape))

        # Decode
        decoder_output = self.decoder(embedded_kin, encoded_cap.reshape(self.batch_size, -1, self.feat_dim))

        # Final layer
        if self.is_mask_enable:
            output = self.final(decoder_output).permute(0, 2, 1)  # input shape [N, W, C]
        else:
            output = self.final_classifier(decoder_output.reshape(-1, self.feat_dim * self.input_frames)).reshape(-1, self.output_frames, self.channel).permute(0, 2, 1)
        # print("fully connected layers output = {}".format(output.shape))

        return output


if __name__ == "__main__":
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(input)
    print(output)