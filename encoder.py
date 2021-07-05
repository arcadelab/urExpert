import torch
import copy
import math
import torch.nn as nn
from einops import repeat
from torch.nn import functional as F
from einops.layers.torch import Rearrange


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


def conv_layer(input_channel, output_channel, conv_kernel_size, conv_stride, pool_kernel_size, pool_stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=conv_kernel_size,
                  stride=conv_stride, padding=padding),
        nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        # nn.LayerNorm(output_channel),
        nn.GELU()
    )


class conv_network(nn.Module):
    def __init__(self, num_conv_layers, input_channel, output_channel, conv_kernel_size, conv_stride, pool_kernel_size,
                 pool_stride, padding):
        super(conv_network, self).__init__()
        self.input_channel = input_channel
        self.conv_network = nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_network.append(
            conv_layer(input_channel=self.input_channel, output_channel=self.input_channel * 4,
                       conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                       pool_stride=pool_stride, padding=padding))
            self.input_channel = self.input_channel * 4
        self.conv_network.append(
            conv_layer(input_channel=self.input_channel, output_channel=output_channel,
                       conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                       pool_stride=pool_stride, padding=padding))

    def forward(self, input):
        x = input
        for module in self.conv_network:
            x = module(x)
        # print("Tensor shape after convolution: {}".format(x.shape))
        return x


class frame_patch_embed(nn.Module):
    def __init__(self, img_height, img_width, patch_height, patch_width, in_dim, feat_dim):
        """
        Patch input frames of shape (H * W) into subsets of (N * PH * PW), where H is input frame height, W is input frame width,
        PH is patch height, PW is patch width, N is the number of subsets in after patch process, i.e. N = (H // PH) * (W // PW).
        Patched frames are embedded using linear layer.
        :param img_height: input frame height
        :param img_width: input frame width
        :param patch_height: patch height
        :param patch_width: patch weight
        :param patch_dim: patch dimension
        :param feat_dim: feature dimension
        """
        super(frame_patch_embed, self).__init__()
        assert img_width % patch_width == 0 and img_height % patch_height == 0, "frame height/width should be divisible by patch height/width"
        self.patch = Rearrange('b l c (h ph) (w pw) -> b l (h w) (ph pw c)', ph=patch_height, pw=patch_width)
        self.embedding = nn.Linear(patch_height * patch_width * in_dim, feat_dim)

    def forward(self, vid):
        x = self.patch(vid)
        # print("Tensor shape after patch: {}".format(x.shape))
        x = self.embedding(x)
        # print("Tensor shape after patch embedding: {}".format(x.shape))
        return x

class pos_embedding(nn.Module):
    def __init__(self, feat_dim, num_patches, batch_size, capture_size, device):
        """
        This module add learnable CLS token and position embedding to the beginning of each patched frame.
        :param feat_dim: feature dimension
        :param num_patches: number of subsets in one patched frame
        :param batch_size: training batch size
        """
        super(pos_embedding, self).__init__()
        self.cls = repeat(nn.Parameter(torch.randn(1, capture_size, 1, feat_dim)), '() l n d -> b l n d', b=batch_size).to(device)
        self.pos_embedding = nn.Parameter(torch.randn(1, capture_size, num_patches+1, feat_dim)).to(device)

    def forward(self, vid):
        x = torch.cat((self.cls, vid), dim=2)
        # print("Tensor shape after cls concatenation: {}".format(x.shape))
        x = x + self.pos_embedding
        # print("Tensor shape after pos embedding: {}".format(x.shape))
        return x


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer:
    Applies self-attention on patched and embedded video captures
    :param x: patched and embedded video captures
    """

    def __init__(self, interm_dim, feat_dim, nhead, dropout):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(feat_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, feat_dim)

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, vid):
        # self-attn
        x = self.norm1(self.attn_layer(q=vid, k=vid, v=vid, mask=None) + vid)
        # feed forward
        x = self.norm2(self.linear2(self.activation(self.linear1(x))) + x)
        return x


class Encoder(nn.Module):
    def __init__(self, is_feature_extract, num_conv_layers, input_channel, output_channel, conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding, img_height, img_width, patch_height, patch_width,
                 in_dim, feat_dim, batch_size, capture_size, dropout, device, interm_dim, nhead, num_encoder_layer, is_encode):
        """
        Jigsaw dataset surgical video frames encoder. Architecture based on vision transformer encoder.
        :param is_feature_extract:
        :param num_conv_layers:
        :param input_channel:
        :param output_channel:
        :param kernel_size:
        :param stride:
        :param padding:
        :param img_height:
        :param img_width:
        :param patch_height:
        :param patch_width:
        :param patch_dim:
        :param feat_dim:
        :param num_patches:
        :param batch_size:
        :param dropout:
        """
        super(Encoder, self).__init__()
        self.is_feature_extract = is_feature_extract
        if is_feature_extract:
            self.conv_network = conv_network(num_conv_layers, input_channel, output_channel,conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding)
        # TODO: currently pooling is assumed to decreased image size by a factor of 0.5, need to adapt to general case
        self.is_encode = is_encode
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.img_height = img_height // pow(2, num_conv_layers + 1) if is_feature_extract else img_height
        self.img_width = img_width // pow(2, num_conv_layers + 1) if is_feature_extract else img_width
        self.in_dim = output_channel if is_feature_extract else in_dim
        self.num_patches = (self.img_height // patch_height) * (self.img_width // patch_width)
        self.frame_patch_embed = frame_patch_embed(self.img_height, self.img_width, patch_height, patch_width, self.in_dim, feat_dim)
        self.pos_embedding = pos_embedding(feat_dim, self.num_patches, batch_size, capture_size, device)
        self.encoder_layer = EncoderLayer(interm_dim, feat_dim, nhead, dropout)
        self.encoders = get_clones(self.encoder_layer, num_encoder_layer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vid):
        # feature extract if desired
        x = self.conv_network(vid) if self.is_feature_extract else vid
        # patch and embed
        x = self.frame_patch_embed(x)
        # position embed
        x = self.pos_embedding(x)
        # concatenate for self-attn
        # before reshape [bs, cs, ps, (ph * pw * c)]
        x = self.dropout(x.reshape(self.batch_size, -1, self.feat_dim))
        # encode
        if self.is_encode:
            for module in self.encoders:
                x = module(x)
        return x


if __name__ == "__main__":
    is_feature_extract = False
    num_conv_layers = 3
    input_channel = 3
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
    feat_dim = 1024
    batch_size = 16
    capture_size = 2
    dropout = 0.1
    device = 'cpu'
    interm_dim = 2048
    nhead = 8
    num_encoder_layer = 6
    is_encode = False
    model = Encoder(is_feature_extract=is_feature_extract, num_conv_layers=num_conv_layers, input_channel=input_channel,
                    output_channel=output_channel, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width, patch_height=patch_height,
                    patch_width=patch_width, in_dim=in_dim, feat_dim=feat_dim, batch_size=batch_size, capture_size=capture_size, dropout=dropout, device=device,
                    interm_dim=interm_dim, nhead=nhead, num_encoder_layer=num_encoder_layer, is_encode=is_encode)

    input = torch.randn(batch_size, capture_size, input_channel, img_height, img_width)
    print(model(input).shape)
