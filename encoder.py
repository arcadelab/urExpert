import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


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

    def forward(self, x):
        for module in self.conv_network:
            x = module(x)
        print("Tensor shape after convolution: {}".format(x.shape))
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
        self.patch = Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width)
        self.embedding = nn.Linear(patch_height * patch_width * in_dim, feat_dim)

    def forward(self, x):
        x = self.patch(x)
        print("Tensor shape after patch: {}".format(x.shape))
        x = self.embedding(x)
        print("Tensor shape after patch embedding: {}".format(x.shape))
        return x

class pos_embedding(nn.Module):
    def __init__(self, feat_dim, num_patches, batch_size):
        """
        This module add learnable CLS token and position embedding to the beginning of each patched frame.
        :param feat_dim: feature dimension
        :param num_patches: number of subsets in one patched frame
        :param batch_size: training batch size
        """
        super(pos_embedding, self).__init__()
        self.cls = repeat(nn.Parameter(torch.randn(1, 1, feat_dim)), '() n d -> b n d', b=batch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, feat_dim))

    def forward(self, x):
        x = torch.cat((self.cls, x), dim=1)
        print("Tensor shape after cls concatenation: {}".format(x.shape))
        x = x + self.pos_embedding
        print("Tensor shape after pos embedding: {}".format(x.shape))
        return x


class Encoder(nn.Module):
    def __init__(self, is_feature_extract, num_conv_layers, input_channel, output_channel, conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding, img_height, img_width, patch_height, patch_width,
                 in_dim, feat_dim, batch_size, dropout):
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
        self.img_height = img_height // pow(2, num_conv_layers+1) if is_feature_extract else img_height
        self.img_width = img_width // pow(2, num_conv_layers + 1) if is_feature_extract else img_width
        self.in_dim = output_channel if is_feature_extract else in_dim
        self.num_patches = (self.img_height // patch_height) * (self.img_width // patch_width)
        self.frame_patch_embed = frame_patch_embed(self.img_height, self.img_width, patch_height, patch_width, self.in_dim, feat_dim)
        self.pos_embedding = pos_embedding(feat_dim, self.num_patches, batch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        if self.is_feature_extract:
            x = self.conv_network(x)
        x = self.frame_patch_embed(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    is_feature_extract = True
    num_conv_layers = 3
    input_channel = 3
    output_channel = 256
    conv_kernel_size = 3
    conv_stride = 1
    pool_kernel_size = 2
    pool_stride = 2
    padding = 1
    img_height = 256
    img_width = 256
    patch_height = 16
    patch_width = 16
    in_dim = 3
    feat_dim = 1024
    batch_size = 16
    dropout = 0.1
    model = Encoder(is_feature_extract=is_feature_extract, num_conv_layers=num_conv_layers, input_channel=input_channel,
                    output_channel=output_channel, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width, patch_height=patch_height,
                    patch_width=patch_width, in_dim=in_dim, feat_dim=feat_dim, batch_size=batch_size, dropout=dropout)

    input = torch.randn(16, 3, 256, 256)
    print(model(input).shape)
