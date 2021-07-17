from attention import *
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
                           conv_kernel_size=conv_kernel_size, conv_stride=conv_stride,
                           pool_kernel_size=pool_kernel_size,
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
    def __init__(self, img_height, img_width, patch_height, patch_width, in_dim, feat_dim, project_type, capture_size):
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
        self.patch_size = (patch_height, patch_width)
        self.project_type = project_type
        # patch and linear layer based embedding
        self.project_linear = nn.Sequential(
            Rearrange('b l c (h ph) (w pw) -> b l (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.Linear(patch_height * patch_width * in_dim, feat_dim)
        )
        # patch and convolution based embedding
        self.project_conv = nn.Sequential(
            Rearrange('b l c H W -> (b l) c H W'),
            nn.Conv2d(in_channels=in_dim, out_channels=feat_dim, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('(b l) e h w -> b l (h w) e', l=capture_size)
        )

    def forward(self, vid):
        if self.project_type == "linear":
            x = self.project_linear(vid)
        elif self.project_type == "conv":
            x = self.project_conv(vid)
        else:
            x = vid
        # print("Tensor shape after patch embedding: {}".format(x.shape))
        return x


class pos_embedding(nn.Module):
    def __init__(self, feat_dim, num_patches, capture_size, device):
        """
        This module add learnable CLS token and position embedding to the beginning of each patched frame.
        :param feat_dim: feature dimension
        :param num_patches: number of subsets in one patched frame
        """
        super(pos_embedding, self).__init__()
        self.cls = nn.Parameter(torch.randn(1, capture_size, 1, feat_dim)).to(device)
        self.pos_embedding = nn.Parameter(torch.randn(1, capture_size, num_patches + 1, feat_dim)).to(device)

    def forward(self, vid, bs):
        x = torch.cat((repeat(self.cls, '() l n d -> b l n d', b=bs), vid), dim=2)
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

    def __init__(self, interm_dim, feat_dim, nhead, dropout, use_encoder_norm):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(feat_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, feat_dim)

        self.norm1 = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(feat_dim, elementwise_affine=False)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.use_encoder_norm = use_encoder_norm

    def forward(self, din):
        """
        :param din: patched and embedded video captures
        """
        # self-attn
        x = self.dropout1(self.attn_layer(q=din, k=din, v=din, mask=None)) + din

        if self.use_encoder_norm:
            x = self.norm1(x)

        # feed forward
        x = self.dropout2(self.linear2(self.activation(self.linear1(x)))) + x

        if self.use_encoder_norm:
            x = self.norm2(x)

        return x


class VideoEncoder(nn.Module):
    def __init__(self, is_feature_extract, num_conv_layers, input_channel, output_channel, conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding, img_height, img_width, patch_height, patch_width,
                 in_dim, feat_dim, capture_size, dropout, device, interm_dim, nhead, num_encoder_layer,
                 is_encode, project_type):
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
        super(VideoEncoder, self).__init__()
        self.is_feature_extract = is_feature_extract
        if is_feature_extract:
            self.conv_network = conv_network(num_conv_layers, input_channel, output_channel, conv_kernel_size,
                                             conv_stride, pool_kernel_size, pool_stride, padding)
        # TODO: currently pooling is assumed to decreased image size by a factor of 0.5, need to adapt to general case
        self.is_encode = is_encode
        self.feat_dim = feat_dim
        self.img_height = img_height // pow(2, num_conv_layers + 1) if is_feature_extract else img_height
        self.img_width = img_width // pow(2, num_conv_layers + 1) if is_feature_extract else img_width
        self.in_dim = output_channel if is_feature_extract else in_dim
        self.num_patches = (self.img_height // patch_height) * (self.img_width // patch_width)
        self.frame_patch_embed = frame_patch_embed(self.img_height, self.img_width, patch_height, patch_width,
                                                   self.in_dim, feat_dim, project_type, capture_size)
        self.pos_embedding = pos_embedding(feat_dim, self.num_patches, capture_size, device)
        self.encoder_layer = EncoderLayer(interm_dim, feat_dim, nhead, dropout, use_encoder_norm=True)
        self.encoders = get_clones(self.encoder_layer, num_encoder_layer)
        self.vid_norm = nn.LayerNorm([img_height, img_width], elementwise_affine=False)

        if is_encode:
            print("self attend encoded video..")

    def forward(self, vid):
        # get batch size
        bs, l, c, h, w = vid.shape
        # feature extract if desired
        x = self.conv_network(vid) if self.is_feature_extract else vid  # [:, 1].unsqueeze(1)

        # patch and embed
        x = self.frame_patch_embed(self.vid_norm(x))

        # position embed
        x = self.pos_embedding(x, bs)

        # concatenate for self-attn, before reshape [bs, cs, ps, (ph * pw * c)]
        x = x.reshape(bs, -1, self.feat_dim)

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
    img_height = 240
    img_width = 320
    patch_height = 16
    patch_width = 16
    in_dim = 3
    feat_dim = 512
    batch_size = 16
    capture_size = 1
    dropout = 0.1
    device = 'cpu'
    interm_dim = 2048
    nhead = 8
    num_encoder_layer = 6
    is_encode = False
    project_type = "conv"
    model = VideoEncoder(is_feature_extract=is_feature_extract, num_conv_layers=num_conv_layers,
                         input_channel=input_channel,
                         output_channel=output_channel, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride,
                         pool_kernel_size=pool_kernel_size,
                         pool_stride=pool_stride, padding=padding, img_height=img_height, img_width=img_width,
                         patch_height=patch_height, patch_width=patch_width, in_dim=in_dim, feat_dim=feat_dim,
                         capture_size=capture_size, dropout=dropout, device=device,
                         interm_dim=interm_dim, nhead=nhead, num_encoder_layer=num_encoder_layer, is_encode=is_encode,
                         project_type=project_type)

    input = torch.randn(batch_size, 2, input_channel, img_height, img_width)
    print(model(input).shape)

    conv = nn.Conv2d(in_channels=3, out_channels=feat_dim, kernel_size=(patch_height, patch_width),
                     stride=(patch_height, patch_width))
    # print(conv(torch.randn(1, 3, 240, 320)).shape)

    input = torch.randn(16, 2, 3, 240, 320)
    # print(conv(input[:, 1]).shape)
