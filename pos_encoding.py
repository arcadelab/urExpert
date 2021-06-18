import torch
from torch import nn, Tensor
import math


class NestedTensor(object):
    def __init__(self, left, right, disp=None, sampled_cols=None, sampled_rows=None, occ_mask=None,
                 occ_mask_right=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.occ_mask = occ_mask
        self.occ_mask_right = occ_mask_right
        self.sampled_cols = sampled_cols
        self.sampled_rows = sampled_rows


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, x):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,2W-1]
        """

        # update h and w if downsampling
        bs, c, w = x.size()

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)

        if self.normalize:
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos


def no_pos_encoding(x):
    return None


def build_position_encoding(args):
    mode = args.position_encoding
    channel_dim = args.channel_dim
    if mode == 'sine1d_rel':
        n_steps = channel_dim
        position_encoding = PositionEncodingSine1DRelative(n_steps, normalize=False)
    elif mode == 'none':
        position_encoding = no_pos_encoding
    else:
        raise ValueError(f"not supported {mode}")

    return position_encoding