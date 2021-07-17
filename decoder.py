from attention import *


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer:
    Applies self-attention on embedded kinematics sequences, followed by cross-attention on encoded video captures
    """

    def __init__(self, interm_dim, feat_dim, nhead, dropout, device, use_norm, attend_vid):
        super(DecoderLayer, self).__init__()
        self.attn_layer1 = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)
        self.attn_layer2 = MultiHeadSelfAttention(d_model=feat_dim, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(feat_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, feat_dim)

        self.norm1 = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(feat_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(feat_dim, elementwise_affine=False)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.device = device
        self.use_norm = use_norm
        self.attend_vid = attend_vid

        if self.attend_vid:
            print("cross attend encoded video..")

    def forward(self, kin, vid):
        """
        :param kin: embedded kinematics psm sequences
        :param vid: encoded video captures
        """
        # self-attn
        x = self.dropout1(self.attn_layer1(q=kin, k=kin, v=kin, mask=None)) + kin

        if self.use_norm:
            x = self.norm1(x)

        # cross-attn on vid
        if self.attend_vid:
            x = self.dropout2(self.attn_layer2(q=x, k=vid, v=vid, mask=None)) + x

        # feed forward
        if self.use_norm:
            x = self.norm2(x)

        x = self.dropout3(self.linear2(self.activation(self.linear1(x)))) + x

        if self.use_norm:
            x = self.norm3(x)
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder:
    Stacked collections of decoder layers
    """

    def __init__(self, interm_dim, feat_dim, nhead, num_decoder_layers, dropout, device, use_norm, attend_vid):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model=feat_dim, dropout=0, max_len=5000)
        self.decoder_layer = DecoderLayer(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead, dropout=dropout,
                                          device=device, use_norm=use_norm, attend_vid=attend_vid)
        self.decoders = get_clones(self.decoder_layer, num_decoder_layers)

    def forward(self, kin, vid):
        """
        :param kin: embedded kinematics psm sequences
        :param vid: encoded video captures
        """
        x = self.pos_encoder.forward(kin)

        for module in self.decoders:
            x = module(x, vid)

        return x
