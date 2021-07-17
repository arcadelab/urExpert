from decoder import *
from encoder import *


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

    def __init__(self, interm_dim, feat_dim, nhead, num_decoder_layers, channel, device, use_norm,
                 input_frames, output_frames, attend_vid, dropout):
        super().__init__()
        self.channel = channel
        self.feat_dim = feat_dim
        self.input_frames = input_frames
        self.output_frames = output_frames

        self.embedding_kin = nn.Linear(channel, feat_dim)
        self.decoder = Decoder(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead,
                               num_decoder_layers=num_decoder_layers,
                               dropout=dropout, device=device, use_norm=use_norm,
                               attend_vid=attend_vid)
        self.final = nn.Linear(feat_dim, channel)
        self.final_classifier = Classifier(in_channels=feat_dim * input_frames, out_channels=channel * output_frames)

    def forward(self, kin, vid):
        """
        :param kin: raw kinematics input from psm
        :param vid: captured surgical video frames
        """

        # kinematics embeddings
        embedded_kin = self.embedding_kin(kin.permute(0, 2, 1))
        # print("embedding shape = {}".format(x.shape))

        # Decode
        decoder_output = self.decoder(embedded_kin, vid)

        # Final layer
        output = self.final_classifier(decoder_output.reshape(-1, self.feat_dim * self.input_frames))
        # print("fully connected layers output = {}".format(output.shape))

        return output.reshape(-1, self.output_frames, self.channel).permute(0, 2, 1)


class VideoEncoderDecoderTransformer(nn.Module):
    def __init__(self, feat_dim, nhead, num_decoder_layers, channel, device, use_norm, input_frames,
                 output_frames, is_feature_extract, num_conv_layers, input_channel, output_channel, conv_kernel_size,
                 conv_stride, pool_kernel_size, pool_stride, padding, img_height, img_width, patch_height, patch_width,
                 in_dim, capture_size, dropout, interm_dim, num_encoder_layer, is_encode, project_type, attend_vid):
        super().__init__()
        self.feat_dim = feat_dim
        self.channel = channel
        self.input_frames = input_frames
        self.output_frames = output_frames

        self.encoder = VideoEncoder(is_feature_extract=is_feature_extract, num_conv_layers=num_conv_layers,
                                    input_channel=input_channel, output_channel=output_channel, conv_kernel_size=conv_kernel_size,
                                    conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                                    pool_stride=pool_stride, padding=padding, img_height=img_height,
                                    img_width=img_width, patch_height=patch_height,
                                    patch_width=patch_width, in_dim=in_dim, feat_dim=feat_dim,
                                    capture_size=capture_size, dropout=dropout, device=device,
                                    interm_dim=interm_dim, nhead=nhead, num_encoder_layer=num_encoder_layer,
                                    is_encode=is_encode, project_type=project_type)
        self.embedding_kin = nn.Linear(channel, feat_dim)
        self.decoder = Decoder(interm_dim=interm_dim, feat_dim=feat_dim, nhead=nhead, num_decoder_layers=num_decoder_layers,
                               dropout=dropout, device=device, use_norm=use_norm, attend_vid=attend_vid)
        self.final = nn.Linear(feat_dim, channel)
        self.final_classifier = Classifier(in_channels=feat_dim * input_frames, out_channels=channel * output_frames)

        print("video encoder-decoder transformer instantiated...")

    def forward(self, kin, vid):
        """
        :param kin: raw kinematics input from psm
        :param vid: captured surgical video frames
        """
        # encoder video captures
        embedded_vid = self.encoder(vid)  # encoded capture shape [N, capture_size, patch_size, dimension]
        # print("encoding shape = {}".format(encoded_cap.shape))

        # kinematics embeddings
        embedded_kin = self.embedding_kin(kin.permute(0, 2, 1))
        # print("embedding shape = {}".format(embedded_kin.shape))

        # Decode
        decoder_output = self.decoder(embedded_kin, embedded_vid)

        # Final layer
        output = self.final_classifier(decoder_output.reshape(-1, self.feat_dim * self.input_frames))
        # print("fully connected layers output = {}".format(output.shape))

        return output.reshape(-1, self.output_frames, self.channel).permute(0, 2, 1)


if __name__ == "__main__":
    m = nn.Softmax(dim=1)
    input = torch.randn(16, 2, 3, 2)
    output = m(input)
    print(output)
