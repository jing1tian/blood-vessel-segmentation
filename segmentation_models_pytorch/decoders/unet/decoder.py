import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        scale_factor=2,
        norm_type=None,
        act_type="ReLU",
        attention_type=None,
        upsample_method="nearest",
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.scale_factor = scale_factor

        self.upsample_method = upsample_method
        if upsample_method == "transposed_conv":
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale_factor, stride=scale_factor),
                md.LayerNorm2d(in_channels),
                nn.GELU()
            )
        

    def forward(self, x, skip=None):
        if self.upsample_method == "transposed_conv":
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.upsample_method)
        # x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_type="BN", act_type="ReLU"):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
            act_type=act_type
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_type=norm_type,
            act_type=act_type
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        norm_type=None,
        act_type="ReLU",
        attention_type=None,
        center=False,
        use_checkpoint=False,
        scale_factor=2,
        upsample_method="nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        self.use_checkpoint = use_checkpoint
        
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, norm_type=norm_type, act_type=act_type)
        else:
            self.center = nn.Identity()

        if not isinstance(scale_factor, (tuple, list)):
            scale_factor = [scale_factor] * len(out_channels)
        assert len(scale_factor) == len(out_channels)
        
        # combine decoder keyword arguments
        kwargs = dict(norm_type=norm_type, act_type=act_type, attention_type=attention_type, upsample_method=upsample_method)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, s, **kwargs)
            for in_ch, skip_ch, out_ch, s in zip(in_channels, skip_channels, out_channels, scale_factor)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            if self.use_checkpoint:
                x = checkpoint.checkpoint(decoder_block, x, skip)
            else:
                x = decoder_block(x, skip)

        return x
