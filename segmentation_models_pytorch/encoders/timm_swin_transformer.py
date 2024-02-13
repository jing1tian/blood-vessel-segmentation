from ._base import EncoderMixin
from timm.models.swin_transformer import SwinTransformer
from timm.layers import resample_patch_embed, resize_rel_pos_bias_table
import torch
import torch.nn as nn
import torch.nn.functional as F


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    old_weights = True
    if 'head.fc.weight' in state_dict:
        old_weights = False
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'attn_mask')]):
            continue  # skip buffers that should not be persistent

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W), # type: ignore
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if k.endswith('relative_position_bias_table'):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(
                    v,
                    new_window_size=m.window_size,
                    new_bias_shape=m.relative_position_bias_table.shape,
                )

        if old_weights:
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = k.replace('head.', 'head.fc.')

        out_dict[k] = v
    return out_dict


class SwinTransformerEncoder(SwinTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.head
        
    def get_stages(self):
        return [
            nn.Identity(),
            self.patch_embed,
            self.layers[0],
            self.layers[1],
            self.layers[2],
            self.layers[3],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if i == 0:
                features.append(x)
            elif i == 1:
                features.append(F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear"))
            else:
                features.append(x.permute(0, 3, 1, 2))
        
        # [2, 4, 8, 16, 32]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = checkpoint_filter_fn(state_dict, self)
        state_dict.pop("head.fc.bias", None)
        state_dict.pop("head.fc.weight", None)
        state_dict.pop("head.norm.weight", None)
        state_dict.pop("head.norm.bias", None)
        super().load_state_dict(state_dict, **kwargs)
        
        
swin_transformer_weights = {
    "swin_tiny_patch4_window7": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",  # noqa
    },
    "swin_small_patch4_window7": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",  # noqa
    },
    "swin_base_patch4_window7": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",  # noqa
    },
    "swin_base_patch4_window12": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth",  # noqa
    },
    "swin_large_patch4_window7": {
        "imagenet22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",  # noqa
    },
    "swin_large_patch4_window12": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in swin_transformer_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
        
timm_swin_transformer_encoders = {
    "swin_tiny_patch4_window7": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_tiny_patch4_window7"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
        },
    },
    "swin_small_patch4_window7": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_small_patch4_window7"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (3, 6, 12, 24),
        },
    },
    "swin_base_patch4_window7": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_base_patch4_window7"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
        },
    },
    "swin_base_patch4_window12": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_base_patch4_window12"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "patch_size": 4,
            "window_size": 12,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
        },
    },
    "swin_large_patch4_window7": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_large_patch4_window7"],
        "params": {
            "out_channels": (3, 192, 192, 384, 768, 1536),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
        },
    },
    "swin_large_patch4_window12": {
        "encoder": SwinTransformerEncoder,
        "pretrained_settings": pretrained_settings["swin_large_patch4_window12"],
        "params": {
            "out_channels": (3, 192, 192, 384, 768, 1536),
            "patch_size": 4,
            "window_size": 12,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
        },
    },
}
