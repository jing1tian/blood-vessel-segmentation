from ._base import EncoderMixin
from timm.models.focalnet import FocalNet
from timm.layers import LayerNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F

def checkpoint_filter_fn(state_dict, model: FocalNet):
    state_dict = state_dict.get('model', state_dict)
    if 'stem.proj.weight' in state_dict:
        return state_dict
    import re
    out_dict = {}
    dest_dict = model.state_dict()
    for k, v in state_dict.items():
        k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        k = k.replace('patch_embed', 'stem')
        k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        if 'norm' in k and k not in dest_dict:
            k = re.sub(r'norm([0-9])', r'norm\1_post', k)
        k = k.replace('ln.', 'norm.')
        k = k.replace('head', 'head.fc')
        if k in dest_dict and dest_dict[k].numel() == v.numel() and dest_dict[k].shape != v.shape:
            v = v.reshape(dest_dict[k].shape)
        out_dict[k] = v
    return out_dict

class FocalNetEncoder(FocalNet, EncoderMixin):
    def __init__(self, out_channels, in_channels=3, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels
        
        self.extra_stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], 3, 2, 1),
            LayerNorm2d(out_channels[1])
        )

        # self.extra_norms = nn.ModuleList([
        #     LayerNorm2d(dim) for dim in out_channels[2:]
        # ])

        del self.head
        del self.norm
        
    def get_stages(self):
        return [
            self.layers[0],
            self.layers[1],
            self.layers[2],
            self.layers[3],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        features.append(x)
        _x = self.extra_stem(x)
        features.append(_x)
        x = self.stem(x)

        for i in range(self._depth - 1):
            x = stages[i](x)
            features.append(x)
                
        # [1, 2, 4, 8, 16, 32]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = checkpoint_filter_fn(state_dict, self)
        state_dict.pop("head.fc.bias", None)
        state_dict.pop("head.fc.weight", None)
        state_dict.pop("head.norm.weight", None)
        state_dict.pop("head.norm.bias", None)
        state_dict.pop("norm.weight", None)
        state_dict.pop("norm.bias", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)
        
        
focalnet_weights = {
    "focalnet_tiny_srf": {
        "imagenet": "",  # noqa
    },
    "focalnet_small_srf": {
        "imagenet": "",  # noqa
    },
    "focalnet_base_srf": {
        "imagenet": "",  # noqa
    },
    "focalnet_tiny_lrf": {
        "imagenet": "focalnet_tiny_lrf.bin",  # noqa
    },
    "focalnet_small_lrf": {
        "imagenet": "",  # noqa
    },
    "focalnet_base_lrf": {
        "imagenet": "",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in focalnet_weights.items():
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
        
timm_focalnet_encoders = {
    "focalnet_tiny_srf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_tiny_srf"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (2, 2, 6, 2),
            "embed_dim": 96,
        },
    },
    "focalnet_tiny_lrf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_tiny_lrf"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (2, 2, 6, 2),
            "embed_dim": 96,
            "focal_levels": (3, 3, 3, 3),
        },
    },
    "focalnet_small_srf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_small_srf"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (2, 2, 18, 2),
            "embed_dim": 96,
        },
    },
    "focalnet_small_lrf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_small_lrf"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (2, 2, 18, 2),
            "embed_dim": 96,
            "focal_levels": (3, 3, 3, 3),
        },
    },
    "focalnet_base_srf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_base_srf"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "depths": (2, 2, 18, 2),
            "embed_dim": 128,
        },
    },
    "focalnet_base_lrf": {
        "encoder": FocalNetEncoder,
        "pretrained_settings": pretrained_settings["focalnet_base_lrf"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "depths": (2, 2, 18, 2),
            "embed_dim": 128,
            "focal_levels": (3, 3, 3, 3),
        },
    },
}
