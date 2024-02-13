from ._base import EncoderMixin
from timm.models.convnext import ConvNeXt, LayerNorm2d
import torch
import torch.nn as nn
import torch.nn.functional as F

def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict

class ConvNeXtEncoder(ConvNeXt, EncoderMixin):
    def __init__(self, out_channels, depth=5, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels

        self.extra_stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], 3, 2, 1),
            nn.GroupNorm(32, out_channels[1]),
        )

        self.extra_norms = nn.ModuleList([
            nn.GroupNorm(32, dim) for dim in out_channels[2:]
        ])

        del self.head
        
    def get_stages(self):
        return [
            self.stages[0],
            self.stages[1],
            self.stages[2],
            self.stages[3],
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
            features.append(self.extra_norms[i](x))
                
        # [1, 2, 4, 8, 16, 32]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = checkpoint_filter_fn(state_dict, self)
        state_dict.pop("head.fc.bias", None)
        state_dict.pop("head.fc.weight", None)
        state_dict.pop("head.norm.weight", None)
        state_dict.pop("head.norm.bias", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)
        
        
convnext_weights = {
    "convnext_atto": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth",  # noqa
    },
    "convnext_atto_ols": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_ols_a2-78d1c8f3.pth",  # noqa
    },
    "convnext_femto": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth",  # noqa
    },
    "convnext_femto_ols": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_ols_d1-246bf2ed.pth",  # noqa
    },
    "convnext_pico": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth",  # noqa
    },
    "convnext_pico_ols": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_ols_d1-611f0ca7.pth",  # noqa
    },
    "convnext_nano": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth",  # noqa
    },
    "convnext_nano_ols": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_ols_d1h-ae424a9a.pth",  # noqa
    },
    "convnext_tiny": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth",  # noqa
    },
    "convnext_small": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth",  # noqa
    },
    "convnext_base": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth",  # noqa
    },
    "convnext_large": {
        "imagenet": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in convnext_weights.items():
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
        
timm_convnext_encoders = {
    "convnext_atto": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_atto"],
        "params": {
            "out_channels": (3, 40, 40, 80, 160, 320),
            "depths": (2, 2, 6, 2),
            "dims": (40, 80, 160, 320),
            "conv_mlp": True
        },
    },
    "convnext_atto_ols": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_atto_ols"],
        "params": {
            "out_channels": (3, 40, 40, 80, 160, 320),
            "depths": (2, 2, 6, 2),
            "dims": (40, 80, 160, 320),
            "conv_mlp": True,
            "stem_type": "overlap_tiered"
        },
    },
    "convnext_femto": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_femto"],
        "params": {
            "out_channels": (3, 48, 48, 96, 192, 384),
            "depths": (2, 2, 6, 2),
            "dims": (48, 96, 192, 384),
            "conv_mlp": True
        },
    },
    "convnext_femto_ols": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_femto_ols"],
        "params": {
            "out_channels": (3, 48, 48, 96, 192, 384),
            "depths": (2, 2, 6, 2),
            "dims": (48, 96, 192, 384),
            "conv_mlp": True,
            "stem_type": "overlap_tiered"
        },
    },
    "convnext_pico": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_pico"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "depths": (2, 2, 6, 2),
            "dims": (64, 128, 256, 512),
            "conv_mlp": True
        },
    },
    "convnext_pico_ols": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_pico_ols"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "depths": (2, 2, 6, 2),
            "dims": (64, 128, 256, 512),
            "conv_mlp": True,
            "stem_type": "overlap_tiered"
        },
    },
    "convnext_nano": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_nano"],
        "params": {
            "out_channels": (3, 80, 80, 160, 320, 640),
            "depths": (2, 2, 8, 2),
            "dims": (80, 160, 320, 640),
            "conv_mlp": True
        },
    },
    "convnext_nano_ols": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_nano_ols"],
        "params": {
            "out_channels": (3, 80, 80, 160, 320, 640),
            "depths": (2, 2, 8, 2),
            "dims": (80, 160, 320, 640),
            "conv_mlp": True,
            "stem_type": "overlap_tiered"
        },
    },
    "convnext_tiny": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_tiny"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (3, 3, 9, 3),
            "dims": (96, 192, 384, 768),
        },
    },
    "convnext_small": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_small"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (3, 3, 27, 3),
            "dims": (96, 192, 384, 768),
        },
    },
    "convnext_base": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_base"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "depths": (3, 3, 27, 3),
            "dims": (128, 256, 512, 1024),
        },
    },
    "convnext_large": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": pretrained_settings["convnext_large"],
        "params": {
            "out_channels": (3, 192, 192, 384, 768, 1536),
            "depths": (3, 3, 27, 3),
            "dims": (192, 384, 768, 1536),
        },
    },
}
