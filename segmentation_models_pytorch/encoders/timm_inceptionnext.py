from ._base import EncoderMixin
from timm.models.inception_next import MetaNeXt, InceptionDWConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d

class MetaNeXtEncoder(MetaNeXt, EncoderMixin):
    def __init__(self, out_channels, depth=5, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = in_channels
        
        self.extra_stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], 3, 2, 1),
            LayerNorm2d(out_channels[1])
        )

        self.extra_norms = nn.ModuleList([
            LayerNorm2d(dim) for dim in out_channels[2:]
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
        state_dict.pop("head.fc1.bias", None)
        state_dict.pop("head.fc1.weight", None)
        state_dict.pop("head.fc2.bias", None)
        state_dict.pop("head.fc2.weight", None)
        state_dict.pop("head.norm.weight", None)
        state_dict.pop("head.norm.bias", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)
        
        
inceptionnext_weights = {
    "inception_next_tiny": {
        "imagenet": "https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth",  # noqa
    },
    "inception_next_small": {
        "imagenet": "https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth",  # noqa
    },
    "inception_next_base": {
        "imagenet": "https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in inceptionnext_weights.items():
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
        
timm_inceptionnextnext_encoders = {
    "inception_next_tiny": {
        "encoder": MetaNeXtEncoder,
        "pretrained_settings": pretrained_settings["inception_next_tiny"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (3, 3, 9, 3),
            "dims": (96, 192, 384, 768),
            "token_mixers": InceptionDWConv2d,
        },
    },
    "inception_next_small": {
        "encoder": MetaNeXtEncoder,
        "pretrained_settings": pretrained_settings["inception_next_small"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "depths": (3, 3, 27, 3),
            "dims": (96, 192, 384, 768),
            "token_mixers": InceptionDWConv2d,
        },
    },
    "inception_next_base": {
        "encoder": MetaNeXtEncoder,
        "pretrained_settings": pretrained_settings["inception_next_base"],
        "params": {
            "out_channels": (3, 128, 128, 256, 512, 1024),
            "depths": (3, 3, 27, 3),
            "dims": (128, 256, 512, 1024),
            "token_mixers": InceptionDWConv2d,
        },
    },
}
