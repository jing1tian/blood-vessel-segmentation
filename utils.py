import os
import math
import random
import numpy as np
import sys
import inspect
from loguru import logger
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as eucl_distance

import torch
from torch import nn
import torch.distributed as dist
from timm.models.convnext import ConvNeXtBlock
from timm.models.swin_transformer import WindowAttention
from timm.models.inception_next import MetaNeXtBlock

from dist_utils import is_dist_avail_and_initialized, is_parallel, de_parallel
from segmentation_models_pytorch.base.modules import LayerNorm2d

def seed_everything(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
def draw_sample(image, mask, image_id, save_dir="."):
    image = image.permute((1, 2, 0)).numpy()*255.0
    image = image.astype('uint8')
    ch = image.shape[-1]
    image = image[..., ch//2]
    mask = (mask*255).numpy().astype('uint8')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='bone')
    plt.subplot(1,3,2)
    plt.imshow(mask[0])
    plt.subplot(1,3,3)
    plt.imshow(image, cmap='bone')
    plt.imshow(mask[0], alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{image_id}.png"))
    plt.close()
    
    
def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        # print(m)
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(
            m, (
                nn.modules.batchnorm._BatchNorm,
                nn.modules.normalization.GroupNorm,
                nn.modules.normalization.LayerNorm,
                nn.modules.instancenorm._InstanceNorm,
                LayerNorm2d
            )):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, ConvNeXtBlock):
            if m.gamma is not None:
                group_no_decay.append(m.gamma)
        elif isinstance(m, WindowAttention):
            if m.relative_position_bias_table is not None:
                group_no_decay.append(m.relative_position_bias_table)
        elif isinstance(m, MetaNeXtBlock):
            if m.gamma is not None:
                group_no_decay.append(m.gamma)
    
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
    
    
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(0, 1))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(0, 1))
    return iou


#https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py#L272C1-L273C1
def onehot2dist(seg, resolution=None, dtype=None) -> np.ndarray:
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask # type: ignore
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def binary2dist(seg, resolution=[1, 1]) -> np.ndarray:
    res = np.zeros_like(seg, dtype=np.float32)
    posmask = seg.astype(bool)
    if posmask.any():
        negmask = ~posmask
        res = eucl_distance(negmask, sampling=resolution) * negmask \
            - (eucl_distance(posmask, sampling=resolution) - 1) * posmask # type: ignore
    
    return res


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=None, format=None):
        if format is None:
            format = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.format = format

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
    def clear(self):
        self.count = 0
        self.total = 0.0
        self.deque.clear()
    
    @torch.no_grad()
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    @torch.no_grad()
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    @torch.no_grad()
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    @torch.no_grad()
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    @property
    @torch.no_grad()
    def max(self):
        if len(self.deque) == 0:
            return 0
        return max(self.deque)

    @property
    @torch.no_grad()
    def value(self):
        if len(self.deque) == 0:
            return 0
        return self.deque[-1]

    def __str__(self):
        return self.format.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        
        
class MetricLogger:
    def __init__(self, metrics, window_size=None, format=None):
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = AverageMeter(window_size=window_size, format=format)
            
    def update(self, metrics, count):
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                val = v.item() if not isinstance(v, float) else v
                self.metrics[k].update(val, count)
        else:
            for i, k in enumerate(self.metrics.keys()):
                val = metrics[i].item() if not isinstance(metrics[i], float) else metrics[i]
                self.metrics[k].update(val, count)
                
    def synchronize_between_processes(self):
        for metric in self.metrics.values():
            metric.synchronize_between_processes()
            
    def get_value(self, metric, name="global_avg"):
        if name == "global_avg":
            return self.metrics[metric].global_avg
        elif name == "median":
            return self.metrics[metric].median
        elif name == "avg":
            return self.metrics[metric].avg
        elif name == "max":
            return self.metrics[metric].max
        else:
            return self.metrics[metric].value
    
    def get_dict(self):
        return_dict = {}
        for k, v in self.metrics.items():
            return_dict[k] = v.__str__()
        return return_dict
    
    def __str__(self):
        result = "\n"
        for k, v in self.metrics.items():
            result += f"{k}: {v.global_avg:0.5f}\n"
        return result


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]
    

class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """
    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")