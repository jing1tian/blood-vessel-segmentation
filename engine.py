import os
import gc
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
from torch import nn
import torch.nn.functional as F

from dist_utils import is_main_process
from utils import MetricLogger, dice_coef, iou_coef


def train_one_epoch(model, optimizer, scheduler, dataloader, loss_fn, scaler, epoch, args):
    model.train()
    
    running_metric = MetricLogger(["loss", "dice_loss", "focal_loss", "boundary_loss", "custom_loss"])
    
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f'Epoch {epoch} | Train ', 
        ncols=250,
        disable=not is_main_process()
    )
    
    for step, data in pbar:         
        images = data["image"].to(dtype=torch.float).cuda()
        masks  = data["mask"].to(dtype=torch.float).cuda()
        dist_maps = data["dist_map"].to(dtype=torch.float).cuda() if "dist_map" in data else None
        
        batch_size = images.size(0)
        
        with torch.cuda.amp.autocast(enabled=True): # type: ignore
            y_pred = model(images)
            loss_dict = loss_fn(y_pred, masks, dist_maps)
            loss = loss_dict["loss"] / args.accumulation_steps
            
        scaler.scale(loss).backward()
    
        if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()
            
        if scheduler is not None:
            scheduler.step()

        running_metric.update(loss_dict, batch_size)
                
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        running_dict = running_metric.get_dict()
        pbar.set_postfix({
            "loss": running_dict["loss"],
            "dice_loss": running_dict["dice_loss"],
            "focal_loss": running_dict["focal_loss"],
            "boundary_loss": running_dict["boundary_loss"],
            "custom_loss": running_dict["custom_loss"],
            "lr": f'{current_lr:0.5f}',
            "gpu_mem": f'{mem:0.2f} GB'
        })    
        
    # reduce across gpus
    running_metric.synchronize_between_processes()
    logger.info(f"Epoch {epoch} training result\n"+running_metric.__str__())

    gc.collect()
    torch.cuda.empty_cache()
    
    
    
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, epoch, args):
    model.eval()
    
    running_metric = MetricLogger(["loss", "dice_loss", "focal_loss", "boundary_loss", "custom_loss", "iou", "dice"])
    
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc='Validating ', 
        ncols=250,
        disable=not is_main_process()
    )
    
    for step, data in pbar:        
        images = data["image"].to(dtype=torch.float).cuda()
        masks  = data["mask"].to(dtype=torch.float).cuda()
        dist_maps = data["dist_map"].to(dtype=torch.float).cuda() if "dist_map" in data else None
        
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=True): # type: ignore
            y_pred  = model(images)
            loss_dict = loss_fn(y_pred, masks, dist_maps)
        
        y_pred = y_pred.sigmoid()
        val_dice = dice_coef(masks, y_pred)
        val_iou = iou_coef(masks, y_pred)
        
        loss_dict["iou"] = val_iou.item()
        loss_dict["dice"] = val_dice.item()
        running_metric.update(loss_dict, batch_size)
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        running_dict = running_metric.get_dict()
        pbar.set_postfix({
            "loss": running_dict["loss"],
            "dice_loss": running_dict["dice_loss"],
            "focal_loss": running_dict["focal_loss"],
            "boundary_loss": running_dict["boundary_loss"],
            "custom_loss": running_dict["custom_loss"],
            "iou": running_dict["iou"],
            "dice": running_dict["dice"],
            "gpu_mem": f'{mem:0.2f} GB'
        })
                
    # reduce across gpus
    running_metric.synchronize_between_processes()
    logger.info(f"Epoch {epoch} validation result\n"+running_metric.__str__())
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return running_metric.get_value("iou", "global_avg")