import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

from surface_distance import create_table_neighbour_code_to_surface_area

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def boundary_loss(inputs, dist_maps=None):
    if dist_maps is None:
        loss = 0.0
    else:
        prob = inputs.sigmoid()
        loss = torch.einsum("bkwh,bkwh->bkwh", prob, dist_maps).mean()
    return loss

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        power = 2**np.arange(0, 8).reshape(1, 1, 2, 2, 2).astype(np.float32)
        area = create_table_neighbour_code_to_surface_area((1, 1, 1)).astype(np.float32)
        self.power = nn.Parameter(torch.from_numpy(power), requires_grad=False)
        self.kernel = nn.Parameter(torch.ones(1, 1, 2, 2, 2), requires_grad=False)
        self.area = nn.Parameter(torch.from_numpy(area), requires_grad=False)
        
    def forward(self, preds, targets, eps=1e-5):
        """
        preds: tensor of shape [bs, 1, d, h, w]
        targets: tensor of shape [bs, 1, d, h, w]
        """
        bsz = preds.shape[0]

        # voxel logits to cube logits
        foreground_probs = F.conv3d(F.logsigmoid(preds), self.kernel).exp().flatten(1)
        background_probs = F.conv3d(F.logsigmoid(-preds), self.kernel).exp().flatten(1)
        surface_probs = 1 - foreground_probs - background_probs

        # ground truth to neighbour code
        with torch.no_grad():
            cubes_byte = F.conv3d(targets, self.power).to(torch.int32)
            gt_area = self.area[cubes_byte.reshape(-1)].reshape(bsz, -1)
            gt_foreground = (cubes_byte == 255).to(torch.float32).reshape(bsz, -1)
            gt_background = (cubes_byte == 0).to(torch.float32).reshape(bsz, -1)
            gt_surface = (gt_area > 0).to(torch.float32).reshape(bsz, -1)
        
        # dice
        foreground_dice = (2*(foreground_probs*gt_foreground).sum(-1)+eps) / (foreground_probs.sum(-1)+gt_foreground.sum(-1)+eps)
        background_dice = (2*(background_probs*gt_background).sum(-1)+eps) / (background_probs.sum(-1)+gt_background.sum(-1)+eps)
        surface_dice = (2*(surface_probs*gt_area).sum(-1)+eps) / (((surface_probs+gt_surface)*gt_area).sum(-1)+eps)
        dice = (foreground_dice + background_dice + surface_dice) / 3
        return 1 - dice.mean()

class Loss(nn.Module):
    def __init__(
        self, 
        focal_coef=1.0, 
        dice_coef=1.0, 
        boundary_coef=0.01, 
        custom_loss_coef=1.0, 
        focal_alpha=0.25,
        focal_gamma=2,
    ):
        super().__init__()
        self.focal_coef = focal_coef
        self.dice_coef = dice_coef
        self.boundary_coef = boundary_coef
        self.custom_loss_coef = custom_loss_coef
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if custom_loss_coef > 0:
            self.custom_loss = CustomLoss()

    def forward(self, preds, targets, dist_maps=None):
        b = preds.shape[0]
        loss_dict = {}
        loss = 0.0

        if self.focal_coef > 0:
            loss_dict["focal_loss"] = self.focal_coef * sigmoid_focal_loss(preds, targets, b, self.focal_alpha, self.focal_gamma)
            loss += loss_dict["focal_loss"]

        if self.dice_coef > 0:
            loss_dict["dice_loss"] = self.dice_coef * dice_loss(preds, targets, b)
            loss += loss_dict["dice_loss"]

        if self.boundary_coef > 0 and dist_maps is not None:
            loss_dict["boundary_loss"] = self.boundary_coef * boundary_loss(preds, dist_maps)
            loss += loss_dict["boundary_loss"]

        if self.custom_loss_coef > 0:
            loss_dict["custom_loss"] = self.custom_loss_coef * self.custom_loss(preds.unsqueeze(1), targets.unsqueeze(1))
            loss += loss_dict["custom_loss"]

        loss_dict["loss"] = loss

        return loss_dict