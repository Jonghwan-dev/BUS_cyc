import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import binary_erosion

def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def get_boundary(mask):
    eroded = binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    return boundary.astype(np.uint8)

def hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_bd = get_boundary(pred)
    gt_bd = get_boundary(gt)
    pred_pts = np.argwhere(pred_bd)
    gt_pts = np.argwhere(gt_bd)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    dists_pred_to_gt = np.min(np.linalg.norm(pred_pts[:, None] - gt_pts[None, :], axis=2), axis=1)
    dists_gt_to_pred = np.min(np.linalg.norm(gt_pts[:, None] - pred_pts[None, :], axis=2), axis=1)
    hd95_val = np.percentile(np.hstack([dists_pred_to_gt, dists_gt_to_pred]), 95)
    return hd95_val

def hd95_batch(preds, targets):
    hd95s = []
    for pred, target in zip(preds, targets):
        pred_bin = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        target_bin = (target.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        val = hd95(pred_bin, target_bin)
        if not np.isnan(val):
            hd95s.append(val)
    return np.mean(hd95s) if hd95s else np.nan