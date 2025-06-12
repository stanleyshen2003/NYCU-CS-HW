import numpy as np
import torch

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    '''
    pred_mask: torch.Tensor, shape=(1, 256, 256), predicted mask
    gt_mask: torch.Tensor, shape=(1, 256, 256), ground truth mask
    '''
    with torch.no_grad():
        sum = 0
        pred_mask = pred_mask > 0.5
        pred_mask = pred_mask.cpu().numpy().astype(np.float32)
        gt_mask = gt_mask.cpu().numpy().astype(np.float32)
        pred = pred_mask.flatten()
        gt = gt_mask
        # intersect = pixel level multiplication
        intersection = np.sum(pred * gt)
        # area of mask = sum of all pixels
        area_pred = np.sum(pred)
        area_gt = np.sum(gt)
        # dice score formula
        sum += 2 * intersection / (area_pred + area_gt)
    return sum