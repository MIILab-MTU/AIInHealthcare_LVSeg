import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
def metric(gt,pred):
    preds = pred.detach().numpy()
    gts = gt.detach().numpy()
    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    smooth = 0.001
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)
    return dice
