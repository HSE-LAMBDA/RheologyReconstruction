import numpy as np
import torch
from scipy.ndimage import sobel


class BaseMetric():

    def __init__(self, name: str):

        self.name = name

    def __call__(self):

        raise NotImplementedError


class Dice(BaseMetric):
    """
    Dice coefficient for binary classification problem over the batch of images

    :param preds: batch of model predictions of pixel classes
                   Shape: (N_batch, W_image, H_image)
    :type preds: np.ndarray

    :param labels: batch of correct pixel clasees for images 
                   Shape: (N_batch, W_image, H_image)
    :type labels: np.ndarray

    :param weights: a dummy weight matrix, not used for this metrics.
                    Added for code uniformity

    :params tolerance (default = 0.5): threshold value to consider a pixel to be in the class 1
    :type tolerance: float

    :returns: DICE - average Dice metrics over the batch
    """

    def __init__(self): 
        super().__init__("average_dice_score")
        self.EPS = 1e-10 # a constant for numerical stability

    @staticmethod
    def _dice(true, pred):

        true = true.astype(bool)
        pred = pred.astype(bool)

        intersection = (true & pred).sum()
        im_sum = true.sum() + pred.sum()

        return 2.0 * intersection / (im_sum + self.EPS)

    def __call__(self, batch, preds, tolerance=0.5):

        # TODO: rewrite to actual weights

        preds_l  = preds[0].data.cpu().numpy()
        preds_m  = preds[1].data.cpu().numpy()
        preds_r  = preds[2].data.cpu().numpy()

        ground_l = batch.mask.cpu().copy().data.numpy()
        ground_m = batch.mask.cpu().copy().data.numpy()
        ground_r = batch.mask.cpu().copy().data.numpy()


        AVG_DICE_l = 0.0
        empty_l    = 0.0

        for t, p in zip(ground_l, preds_l):
            if not np.sum(t): empty_l += 1.
            AVG_DICE_l += self._dice(t, p > tolerance)

        AVG_DICE_l = AVG_DICE_l / (preds_l.shape[0] - empty_l) if empty_l != preds_l.shape[0] else 0.0

        AVG_DICE_m = 0.0
        empty_m    = 0.0

        for t, p in zip(ground_m, preds_m):
            if not np.sum(t): empty_m += 1.
            AVG_DICE_m += self._dice(t, p > tolerance)

        AVG_DICE_m = AVG_DICE_m / (preds_m.shape[0] - empty_m) if empty_m != preds_m.shape[0] else 0.0

        AVG_DICE_r = 0.0
        empty_r    = 0.0

        for t, p in zip(ground_r, preds_r):
            if not np.sum(t): empty_r += 1.
            AVG_DICE_m += self._dice(t, p > tolerance)

        AVG_DICE_r = AVG_DICE_r / (preds_r.shape[0] - empty_r) if empty_r != preds_r.shape[0] else 0.0

        AVG_DICE = (AVG_DICE_l + AVG_DICE_m + AVG_DICE_r) / 3.

        return AVG_DICE


class WeightedDice(BaseMetric):

    """
    A weighted Dice coefficient for binary classification problem over the batch of images

    :param preds: batch of model predictions of pixel classes
                  Shape: (N_batch, W_image, H_image)
    :type preds: np.ndarray

    :param labels: batch of correct pixel clasees for images 
                   Shape: (N_batch, W_image, H_image)
    :type labels: np.ndarray

    :param weights: batch of weight matrices for images
                    Shape: (N_batch, W_image, H_image)
    :type weights: np.ndarray

    :params tolerance (default = 0.5): threshold value to consider a pixel to be in the class 1
    :type tolerance: float

    :returns: 
        DICE - average weighted Dice metrics over the batch
    """

    def __init__(self): 
        super().__init__("average_weighted_dice_score")
        self.EPS = 1e-10 # a constant for numerical stability
    
   
    @staticmethod
    def _weighted_dice(true, pred, weights):

        true = true.astype(bool)
        pred = pred.astype(bool)

        intersection = (weights * (true & pred)).sum()
        im_sum = (weights * true).sum() + (weights * pred).sum()

        return 2.0 * intersection / (im_sum + self.EPS)

    def __call__(self, preds, labels, weights, tolerance=0.5):

        # TODO: rewrite to actual weights an masks

        AVG_DICE = 0.0
        empty    = 0.0

        preds_l  = preds[0].data.cpu().numpy()
        preds_m  = preds[1].data.cpu().numpy()
        preds_r  = preds[2].data.cpu().numpy()

        ground_l = batch.mask.cpu().copy().data.numpy()
        ground_m = batch.mask.cpu().copy().data.numpy()
        ground_r = batch.mask.cpu().copy().data.numpy()

        weights_l = batch.weights.cpu().copy().data.numpy()
        weights_m = batch.weights.cpu().copy().data.numpy()
        weights_r = batch.weights.cpu().copy().data.numpy()


        AVG_DICE_l = 0.0
        empty_l    = 0.0

        for t, p, w in zip(ground_l, preds_l, weights_l):
            if not np.sum(t): empty_l += 1.
            AVG_DICE_l += self._weighted_dice(t, p > tolerance, w)

        AVG_DICE_l = AVG_DICE_l / (preds_l.shape[0] - empty_l) if empty_l != preds_l.shape[0] else 0.0

        AVG_DICE_m = 0.0
        empty_m    = 0.0

        for t, p in zip(ground_m, preds_m, weights_m):
            if not np.sum(t): empty_m += 1.
            AVG_DICE_m += self._weighted_dice(t, p > tolerance, w)

        AVG_DICE_m = AVG_DICE_m / (preds_m.shape[0] - empty_m) if empty_m != preds_m.shape[0] else 0.0

        AVG_DICE_r = 0.0
        empty_r    = 0.0

        for t, p in zip(ground_r, preds_r, weights_r):
            if not np.sum(t): empty_r += 1.
            AVG_DICE_m += self._weighted_dice(t, p > tolerance, w)

        AVG_DICE_r = AVG_DICE_r / (preds_r.shape[0] - empty_r) if empty_r != preds_r.shape[0] else 0.0

        AVG_DICE = (AVG_DICE_l + AVG_DICE_m + AVG_DICE_r) / 3.

        return AVG_DICE