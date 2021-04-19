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
        self.EPS = 1e-10  # a constant for numerical stability

    @staticmethod
    def _dice(true, pred):

        true = true.astype(bool)
        pred = pred.astype(bool)

        intersection = (true & pred).sum()
        im_sum = true.sum() + pred.sum()

        return 2.0 * intersection / (im_sum + self.EPS)

    def __call__(self, preds, labels, weights, tolerance=0.5):

        AVG_DICE = 0.0
        empty = 0.0

        for t, p in zip(labels, preds):
            if not np.sum(t): empty += 1.
            AVG_DICE += self._dice(t, p > tolerance)

        return AVG_DICE / (preds.shape[0] - empty) if empty != preds.shape[0] else 0.0


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
        self.EPS = 1e-10  # a constant for numerical stability

    @staticmethod
    def _weighted_dice(true, pred, weights):

        true = true.astype(bool)
        pred = pred.astype(bool)

        intersection = (weights * (true & pred)).sum()
        im_sum = (weights * true).sum() + (weights * pred).sum()

        return 2.0 * intersection / (im_sum + self.EPS)

    def __call__(self, preds, labels, weights, tolerance=0.5):

        AVG_DICE = 0.0
        empty = 0.0

        for t, p, w in zip(labels, preds, weights):
            if not np.sum(t): empty += 1.
            AVG_DICE += self._weighted_dice(t, p > tolerance, w)

        return AVG_DICE / (preds.shape[0] - empty) if empty != preds.shape[0] else 0.0
