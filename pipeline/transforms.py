import torch
import torch.nn.functional as F

class DifferentialTransform:
    """
    For each pixel computes l2 norm of numerical gradient 
    of intensity of a 2d tensor with 1 color channel, 
    which is equivalent to application of two filters
    Filter along x is: [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
    Filter along y us: [[0, 0, 0], [-1, 0, 1], [0, 0, 0]] 
    """

    def __init__(self):
        self.filter_x = torch.tensor(
            [[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]]
        ).float()
        self.filter_y = torch.tensor(
            [[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]]
        ).float() 

    def __call__(self, x):
        """
        :returns:
            2d tensor, each pixel of which contains l2 norm of the
            'gradient' of image in current pount
        """

        grad_x = F.conv2d(x, self.filter_x, padding=1)
        grad_y = F.conv2d(x, self.filter_y, padding=1)     
        res = torch.sqrt(grad_x ** 2 + grad_y ** 2) > 0
        return res.float() 