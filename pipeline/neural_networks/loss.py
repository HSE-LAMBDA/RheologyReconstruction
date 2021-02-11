import torch


class BaseLoss():

	def __init__(self, name: str):  self.name = name
	def __call__(self, preds, labels, weights): raise NotImplementedError


class BCELoss(BaseLoss):
	"""	
	Plain old binary cross entropy loss.
	"""

	def __init__(self): super().__init__("binary_cross_entropy")

	def __call__(self, preds, labels, weights):	
		"""
		:notes: 
			argument 'weights' here is redundant, and added just for code uniformity
		"""

		return torch.mean((-1) * (
    		labels * torch.log(torch.max(torch.ones_like(preds) * 1e-7, preds)) +\
    		(1. - labels) * torch.log(torch.max(torch.ones_like(preds) * 1e-7, 1. - preds))))


class weightedBCELoss(BaseLoss):
	"""
	BCE with weights.
	"""

	def __init__(self): super().__init__("weighted_binary_cross_entropy")

	def __call__(self, preds, labels, weights):

		return torch.mean((-1) * weights * (
	         labels * torch.log(torch.max(torch.ones_like(preds) * 1e-7, preds)) +\
	         (1. - labels) * torch.log(torch.max(torch.ones_like(preds) * 1e-7, 1. - preds))))


class focalLoss(BaseLoss):
	"""
	Plain version of focal loss.
	See https://arxiv.org/pdf/1708.02002.pdf.
	"""

	def __init__(self, focal_coeff): 
		super().__init__('focal_loss')
		self.focal_coeff = focal_coeff

	def __call__(self, preds, labels, weights):

		return torch.mean((-1) * (
		    (
		    	labels * torch.pow((1. - preds), focal_coeff) *\
		     	torch.log(torch.max(torch.ones_like(preds) * 1e-7, preds))
		    ) +\
		    (
		     	(1. - labels) * torch.pow(preds, focal_coeff) *\
		     	 torch.log(torch.max(torch.ones_like(preds) * 1e-7, 1. - preds))
		    )
		))


class weightedFocalLoss(BaseLoss):
 	"""
 	Weighted version of focal loss.
	See https://arxiv.org/pdf/1708.02002.pdf.
 	"""

 	def __init__(self, focal_coeff):
 		super().__init__('weighted_focal_loss')
 		self.focal_coeff = focal_coeff

 	def __call__(self, preds, labels, weights):

 		return torch.mean((-1) * weights * (
			(
		    	labels * torch.pow((1. - preds), focal_coeff) *\
		     	torch.log(torch.max(torch.ones_like(preds) * 1e-7, preds))
		    ) +\
		    (
		     	(1. - labels) * torch.pow(preds, focal_coeff) *\
		     	 torch.log(torch.max(torch.ones_like(preds) * 1e-7, 1. - preds))
		    )
		))


class weightedDiceLoss(BaseLoss):
	"""
    1. - Dice(X, Y, W). where X is the predicitions of the model,
    Y - ground truth, W - weight masks
    """

	def __init__(self): super().__init__('weighted_dice_loss')

	def __call__(self, preds, labels, weights):

		num = torch.sum(2. * weights * preds * labels)
		den = torch.sum(weights * (preds + labels)) + 1e-8

		return 1. - num / den