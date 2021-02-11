import torch
import torch.nn as nn


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDropConv2d(torch.nn.Conv2d):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


class NormFactory():
    
    def __init__(self): self.keys = ['batch', 'layer', 'instance']

    def __call__(self, name, *args, **kwargs):

        if name not in self.keys: 
            raise ValueError(f'Invalid layer name: {name}'
                             f'Possible values are: {self.keys}')

        if name == 'batch'   : return nn.BatchNorm2d(*args, **kwargs)
        if name == 'layer'   : return nn.LayerNorm(*args, **kwargs)
        if name == 'instance': return nn.InstanceNorm2d(*args, **kwargs)


class ActivationFactory():

    def __init__(self): self.keys = ['ReLU', 'LeakyReLU', 'Tanh']

    def __call__(self, name, *args, **kwargs):

        if name not in self.keys:
            raise ValueError(f'Invalid activation name: {name}'
                             f'Possible values are: {self.keys}')

        if name == 'ReLU': return nn.ReLU(*args, **kwargs)
        if name == 'LeakyReLU'   : return nn.LeakyReLU(*args, **kwargs)
        if name == 'Tanh': return nn.Tanh(*args, **kwargs)





