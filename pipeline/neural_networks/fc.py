import torch
import torch.nn as nn

from .modules import NormFactory, ActivationFactory
from .func_utils import constant
from functools import partial
from collections import OrderedDict


class FC(nn.Module):

    def __init__(
        self,
        n_layers,
        n_hid,
        activation='Tanh',

    ):

        super().__init__()

        self.n_layers   = n_layers
        self.n_hid      = n_hid
        self.activation = activation
        self.input_size = 64

        act_factory   = ActivationFactory()
        make_act      = partial(act_factory, self.activation)
        
        self.adapter = nn.Sequential(OrderedDict([
            ('adaptive_pool', nn.AdaptiveAvgPool2d((self.input_size, self.input_size)))
        ]))
                
        fc_stack = []

        for j in range(self.n_layers):
            fc_stack.extend([
                (f'linear_{j+1}', nn.Linear(
                    2 * self.input_size ** 2 if j == 0 else self.n_hid, self.n_hid)),
                (f'act_{j+1}', make_act())
            ])
            
        self.fc_stack = nn.Sequential(OrderedDict(fc_stack))
        self.output   = nn.Linear(self.n_hid, self.input_size ** 2)
    
    def forward(self, x):

        x = self.adapter(x)
        x = x.flatten(start_dim=1, end_dim=3)
        x = self.fc_stack(x)
        
        lmbda = self.output(x).view(-1, self.input_size, self.input_size)

        return torch.sigmoid(lmbda)