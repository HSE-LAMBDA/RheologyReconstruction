import torch
import torch.nn as nn

from .modules import NormFactory, ActivationFactory
from .func_utils import constant
from functools import partial
from collections import OrderedDict

import yaml


class SegNet_3Head_conf(nn.Module):

    """
    See https://arxiv.org/abs/1511.00561 for base article
    :notes:
        Method 'forward' assumes fixed size input 
    """


    @staticmethod
    def create_encoder_section(
        inc, 
        outc,
        make_conv, 
        make_norm, 
        make_act,
        dropout_rate
    ):

        return nn.Sequential(
            OrderedDict([
                ('conv1'   , make_conv(inc, outc)),
                ('norm1'   , make_norm(outc)),
                ('act1'    , make_act()),
                ('dropout1', nn.Dropout2d(dropout_rate)),
                ('conv2'   , make_conv(outc, outc)),
                ('norm2'   , make_norm(outc)),
                ('act2'    , make_act()),
                ('dropout2', nn.Dropout2d(dropout_rate)),
                ('conv3'   , make_conv(outc, outc)),
                ('norm3'   , make_norm(outc)),
                ('act3'    , make_act()),
                ('dropout3', nn.Dropout2d(dropout_rate)),
                ('convpool', nn.Conv2d(outc, outc, kernel_size=2, stride=2)), # convpool
                ('norm4'   , make_norm(outc)),
                ('act4'    , make_act()),
                ('dropout4', nn.Dropout2d(dropout_rate))
            ]))

    @staticmethod
    def create_decoder_section(
        inc, 
        outc,
        make_conv, 
        make_norm, 
        make_act,
        dropout_rate
    ):  

        return nn.Sequential(
            OrderedDict([
                ('upsample', nn.ConvTranspose2d(inc, inc, kernel_size=2, stride=2)), 
                ('norm1'   , make_norm(inc)),
                ('act1'    , make_act()),
                ('dropout1', nn.Dropout2d(dropout_rate)),
                ('conv1'   , make_conv(inc, inc)),
                ('norm2'   , make_norm(inc)),
                ('act2'    , make_act()),
                ('dropout2', nn.Dropout2d(dropout_rate)),
                ('conv2'   , make_conv(inc, inc)),
                ('norm3'   , make_norm(inc)),
                ('act3'    , make_act()),
                ('dropout3', nn.Dropout2d(dropout_rate)),
                ('conv3'   , make_conv(inc, outc)), 
                ('norm4'   , make_norm(outc)),    
                ('act4'    , make_act()),
                ('dropout4', nn.Dropout2d(dropout_rate))
            ]))

    def __init__(
        self,
        conv_type='regular',
        norm_layer='instance',
        activation='CELU',
        yaml_path = "RheologyReconstruction/pipeline/dolfin_adjoint/solver_config.yaml"
    ):

        assert conv_type in ('regular', 'dilated')
 
        super().__init__()

        self.input_size = 128
       
        with open(yaml_path, 'r') as stream:
            conf = yaml.safe_load(stream)
            
        encoder_channels = conf["Neural Network"]["encoder_channels"]
        decoder_channels = conf["Neural Network"]["decoder_channels"]
        dropout_rate = conf["Neural Network"]["dropout_rate"] 

        self.conv_type  = conv_type
        self.norm_layer = norm_layer
        self.activation = activation

        norm_factory   = NormFactory()
        make_norm      = partial(norm_factory, self.norm_layer)
        
        act_factory   = ActivationFactory()
        make_act      = partial(act_factory, self.activation)

        if self.conv_type == 'regular':
            make_conv = partial(nn.Conv2d, kernel_size=3, padding=1)
        else:
            make_conv = partial(nn.Conv2d, kernel_size=3, dilation=2, padding=2)

        make_e_block = partial(
            self.create_encoder_section, 
            make_conv=make_conv, make_act=make_act, make_norm=make_norm, dropout_rate=dropout_rate
        )

        make_d_block = partial(
            self.create_decoder_section, 
            make_conv=make_conv, make_act=make_act, make_norm=make_norm, dropout_rate=dropout_rate
        )

        # ----------------- Encoder -------------------

        self.adapter = nn.Sequential(OrderedDict([
            ('adaptive_pool', nn.AdaptiveAvgPool2d((self.input_size, self.input_size))),
            ('input', make_conv(2, encoder_channels[0])),
            ('norm', make_norm(encoder_channels[0])),
            ('activation', make_act())
        ]))
                
        encoder = []

        for j, (inc, outc) in enumerate(zip(encoder_channels[:-1], encoder_channels[1:])):
            encoder.append((f'block_{j+1}', make_e_block(inc, outc)))
            
        self.encoder = nn.Sequential(OrderedDict(encoder))

        # ----------------- Decoder -------------------

        decoder = []

        for j, (inc, outc) in enumerate(zip(decoder_channels[:-1], decoder_channels[1:])):
            decoder.append((f'block_{j + 1}', make_d_block(inc, outc)))

        self.decoder = nn.Sequential(OrderedDict(decoder))

        self.head_lambda = nn.Sequential(
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Dropout2d(dropout_rate),
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.head_mu = nn.Sequential(
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Dropout2d(dropout_rate),
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.head_rho = nn.Sequential(
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Dropout2d(dropout_rate),
            make_conv(decoder_channels[-1], decoder_channels[-1]),
            make_norm(decoder_channels[-1]),
            make_act(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        # TODO: check whether we require asserts, or 
        # resize via avg. pooling is good enough to use
        # assert h // 8 > 0
        # assert w // 8 > 0

        x = self.adapter(x)
        x = self.encoder(x)
        x = self.decoder(x)

        lmbda = self.head_lambda(x).view(-1, self.input_size, self.input_size)
        mu    = self.head_mu(x).view(-1, self.input_size, self.input_size)
        rho   = self.head_rho(x).view(-1, self.input_size, self.input_size)

        return lmbda, mu, rho

