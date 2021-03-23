import torch
import torch.nn as nn
from .modules import NormFactory, ActivationFactory
from .func_utils import constant


class SegNet_3Head(nn.Module):

    """
    See https://arxiv.org/abs/1511.00561 for base article
    :notes:
        Method 'forward' assumes fixed size input 

    """


    @staticmethod
    def create_encoder_section(
        inc, outc, 
        norm_generator, 
        activation_generator,
        dropout_rate
    ):

        return nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, dilation=2, padding=2),
                norm_generator(outc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(outc, outc, kernel_size=3, dilation=2, padding=2),
                norm_generator(outc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(outc, outc, kernel_size=3, dilation=2, padding=2),
                norm_generator(outc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(outc, outc, kernel_size=2, stride=2), # convpool
                norm_generator(outc),
                activation_generator(),
                nn.Dropout2d(dropout_rate)
            )

    @staticmethod
    def create_decoder_section(
        inc, outc, 
        norm_generator, 
        activation_generator,
        dropout_rate
    ):  

        return nn.Sequential(
                nn.ConvTranspose2d(inc, inc, kernel_size=2, stride=2), # upsample via transposed conv
                norm_generator(inc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(inc, inc, kernel_size=3, dilation=2, padding=2),
                norm_generator(inc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(inc, inc, kernel_size=3, dilation=2, padding=2),
                norm_generator(inc),
                activation_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(inc, outc, kernel_size=3, dilation=2, padding=2), 
                norm_generator(outc),    
                activation_generator(),
                nn.Dropout2d(dropout_rate)
            )

    def __init__(self):

        #TODO: parameters from config 
        super().__init__()

        self.input_size = 128

        encoder_channels = [64, 128, 256]
        decoder_channels = [256, 128, 64]
        dropout_rate     = 0.

        norm_layer = 'instance'
        activation = 'CELU'

        norm_factory   = NormFactory()
        norm_generator = lambda x: norm_factory(norm_layer, x)
        
        act_factory = ActivationFactory()
        def act_generator(): return act_factory(activation)

        # ----------------- Encoder -------------------

        encoder = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d((self.input_size, self.input_size)),
                nn.Conv2d(2, 64, kernel_size=3, dilation=2, padding=2), 
                norm_generator(64),
                act_generator()
            )
        ]
        

        for inc, outc in zip(encoder_channels[:-1], encoder_channels[1:]):
            encoder.append(
                self.create_encoder_section(inc, outc, norm_generator, act_generator, dropout_rate))
            
        self.encoder = nn.Sequential(*encoder)

        # ----------------- Decoder -------------------

        decoder = []

        for inc, outc in zip(decoder_channels[:-1], decoder_channels[1:]):
            decoder.append(
                self.create_decoder_section(inc, outc, norm_generator, act_generator, dropout_rate))

        self.decoder = nn.Sequential(*decoder)

        self.head_lambda = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.head_mu = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.head_rho = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        # TODO: check whether we require asserts, or 
        # resize via avg. pooling is good enough to use
        # assert h // 8 > 0
        # assert w // 8 > 0

        x = self.encoder(x)
        x = self.decoder(x)

        lmbda = self.head_lambda(x).view(-1, self.input_size, self.input_size)
        mu    = self.head_mu(x).view(-1, self.input_size, self.input_size)
        rho   = self.head_rho(x).view(-1, self.input_size, self.input_size)

        return lmbda, mu, rho