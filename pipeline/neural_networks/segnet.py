import torch
import torch.nn as nn
from .modules import NormFactory, ActivationFactory
from .func_utils import constant


class SegNet(nn.Module):

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

    @constant
    @staticmethod
    def dim_x(): return 1024

    @constant
    @staticmethod
    def dim_y(): return 64 


    def __init__(self):

        #TODO: parameters from markup language
	  
        super().__init__()

        # hyperparameter


        encoder_channels = [64, 128, 256]
        decoder_channels = [256, 128, 64]
        dropout_rate     = 0.15

        norm_layer = 'batch'
        activation = 'ReLU'

        norm_factory   = NormFactory()
        norm_generator = lambda x: norm_factory(norm_layer, x)
        
        act_factory = ActivationFactory()
        def act_generator(): return act_factory(activation)

        # ----------------- Encoder -------------------

        encoder = [
            nn.Sequential(
                nn.Conv2d(1, encoder_channels[0],   kernel_size=3, dilation=2, padding=2),
                norm_generator(encoder_channels[0]),
                act_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, dilation=2, padding=2),
                norm_generator(encoder_channels[0]),
                act_generator(),
                nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, dilation=2, padding=2),
                norm_generator(encoder_channels[0]),
                act_generator(),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=(16, 1), stride=(16, 1)),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
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
        

        decoder.append(nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1], kernel_size=2, stride=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, dilation=2, padding=2),
            norm_generator(decoder_channels[-1]),
            act_generator(),
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        ))

        self.decoder = nn.Sequential(*decoder)
        
    
    def forward(self, x):

        return self.decoder(self.encoder(x.unsqueeze(dim=1))).view(-1, 128, 128)