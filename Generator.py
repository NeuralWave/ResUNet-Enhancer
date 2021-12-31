import torch
import torch.nn as nn
from Blocks import ResUNetConvBlock, Upsample

class Encoder(nn.Module):
    def __init__(self, filters, input_channels):
        super(Encoder, self).__init__()

        self.input_layer = nn.Sequential(nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1),
                                         nn.BatchNorm2d(filters[0]),
                                         nn.ReLU(),
                                         nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1))

        self.input_skip = nn.Sequential(nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1))

        self.conv_blocks = nn.ModuleList(
            [ResUNetConvBlock(filters[i], filters[i+1], first_conv_stride=2) for i in range(len(filters)-2)] # ultimos 2 son para bridge
        ) 

    def forward(self, x):
        x = self.input_layer(x) + self.input_skip(x)

        y = []
        y.append(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            y.append(x)

        return y


class Decoder(nn.Module):
    def __init__(self, filters):
        super(Decoder, self).__init__()

        self.upsample_blocks = nn.ModuleList(
            [Upsample(filters[i], filters[i]) for i in reversed(range(1, len(filters)))] 
        ) 

        self.conv_blocks = nn.ModuleList(
            [ResUNetConvBlock(filters[i] + filters[i-1], filters[i-1], first_conv_stride=1) for i in reversed(range(1, len(filters)))]
        ) 
    
    def forward(self, x, encoder_outputs):
        N_enc_out = len(encoder_outputs)
        for i in range(len(self.upsample_blocks)):
            x = self.upsample_blocks[i](x)
            enc_out = encoder_outputs[N_enc_out - 1 - i]

            x = torch.cat([x, enc_out], dim=1)
            x = self.conv_blocks[i](x)

        return x



class ResUNet(nn.Module):
    def __init__(self, hparams):
        super(ResUNet, self).__init__()

        input_channels = hparams.input_channels
        filters = hparams.filters

        self.encoder = Encoder(filters, input_channels)
        self.bridge = ResUNetConvBlock(filters[-2], filters[-1], first_conv_stride=2)
        self.decoder = Decoder(filters)
        
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=filters[0], out_channels=input_channels, kernel_size=1),
                                          nn.Tanh())

        print("Number of G parameters =", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        bridge_output = self.bridge(encoder_outputs[-1])
        decoder_output = self.decoder(bridge_output, encoder_outputs)

        y = self.output_layer(decoder_output)

        return y    
