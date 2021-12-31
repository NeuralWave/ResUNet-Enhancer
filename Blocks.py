import torch.nn as nn

class ResUNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first_conv_stride):
        super(ResUNetConvBlock, self).__init__()

        self.conv_block = nn.Sequential(nn.BatchNorm2d(in_ch),
                                        nn.ReLU(),
                                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=first_conv_stride, padding=1),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(),
                                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))

        self.conv_skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=first_conv_stride, padding=1),
                                       nn.BatchNorm2d(out_ch))                            

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
        

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2):
        super(Upsample, self).__init__()

        # self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))   

    def forward(self, x):
        return self.upsample(x)