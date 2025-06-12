# Implement your UNet model here
import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        output = self.down(x)
        return output, x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_x):
        x = self.upconv(x)
        # crop skip_x to match x
        diff = skip_x.size()[3] - x.size()[3]
        skip_x = skip_x[:, :, diff // 2: x.size()[3] + diff // 2, diff // 2: x.size()[3] + diff // 2]
        x = torch.cat((x, skip_x), dim=1)
        return self.block(x)




class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encode1 = EncoderBlock(in_channels, 64)
        self.encode2 = EncoderBlock(64, 128)
        self.encode3 = EncoderBlock(128, 256)
        self.encode4 = EncoderBlock(256, 512)

        self.middle = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decode4 = DecoderBlock(1024, 512)
        self.decode3 = DecoderBlock(512, 256)
        self.decode2 = DecoderBlock(256, 128)
        self.decode1 = DecoderBlock(128, 64)

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encode1, skip1 = self.encode1(x)
        encode2, skip2 = self.encode2(encode1)
        encode3, skip3 = self.encode3(encode2)
        encode4, skip4 = self.encode4(encode3)

        middle = self.middle(encode4)

        decode4 = self.decode4(middle, skip4)
        decode3 = self.decode3(decode4, skip3)
        decode2 = self.decode2(decode3, skip2)
        decode1 = self.decode1(decode2, skip1)

        output = self.conv_out(decode1)
        return output
