from torch import nn

def ConvBlock(in_channels, out_channels, kernel_size, size):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    for i in range(1, size):
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG19(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG19, self).__init__()
        
        self.conv_block1 = ConvBlock(3, 64, 3, 2)
        self.conv_block2 = ConvBlock(64, 128, 3, 2)
        self.conv_block3 = ConvBlock(128, 256, 3, 4)
        self.conv_block4 = ConvBlock(256, 512, 3, 4)
        self.conv_block5 = ConvBlock(512, 512, 3, 4)

        self.linear = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, mode='train'):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.linear(x)
        if mode == 'train':
            return x
        return x.argmax(dim=1)
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)