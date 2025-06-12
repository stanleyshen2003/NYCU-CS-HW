import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, base_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(base_channels, base_channels*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != base_channels*4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, base_channels*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels*4)
            )
            
    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            x_in = self.downsample(x_in)
        x += x_in
        x = self.relu3(x)
        return x
    

class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_amount = [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, self.block_amount[0])
        self.layer2 = self._make_layer(128, self.block_amount[1], stride=2)
        self.layer3 = self._make_layer(256, self.block_amount[2], stride=2)
        self.layer4 = self._make_layer(512, self.block_amount[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def _make_layer(self, base_channel, blocks, stride=1):
        layers = []
        '''
        first block 
            input channel -> last base_channel * 4
            base channel -> base_channel
            output channel -> base_channel * 4
        other blocks
            input channel -> base_channel * 4
            base channel -> base_channel
            output channel -> base_channel * 4
        '''
        layers.append(BottleneckBlock(self.in_channels, base_channel, stride))
        self.in_channels = base_channel*4
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.in_channels, base_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x, mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = self.softmax(x)
        if mode == 'train':
            return x
        return x.argmax(dim=1)