from torch.utils.data import Dataset as torchData
from glob import glob
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
import os
import torch.nn as nn

class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, C_size, partial=1.0):
        super().__init__()
        assert C_size[0] <= 64 and C_size[1] <= 64, "crop size is too large !!!"

        self.transform = transforms.Compose([
                transforms.ToTensor(), #Convert to tensor
                transforms.Normalize(mean=[0.4816, 0.4324, 0.3845],std=[0.2602, 0.2518, 0.2537]),# Normalize the pixel values
        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        #self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4868, 0.4341, 0.3844],std=[0.2620, 0.2527, 0.2543]),
                        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        from torch.utils.data import Dataset as torchData
from glob import glob
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
import os
import torch.nn as nn

class LoadTrainData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, C_size, partial=1.0):
        super().__init__()
        assert C_size[0] <= 64 and C_size[1] <= 64, "crop size is too large !!!"

        self.transform = transforms.Compose([
                transforms.ToTensor(), #Convert to tensor
                transforms.Normalize(mean=[0.4816, 0.4324, 0.3845],std=[0.2602, 0.2518, 0.2537]),# Normalize the pixel values
        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        #self.folder = glob(os.path.join(root + '/*.png'))
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Training Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
class LoadTestData(torchData):
    """Training Dataset Loader

    Args:
        root: Dataset Path
        partial: Only train pat of the dataset
    """

    def __init__(self, root, partial=1.0):
        super().__init__()

        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.4868, 0.4341, 0.3844],std=[0.2620, 0.2527, 0.2543]),
                        ])
        self.folder = sorted([os.path.join(root, file) for file in os.listdir(root)])
        self.partial = partial

    def __len__(self):
        return int(len(self.folder) * self.partial)
    
    @property
    def info(self):
        return f"\nNumber of Testing Data: {int(len(self.folder) * self.partial)}"
    
    def __getitem__(self, index):
        path = self.folder[index]
        return self.transform(imgloader(path))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        