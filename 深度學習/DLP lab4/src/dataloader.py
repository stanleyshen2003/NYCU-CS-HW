
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        
        # get a random seed
        #new_seed = torch.randint(0, 100000, (1,)).item()
        
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('/')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[(index*self.video_len)+i]
            label_name = '/'.join(label_list)
            #torch.manual_seed(new_seed)
            imgs.append(self.transform(imgloader(img_name)))
            #torch.manual_seed(new_seed)
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)
    
    

