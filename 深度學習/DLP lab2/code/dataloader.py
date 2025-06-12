import pandas as pd
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision.transforms import v2
import torch
import numpy as np

def getData(mode, root):
    df = pd.read_csv(root + '/' + mode + '.csv')
    path = df['filepaths'].tolist()
    label = df['label_id'].tolist()
    return path, label

class ButterflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode, root)
        self.mode = mode
        # save the images and labels to avoid loading them every time
        self.images = []
        self.labels = []
        for index in range(len(self.img_name)):
            img = Image.open(self.root + '/' + self.img_name[index])
            label = self.label[index]
            # RGB, resize, /255.0, transpose
            img = img.convert('RGB')
            img = img.resize((224, 224))
            self.images.append(img)
            self.labels.append(label)
            
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img = self.images[index]
        
        if self.mode == 'train':
            flip = np.random.randint(0, 2)
            rotate_degree = np.random.randint(0, 45)
            rotate_degree = np.random.choice([rotate_degree, -rotate_degree])
            img = img.rotate(rotate_degree)
            if flip == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        
        return img, self.labels[index]
