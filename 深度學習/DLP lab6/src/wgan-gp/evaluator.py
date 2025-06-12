import torch
import torch.nn as nn
import torchvision.models as models
import json
from model import Generator
from PIL import Image
import numpy as np

'''===============================================================
1. Title:     

DL spring 2024 Lab6 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You may need to modify the checkpoint's path at line 40.
You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]
Images should be normalized with:
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

==============================================================='''


class evaluation_model():
    def __init__(self, seed=30):
        #modify the path to your own path
        self.mode = 'GAN'
        model_path = 'output_smaller_noise2/checkpoint/500/generator.pth'
        self.test_path = 'test.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        
        checkpoint = torch.load('./checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.to(self.device)
        self.resnet18.eval()
        self.classnum = 24
        self.object_dict = json.load(open('objects.json'))
        self.data = json.load(open(self.test_path))
        
        labels = torch.zeros(len(self.data), self.classnum)
        for i in range(len(self.data)):
            for obj in self.data[i]:
                labels[i][self.object_dict[obj]] = 1
        labels = labels.to(self.device)
        self.generator = Generator(noise_dim=4).to(self.device)
        self.generator.load_state_dict(torch.load(model_path))
        self.generator.eval()

        images = self.generate_images(labels)
        self.generate_grid(images)

        print(self.eval(images, labels))
        
    def generate_images(self, labels):
        if self.mode == 'GAN':
            noise = torch.randn(len(self.data), 4).to(self.device)
            images = self.generator(noise, labels).to(self.device)
            with torch.no_grad():
                images = self.generator(noise, labels)
        else:
            images = self.generator.generate_samples(4, 4, 4)
        return images
    
    def generate_grid(self, images):
        # create image grid for the 32 images
        import os
        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{self.mode}', exist_ok=True)
        image_grid = np.zeros((4*64, 8*64, 3))
        for i in range(4):
            for j in range(8):
                image_grid[i*64:(i+1)*64, j*64:(j+1)*64] = images[i*8+j].detach().cpu().numpy().transpose(1,2,0)
        image_grid = (image_grid + 1) / 2
        image_grid = (image_grid * 255).astype(np.uint8)
        image_grid = Image.fromarray(image_grid)
        image_grid.save(f'results/{self.mode}/{self.test_path}.png')
                
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc
        
        
if __name__ == '__main__':
    evaluation_model()
    