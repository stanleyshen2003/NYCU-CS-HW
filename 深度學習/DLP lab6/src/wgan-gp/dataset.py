import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image


class Iclevr_Dataset(Dataset):
    def __init__(self, mode='train'):
        self.object_dict = json.load(open('objects.json'))
        self.data = json.load(open('train.json'))
        # convert dict to list of list
        self.data = [['iclevr/'+keys, self.data[keys]] for keys in self.data.keys()]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0]).convert('RGB')
        image = self.transform(image)
        label = torch.zeros(24)
        for obj in self.data[idx][1]:
            label[self.object_dict[obj]] = 1
        return image, label
    
if __name__ == '__main__':
    dataset = Iclevr_Dataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[1][0].shape)