import torch
from torch.utils.data import DataLoader
import numpy as np

from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from oxford_pet import load_dataset
from utils import dice_score

def evaluate(net, data, device):
    # implement the evaluation function here
    '''
    input: model, dataloader, device
    output: average dice score
    
    loop through the data and calculate the dice score for each image
    '''
    score = 0
    pic_num = 0
    with torch.no_grad():
        for _, sample in enumerate(data):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            outputs = net(image)
            for i in range(outputs.shape[0]):
                score += dice_score(outputs[i], mask[i])
            pic_num += outputs.shape[0]
    return score / pic_num


if __name__ == '__main__':
    dataset = load_dataset('../dataset/oxford-iiit-pet', 'test')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    model_name = 'unet'
    model_path = '../saved_models/'+model_name+'.pth'
    
    if model_name == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif model_name == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1)
        
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    score = evaluate(model, dataloader, device)
    print(score)
        