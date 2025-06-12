import argparse
import numpy as np
import os
import torch
from PIL import Image

from evaluate import evaluate
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset


def set_model(args):
    model_name = args.model_path.split('_')[-4]    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'ResNet34':
        model = ResNet34_UNet(in_channels=3, out_channels=1)
    else:
        model = UNet(in_channels=3, out_channels=1)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    return model

def load_preprocess(image_path):
    data = Image.open(image_path).convert("RGB")
    data = np.array(data.resize((256, 256), Image.BILINEAR))
    data = (np.moveaxis(data, -1, 0) / 255.0).astype(np.float32)
    return data

def predict(model, data):
    prediction = model(data).cpu().detach().numpy().reshape(256, 256)
    prediction = prediction > 0.5
    return prediction

def visualize(data, mask):
    data = data.squeeze(0)
    data = np.transpose(data, (1, 2, 0))
    mask = np.stack([mask, mask, mask], axis=-1)
    mask = mask * 255
    data = data * 255
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    data = Image.fromarray((data).astype(np.uint8))
    new_img = Image.blend(data, mask, 0.5)
    return new_img

def inference_txt(args):
    path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_dataset(path, mode='test')
    dataset = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    model = set_model(args)
    acc = evaluate(model, dataset, device)
    print(f'Dice score: {acc}')
    
    # I need the file names to save the images
    csv_path = path + '/annotations/test.txt'
    with open(csv_path) as f:
        split_data = f.read().strip("\n").split("\n")
    filenames = [x.split(" ")[0] for x in split_data]
    
    os.makedirs('inferenced_image', exist_ok=True)
    for file in filenames:
        complete_path = path + '/images/' + file + '.jpg'
        data = load_preprocess(complete_path)
        data = torch.tensor(data).unsqueeze(0)
        data = data.to(device)
        prediction = predict(model, data)
        new_img = visualize(data.cpu().numpy(), prediction)
        new_img.save('inferenced_image/'+ file + "_mask.png")

def inference_png(args):
    path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = set_model(args)
    
    os.makedirs('inferenced_image', exist_ok=True)
    data = load_preprocess(path)
    data = torch.tensor(data).unsqueeze(0)
    data = data.to(device)
    prediction = predict(model, data)
    save_name = path.split('/')[-1].split('.')[0]
    new_img = visualize(data.cpu().numpy(), prediction)
    new_img.save('inferenced_image/'+ save_name + "_mask.png")
        
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_path', default='../saved_models/DL_Lab3_UNet_110705013_沈昱宏.pth', help='path to the stored model weight', choices=['../saved_models/DL_Lab3_UNet_110705013_沈昱宏.pth', '../saved_models/DL_Lab3_ResNet34_UNet_110705013_沈昱宏.pth'])
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', help='path to the input data')
    parser.add_argument('--mode', '-m', type=str, default='txt', help='txt or png', choices=['txt', 'png'])
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'txt':
        inference_txt(args)
    else:
        inference_png(args)
    