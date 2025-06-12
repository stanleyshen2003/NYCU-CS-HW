import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

from evaluate import evaluate
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset


def train(args):
    # implement the training function here
    # extract arguments
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_n = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare data, model, and loggers
    train_writer = SummaryWriter('log/'+model_n+'/train_lr' + str(learning_rate))
    valid_writer = SummaryWriter('log/'+model_n+'/valid_lr' + str(learning_rate))
    train_dataset = load_dataset(data_path, mode='train')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = load_dataset(data_path, mode='valid')
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    if model_n == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(epochs):
        running_loss = 0.0
        for _, sample in enumerate(train_data_loader):
            image = sample['image'].to(device)
            mask = sample['mask'].to(device)
            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimize (I already flatten the mask in preprocessing)
            outputs = model(image)
            outputs = outputs.flatten(start_dim=1, end_dim=3)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_writer.add_scalar('training loss', running_loss / len(train_data_loader), epoch)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_data_loader)}')

        # I originally set it to 2, but eventually I found that I need to evaluate every epoch
        if epoch % 1 == 0:
            train_score = evaluate(model, train_data_loader, device)
            train_writer.add_scalar('dice score', train_score, epoch)
            print(f'Dice score: {train_score}')
            validate_score = evaluate(model, valid_data_loader, device)
            valid_writer.add_scalar('dice score', validate_score, epoch)
            print(f'Validation Dice score: {validate_score}')
        
        # save the last 5 models
        if(epoch >= epochs - 5):
            torch.save(model.state_dict(), f'../saved_models/{model_n}_{epoch}_{validate_score}.pth')
    print('Finished Training')


    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, default='unet', help='model to use', choices=['unet', 'resnet34_unet'])

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)