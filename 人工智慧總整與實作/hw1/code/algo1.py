import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class CnnNet(nn.Module):
    def __init__(self, num_classes=5, init_weights=True):
        super(CnnNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        self.class_nn = nn.Sequential(nn.Linear(1024, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        C_value = self.class_nn(x)
        C_value = torch.squeeze(C_value)
        return F.softmax(C_value, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

def evaluate(model, dataloader):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == torch.max(targets, 1)[1]).sum().item()
    return correct / total

class CustomDataset(Dataset):
    def __init__(self, root_dir, folds, size = 100):
        self.folds = folds
        self.root_dir = root_dir
        self.classes = ['traffic_light', 'zebra_crossing', 'bus', 'double_yellow_line', 'nothing']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self.get_images(size)

    def get_images(self, size):
        images = []
        image_name = []
        for i in self.folds:
            for j in range(int(i*size/5), int((i+1)*size/5)):
                image_name.append(str(j+1).zfill(3)+'.png')
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            for img in image_name:
                img_path = os.path.join(class_dir, img)
                image = Image.open(img_path)
                y = np.zeros(len(self.classes))
                y[self.class_to_idx[cls_name]] = 1
                images.append((np.array(image).transpose((2,0,1)), y))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        return image, label

if __name__ == "__main__":
    # set some parameters
    epochs = 20000
    batch_size = 16
    lr = 0.00005
    fold_n = 1
    dataset_size = 100

    dataset = CustomDataset(root_dir='data/dataset/train/',folds=[0,1,2,3], size = dataset_size)
    dataset_val = CustomDataset(root_dir='data/dataset/test/', folds=[4], size = dataset_size)
    dataset_test = CustomDataset(root_dir='data/dataset/test/', folds=[4], size = dataset_size)
    # create tensorboard writer
    writer = SummaryWriter()

    # create dataLoader    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # create model
    net = CnnNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    total_samples = len(dataset)
    n_iterations = np.ceil(total_samples / 4)

    # training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (features, targets) in enumerate(dataloader):
            # forward to model
            outputs = net(features)
            # calculate loss
            loss = criterion(outputs, targets)
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss)
            running_loss += loss.item()

        writer.add_scalar("Loss/train", running_loss, epoch)
        if(epoch % 100 == 0):
            print(f'epoch {epoch}/{epochs}')
            print(running_loss)
            train_eval = evaluate(net, dataloader)
            print("Accuracy: ", train_eval)
            writer.add_scalar("Acc/train", train_eval, epoch)
            val_eval = evaluate(net, dataloader_val)
            print("Validation Accuracy: ", val_eval)
            writer.add_scalar("Acc/validation", val_eval, epoch)
    
    test_eval = evaluate(net, dataloader_test)
    print("test accuracy: ", test_eval)
    torch.save(net.state_dict(), f"model_all_val_{val_eval}_test_{test_eval}.pth")
            



                