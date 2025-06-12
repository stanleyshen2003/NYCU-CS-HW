from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import torch
from dataloader import ButterflyMothLoader
from ResNet50 import ResNet50
import numpy as np
from VGG19 import VGG19

def evaluate(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total_size = 0
        correct_sum = 0
        for _, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model.forward(images, mode='evaluate')
            correct = (predictions == labels).sum().item()
            total_size += len(labels)
            correct_sum += correct
        acc = correct_sum/total_size

    return acc
            

def test(model_name='resnet50'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ButterflyMothLoader(root='dataset', mode='test')
    data = DataLoader(dataset, batch_size=32, shuffle=False)
    model_path = 'saved_model/' + model_name + '.pth'
    if model_name == 'resnet50':
        model = ResNet50().to(device)
    else:
        model = VGG19().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    acc = evaluate(model, data)
    print(f"Test Accuracy: {acc}")

def train():
    epochs = 50
    lr = 0.00001
    model_name = 'resnet'
    assert model_name in ['VGG19', 'resnet']
    load_name = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(0)
    dataset = ButterflyMothLoader(root='dataset', mode='train')
    validset = ButterflyMothLoader(root='dataset', mode='valid')
    testset = ButterflyMothLoader(root='dataset', mode='test')
    data = DataLoader(dataset, batch_size=32, shuffle=True)
    validdata = DataLoader(validset, batch_size=32, shuffle=True)
    testdata = DataLoader(testset, batch_size=32, shuffle=False)
    
    if model_name == 'resnet':
        model = ResNet50().to(device)
    else:
        model = VGG19().to(device)
    validate_writer = SummaryWriter('log/'+model_name+'/validate_'+str(lr))
    train_writer = SummaryWriter('log/'+model_name+'/train_'+str(lr))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    criterion = CrossEntropyLoss()
    
    if load_name is not None:
        model.load_state_dict(torch.load(load_name))
    print("Training")
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        loss_sum = 0
        for i, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Loss: {loss_sum}")
        train_writer.add_scalar('Loss', loss_sum, epoch)
        
        if epoch % 1 == 0:
            train_acc= evaluate(model, data)
            train_writer.add_scalar('Accuracy', train_acc, epoch)
            print(f"Train Accuracy: {train_acc}")
            eval_acc = evaluate(model, validdata)
            validate_writer.add_scalar('Accuracy', eval_acc, epoch)
            print(f"Valid Accuracy: {eval_acc}")
        if epoch >46:
            test_acc = evaluate(model, testdata)
            torch.save(model.state_dict(), 'model/'+model_name+'/'+str(epoch) +'_'+str(eval_acc)+ '_' + str(test_acc)+'.pth')
    torch.save(model.state_dict(), 'model.pth')
    print("Training Done")



if __name__ == "__main__":
    #train()
    print("Test Resnet50")
    test('resnet50')
    print("")
    print("Test VGG19")
    test('vgg19')