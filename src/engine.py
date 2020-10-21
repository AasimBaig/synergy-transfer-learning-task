from tqdm import tqdm
import torch
import config
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable


def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0.0
    fin_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for i, (image, label) in tqdm(enumerate(data_loader)):
        if torch.cuda.is_available():
            image = Variable(image.cuda())
            label = Variable(label.cuda())
        else:
            image = Variable(image)
            label = Variable(label)

        # forward
        score = model(image)
        loss = criterion(score, label)

        # backward
        optimizer.zero_grad()

        _, pred = torch.max(score.data, 1)
        loss.backward()

        # gradient descent <3
        optimizer.step()

        fin_loss += loss.item()

        fin_acc += torch.sum(pred == label)
    return (fin_loss / len(data_loader)), ((fin_acc) / len(data_loader))


def check_accuracy(data_loader):
    model = torch.load("model.pth")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in tqdm(enumerate(data_loader)):
            if torch.cuda.is_available():
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(
        f"\nGot {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}\n")


def save_checkpoint(model, filename="my_checkpoint.bin"):
    print("=> Saving checkpoint")
    PATH = 'model.pth'
    torch.save(model, PATH)
