import os
import wandb
import time
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from models.simplenet import SimpleNet, SimpleMLP
from utils import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Implementation of Catastrophic Forgetting')
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--sequential', default=True, type=bool)
args = parser.parse_args()

name = "dataset_{}-sequential_{}".format(args.dataset, args.sequential)

wandb.init(
    project="Catastrophic-Forgetting",
    config=vars(args),
    name=name,
)

device = getDevice()
model = SimpleNet().to(device)
# model = SimpleMLP(input_dim=784).to(device)

wandb.watch(model)

dataset = args.dataset
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if dataset == "mnist":
    dataset = torchvision.datasets.MNIST(root='./data', download=False, train=True, transform=torchvision.transforms.ToTensor())
elif dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root='./data', download=False, train=True, transform=torchvision.transforms.ToTensor())

train_set_0, val_set_0 = getDataset(dataset, label=0)
train_set_1, val_set_1 = getDataset(dataset, label=1)

trainset_all = ConcatDataset([train_set_0, train_set_1])
trainLoader = DataLoader(trainset_all, batch_size=batch_size, shuffle=True)


trainLoader_0 = DataLoader(train_set_0, batch_size=batch_size, shuffle=True)
valLoader_0 = DataLoader(val_set_0, batch_size=batch_size, shuffle=True)
trainLoader_1 = DataLoader(train_set_1, batch_size=batch_size, shuffle=True)
valLoader_1 = DataLoader(val_set_1, batch_size=batch_size, shuffle=True)


start = time.process_time()
for epoch in range(num_epochs):
    model.train()
    train_total = 0
    train_loss, train_correct = 0, 0
    if args.sequential:
        if epoch < int(num_epochs / 2):
            train_acc = train(model, trainLoader_0, optimizer, device)
            wandb.log({'Train acc 1': train_acc, 'Epoch': epoch})

        else:
            train_acc = train(model, trainLoader_1, optimizer, device)
            wandb.log({'Train acc 2': train_acc, 'Epoch': epoch})
    else:
        train_acc = train(model, trainLoader, optimizer, device)
        wandb.log({'Train total': train_acc, 'Epoch': epoch})
        

    val_acc_0 = evaluate(model, valLoader_0, device)
    val_acc_1 = evaluate(model, valLoader_1, device)
    wandb.log({'Val acc 1': val_acc_0, 'Val acc 2': val_acc_1, 'Epoch': epoch})

    print('Epoch: {}/{} | Val 1: {:.2f}, Val 2: {:.2f}'.format(epoch+1, num_epochs, val_acc_0, val_acc_1))