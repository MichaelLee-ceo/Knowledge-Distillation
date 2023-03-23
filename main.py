import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from model import TeacherNet
from utils import *

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
print(torch.cuda.get_device_properties(device), '\n')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset = torchvision.datasets.CIFAR10(root='./data', download=False, train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', download=False, train=False, transform=transform)

train_size = int(len(dataset) * 0.8)
val_size= len(dataset) - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

model = TeacherNet().to(device)
print(model)

num_epochs = 50
lr = 0.003
batch_size = 64
criterior = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

train_total_loss, train_total_acc, val_total_loss, val_total_acc = [], [], [], []
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0, 0
    for x, label in train_loader:
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)

        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == label).sum().item()
        
        loss = criterior(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_total_loss.append(train_loss / train_size)
    train_total_acc.append(train_correct / train_size)


    val_loss, val_correct = 0, 0
    with torch.no_grad():
        model.eval()
        for data, labels in val_loader:
            data, target = data.to(device), labels.to(device)
            outputs = model(data)
            val_loss += criterior(outputs, target).item()

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == target).sum().item()

        val_total_loss.append(val_loss / val_size)
        val_total_acc.append(val_correct / val_size)
    
    print(f'Epoch {epoch+1} / {num_epochs}')
    print(f'[Train] loss: {train_loss / train_size}, acc: {train_correct / train_size} %')
    print(f'[Val]   loss: {val_loss / val_size}, acc: {val_correct / val_size} %')


torch.save(model.state_dict(), './cifar_model.pth')
model.load_state_dict(torch.load('./cifar_model.pth'))

with torch.no_grad():
    correct = 0
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print(f'\n[Test] Accuracy: {100 * correct / len(testset)}%')

show_train_result(num_epochs, train_total_loss, train_total_acc, val_total_loss, val_total_acc)