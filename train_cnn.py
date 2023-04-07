import os
import time
import torch
from data_loader import DataLoader
from models.simplenet import SimpleNet
from utils import *

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
print(torch.cuda.get_device_properties(device), '\n')

train_loader, val_loader, test_loader = DataLoader(batch_size=128, train_val_split=0.8)

model = SimpleNet().to(device)
# print(model)

num_epochs = 100
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_acc = 0.0
print('----- start training -----')
start = time.process_time()
train_total_loss, train_total_acc, val_total_loss, val_total_acc = [], [], [], []
for epoch in range(num_epochs):
    model.train()
    train_total = 0
    train_loss, train_correct = 0, 0
    for x, label in train_loader:
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x)

        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_total_loss.append(train_loss / len(train_loader))
    train_total_acc.append(100 * train_correct / train_total)

    val_total = 0
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            val_loss += loss_fn(outputs, target).item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

        val_total_loss.append(val_loss / len(val_loader))
        val_total_acc.append(100 * val_correct / val_total)

    scheduler.step()
    
    print('Epoch: {}/{}'.format(epoch+1, num_epochs))
    print('[Train] loss: {:.5f}, acc: {:.2f}%'.format(train_total_loss[-1], train_total_acc[-1]))
    print('[Val]   loss: {:.5f}, acc: {:.2f}%'.format(val_total_loss[-1], val_total_acc[-1]))

     # save checkpoint
    if val_total_acc[-1] > best_acc:
        state = {
            'model': model.state_dict(),
            'acc': val_total_acc[-1],
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/student_cpkt')
        best_acc = val_total_acc[-1]
        print('- New checkpoint -')

print(f'Traing time: {time.process_time() - start} s')

print('\nLoading best model...')
checkpoint = torch.load('./checkpoint/student_cpkt')
model.load_state_dict(checkpoint['model'])

with torch.no_grad():
    total, correct = 0, 0
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'\n[Test] Accuracy: {100 * correct / total}%')

show_train_result(num_epochs, train_total_loss, train_total_acc, val_total_loss, val_total_acc, 'SimpleNet')
