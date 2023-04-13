import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
    print(torch.cuda.get_device_properties(device), '\n')
    return device

def mixup(img, label, alpha=0.2):
    lamb = np.random.beta(alpha, alpha)
    mixup_idx = torch.randperm(len(img))

    mixup_img = img[mixup_idx]
    mixup_label = label[mixup_idx]

    labels = F.one_hot(label).long()
    mixup_labels = F.one_hot(mixup_label).long()
    return lamb * img + (1 - lamb) * mixup_img, lamb * labels + (1 - lamb) * mixup_labels

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs, T=4, alpha=0.9):
    KD_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(outputs/T, dim=1), 
                             nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha*T*T)
    KD_loss += nn.functional.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_train_result(num_epochs, train_loss, train_acc, val_loss, val_acc, name):
    x_range = np.arange(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(x_range, train_loss, label='train loss')
    plt.plot(x_range, val_loss, label='validation loss')

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(x_range, train_acc, label='train acc')
    plt.plot(x_range, val_acc, label='validation acc')
    plt.legend()
    plt.savefig('./figures/' + name + '.png')
    # plt.show()