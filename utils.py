import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def reset_wieghts(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer {layer}')
            layer.reset_parameters()

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs, T=1, alpha=0.25):
    KD_loss = nn.KLDivLoss()(nn.functional.log_softmax(outputs/T, dim=1), 
                             nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha*T*T)
    KD_loss += nn.functional.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss

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
    plt.savefig(name + '.png')
    # plt.show()