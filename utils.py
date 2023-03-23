import numpy as np
import matplotlib.pyplot as plt

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

def show_train_result(num_epochs, train_loss, train_acc, val_loss, val_acc):
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
    plt.show()