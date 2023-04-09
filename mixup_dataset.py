import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class Mixup_Dataset(Dataset):
    def __init__(self, data_dir, train, mixup, transform):
        self.data_dir = data_dir
        self.train = train
        self.mixup = mixup
        self.transform = transform
        self.data = []
        self.targets = []

        if self.train:
            for i in range(5):
                with open(data_dir + 'data_batch_' + str(i+1), 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
        else:
            with open(data_dir + 'test_batch', 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        # reshape and turn into the HWC format for PyTorch
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # create a one-hot label vector
        label = torch.zeros(10)
        label[self.targets[idx]] = 1.

        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)

        if self.mixup:
            mixup_idx = np.random.randint(0, len(self.data) - 1)
            mixup_label = torch.zeros(10)
            mixup_label[self.targets[mixup_idx]] = 1.

            mixup_image = Image.fromarray(self.data[mixup_idx])
            if self.transform:
                mixup_image = self.transform(mixup_image)

            # select a random number from the given beta distribution
            # and mixup the images accordingly
            alpha = 0.2
            lamb = np.random.beta(alpha, alpha)
            
            image = lamb * image + (1 - lamb) * mixup_image
            label = lamb * label + (1 - lamb) * mixup_label

        return image, label
