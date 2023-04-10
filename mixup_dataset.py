import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset, ratio=1, mixup=False):
        if ratio == 1:
            self.dataset = dataset
        else:
            num_samples = int(len(dataset) * ratio)
            indices = torch.randperm(len(dataset))[:num_samples]
            self.dataset = torch.utils.data.Subset(dataset, indices)
        self.mixup = mixup

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # create a one-hot label vector
        label = torch.zeros(10)
        label[self.dataset[idx][1]] = 1.
        image = self.dataset[idx][0]
        
        if self.mixup:
            mixup_idx = np.random.randint(0, len(self.dataset) - 1)

            mixup_label = torch.zeros(10)
            mixup_label[self.dataset[mixup_idx][1]] = 1.
            mixup_image =self.dataset[mixup_idx][0]

            # select a random number from the given beta distribution
            # and mixup the images accordingly
            alpha = 0.2
            lamb = np.random.beta(alpha, alpha)
            
            image = lamb * image + (1 - lamb) * mixup_image
            label = lamb * label + (1 - lamb) * mixup_label

        return image, label
