import torch
import torchvision

torch.manual_seed(0)

def DataLoader(batch_size=128, train_val_split=0.8):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', download=False, train=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', download=False, train=False, transform=transform_test)

    train_size = int(len(dataset) * train_val_split)
    val_size= len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader