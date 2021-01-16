import numpy as np
import torch
from torchvision import datasets, transforms
import os
import random

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def naiveBaseline():
    print("Calculating naive baseline ...")
    data_dir = 'archive/dice'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloaders['train']:
            images, labels = data
            predicted = torch.tensor([random.randint(0,5), random.randint(0,5), random.randint(0,5), random.randint(0,5)])
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = 100*correct/total
    print("Finished calculating naive baseline!")
    return accuracy

def weakBaseline():
    print("Calculating weak baseline ...")
    data_dir = 'archive/dice'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloaders['train']:
            images, labels = data
            weightedlist = [0]*12 + [1]*12 + [2]*23 + [3]*14 + [4]*28 + [5]*11
            predicted = torch.tensor([random.choice(weightedlist), random.choice(weightedlist), random.choice(weightedlist), random.choice(weightedlist)])
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = 100*correct/total
    print("Finished calculating weak baseline!")
    return accuracy


if __name__ == '__main__':
    naive_Acc = naiveBaseline()
    weak_Acc = weakBaseline()
    print('Accuracy of the naive baseline: %d %%' % naive_Acc)
    print('Accuracy of the weak baseline: %d %%' % weak_Acc)
