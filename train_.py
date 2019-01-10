from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

class ConvLayer(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

    def forward(self, x): return F.relu(self.conv(x))


class ConvNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.layers = nn.ModuleList([ConvLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)


    def forward(self, x):
        for l in self.layers: x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.relu(self.out(x))

class BnLayer(nn.Module):
    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
                              bias=False, padding=1)
        # self.a = nn.Parameter(torch.zeros(nf,1,1))
        # self.m = nn.Parameter(torch.ones(nf,1,1))
        self.bn = nn.BatchNorm2d(nf)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        # x = self.bn(x)
        # x_chan = x.transpose(0,1).contiguous().view(x.size(1), -1)
        # if self.training:
        #     self.means = x_chan.mean(1)[:,None,None]
        #     self.stds  = x_chan.std (1)[:,None,None]
        # return (x-self.means) / (self.stds+1e-4) *self.m + self.a
        return x

class ConvBnNet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([BnLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2 in zip(self.layers, self.layers2):
            x = l(x)
            x = l2(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        # x = self.out(x)
        # print(x.size())
        return F.softmax(self.out(x), dim=-1)


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    data_dir = 'data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0")


    model = ConvBnNet2([10, 20, 40], 10)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)
    torch.save(model, 'torch_model_.pkl')
